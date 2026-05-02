import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import anthropic
from pydantic import ValidationError

from app.config import settings
from app.prompts import SYSTEM_PROMPT, TOOLS, build_initial_prompt
from app.schemas import ResearchResult
from app.tools.extract_product import ProductPageExtraction
from app.tools.fetch_page import FetchedPage, fetch_page
from app.tools.normalize_url import fetch_page_cache_key, normalize_url
from app.tools.offer_candidates import (
    OfferCandidate,
    candidate_from_page,
    candidate_from_search_result,
    format_offer_table,
    is_same_size,
    is_unavailable,
    parse_price_amount,
)
from app.tools.product_identity import ProductIdentity
from app.tools.search_web import search_web

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 10


@dataclass
class _RemainingToolBudget:
    searches_remaining: int
    fetches_remaining: int


def _observe_url(url: str, seen_urls: set[str]) -> None:
    seen_urls.add(url)
    try:
        seen_urls.add(normalize_url(url))
    except ValueError:
        pass

def _candidate_key(candidate: OfferCandidate) -> str:
    try:
        return normalize_url(candidate.source_url)
    except ValueError:
        return candidate.source_url.strip()
    
def _add_offer_candidate(
    candidates: list[OfferCandidate],
    candidate: OfferCandidate | None,
) -> None:
    if candidate is None or candidate.price_amount is None:
        return

    new_key = _candidate_key(candidate)

    for index, existing in enumerate(candidates):
        if _candidate_key(existing) != new_key:
            continue

        # Prefer fetched pages over search snippets when both describe the same URL.
        if existing.source_type == "search_result" and candidate.source_type == "fetched_page":
            candidates[index] = candidate
        return

    candidates.append(candidate)


def _url_was_observed(url: str, seen_urls: set[str]) -> bool:
    if url in seen_urls:
        return True
    try:
        return normalize_url(url) in seen_urls
    except ValueError:
        return False


def _format_search_results(results) -> str:
    if not results:
        return "No results found."
    return json.dumps([
        {
            "title": result.title,
            "url": str(result.url),
            "snippet": result.snippet,
            "score": result.metadata.get("score"),
        }
        for result in results
    ])


def _urls_are_equivalent(url_a: str, url_b: str) -> bool:
    """Return True if two URLs resolve to the same resource.

    Uses fetch_page_cache_key so Amazon URL variants with different tracking
    or session params all match when they share the same ASIN.
    """
    if not url_a or not url_b:
        return False
    try:
        return fetch_page_cache_key(url_a) == fetch_page_cache_key(url_b)
    except ValueError:
        return url_a.strip() == url_b.strip()


def _build_eligible_candidates(
    candidates: list[OfferCandidate],
    submitted_price: float,
    identity_value: str,
    seen_urls: set[str],
) -> list[OfferCandidate]:
    eligible = []
    for candidate in candidates:
        if candidate.price_amount is None:
            continue
        if not _url_was_observed(candidate.source_url, seen_urls):
            continue
        savings = (submitted_price - candidate.price_amount) / submitted_price
        if savings < settings.meaningful_savings_threshold:
            continue
        if is_unavailable(candidate):
            continue
        if not is_same_size(identity_value, candidate.product_name, candidate.match_text):
            continue
        eligible.append(candidate)
    eligible.sort(key=lambda c: c.price_amount)
    return eligible


def _find_candidate_by_url(
    url: str, candidates: list[OfferCandidate]
) -> OfferCandidate | None:
    for candidate in candidates:
        if _urls_are_equivalent(url, candidate.source_url):
            return candidate
    return None


def _validate_alternative_is_lowest(
    alternative: dict | None,
    eligible_candidates: list[OfferCandidate],
) -> str | None:
    """Return a rejection message if the alternative is not the lowest eligible offer.

    Checks both price (within 5% tolerance) and source URL against the lowest
    eligible candidate.
    """
    if not eligible_candidates or alternative is None:
        return None
    lowest = eligible_candidates[0]
    alt_price = parse_price_amount(str(alternative.get("price") or ""))
    if alt_price is None:
        return None
    # Price check: allow 5% tolerance for display/rounding differences.
    if alt_price > lowest.price_amount * 1.05:
        return (
            f"Alternative must use the lowest eligible comparable offer. "
            f"{lowest.merchant} at ${lowest.price_amount:.2f} is cheaper than "
            f"the submitted alternative at ${alt_price:.2f}."
        )
    # URL check: the alternative must point to the lowest eligible offer.
    alt_url = str(alternative.get("source_url") or "")
    if not _urls_are_equivalent(alt_url, lowest.source_url):
        return (
            f"Alternative source URL does not match the lowest eligible offer. "
            f"Please use {lowest.merchant} at ${lowest.price_amount:.2f} "
            f"({lowest.source_url}) as the alternative."
        )
    return None


def _format_fetched_page(page) -> str:
    lines = [
        f"Title: {page.title or 'N/A'}",
        f"URL: {page.url}",
        f"Price: {page.price_guess or 'N/A'}",
    ]
    if getattr(page, "lowest_visible_price", None):
        lines.append(f"Lowest visible: {page.lowest_visible_price}")
    candidates = getattr(page, "price_candidates", [])
    labels = getattr(page, "price_candidate_labels", [])
    if candidates:
        parts = []
        for i, p in enumerate(candidates[:5]):
            lbl = labels[i] if i < len(labels) else ""
            parts.append(f"{lbl}: {p}" if lbl else p)
        lines.append(f"Price candidates: {', '.join(parts)}")
    if getattr(page, "shipping_text", None):
        lines.append(f"Shipping: {page.shipping_text}")
    if getattr(page, "listing_count", None):
        lines.append(f"Listings: {page.listing_count}")
    if getattr(page, "condition_text", None):
        lines.append(f"Condition: {page.condition_text}")
    content_preview = (page.content or "")[:1000]
    lines.append(f"Content:\n{content_preview}")
    return "\n".join(lines)


async def _execute_tool(
    tool_name: str,
    tool_input: dict,
    seen_urls: set[str],
    budget: _RemainingToolBudget,
    normalized_submitted_url: str,
    initial_fetched_page: FetchedPage | None,
    candidates: list[OfferCandidate],
) -> str:
    if tool_name == "search_web":
        if budget.searches_remaining <= 0:
            logger.warning("Search budget exhausted (max_searches=%d)", settings.max_searches)
            return (
                "Search budget exhausted. You have reached the maximum number of searches. "
                "Call submit_verdict with the evidence gathered so far."
            )
        # Decrement before executing so malformed or failed calls still consume budget.
        budget.searches_remaining -= 1
        query = tool_input.get("query")
        if not query:
            return "Tool error: 'query' is required for search_web."
        try:
            results = await search_web(query)
            for result in results:
                _observe_url(str(result.url), seen_urls)
                _add_offer_candidate(candidates, candidate_from_search_result(result))
            logger.info(
                "search_web: query=%r results=%d searches_remaining=%d",
                query,
                len(results),
                budget.searches_remaining,
            )
            return _format_search_results(results)
        except ValueError as exc:
            logger.warning("search_web failed: query=%r error=%s", query, exc)
            return f"Search error: {exc}"

    if tool_name == "fetch_page":
        url = tool_input.get("url")
        if not url:
            return "Tool error: 'url' is required for fetch_page."

        if initial_fetched_page is not None:
            requested_cache_key = fetch_page_cache_key(url)
            submitted_cache_key = fetch_page_cache_key(normalized_submitted_url)
            if requested_cache_key == submitted_cache_key:
                logger.info("fetch_page cache hit for initial submitted URL: url=%r", url)
                _observe_url(url, seen_urls)
                _observe_url(initial_fetched_page.url, seen_urls)
                _add_offer_candidate(candidates, candidate_from_page(initial_fetched_page))
                return _format_fetched_page(initial_fetched_page)

        if budget.fetches_remaining <= 0:
            logger.warning(
                "Fetch budget exhausted (max_fetched_pages=%d)", settings.max_fetched_pages
            )
            return (
                "Fetch budget exhausted. You have reached the maximum number of page fetches. "
                "Call submit_verdict with the evidence gathered so far."
            )

        budget.fetches_remaining -= 1
        logger.info("fetch_page: url=%r fetches_remaining=%d", url, budget.fetches_remaining)
        try:
            page = await fetch_page(url)
            _observe_url(page.url, seen_urls)
            _add_offer_candidate(candidates, candidate_from_page(page))
            return _format_fetched_page(page)
        except ValueError as exc:
            logger.warning("fetch_page failed: url=%r error=%s", url, exc)
            return f"Fetch error: {exc}"

    return f"Unknown tool '{tool_name}'."


def _filter_evidence_to_observed(
    evidence: list[dict], seen_urls: set[str]
) -> list[dict]:
    submitted_urls = [str(item.get("source_url") or "") for item in evidence]
    valid = []
    had_drop = False
    for item in evidence:
        url = str(item.get("source_url") or "")
        if url and _url_was_observed(url, seen_urls):
            valid.append(item)
        else:
            had_drop = True
            logger.warning(
                "Evidence item dropped: source_url %r was not observed in tool results", url
            )
    if had_drop:
        logger.warning("Observed URL set at validation: %s", seen_urls)
        logger.warning("Submitted evidence URLs: %s", submitted_urls)
    return valid


def _sanitize_alternative(
    alternative: dict | None, seen_urls: set[str]
) -> dict | None:
    if alternative is None:
        return None

    if not isinstance(alternative, dict):
        logger.warning(
            "Alternative dropped: expected dict but got %s", type(alternative).__name__
        )
        return None

    for field in ("product_name", "price", "reason", "source_url"):
        if not str(alternative.get(field) or "").strip():
            logger.warning("Alternative dropped: missing or blank field %r", field)
            return None

    alt_url = str(alternative["source_url"]).strip()
    if not _url_was_observed(alt_url, seen_urls):
        logger.warning(
            "Alternative dropped: source_url %r was not observed in tool results", alt_url
        )
        return None

    if not alternative.get("is_cheaper") and not alternative.get("is_better_reviewed"):
        logger.warning(
            "Alternative dropped: neither is_cheaper nor is_better_reviewed is true"
        )
        return None

    return alternative


def _validate_verdict_rules(verdict_input: dict, seen_urls: set[str]) -> None:
    verdict = verdict_input.get("verdict", "")
    evidence = verdict_input.get("evidence", [])
    confidence = verdict_input.get("confidence", "")
    count = len(evidence)

    if verdict != "insufficient_data" and count < 3:
        raise ValueError(
            f"Supported verdicts require at least 3 evidence items, got {count}."
        )

    if confidence == "high":
        distinct_urls = {
            item.get("source_url") for item in evidence if item.get("source_url")
        }
        if len(distinct_urls) < 2:
            raise ValueError(
                "High confidence requires at least 2 distinct evidence source URLs."
            )


def _build_timeout_result(
    extraction: ProductPageExtraction,
    identity: ProductIdentity,
) -> ResearchResult:
    identity_name = identity.value if identity.source != "insufficient_data" else None
    product_name = identity_name or extraction.product_name or "unknown"
    merchant = extraction.merchant or "unknown"
    return ResearchResult(
        product_name=product_name,
        merchant=merchant,
        listed_price=extraction.listed_price,
        verdict="failed",
        confidence="low",
        summary=(
            "Research timed out before a verdict could be reached. "
            "Please try again — results may vary based on current page load times."
        ),
        evidence=[],
        alternative=None,
        last_checked=datetime.now(timezone.utc),
    )


def _build_fallback_result(
    verdict_input: dict,
    extraction: ProductPageExtraction,
    identity: ProductIdentity,
) -> ResearchResult:
    identity_name = identity.value if identity.source != "insufficient_data" else None
    product_name = verdict_input.get("product_name") or identity_name or "unknown"
    merchant = verdict_input.get("merchant") or extraction.merchant or "unknown"
    listed_price = verdict_input.get("listed_price") or extraction.listed_price
    return ResearchResult(
        product_name=product_name,
        merchant=merchant,
        listed_price=listed_price,
        verdict="insufficient_data",
        confidence="low",
        summary=(
            "Research could not produce enough validated evidence for a supported verdict."
        ),
        evidence=[],
        alternative=None,
        last_checked=datetime.now(timezone.utc),
    )


def _build_research_result(
    verdict_input: dict,
    seen_urls: set[str],
    extraction: ProductPageExtraction,
    identity: ProductIdentity,
) -> ResearchResult:
    try:
        evidence_truncated = verdict_input.get("evidence", [])[:5]
        valid_evidence = _filter_evidence_to_observed(evidence_truncated, seen_urls)
        valid_alternative = _sanitize_alternative(verdict_input.get("alternative"), seen_urls)
        filtered_verdict = {
            **verdict_input,
            "evidence": valid_evidence,
            "alternative": valid_alternative,
        }
        _validate_verdict_rules(filtered_verdict, seen_urls)
        result = ResearchResult(
            product_name=filtered_verdict["product_name"],
            merchant=filtered_verdict["merchant"],
            listed_price=filtered_verdict.get("listed_price"),
            verdict=filtered_verdict["verdict"],
            confidence=filtered_verdict["confidence"],
            summary=filtered_verdict["summary"],
            evidence=valid_evidence,
            alternative=filtered_verdict.get("alternative"),
            last_checked=datetime.now(timezone.utc),
        )
        logger.info(
            "Verdict accepted: verdict=%s confidence=%s evidence=%d product=%r",
            result.verdict.value,
            result.confidence.value,
            len(result.evidence),
            result.product_name,
        )
        return result
    except (ValueError, ValidationError, KeyError) as exc:
        logger.warning(
            "Agent verdict failed validation, falling back to insufficient_data: %s", exc
        )
        return _build_fallback_result(verdict_input, extraction, identity)


def _evaluate_verdict_submission(
    verdict_input: dict,
    candidates: list[OfferCandidate],
    submitted_price: float | None,
    identity_value: str,
    seen_urls: set[str],
    offer_table_shown: bool,
) -> str:
    """Return the tool result content for a submit_verdict call.

    Returns "Verdict received." when the verdict should be accepted.
    Returns a rejection/table message when the agent must resubmit.
    The offer table is shown at most once (on the first rejection or first
    submission when eligible candidates exist and an alternative is present).
    """
    alternative = verdict_input.get("alternative")

    if alternative is not None:
        alt_url = str(alternative.get("source_url") or "")
        alt_candidate = _find_candidate_by_url(alt_url, candidates)
        if alt_candidate is not None and is_unavailable(alt_candidate):
            unavailable_rejection = (
                "The submitted alternative is sold out or unavailable. "
                "Please omit the alternative or choose a currently available listing."
            )
            return f"{unavailable_rejection}\n\nPlease resubmit your verdict."

    if submitted_price is None:
        return "Verdict received."

    eligible = _build_eligible_candidates(
        candidates, submitted_price, identity_value, seen_urls
    )
    if not eligible:
        return "Verdict received."

    table_text = format_offer_table(eligible)

    if alternative is None:
        lowest = eligible[0]
        missing_alternative_message = (
            f"Eligible cheaper alternatives were found. Please include the lowest eligible "
            f"comparable offer as the alternative: {lowest.merchant} at ${lowest.price_amount:.2f}."
        )
        return f"{table_text}\n\n{missing_alternative_message}\n\nPlease resubmit your verdict."

    rejection = _validate_alternative_is_lowest(alternative, eligible)

    if not offer_table_shown:
        if rejection:
            return f"{table_text}\n\n{rejection}\n\nPlease resubmit your verdict."
        return f"{table_text}\n\nPlease confirm your verdict and resubmit."

    if rejection:
        return f"{table_text}\n\n{rejection}\n\nPlease resubmit your verdict."

    return "Verdict received."


async def _run_agent_loop(
    normalized_url: str,
    extraction: ProductPageExtraction,
    identity: ProductIdentity,
    initial_fetched_page: FetchedPage | None = None,
) -> ResearchResult:
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    seen_urls: set[str] = set()
    _observe_url(normalized_url, seen_urls)
    budget = _RemainingToolBudget(
        searches_remaining=settings.max_searches,
        fetches_remaining=settings.max_fetched_pages,
    )
    candidates: list[OfferCandidate] = []
    submitted_price = parse_price_amount(extraction.listed_price)
    offer_table_shown = False
    logger.info(
        "Agent loop starting: url=%s identity=%r source=%s",
        normalized_url,
        identity.value,
        identity.source,
    )
    messages = [
        {
            "role": "user",
            "content": build_initial_prompt(normalized_url, extraction, identity),
        }
    ]

    for _ in range(_MAX_ITERATIONS):
        response = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_use_blocks = [block for block in response.content if block.type == "tool_use"]

        if not tool_use_blocks:
            logger.warning(
                "Agent stopped without calling submit_verdict, returning insufficient_data"
            )
            return _build_fallback_result({}, extraction, identity)

        tool_results = []
        verdict_input = None

        for block in tool_use_blocks:
            if block.name == "submit_verdict":
                verdict_input = block.input
                logger.info(
                    "submit_verdict received: verdict=%r confidence=%r evidence=%d",
                    verdict_input.get("verdict"),
                    verdict_input.get("confidence"),
                    len(verdict_input.get("evidence", [])),
                )
                # Exclude the submitted product's own URL from comparable offers
                # so the agent is never asked to recommend the same listing as an alternative.
                comparable_candidates = [
                    c for c in candidates
                    if not _urls_are_equivalent(c.source_url, normalized_url)
                ]
                tool_result_content = _evaluate_verdict_submission(
                    verdict_input, comparable_candidates, submitted_price,
                    identity.value, seen_urls, offer_table_shown,
                )
                if tool_result_content != "Verdict received.":
                    offer_table_shown = True
                    verdict_input = None
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_result_content,
                })
            else:
                result_content = await _execute_tool(
                    block.name, block.input, seen_urls, budget,
                    normalized_url, initial_fetched_page, candidates,
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                })

        if verdict_input is not None:
            return _build_research_result(verdict_input, seen_urls, extraction, identity)

        messages.append({"role": "user", "content": tool_results})

    searches_used = settings.max_searches - budget.searches_remaining
    fetches_used = settings.max_fetched_pages - budget.fetches_remaining
    logger.warning(
        "Agent reached maximum iterations (%d), returning insufficient_data "
        "(searches=%d/%d fetches=%d/%d)",
        _MAX_ITERATIONS,
        searches_used,
        settings.max_searches,
        fetches_used,
        settings.max_fetched_pages,
    )
    return _build_fallback_result({}, extraction, identity)


async def run_research_agent(
    normalized_url: str,
    extraction: ProductPageExtraction,
    identity: ProductIdentity,
    initial_fetched_page: FetchedPage | None = None,
) -> ResearchResult:
    try:
        return await asyncio.wait_for(
            _run_agent_loop(normalized_url, extraction, identity, initial_fetched_page),
            timeout=settings.agent_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Agent timed out after %d seconds, returning failed",
            settings.agent_timeout_seconds,
        )
        return _build_timeout_result(extraction, identity)
    except anthropic.APIError as exc:
        logger.warning(
            "Anthropic API error (%s), returning insufficient_data: %s",
            type(exc).__name__,
            exc,
        )
        return _build_fallback_result({}, extraction, identity)
