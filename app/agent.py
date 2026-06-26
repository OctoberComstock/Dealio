import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

import anthropic
from pydantic import ValidationError

from app.config import settings
from app.prompts import SYSTEM_PROMPT, TOOLS, build_initial_prompt
from app.schemas import ComparableOffer, ResearchResult, Verdict
from app.services.offer_validation import (
    alternative_eligible_offers,
    current_market_offers,
    validate_offer_candidates,
)
from app.tools.extract_product import ProductPageExtraction
from app.tools.fetch_page import FetchedPage, fetch_page
from app.tools.normalize_url import fetch_page_cache_key, normalize_url
from app.tools.offer_candidates import (
    OfferCandidate,
    assess_product_identity,
    candidate_from_page,
    candidate_from_search_result,
    is_accessory_or_partial_listing,
    is_installment_price,
    is_purchasable_offer_candidate,
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


def _mark_search_candidate_fetch_failed(
    candidates: list[OfferCandidate],
    url: str,
) -> None:
    for candidate in candidates:
        if candidate.source_type != "search_result":
            continue
        if _urls_are_equivalent(candidate.source_url, url):
            candidate.verification_status = "fetch_failed"


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
    verified_candidates = []
    search_candidates = []

    def exclude(candidate: OfferCandidate, reason: str, identity_reason: str = "") -> None:
        logger.info(
            "Offer candidate excluded: url=%s title=%r price=%s source_type=%s "
            "verification=%s identity=%r reason=%s",
            candidate.source_url,
            candidate.product_name,
            candidate.price_amount,
            candidate.source_type,
            candidate.verification_status,
            identity_reason,
            reason,
        )

    for candidate in candidates:
        if candidate.price_amount is None:
            exclude(candidate, "missing parseable price")
            continue
        if not _url_was_observed(candidate.source_url, seen_urls):
            exclude(candidate, "URL was not observed")
            continue
        savings = (submitted_price - candidate.price_amount) / submitted_price
        if savings < settings.meaningful_savings_threshold:
            exclude(candidate, "savings below threshold")
            continue
        if is_unavailable(candidate):
            exclude(candidate, "listing unavailable")
            continue
        if not is_same_size(identity_value, candidate.product_name, candidate.match_text):
            exclude(candidate, "size mismatch")
            continue
        if not is_purchasable_offer_candidate(candidate):
            exclude(candidate, "not a concrete purchasable offer")
            continue
        if is_accessory_or_partial_listing(candidate):
            exclude(candidate, "accessory, replacement part, or partial product")
            continue
        if is_installment_price(candidate):
            exclude(candidate, "installment or payment amount")
            continue

        identity_match = assess_product_identity(
            identity_value,
            candidate.product_name,
            candidate.match_text,
        )
        if not identity_match.matches:
            exclude(candidate, "product identity mismatch", identity_match.reason)
            continue
        if candidate.source_type == "search_result" and not identity_match.is_strong:
            exclude(candidate, "search result lacks strong identity", identity_match.reason)
            continue

        logger.info(
            "Offer candidate eligible: url=%s title=%r price=%s source_type=%s "
            "verification=%s identity=%r",
            candidate.source_url,
            candidate.product_name,
            candidate.price_amount,
            candidate.source_type,
            candidate.verification_status,
            identity_match.reason,
        )
        if candidate.verification_status == "verified":
            verified_candidates.append(candidate)
        else:
            search_candidates.append(candidate)

    if verified_candidates:
        for candidate in search_candidates:
            exclude(candidate, "verified candidate precedence")
        eligible = verified_candidates
    else:
        eligible = search_candidates

    eligible.sort(key=lambda candidate: candidate.price_amount)
    return eligible


def _format_validated_offer_table(offers: list[ComparableOffer]) -> str:
    if not offers:
        return ""

    lines = ["Comparable offers found (sorted by delivered price):"]
    for index, offer in enumerate(offers, 1):
        comparison_price = offer.delivered_price or offer.current_item_price
        price_text = f"${comparison_price:.2f}" if comparison_price is not None else "price unknown"
        shipping_text = "shipping unknown"
        if offer.shipping_price is not None:
            shipping_text = f"shipping ${offer.shipping_price:.2f}"
        lines.append(
            f"{index}. {offer.merchant} — {price_text} delivered ({shipping_text}) — "
            f"{offer.product_name} ({offer.source_type})"
        )
    lines.append(
        "If recommending an alternative, use the lowest eligible offer from this list."
    )
    return "\n".join(lines)


def _find_candidate_by_url(
    url: str, candidates: list[OfferCandidate]
) -> OfferCandidate | None:
    for candidate in candidates:
        if _urls_are_equivalent(url, candidate.source_url):
            return candidate
    return None


def _validate_alternative_is_lowest(
    alternative: dict | None,
    eligible_offers: list[ComparableOffer | OfferCandidate],
) -> str | None:
    """Return a rejection message if the alternative is not the lowest eligible offer.

    Checks both price (within 5% tolerance) and source URL against the lowest
    eligible candidate.
    """
    if not eligible_offers or alternative is None:
        return None
    lowest = eligible_offers[0]
    alt_price = parse_price_amount(str(alternative.get("price") or ""))
    if alt_price is None:
        return None
    lowest_price = _offer_comparison_price(lowest)
    if lowest_price is None:
        return None
    # Price check: allow 5% tolerance for display/rounding differences.
    if Decimal(str(alt_price)) > lowest_price * Decimal("1.05"):
        return (
            f"Alternative must use the lowest eligible comparable offer. "
            f"{lowest.merchant} at ${lowest_price:.2f} delivered is cheaper than "
            f"the submitted alternative at ${alt_price:.2f}."
        )
    # URL check: the alternative must point to the lowest eligible offer.
    alt_url = str(alternative.get("source_url") or "")
    lowest_url = _offer_source_url(lowest)
    if not _urls_are_equivalent(alt_url, lowest_url):
        return (
            f"Alternative source URL does not match the lowest eligible offer. "
            f"Please use {lowest.merchant} at ${lowest_price:.2f} delivered "
            f"({lowest_url}) as the alternative."
        )
    return None


def _offer_comparison_price(offer: ComparableOffer | OfferCandidate) -> Decimal | None:
    if isinstance(offer, ComparableOffer):
        return offer.delivered_price or offer.current_item_price
    if offer.price_amount is None:
        return None
    return Decimal(str(offer.price_amount)).quantize(Decimal("0.01"))


def _offer_source_url(offer: ComparableOffer | OfferCandidate) -> str:
    return str(offer.source_url)


def _market_verdict_from_validated_offers(
    submitted_price: float | None,
    offers: list[ComparableOffer],
) -> Verdict | None:
    if submitted_price is None:
        return None

    market_offers = current_market_offers(offers)
    delivered_prices = [
        offer.delivered_price
        for offer in market_offers
        if offer.delivered_price is not None
    ]
    if len(delivered_prices) < 2:
        return None

    ordered_prices = sorted(delivered_prices)
    midpoint = len(ordered_prices) // 2
    if len(ordered_prices) % 2 == 1:
        typical_price = ordered_prices[midpoint]
    else:
        typical_price = (ordered_prices[midpoint - 1] + ordered_prices[midpoint]) / Decimal("2")

    submitted = Decimal(str(submitted_price)).quantize(Decimal("0.01"))
    threshold = Decimal(str(settings.meaningful_savings_threshold))
    savings = (typical_price - submitted) / typical_price
    premium = (submitted - typical_price) / typical_price

    if savings >= threshold:
        return Verdict.good_deal
    if premium >= threshold:
        return Verdict.overpriced
    return Verdict.fair


def _summary_with_validated_context(
    original_summary: str,
    submitted_price: float | None,
    offers: list[ComparableOffer],
    verdict: Verdict,
) -> str:
    if verdict != Verdict.fair or submitted_price is None:
        return original_summary

    has_msrp_context = any(offer.price_classification.value == "msrp" for offer in offers)
    current_offer_count = len([offer for offer in offers if offer.delivered_price is not None])
    has_unknown_shipping = any(
        offer.current_item_price is not None and offer.shipping_price is None
        for offer in offers
    )

    if not has_msrp_context:
        return original_summary

    context = (
        " This is a fair current-market price and a verified discount from MSRP, "
        "but the discount is not unique when comparable current offers are similar."
    )
    if current_offer_count > 0 and has_unknown_shipping:
        context += " Some competing shipping costs were not fully verified."
    return f"{original_summary.rstrip()}{context}"


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
            page = await fetch_page(
                url,
                request_timeout_seconds=settings.supporting_page_timeout_seconds,
            )
            _observe_url(page.url, seen_urls)
            _add_offer_candidate(candidates, candidate_from_page(page))
            return _format_fetched_page(page)
        except ValueError as exc:
            _mark_search_candidate_fetch_failed(candidates, url)
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
    comparable_offers: list[ComparableOffer] | None = None,
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
        submitted_price = parse_price_amount(extraction.listed_price)
        validated_verdict = _market_verdict_from_validated_offers(
            submitted_price,
            comparable_offers or [],
        )
        verdict = validated_verdict or filtered_verdict["verdict"]
        summary = _summary_with_validated_context(
            filtered_verdict["summary"],
            submitted_price,
            comparable_offers or [],
            verdict,
        )
        result = ResearchResult(
            product_name=filtered_verdict["product_name"],
            merchant=filtered_verdict["merchant"],
            listed_price=filtered_verdict.get("listed_price"),
            verdict=verdict,
            confidence=filtered_verdict["confidence"],
            summary=summary,
            evidence=valid_evidence,
            alternative=filtered_verdict.get("alternative"),
            comparable_offers=comparable_offers or [],
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
) -> str:
    """Return the tool result content for a submit_verdict call.

    Returns "Verdict received." when the verdict should be accepted.
    Returns a rejection/table message when the agent must resubmit.
    """
    def reject(reason: str, table_text: str | None = None) -> str:
        logger.warning("Verdict submission rejected: %s", reason)
        parts = [part for part in (table_text, reason, "Please resubmit your verdict.") if part]
        return "\n\n".join(parts)

    alternative = verdict_input.get("alternative")

    validated_offers = validate_offer_candidates(candidates, identity_value)
    validated_alt_offer = None
    if alternative is not None:
        alt_url = str(alternative.get("source_url") or "")
        alt_candidate = _find_candidate_by_url(alt_url, candidates)
        for offer in validated_offers:
            if _urls_are_equivalent(alt_url, str(offer.source_url)):
                validated_alt_offer = offer
                break
        if alt_candidate is not None and is_unavailable(alt_candidate):
            unavailable_rejection = (
                "The submitted alternative is sold out or unavailable. "
                "Please omit the alternative or choose a currently available listing."
            )
            return reject(unavailable_rejection)
        if alt_candidate is not None and not is_purchasable_offer_candidate(alt_candidate):
            return reject(
                "The submitted alternative appears to be an evidence-only page or "
                "market-summary page, not a concrete purchasable listing. "
                "Please omit the alternative or choose a specific purchasable product/listing page."
            )
        if alt_candidate is not None and is_accessory_or_partial_listing(alt_candidate):
            return reject(
                "The submitted alternative appears to be an accessory, replacement part, "
                "or partial product. Please omit it or choose the complete product."
            )
        if alt_candidate is not None and is_installment_price(alt_candidate):
            return reject(
                "The submitted alternative price appears to be an installment or payment "
                "amount rather than the full product price."
            )
        if alt_candidate is not None:
            identity_match = assess_product_identity(
                identity_value,
                alt_candidate.product_name,
                alt_candidate.match_text,
            )
            if not identity_match.matches:
                return reject(
                    "The submitted alternative does not have sufficient product identity "
                    "overlap with the submitted product."
                )
        if (
            alt_candidate is not None
            and submitted_price is not None
            and alt_candidate.price_amount is not None
            and alternative.get("is_cheaper")
        ):
            savings = (submitted_price - alt_candidate.price_amount) / submitted_price
            if savings < settings.meaningful_savings_threshold:
                return reject(
                    f"The submitted alternative is not meaningfully cheaper "
                    f"({savings:.1%} savings is below the "
                    f"{settings.meaningful_savings_threshold:.0%} threshold). "
                    "Please omit the alternative or find a substantially cheaper offer."
                )
            alternative_survives_shipping_uncertainty = (
                alternative_eligible_offers(
                    [validated_alt_offer],
                    Decimal(str(submitted_price)).quantize(Decimal("0.01")),
                    settings.meaningful_savings_threshold,
                )
                if validated_alt_offer is not None
                else []
            )
            if (
                validated_alt_offer is not None
                and validated_alt_offer.shipping_price is None
                and not alternative_survives_shipping_uncertainty
            ):
                return reject(
                    "The submitted alternative has unknown shipping, so the claimed savings "
                    "could disappear at checkout. Please omit it or verify delivered cost."
                )

    if submitted_price is None:
        return "Verdict received."

    eligible = alternative_eligible_offers(
        validated_offers,
        Decimal(str(submitted_price)).quantize(Decimal("0.01")),
        settings.meaningful_savings_threshold,
    )
    if not eligible:
        return "Verdict received."

    table_text = _format_validated_offer_table(eligible)

    if alternative is None:
        lowest = eligible[0]
        lowest_price = lowest.delivered_price or lowest.current_item_price
        missing_alternative_message = (
            f"Eligible cheaper alternatives were found. Please include the lowest eligible "
            f"comparable offer as the alternative: {lowest.merchant} at "
            f"${lowest_price:.2f} delivered."
        )
        return reject(missing_alternative_message, table_text)

    rejection = _validate_alternative_is_lowest(alternative, eligible)
    if rejection:
        return reject(rejection, table_text)

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
                    identity.value, seen_urls,
                )
                if tool_result_content != "Verdict received.":
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
            comparable_candidates = [
                candidate for candidate in candidates
                if not _urls_are_equivalent(candidate.source_url, normalized_url)
            ]
            comparable_offers = validate_offer_candidates(
                comparable_candidates,
                identity.value,
            )
            return _build_research_result(
                verdict_input,
                seen_urls,
                extraction,
                identity,
                comparable_offers,
            )

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
