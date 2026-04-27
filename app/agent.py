import json
import logging
from datetime import datetime, timezone

import anthropic
from pydantic import ValidationError

from app.config import settings
from app.prompts import SYSTEM_PROMPT, TOOLS, build_initial_prompt
from app.schemas import ResearchResult
from app.tools.extract_product import ProductPageExtraction
from app.tools.fetch_page import fetch_page
from app.tools.normalize_url import normalize_url
from app.tools.product_identity import ProductIdentity
from app.tools.search_web import search_web

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 10


def _observe_url(url: str, seen_urls: set[str]) -> None:
    seen_urls.add(url)
    try:
        seen_urls.add(normalize_url(url))
    except ValueError:
        pass


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


def _format_fetched_page(page) -> str:
    content_preview = (page.content or "")[:1000]
    return "\n".join([
        f"Title: {page.title or 'N/A'}",
        f"URL: {page.url}",
        f"Price: {page.price_guess or 'N/A'}",
        f"Content:\n{content_preview}",
    ])


async def _execute_tool(tool_name: str, tool_input: dict, seen_urls: set[str]) -> str:
    if tool_name == "search_web":
        query = tool_input.get("query")
        if not query:
            return "Tool error: 'query' is required for search_web."
        try:
            results = await search_web(query)
            for result in results:
                _observe_url(str(result.url), seen_urls)
            return _format_search_results(results)
        except ValueError as exc:
            return f"Search error: {exc}"

    if tool_name == "fetch_page":
        url = tool_input.get("url")
        if not url:
            return "Tool error: 'url' is required for fetch_page."
        try:
            page = await fetch_page(url)
            _observe_url(page.url, seen_urls)
            return _format_fetched_page(page)
        except ValueError as exc:
            return f"Fetch error: {exc}"

    return f"Unknown tool '{tool_name}'."


def _validate_verdict_rules(verdict_input: dict, seen_urls: set[str]) -> None:
    verdict = verdict_input.get("verdict", "")
    evidence = verdict_input.get("evidence", [])
    confidence = verdict_input.get("confidence", "")
    count = len(evidence)

    if count > 5:
        raise ValueError(f"Evidence must contain at most 5 items, got {count}.")

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

    for item in evidence:
        url = str(item.get("source_url") or "")
        if url and not _url_was_observed(url, seen_urls):
            raise ValueError(
                f"Evidence source_url was not observed in tool results: {url}"
            )

    alternative = verdict_input.get("alternative")
    if alternative:
        alt_url = str(alternative.get("source_url") or "")
        if alt_url and not _url_was_observed(alt_url, seen_urls):
            raise ValueError(
                f"Alternative source_url was not observed in tool results: {alt_url}"
            )
        if not alternative.get("is_cheaper") and not alternative.get("is_better_reviewed"):
            raise ValueError(
                "Alternative must have is_cheaper=True or is_better_reviewed=True."
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
        _validate_verdict_rules(verdict_input, seen_urls)
        return ResearchResult(
            product_name=verdict_input["product_name"],
            merchant=verdict_input["merchant"],
            listed_price=verdict_input.get("listed_price"),
            verdict=verdict_input["verdict"],
            confidence=verdict_input["confidence"],
            summary=verdict_input["summary"],
            evidence=verdict_input.get("evidence", []),
            alternative=verdict_input.get("alternative"),
            last_checked=datetime.now(timezone.utc),
        )
    except (ValueError, ValidationError, KeyError) as exc:
        logger.warning(
            "Agent verdict failed validation, falling back to insufficient_data: %s", exc
        )
        return _build_fallback_result(verdict_input, extraction, identity)


async def run_research_agent(
    normalized_url: str,
    extraction: ProductPageExtraction,
    identity: ProductIdentity,
) -> ResearchResult:
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    seen_urls: set[str] = set()
    _observe_url(normalized_url, seen_urls)
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
            raise ValueError("Agent stopped without calling submit_verdict.")

        tool_results = []
        verdict_input = None

        for block in tool_use_blocks:
            if block.name == "submit_verdict":
                verdict_input = block.input
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Verdict received.",
                })
            else:
                result_content = await _execute_tool(block.name, block.input, seen_urls)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                })

        if verdict_input is not None:
            return _build_research_result(verdict_input, seen_urls, extraction, identity)

        messages.append({"role": "user", "content": tool_results})

    raise ValueError(f"Agent exceeded the maximum number of iterations ({_MAX_ITERATIONS}).")
