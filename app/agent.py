import json
from datetime import datetime, timezone

import anthropic
from pydantic import ValidationError

from app.config import settings
from app.prompts import SYSTEM_PROMPT, TOOLS, build_initial_prompt
from app.schemas import ResearchResult
from app.tools.extract_product import ProductPageExtraction
from app.tools.fetch_page import fetch_page
from app.tools.product_identity import ProductIdentity
from app.tools.search_web import search_web

_MAX_ITERATIONS = 10


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
                seen_urls.add(str(result.url))
            return _format_search_results(results)
        except ValueError as exc:
            return f"Search error: {exc}"

    if tool_name == "fetch_page":
        url = tool_input.get("url")
        if not url:
            return "Tool error: 'url' is required for fetch_page."
        try:
            page = await fetch_page(url)
            seen_urls.add(page.url)
            return _format_fetched_page(page)
        except ValueError as exc:
            return f"Fetch error: {exc}"

    return f"Unknown tool '{tool_name}'."


def _validate_evidence_urls(verdict_input: dict, seen_urls: set[str]) -> None:
    for item in verdict_input.get("evidence", []):
        url = str(item.get("source_url") or "")
        if url and url not in seen_urls:
            raise ValueError(
                f"Evidence source_url was not observed in tool results: {url}"
            )
    alternative = verdict_input.get("alternative")
    if alternative:
        alt_url = str(alternative.get("source_url") or "")
        if alt_url and alt_url not in seen_urls:
            raise ValueError(
                f"Alternative source_url was not observed in tool results: {alt_url}"
            )


def _build_research_result(verdict_input: dict, seen_urls: set[str]) -> ResearchResult:
    _validate_evidence_urls(verdict_input, seen_urls)
    try:
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
    except (ValidationError, KeyError) as exc:
        raise ValueError(f"Invalid verdict payload from agent: {exc}") from exc


async def run_research_agent(
    normalized_url: str,
    extraction: ProductPageExtraction,
    identity: ProductIdentity,
) -> ResearchResult:
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    seen_urls: set[str] = {normalized_url}
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
            return _build_research_result(verdict_input, seen_urls)

        messages.append({"role": "user", "content": tool_results})

    raise ValueError(f"Agent exceeded the maximum number of iterations ({_MAX_ITERATIONS}).")
