"""
Dealio eval harness — evaluates agent research quality against curated fixture cases.

Usage:
    python evals/run_dealio_evals.py
    python evals/run_dealio_evals.py --case fair_verdict_brand_page
    python evals/run_dealio_evals.py --case fair_verdict_brand_page clear_good_deal

Requires ANTHROPIC_API_KEY to be set. TAVILY_API_KEY is not needed since tool
calls are mocked per fixture. Each case runs the real LLM against controlled
search results and fetched pages defined in evals/fixtures/<case_id>.json.

This harness is separate from unit/integration tests:
- Unit/integration tests (tests/):  deterministic backend behavior, mocked LLM.
- Mocked evals (this file):         real LLM, mocked tool outputs (repeatable).
- Live smoke tests:                  run_dealio_evals.py --live (not yet implemented).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.agent import run_research_agent  # noqa: E402
from app.tools.extract_product import ProductPageExtraction  # noqa: E402
from app.tools.fetch_page import FetchedPage  # noqa: E402
from app.tools.offer_candidates import parse_price_amount  # noqa: E402
from app.tools.product_identity import ProductIdentity  # noqa: E402
from app.tools.search_web import SearchResult  # noqa: E402

EVALS_DIR = Path(__file__).parent
FIXTURES_DIR = EVALS_DIR / "fixtures"
CASES_FILE = EVALS_DIR / "cases.json"


def load_cases() -> list[dict]:
    with open(CASES_FILE) as f:
        return json.load(f)


def load_fixture(case_id: str) -> dict | None:
    fixture_path = FIXTURES_DIR / f"{case_id}.json"
    if not fixture_path.exists():
        return None
    with open(fixture_path) as f:
        return json.load(f)


def build_fetched_page(data: dict) -> FetchedPage:
    return FetchedPage(
        url=data["url"],
        title=data.get("title"),
        content=data.get("content", ""),
        price_guess=data.get("price_guess"),
    )


def build_search_result(data: dict) -> SearchResult:
    return SearchResult(
        title=data["title"],
        url=data["url"],
        snippet=data["snippet"],
        metadata={"source": "fixture", "score": data.get("score", 0.9)},
    )


def check_assertions(result, expected: dict, observed_urls: set[str]) -> list[str]:
    failures = []

    if "verdict_any_of" in expected:
        allowed = expected["verdict_any_of"]
        if result.verdict.value not in allowed:
            failures.append(
                f"Verdict '{result.verdict.value}' not in allowed set {allowed}"
            )

    if "verdict_not" in expected:
        forbidden = expected["verdict_not"]
        if isinstance(forbidden, str):
            forbidden = [forbidden]
        if result.verdict.value in forbidden:
            failures.append(
                f"Verdict '{result.verdict.value}' is in forbidden set {forbidden}"
            )

    if "confidence_any_of" in expected:
        allowed = expected["confidence_any_of"]
        if result.confidence.value not in allowed:
            failures.append(
                f"Confidence '{result.confidence.value}' not in allowed set {allowed}"
            )

    if "confidence_not" in expected:
        forbidden = expected["confidence_not"]
        if isinstance(forbidden, str):
            forbidden = [forbidden]
        if result.confidence.value in forbidden:
            failures.append(
                f"Confidence '{result.confidence.value}' is in forbidden set {forbidden}"
            )

    if "min_evidence_count" in expected:
        count = len(result.evidence)
        minimum = expected["min_evidence_count"]
        if count < minimum:
            failures.append(
                f"Evidence count {count} is below minimum {minimum}"
            )

    if expected.get("must_have_alternative"):
        if result.alternative is None:
            failures.append("Expected an alternative but got none")

    if expected.get("must_not_have_alternative"):
        if result.alternative is not None:
            failures.append(
                f"Expected no alternative but got: {result.alternative.source_url}"
            )

    if expected.get("alternative_must_be_cheaper"):
        if result.alternative is not None and not result.alternative.is_cheaper:
            failures.append("Alternative has is_cheaper=False")

    if "alternative_price_at_most" in expected:
        threshold = expected["alternative_price_at_most"]
        if result.alternative is None:
            failures.append(
                f"alternative_price_at_most={threshold} requires an alternative, but got none"
            )
        else:
            actual_price = parse_price_amount(result.alternative.price)
            if actual_price is None:
                failures.append(
                    f"Alternative price '{result.alternative.price}' could not be parsed"
                )
            elif actual_price > threshold:
                failures.append(
                    f"Alternative price ${actual_price:.2f} exceeds max ${threshold:.2f}"
                )

    if "alternative_url_any_of" in expected:
        allowed_urls = expected["alternative_url_any_of"]
        if result.alternative is None:
            failures.append(
                "alternative_url_any_of requires an alternative, but got none"
            )
        else:
            alt_url = str(result.alternative.source_url)
            if alt_url not in allowed_urls:
                failures.append(
                    f"Alternative URL '{alt_url}' not in allowed set {allowed_urls}"
                )

    if expected.get("all_evidence_urls_observed"):
        for item in result.evidence:
            url = str(item.source_url)
            if url not in observed_urls:
                failures.append(
                    f"Evidence URL '{url}' was not in the set of observed tool-call URLs"
                )

    return failures


async def run_case(case: dict, fixture: dict) -> dict:
    extraction = ProductPageExtraction(**fixture["extraction"])
    identity = ProductIdentity(**fixture["identity"])

    submitted_page_data = fixture.get("submitted_page")
    initial_page = build_fetched_page(submitted_page_data) if submitted_page_data else None

    search_results = [build_search_result(r) for r in fixture.get("search_results", [])]

    fetched_pages: dict[str, FetchedPage] = {
        url: build_fetched_page(data)
        for url, data in fixture.get("fetched_pages", {}).items()
    }

    observed_urls: set[str] = {case["input_url"]}
    if initial_page:
        observed_urls.add(initial_page.url)

    async def fake_search(query: str) -> list[SearchResult]:
        for result in search_results:
            observed_urls.add(str(result.url))
        return search_results

    async def fake_fetch(url: str) -> FetchedPage:
        observed_urls.add(url)
        if url in fetched_pages:
            observed_urls.add(fetched_pages[url].url)
            return fetched_pages[url]
        for known_url, page in fetched_pages.items():
            if known_url in url or url in known_url:
                observed_urls.add(page.url)
                return page
        raise ValueError(f"No fixture data for URL: {url}")

    with (
        patch("app.agent.search_web", new_callable=AsyncMock) as mock_search,
        patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch,
    ):
        mock_search.side_effect = fake_search
        mock_fetch.side_effect = fake_fetch

        result = await run_research_agent(
            case["input_url"],
            extraction,
            identity,
            initial_fetched_page=initial_page,
        )

    failures = check_assertions(result, case.get("expected", {}), observed_urls)

    return {
        "case_id": case["id"],
        "description": case.get("description", ""),
        "passed": len(failures) == 0,
        "failures": failures,
        "verdict": result.verdict.value,
        "confidence": result.confidence.value,
        "evidence_count": len(result.evidence),
        "has_alternative": result.alternative is not None,
        "alternative_price": result.alternative.price if result.alternative else None,
        "alternative_url": str(result.alternative.source_url) if result.alternative else None,
    }


def print_case_result(result: dict) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    print(f"  [{status}] {result['case_id']}")
    verdict = result["verdict"]
    confidence = result["confidence"]
    count = result["evidence_count"]
    print(f"         verdict={verdict}  confidence={confidence}  evidence={count}")
    if result["has_alternative"]:
        print(f"         alternative: {result['alternative_price']} @ {result['alternative_url']}")
    for failure in result["failures"]:
        print(f"         FAILURE: {failure}")


async def main(selected_case_ids: list[str] | None) -> int:
    cases = load_cases()

    if selected_case_ids:
        available_ids = {c["id"] for c in cases}
        for case_id in selected_case_ids:
            if case_id not in available_ids:
                print(f"Error: unknown case '{case_id}'. Available: {sorted(available_ids)}")
                return 1
        cases = [c for c in cases if c["id"] in selected_case_ids]

    passed = 0
    failed = 0
    skipped = 0

    print(f"Running {len(cases)} eval case(s)...\n")

    for case in cases:
        case_id = case["id"]
        fixture = load_fixture(case_id)

        if fixture is None:
            print(f"  [SKIP] {case_id}  (no fixture file found at fixtures/{case_id}.json)")
            skipped += 1
            continue

        print(f"  Running: {case_id}...")
        result = await run_case(case, fixture)
        print_case_result(result)
        print()

        if result["passed"]:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print(f"Results: {passed}/{total} passed", end="")
    if skipped:
        print(f"  ({skipped} skipped)", end="")
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dealio agent eval cases")
    parser.add_argument(
        "--case",
        nargs="+",
        metavar="CASE_ID",
        help="Run only the specified case(s). Omit to run all cases.",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(main(args.case))
    sys.exit(exit_code)
