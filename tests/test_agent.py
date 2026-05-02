import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import pytest

from app.agent import run_research_agent
from app.prompts import SYSTEM_PROMPT
from app.schemas import ResearchResult
from app.tools.extract_product import ProductPageExtraction
from app.tools.fetch_page import FetchedPage
from app.tools.product_identity import ProductIdentity
from app.tools.search_web import SearchResult


def make_tool_block(name: str, input_data: dict, block_id: str = "block_1"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = block_id
    block.input = input_data
    return block


def make_response(*blocks):
    response = MagicMock()
    response.content = list(blocks)
    return response

def tool_result_contents(mock_create: AsyncMock) -> list[str]:
    contents: list[str] = []

    for call in mock_create.call_args_list:
        messages = call.kwargs["messages"]

        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue

            content = message.get("content")
            if not isinstance(content, list):
                continue

            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "tool_result":
                    continue
                result_content = item.get("content")
                if isinstance(result_content, str):
                    contents.append(result_content)

    return contents


VALID_VERDICT = {
    "product_name": "Great Widget",
    "merchant": "example.com",
    "listed_price": "$29.99",
    "verdict": "good_deal",
    "confidence": "medium",
    "summary": "Competitively priced across comparable listings.",
    "evidence": [
        {"text": "Listed at $29.99, below market average.", "source_url": "https://example.com/product"},
        {"text": "Comparable models sell for $35–$40.", "source_url": "https://example.com/product"},
        {"text": "Customer reviews are consistently positive.", "source_url": "https://example.com/product"},
    ],
    "alternative": None,
}


@pytest.fixture
def extraction():
    return ProductPageExtraction(
        product_name="Great Widget",
        listed_price="$29.99",
        merchant="example.com",
    )


@pytest.fixture
def identity():
    return ProductIdentity(value="Great Widget", source="product_name")


@pytest.fixture
def mock_create():
    with patch("app.agent.anthropic.AsyncAnthropic") as mock_cls:
        mock_instance = MagicMock()
        create = AsyncMock()
        mock_instance.messages.create = create
        mock_cls.return_value = mock_instance
        yield create


# --- Agent behavior tests ---


async def test_agent_sends_initial_product_context(mock_create, extraction, identity):
    mock_create.return_value = make_response(make_tool_block("submit_verdict", VALID_VERDICT))
    await run_research_agent("https://example.com/product", extraction, identity)
    messages = mock_create.call_args.kwargs["messages"]
    content = messages[0]["content"]
    assert messages[0]["role"] == "user"
    assert "https://example.com/product" in content
    assert "Great Widget" in content
    assert "$29.99" in content
    assert "example.com" in content
    assert "extracted from the product page" in content


async def test_agent_terminates_on_submit_verdict(mock_create, extraction, identity):
    mock_create.return_value = make_response(make_tool_block("submit_verdict", VALID_VERDICT))
    await run_research_agent("https://example.com/product", extraction, identity)
    assert mock_create.call_count == 1


async def test_submit_verdict_validates_into_research_result(mock_create, extraction, identity):
    mock_create.return_value = make_response(make_tool_block("submit_verdict", VALID_VERDICT))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert isinstance(result, ResearchResult)
    assert result.verdict.value == "good_deal"
    assert result.confidence.value == "medium"
    assert result.product_name == "Great Widget"


async def test_agent_executes_search_web_tool(mock_create, extraction, identity):
    search_result = SearchResult(
        title="Widget Review",
        url="https://reviews.example.com/widget",
        snippet="Great widget at a fair price.",
        metadata={"source": "tavily", "score": 0.9},
    )
    mock_create.side_effect = [
        make_response(make_tool_block("search_web", {"query": "Great Widget price"}, "b1")),
        make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b2")),
    ]
    with patch("app.agent.search_web", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [search_result]
        result = await run_research_agent("https://example.com/product", extraction, identity)
    mock_search.assert_called_once_with("Great Widget price")
    assert isinstance(result, ResearchResult)


async def test_agent_executes_fetch_page_tool(mock_create, extraction, identity):
    fetched = FetchedPage(
        url="https://reviews.example.com/widget",
        title="Widget Review",
        content="This widget is excellent.",
        price_guess="$29.99",
    )
    mock_create.side_effect = [
        make_response(
            make_tool_block(
                "fetch_page",
                {"url": "https://reviews.example.com/widget"},
                "b1",
            )
        ),
        make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b2")),
    ]
    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = fetched
        result = await run_research_agent("https://example.com/product", extraction, identity)
    mock_fetch.assert_called_once_with("https://reviews.example.com/widget")
    assert isinstance(result, ResearchResult)


async def test_fetch_page_cache_hit_for_submitted_url_skips_network_call(
    mock_create, extraction, identity
):
    cached_page = FetchedPage(
        url="https://example.com/product",
        title="Great Widget",
        content="Product content.",
        price_guess="$29.99",
    )
    mock_create.side_effect = [
        make_response(
            make_tool_block("fetch_page", {"url": "https://example.com/product"}, "b1")
        ),
        make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b2")),
    ]
    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        await run_research_agent(
            "https://example.com/product", extraction, identity,
            initial_fetched_page=cached_page,
        )
    mock_fetch.assert_not_called()


async def test_fetch_page_cache_hit_returns_cached_content(mock_create, extraction, identity):
    cached_page = FetchedPage(
        url="https://example.com/product",
        title="Cached Widget Title",
        content="Cached content here.",
        price_guess="$29.99",
    )
    mock_create.side_effect = [
        make_response(
            make_tool_block("fetch_page", {"url": "https://example.com/product"}, "b1")
        ),
        make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b2")),
    ]
    with patch("app.agent.fetch_page", new_callable=AsyncMock):
        await run_research_agent(
            "https://example.com/product", extraction, identity,
            initial_fetched_page=cached_page,
        )
    messages = mock_create.call_args_list[1].kwargs["messages"]
    tool_result_content = messages[-2]["content"][0]["content"]
    assert "Cached Widget Title" in tool_result_content


async def test_fetch_page_cache_hit_does_not_decrement_budget(mock_create, extraction, identity):
    cached_page = FetchedPage(
        url="https://example.com/product",
        title="Great Widget",
        content="Product content.",
        price_guess="$29.99",
    )
    other_page = FetchedPage(
        url="https://other.example.com/page",
        title="Other Page",
        content="Other content.",
        price_guess=None,
    )
    mock_create.side_effect = [
        make_response(
            make_tool_block("fetch_page", {"url": "https://example.com/product"}, "b1")
        ),
        make_response(
            make_tool_block("fetch_page", {"url": "https://other.example.com/page"}, "b2")
        ),
        make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b3")),
    ]
    with patch("app.agent.settings") as mock_settings:
        mock_settings.max_searches = 8
        mock_settings.max_fetched_pages = 1
        mock_settings.agent_timeout_seconds = 120
        with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = other_page
            result = await run_research_agent(
                "https://example.com/product", extraction, identity,
                initial_fetched_page=cached_page,
            )
    # Cache hit doesn't consume the one fetch slot, so the real fetch still executes.
    mock_fetch.assert_called_once_with("https://other.example.com/page")
    assert isinstance(result, ResearchResult)


async def test_fetch_page_cache_hit_with_tracking_param_variant_matches(
    mock_create, extraction, identity
):
    cached_page = FetchedPage(
        url="https://example.com/product",
        title="Great Widget",
        content="Product content.",
        price_guess="$29.99",
    )
    mock_create.side_effect = [
        make_response(
            make_tool_block(
                "fetch_page",
                {"url": "https://example.com/product?utm_source=google"},
                "b1",
            )
        ),
        make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b2")),
    ]
    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        await run_research_agent(
            "https://example.com/product", extraction, identity,
            initial_fetched_page=cached_page,
        )
    mock_fetch.assert_not_called()


async def test_fetch_page_cache_hit_for_amazon_url_with_extra_tracking_params(
    mock_create, extraction, identity
):
    # Submitted URL has Amazon session/detail-page params; agent requests the same
    # ASIN with only ?th=1. Both should resolve to the same cache key.
    cached_page = FetchedPage(
        url="https://www.amazon.com/Product-Name/dp/B08W2HG4WP?th=1",
        title="Haruharu Cleanser",
        content="Product content.",
        price_guess="$18.99",
    )
    submitted_url = (
        "https://www.amazon.com/Product-Name/dp/B08W2HG4WP"
        "?_encoding=UTF8&pd_rd_r=abc&pd_rd_w=def&pf_rd_p=ghi&pf_rd_r=jkl"
    )
    agent_requested_url = "https://www.amazon.com/Haruharu/dp/B08W2HG4WP?th=1"
    mock_create.side_effect = [
        make_response(
            make_tool_block("fetch_page", {"url": agent_requested_url}, "b1")
        ),
        make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b2")),
    ]
    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        await run_research_agent(
            submitted_url, extraction, identity,
            initial_fetched_page=cached_page,
        )
    mock_fetch.assert_not_called()


async def test_fetch_page_cache_miss_for_different_url_executes_normally(
    mock_create, extraction, identity
):
    cached_page = FetchedPage(
        url="https://example.com/product",
        title="Great Widget",
        content="Product content.",
        price_guess="$29.99",
    )
    other_page = FetchedPage(
        url="https://other.example.com/page",
        title="Other Page",
        content="Other content.",
        price_guess=None,
    )
    mock_create.side_effect = [
        make_response(
            make_tool_block("fetch_page", {"url": "https://other.example.com/page"}, "b1")
        ),
        make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b2")),
    ]
    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = other_page
        await run_research_agent(
            "https://example.com/product", extraction, identity,
            initial_fetched_page=cached_page,
        )
    mock_fetch.assert_called_once_with("https://other.example.com/page")


async def test_fetch_page_cache_hit_logs_cache_hit_message(
    mock_create, extraction, identity, caplog
):
    cached_page = FetchedPage(
        url="https://example.com/product",
        title="Great Widget",
        content="Product content.",
        price_guess="$29.99",
    )
    mock_create.side_effect = [
        make_response(
            make_tool_block("fetch_page", {"url": "https://example.com/product"}, "b1")
        ),
        make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b2")),
    ]
    with patch("app.agent.fetch_page", new_callable=AsyncMock):
        with caplog.at_level(logging.INFO, logger="app.agent"):
            await run_research_agent(
                "https://example.com/product", extraction, identity,
                initial_fetched_page=cached_page,
            )
    messages = [r.message for r in caplog.records]
    assert any("cache hit" in m.lower() for m in messages)


async def test_fetch_page_with_no_initial_page_executes_normally(mock_create, extraction, identity):
    fetched = FetchedPage(
        url="https://example.com/product",
        title="Great Widget",
        content="Product content.",
        price_guess="$29.99",
    )
    mock_create.side_effect = [
        make_response(
            make_tool_block("fetch_page", {"url": "https://example.com/product"}, "b1")
        ),
        make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b2")),
    ]
    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = fetched
        await run_research_agent(
            "https://example.com/product", extraction, identity,
            initial_fetched_page=None,
        )
    mock_fetch.assert_called_once_with("https://example.com/product")


async def test_fetch_page_cache_hit_observes_both_requested_and_cached_url(
    mock_create, extraction, identity
):
    cached_page = FetchedPage(
        url="https://example.com/product",
        title="Great Widget",
        content="Product content.",
        price_guess="$29.99",
    )
    # Agent requests the URL with a tracking param, then cites the bare requested URL in evidence.
    requested_url = "https://example.com/product?utm_source=google"
    verdict = {
        **VALID_VERDICT,
        "evidence": [
            {"text": "Price confirmed.", "source_url": requested_url},
            {"text": "Market check.", "source_url": "https://example.com/product"},
            {"text": "Review found.", "source_url": "https://example.com/product"},
        ],
    }
    mock_create.side_effect = [
        make_response(make_tool_block("fetch_page", {"url": requested_url}, "b1")),
        make_response(make_tool_block("submit_verdict", verdict, "b2")),
    ]
    with patch("app.agent.fetch_page", new_callable=AsyncMock):
        result = await run_research_agent(
            "https://example.com/product", extraction, identity,
            initial_fetched_page=cached_page,
        )
    assert result.verdict.value == "good_deal"
    assert any(str(e.source_url) == requested_url for e in result.evidence)

async def test_search_web_priced_result_can_drive_offer_table(
    mock_create, extraction, identity
):
    amazon_result = SearchResult(
        title="Amazon.com: Great Widget",
        url="https://www.amazon.com/great-widget/dp/B000000001",
        snippet="Great Widget 100ml is available for $7.95.",
        metadata={"source": "tavily", "score": 0.95},
    )
    stylevana_page = FetchedPage(
        url="https://www.stylevana.com/great-widget.html",
        title="Great Widget 100ml on Stylevana",
        content="Great Widget 100ml for $11.90.",
        price_guess="$11.90",
    )
    stylevana_verdict = {
        **VALID_VERDICT,
        "alternative": {
            "product_name": "Great Widget 100ml on Stylevana",
            "price": "$11.90",
            "reason": "Cheaper than submitted listing.",
            "source_url": "https://www.stylevana.com/great-widget.html",
            "is_cheaper": True,
            "is_better_reviewed": False,
        },
    }
    amazon_verdict = {
        **VALID_VERDICT,
        "alternative": {
            "product_name": "Great Widget on Amazon",
            "price": "$7.95",
            "reason": "Lowest eligible comparable offer.",
            "source_url": "https://www.amazon.com/great-widget/dp/B000000001",
            "is_cheaper": True,
            "is_better_reviewed": False,
        },
    }

    mock_create.side_effect = [
        make_response(make_tool_block("search_web", {"query": "Great Widget Amazon"}, "b1")),
        make_response(
            make_tool_block(
                "fetch_page",
                {"url": "https://www.stylevana.com/great-widget.html"},
                "b2",
            )
        ),
        make_response(make_tool_block("submit_verdict", stylevana_verdict, "b3")),
        make_response(make_tool_block("submit_verdict", amazon_verdict, "b4")),
    ]

    with (
        patch("app.agent.search_web", new_callable=AsyncMock) as mock_search,
        patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch,
    ):
        mock_search.return_value = [amazon_result]
        mock_fetch.return_value = stylevana_page

        result = await run_research_agent(
            "https://example.com/product",
            extraction,
            identity,
        )

    assert result.alternative is not None
    assert result.alternative.price == "$7.95"

    messages = mock_create.call_args_list[3].kwargs["messages"]
    rejection_tool_result = messages[-2]["content"][0]["content"]
    assert "Comparable offers found" in rejection_tool_result
    assert "www.amazon.com" in rejection_tool_result
    assert "$7.95" in rejection_tool_result
    assert "www.stylevana.com" in rejection_tool_result
    assert "$11.90" in rejection_tool_result


async def test_search_web_result_without_price_does_not_become_offer_candidate(
    mock_create, extraction, identity
):
    amazon_result_without_price = SearchResult(
        title="Amazon.com: Great Widget 100ml",
        url="https://www.amazon.com/great-widget/dp/B000000001",
        snippet="Great Widget 100ml. Price not shown.",
        metadata={"source": "tavily", "score": 0.95},
    )
    stylevana_page = FetchedPage(
        url="https://www.stylevana.com/great-widget.html",
        title="Great Widget 100ml on Stylevana",
        content="Great Widget 100ml for $11.90.",
        price_guess="$11.90",
    )
    stylevana_verdict = {
        **VALID_VERDICT,
        "alternative": {
            "product_name": "Great Widget 100ml on Stylevana",
            "price": "$11.90",
            "reason": "Cheaper than submitted listing.",
            "source_url": "https://www.stylevana.com/great-widget.html",
            "is_cheaper": True,
            "is_better_reviewed": False,
        },
    }

    mock_create.side_effect = [
        make_response(make_tool_block("search_web", {"query": "Great Widget Amazon"}, "b1")),
        make_response(
            make_tool_block(
                "fetch_page",
                {"url": "https://www.stylevana.com/great-widget.html"},
                "b2",
            )
        ),
        make_response(make_tool_block("submit_verdict", stylevana_verdict, "b3")),
        make_response(make_tool_block("submit_verdict", stylevana_verdict, "b4")),
    ]

    with (
        patch("app.agent.search_web", new_callable=AsyncMock) as mock_search,
        patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch,
    ):
        mock_search.return_value = [amazon_result_without_price]
        mock_fetch.return_value = stylevana_page

        result = await run_research_agent(
            "https://example.com/product",
            extraction,
            identity,
        )

    assert result.alternative is not None
    assert result.alternative.price == "$11.90"

    messages = mock_create.call_args_list[3].kwargs["messages"]
    confirmation_tool_result = messages[-2]["content"][0]["content"]
    assert "www.amazon.com" not in confirmation_tool_result
    assert "$100" not in confirmation_tool_result


async def test_fetch_page_cache_hit_skips_network_and_comparable_candidates_are_found(
    mock_create, extraction, identity
):
    """When the submitted URL is served from the cache, no network call is made for it.
    The agent can still find comparable offers by fetching other pages normally.
    """
    submitted_url = "https://example.com/product"
    amazon_url = "https://www.amazon.com/great-widget/dp/B000000001"

    cached_submitted_page = FetchedPage(
        url=submitted_url,
        title="Great Widget 100ml",
        content="Great Widget 100ml. Listed at $29.99.",
        price_guess="$29.99",
    )
    amazon_page = FetchedPage(
        url=amazon_url,
        title="Great Widget 100ml on Amazon",
        content="Great Widget 100ml for $7.95.",
        price_guess="$7.95",
    )
    evidence_urls = [amazon_url, submitted_url, submitted_url]
    amazon_verdict = _verdict_with_evidence_and_alternative(
        _make_alternative(amazon_url, "$7.95"), evidence_urls
    )

    mock_create.side_effect = [
        make_response(make_tool_block("fetch_page", {"url": submitted_url}, "b1")),
        make_response(make_tool_block("fetch_page", {"url": amazon_url}, "b2")),
        make_response(make_tool_block("submit_verdict", amazon_verdict, "b3")),
        make_response(make_tool_block("submit_verdict", amazon_verdict, "b4")),
    ]

    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = amazon_page

        result = await run_research_agent(
            submitted_url,
            extraction,
            identity,
            initial_fetched_page=cached_submitted_page,
        )

    # The cache hit should have prevented a network call for the submitted URL.
    mock_fetch.assert_called_once_with(amazon_url)
    assert result.alternative is not None
    assert "amazon" in str(result.alternative.source_url).lower()


async def test_duplicate_search_and_fetch_url_only_appears_once_in_offer_table(
    mock_create, extraction, identity
):
    amazon_url = "https://www.amazon.com/great-widget/dp/B000000001"
    amazon_result = SearchResult(
        title="Amazon.com: Great Widget",
        url=amazon_url,
        snippet="Great Widget 100ml is available for $7.95.",
        metadata={"source": "tavily", "score": 0.95},
    )
    amazon_page = FetchedPage(
        url=amazon_url,
        title="Great Widget 100ml on Amazon",
        content="Great Widget 100ml for $7.95.",
        price_guess="$7.95",
    )
    stylevana_page = FetchedPage(
        url="https://www.stylevana.com/great-widget.html",
        title="Great Widget 100ml on Stylevana",
        content="Great Widget 100ml for $11.90.",
        price_guess="$11.90",
    )
    stylevana_verdict = {
        **VALID_VERDICT,
        "alternative": {
            "product_name": "Great Widget 100ml on Stylevana",
            "price": "$11.90",
            "reason": "Cheaper than submitted listing.",
            "source_url": "https://www.stylevana.com/great-widget.html",
            "is_cheaper": True,
            "is_better_reviewed": False,
        },
    }
    amazon_verdict = {
        **VALID_VERDICT,
        "alternative": {
            "product_name": "Great Widget 100ml on Amazon",
            "price": "$7.95",
            "reason": "Lowest eligible comparable offer.",
            "source_url": amazon_url,
            "is_cheaper": True,
            "is_better_reviewed": False,
        },
    }

    mock_create.side_effect = [
        make_response(make_tool_block("search_web", {"query": "Great Widget Amazon"}, "b1")),
        make_response(make_tool_block("fetch_page", {"url": amazon_url}, "b2")),
        make_response(
            make_tool_block(
                "fetch_page",
                {"url": "https://www.stylevana.com/great-widget.html"},
                "b3",
            )
        ),
        make_response(make_tool_block("submit_verdict", stylevana_verdict, "b4")),
        make_response(make_tool_block("submit_verdict", amazon_verdict, "b5")),
    ]

    async def fake_fetch(url: str) -> FetchedPage:
        if url == amazon_url:
            return amazon_page
        return stylevana_page

    with (
        patch("app.agent.search_web", new_callable=AsyncMock) as mock_search,
        patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch,
    ):
        mock_search.return_value = [amazon_result]
        mock_fetch.side_effect = fake_fetch

        result = await run_research_agent(
            "https://example.com/product",
            extraction,
            identity,
        )

    assert result.alternative is not None
    assert result.alternative.price == "$7.95"

    messages = mock_create.call_args_list[3].kwargs["messages"]
    rejection_tool_result = messages[-2]["content"][0]["content"]
    # The rejection text also mentions www.amazon.com in the reason sentence, so count
    # only in the offer table section (before the guidance line).
    offer_table_section = rejection_tool_result.split("If recommending")[0]
    assert offer_table_section.count("www.amazon.com") == 1
    assert "$7.95" in rejection_tool_result


async def test_out_of_stock_candidate_excluded_from_offer_table(
    mock_create, extraction, identity
):
    walmart_page = FetchedPage(
        url="https://www.walmart.com/ip/great-widget",
        title="Great Widget",
        content="Great Widget. Out of stock.",
        price_guess="$5.99",
    )
    amazon_page = FetchedPage(
        url="https://www.amazon.com/great-widget/dp/B000000001",
        title="Great Widget",
        content="Great Widget. In stock.",
        price_guess="$7.95",
    )
    amazon_verdict = {
        **VALID_VERDICT,
        "alternative": {
            "product_name": "Great Widget on Amazon",
            "price": "$7.95",
            "reason": "Lowest eligible comparable offer.",
            "source_url": "https://www.amazon.com/great-widget/dp/B000000001",
            "is_cheaper": True,
            "is_better_reviewed": False,
        },
    }

    walmart_fetch_url = "https://www.walmart.com/ip/great-widget"
    amazon_fetch_url = "https://www.amazon.com/great-widget/dp/B000000001"
    mock_create.side_effect = [
        make_response(make_tool_block("fetch_page", {"url": walmart_fetch_url}, "b1")),
        make_response(make_tool_block("fetch_page", {"url": amazon_fetch_url}, "b2")),
        make_response(make_tool_block("submit_verdict", amazon_verdict, "b3")),
        make_response(make_tool_block("submit_verdict", amazon_verdict, "b4")),
    ]

    async def fake_fetch(url: str) -> FetchedPage:
        if "walmart" in url:
            return walmart_page
        return amazon_page

    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = fake_fetch

        result = await run_research_agent(
            "https://example.com/product",
            extraction,
            identity,
        )

    assert result.alternative is not None
    assert result.alternative.price == "$7.95"

    # The offer table sent to the agent should include Amazon but exclude the out-of-stock Walmart.
    messages = mock_create.call_args_list[3].kwargs["messages"]
    offer_table_message = messages[-2]["content"][0]["content"]
    assert "www.amazon.com" in offer_table_message
    assert "www.walmart.com" not in offer_table_message


async def test_candidate_eligible_when_size_in_content_not_title(
    mock_create, extraction, identity
):
    """Amazon title lacks '100ml' but page content includes '3.4 fl oz / 100ml'.
    The candidate should remain eligible when the submitted product specifies 100ml.
    """
    identity_with_size = ProductIdentity(value="Great Widget 100ml", source="product_name")

    amazon_page = FetchedPage(
        url="https://www.amazon.com/great-widget/dp/B000000001",
        title="Great Widget",
        content="Great Widget 3.4 fl oz / 100ml. $7.95.",
        price_guess="$7.95",
    )
    amazon_verdict = {
        **VALID_VERDICT,
        "alternative": {
            "product_name": "Great Widget on Amazon",
            "price": "$7.95",
            "reason": "Lowest eligible comparable offer.",
            "source_url": "https://www.amazon.com/great-widget/dp/B000000001",
            "is_cheaper": True,
            "is_better_reviewed": False,
        },
    }

    amazon_fetch_url = "https://www.amazon.com/great-widget/dp/B000000001"
    mock_create.side_effect = [
        make_response(make_tool_block("fetch_page", {"url": amazon_fetch_url}, "b1")),
        make_response(make_tool_block("submit_verdict", amazon_verdict, "b2")),
        make_response(make_tool_block("submit_verdict", amazon_verdict, "b3")),
    ]

    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = amazon_page

        result = await run_research_agent(
            "https://example.com/product",
            extraction,
            identity_with_size,
        )

    assert result.alternative is not None
    assert result.alternative.price == "$7.95"

    # Amazon should appear in the offer table (eligible despite title lacking '100ml').
    messages = mock_create.call_args_list[2].kwargs["messages"]
    offer_table_message = messages[-2]["content"][0]["content"]
    assert "www.amazon.com" in offer_table_message


async def test_tool_errors_are_returned_as_tool_results(mock_create, extraction, identity):
    mock_create.side_effect = [
        make_response(make_tool_block("search_web", {"query": "widget"}, "b1")),
        make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b2")),
    ]
    with patch("app.agent.search_web", new_callable=AsyncMock) as mock_search:
        mock_search.side_effect = ValueError("Tavily API key is not configured")
        await run_research_agent("https://example.com/product", extraction, identity)
    assert mock_create.call_count == 2
    tool_result = mock_create.call_args_list[1].kwargs["messages"][-2]["content"][0]["content"]
    assert "Search error" in tool_result


async def test_missing_tool_input_returns_clean_error(mock_create, extraction, identity):
    mock_create.side_effect = [
        make_response(make_tool_block("search_web", {}, "b1")),
        make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b2")),
    ]
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert isinstance(result, ResearchResult)
    tool_result = mock_create.call_args_list[1].kwargs["messages"][-2]["content"][0]["content"]
    assert "Tool error" in tool_result


async def test_unknown_tool_returns_clean_error(mock_create, extraction, identity):
    mock_create.side_effect = [
        make_response(make_tool_block("unknown_tool", {}, "b1")),
        make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b2")),
    ]
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert isinstance(result, ResearchResult)
    tool_result = mock_create.call_args_list[1].kwargs["messages"][-2]["content"][0]["content"]
    assert "Unknown tool" in tool_result


async def test_loop_guard_falls_back_to_insufficient_data(mock_create, extraction, identity):
    mock_create.return_value = make_response(
        make_tool_block("search_web", {"query": "widget"}, "b1")
    )
    with patch("app.agent.search_web", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []
        result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "insufficient_data"
    assert isinstance(result, ResearchResult)


async def test_search_budget_exhaustion_does_not_execute_extra_searches(
    mock_create, extraction, identity
):
    with patch("app.agent.settings") as mock_settings:
        mock_settings.max_searches = 1
        mock_settings.max_fetched_pages = 8
        mock_settings.agent_timeout_seconds = 120
        mock_create.side_effect = [
            make_response(make_tool_block("search_web", {"query": "s1"}, "b1")),
            make_response(make_tool_block("search_web", {"query": "s2"}, "b2")),
            make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b3")),
        ]
        with patch("app.agent.search_web", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []
            result = await run_research_agent("https://example.com/product", extraction, identity)
    assert mock_search.call_count == 1
    assert isinstance(result, ResearchResult)


async def test_search_budget_exhaustion_returns_submit_verdict_guidance(
    mock_create, extraction, identity
):
    with patch("app.agent.settings") as mock_settings:
        mock_settings.max_searches = 1
        mock_settings.max_fetched_pages = 8
        mock_settings.agent_timeout_seconds = 120
        mock_create.side_effect = [
            make_response(make_tool_block("search_web", {"query": "s1"}, "b1")),
            make_response(make_tool_block("search_web", {"query": "s2"}, "b2")),
            make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b3")),
        ]
        with patch("app.agent.search_web", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []
            await run_research_agent("https://example.com/product", extraction, identity)
    # The second call's messages contain the budget exhaustion tool result
    tool_result = mock_create.call_args_list[2].kwargs["messages"][-2]["content"][0]["content"]
    assert "submit_verdict" in tool_result


async def test_fetch_budget_exhaustion_does_not_execute_extra_fetches(
    mock_create, extraction, identity
):
    with patch("app.agent.settings") as mock_settings:
        mock_settings.max_searches = 8
        mock_settings.max_fetched_pages = 1
        mock_settings.agent_timeout_seconds = 120
        mock_settings.meaningful_savings_threshold = 0.10
        mock_create.side_effect = [
            make_response(make_tool_block("fetch_page", {"url": "https://example.com/p1"}, "b1")),
            make_response(make_tool_block("fetch_page", {"url": "https://example.com/p2"}, "b2")),
            make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b3")),
        ]
        with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = FetchedPage(
                url="https://example.com/p1",
                title="Product",
                content="Great widget.",
                price_guess="$29.99",
            )
            result = await run_research_agent("https://example.com/product", extraction, identity)
    assert mock_fetch.call_count == 1
    assert isinstance(result, ResearchResult)


async def test_runtime_timeout_returns_failed_verdict(extraction, identity):
    def raise_timeout(coro, **kwargs):
        coro.close()
        raise asyncio.TimeoutError()

    with patch("asyncio.wait_for", side_effect=raise_timeout):
        result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "failed"
    assert isinstance(result, ResearchResult)


async def test_runtime_timeout_result_has_user_safe_retry_message(extraction, identity):
    def raise_timeout(coro, **kwargs):
        coro.close()
        raise asyncio.TimeoutError()

    with patch("asyncio.wait_for", side_effect=raise_timeout):
        result = await run_research_agent("https://example.com/product", extraction, identity)
    assert "timed out" in result.summary.lower()
    assert "try again" in result.summary.lower()


async def test_anthropic_rate_limit_error_falls_back_to_insufficient_data(
    mock_create, extraction, identity
):
    response = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    mock_create.side_effect = anthropic.RateLimitError(
        "Rate limit exceeded", response=response, body=None
    )
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "insufficient_data"
    assert isinstance(result, ResearchResult)


# --- Validation fallback tests ---


async def test_malformed_verdict_payload_falls_back_to_insufficient_data(
    mock_create, extraction, identity
):
    bad_verdict = {"product_name": "Widget"}  # missing required fields
    mock_create.return_value = make_response(make_tool_block("submit_verdict", bad_verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "insufficient_data"
    assert isinstance(result, ResearchResult)


async def test_too_few_evidence_items_falls_back_to_insufficient_data(
    mock_create, extraction, identity
):
    verdict = {
        **VALID_VERDICT,
        "evidence": [
            {"text": "Only one source.", "source_url": "https://example.com/product"}
        ],
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "insufficient_data"


async def test_supported_verdict_with_five_evidence_items_passes(
    mock_create, extraction, identity
):
    verdict = {
        **VALID_VERDICT,
        "evidence": [
            {"text": f"Source {i}.", "source_url": "https://example.com/product"}
            for i in range(5)
        ],
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "good_deal"
    assert len(result.evidence) == 5


async def test_six_evidence_items_truncated_to_five_with_valid_verdict(
    mock_create, extraction, identity
):
    verdict = {
        **VALID_VERDICT,
        "evidence": [
            {"text": f"Source {i}.", "source_url": "https://example.com/product"}
            for i in range(6)
        ],
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "good_deal"
    assert len(result.evidence) == 5


async def test_evidence_url_in_position_six_is_truncated_before_validation(
    mock_create, extraction, identity
):
    # Evidence is truncated to 5 before URL validation, so the fabricated 6th item is never seen.
    verdict = {
        **VALID_VERDICT,
        "evidence": [
            {"text": f"Valid source {i}.", "source_url": "https://example.com/product"}
            for i in range(5)
        ] + [
            {"text": "Fabricated.", "source_url": "https://fabricated.example.com/bad"}
        ],
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "good_deal"
    assert len(result.evidence) == 5


async def test_insufficient_data_with_zero_evidence_passes(
    mock_create, extraction, identity
):
    verdict = {
        **VALID_VERDICT,
        "verdict": "insufficient_data",
        "confidence": "low",
        "summary": "Could not find enough pricing data.",
        "evidence": [],
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert isinstance(result, ResearchResult)
    assert result.verdict.value == "insufficient_data"
    assert result.evidence == []


async def test_high_confidence_single_distinct_source_falls_back_to_insufficient_data(
    mock_create, extraction, identity
):
    verdict = {
        **VALID_VERDICT,
        "confidence": "high",
        "evidence": [
            {"text": f"Source {i}.", "source_url": "https://example.com/product"}
            for i in range(3)
        ],
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "insufficient_data"


async def test_unobserved_evidence_url_falls_back_to_insufficient_data(
    mock_create, extraction, identity
):
    verdict = {
        **VALID_VERDICT,
        "evidence": [
            {"text": "Fake source.", "source_url": "https://fabricated.example.com/fake"},
            {"text": "Another.", "source_url": "https://fabricated.example.com/fake2"},
            {"text": "Third.", "source_url": "https://fabricated.example.com/fake3"},
        ],
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "insufficient_data"


async def test_alternative_with_unsupported_flags_keeps_verdict_with_no_alternative(
    mock_create, extraction, identity
):
    verdict = {
        **VALID_VERDICT,
        "alternative": {
            "product_name": "Widget Pro",
            "price": "$35.00",
            "reason": "Not actually better",
            "source_url": "https://example.com/product",
            "is_cheaper": False,
            "is_better_reviewed": False,
        },
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "good_deal"
    assert result.alternative is None


async def test_fallback_does_not_preserve_misleading_summary(
    mock_create, extraction, identity
):
    verdict = {
        **VALID_VERDICT,
        "verdict": "good_deal",
        "summary": "This is a great deal at a very low price.",
        "evidence": [
            {"text": "Only one piece of evidence.", "source_url": "https://example.com/product"}
        ],
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "insufficient_data"
    assert "great deal" not in result.summary
    assert "good" not in result.summary.lower() or "not" in result.summary.lower()


async def test_high_confidence_two_distinct_observed_sources_passes(
    mock_create, extraction, identity
):
    search_result = SearchResult(
        title="Widget Review",
        url="https://reviews.example.com/widget",
        snippet="Solid product.",
        metadata={"source": "tavily", "score": 0.9},
    )
    verdict = {
        **VALID_VERDICT,
        "confidence": "high",
        "evidence": [
            {"text": "Price check.", "source_url": "https://example.com/product"},
            {"text": "Review found.", "source_url": "https://reviews.example.com/widget"},
            {"text": "Market average.", "source_url": "https://example.com/product"},
        ],
    }
    mock_create.side_effect = [
        make_response(make_tool_block("search_web", {"query": "widget"}, "b1")),
        make_response(make_tool_block("submit_verdict", verdict, "b2")),
    ]
    with patch("app.agent.search_web", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [search_result]
        result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "good_deal"
    assert result.confidence.value == "high"


async def test_alternative_with_unobserved_url_keeps_verdict_with_no_alternative(
    mock_create, extraction, identity
):
    verdict = {
        **VALID_VERDICT,
        "alternative": {
            "product_name": "Widget Pro",
            "price": "$25.00",
            "reason": "Cheaper option",
            "source_url": "https://fabricated.example.com/widget-pro",
            "is_cheaper": True,
            "is_better_reviewed": False,
        },
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "good_deal"
    assert result.alternative is None


async def test_url_validation_accepts_normalized_variant(mock_create, extraction, identity):
    fetched = FetchedPage(
        url="https://shop.example.com/widget/",
        title="Widget",
        content="Great product at $29.99",
        price_guess="$29.99",
    )
    verdict = {
        **VALID_VERDICT,
        "evidence": [
            {"text": "Price found.", "source_url": "https://shop.example.com/widget"},
            {"text": "Market check.", "source_url": "https://example.com/product"},
            {"text": "Reviews checked.", "source_url": "https://example.com/product"},
        ],
    }
    mock_create.side_effect = [
        make_response(
            make_tool_block("fetch_page", {"url": "https://shop.example.com/widget/"}, "b1")
        ),
        make_response(make_tool_block("submit_verdict", verdict, "b2")),
    ]
    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = fetched
        result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "good_deal"


async def test_fallback_result_is_valid_research_result(mock_create, extraction, identity):
    verdict = {**VALID_VERDICT, "evidence": []}
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert isinstance(result, ResearchResult)
    assert result.verdict.value == "insufficient_data"
    assert result.confidence.value == "low"
    assert result.evidence == []
    assert result.alternative is None


async def test_evidence_url_from_search_result_passes_validation(
    mock_create, extraction, identity
):
    search_url = "https://pricecheck.example.com/product"
    search_result = SearchResult(
        title="Price Check",
        url=search_url,
        snippet="Listed at $29.99.",
        metadata={"source": "tavily", "score": 0.9},
    )
    verdict = {
        **VALID_VERDICT,
        "evidence": [
            {"text": "Price confirmed.", "source_url": search_url},
            {"text": "Market check.", "source_url": "https://example.com/product"},
            {"text": "Review found.", "source_url": "https://example.com/product"},
        ],
    }
    mock_create.side_effect = [
        make_response(make_tool_block("search_web", {"query": "widget price"}, "b1")),
        make_response(make_tool_block("submit_verdict", verdict, "b2")),
    ]
    with patch("app.agent.search_web", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [search_result]
        result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "good_deal"


async def test_evidence_url_from_fetched_page_passes_validation(
    mock_create, extraction, identity
):
    fetched_url = "https://pricecheck.example.com/product"
    fetched = FetchedPage(
        url=fetched_url,
        title="Price Check",
        content="Listed at $29.99.",
        price_guess="$29.99",
    )
    verdict = {
        **VALID_VERDICT,
        "evidence": [
            {"text": "Price confirmed.", "source_url": fetched_url},
            {"text": "Market check.", "source_url": "https://example.com/product"},
            {"text": "Review found.", "source_url": "https://example.com/product"},
        ],
    }
    mock_create.side_effect = [
        make_response(make_tool_block("fetch_page", {"url": fetched_url}, "b1")),
        make_response(make_tool_block("submit_verdict", verdict, "b2")),
    ]
    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = fetched
        result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "good_deal"


async def test_unobserved_evidence_item_dropped_if_three_valid_remain(
    mock_create, extraction, identity
):
    verdict = {
        **VALID_VERDICT,
        "evidence": [
            {"text": "Valid 1.", "source_url": "https://example.com/product"},
            {"text": "Valid 2.", "source_url": "https://example.com/product"},
            {"text": "Valid 3.", "source_url": "https://example.com/product"},
            {"text": "Fabricated.", "source_url": "https://fabricated.example.com/bad"},
        ],
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "good_deal"
    assert len(result.evidence) == 3


async def test_verdict_falls_back_when_pruning_leaves_fewer_than_three(
    mock_create, extraction, identity
):
    verdict = {
        **VALID_VERDICT,
        "evidence": [
            {"text": "Valid 1.", "source_url": "https://example.com/product"},
            {"text": "Valid 2.", "source_url": "https://example.com/product"},
            {"text": "Fabricated 1.", "source_url": "https://fabricated.example.com/bad1"},
            {"text": "Fabricated 2.", "source_url": "https://fabricated.example.com/bad2"},
        ],
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "insufficient_data"


async def test_evidence_with_missing_source_url_is_dropped(
    mock_create, extraction, identity
):
    verdict = {
        **VALID_VERDICT,
        "evidence": [
            {"text": "Valid.", "source_url": "https://example.com/product"},
            {"text": "Valid.", "source_url": "https://example.com/product"},
            {"text": "Valid.", "source_url": "https://example.com/product"},
            {"text": "No URL.", "source_url": ""},
        ],
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "good_deal"
    assert len(result.evidence) == 3


async def test_validation_logs_dropped_evidence_url(
    mock_create, extraction, identity, caplog
):
    verdict = {
        **VALID_VERDICT,
        "evidence": [
            {"text": "Valid.", "source_url": "https://example.com/product"},
            {"text": "Valid.", "source_url": "https://example.com/product"},
            {"text": "Valid.", "source_url": "https://example.com/product"},
            {"text": "Fabricated.", "source_url": "https://fabricated.example.com/bad"},
        ],
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    with caplog.at_level(logging.WARNING, logger="app.agent"):
        await run_research_agent("https://example.com/product", extraction, identity)
    messages = [r.message for r in caplog.records]
    assert any("fabricated.example.com" in m for m in messages)
    assert any("observed" in m.lower() for m in messages)


# --- Alternative sanitization tests ---


def _make_valid_alternative(source_url: str = "https://example.com/product") -> dict:
    return {
        "product_name": "Widget Pro",
        "price": "$25.00",
        "reason": "Cheaper option",
        "source_url": source_url,
        "is_cheaper": True,
        "is_better_reviewed": False,
    }


async def test_valid_verdict_with_complete_alternative_keeps_alternative(
    mock_create, extraction, identity
):
    verdict = {**VALID_VERDICT, "alternative": _make_valid_alternative()}
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "good_deal"
    assert result.alternative is not None
    assert result.alternative.price == "$25.00"


async def test_alternative_missing_price_keeps_verdict_with_no_alternative(
    mock_create, extraction, identity
):
    alt = _make_valid_alternative()
    del alt["price"]
    verdict = {**VALID_VERDICT, "alternative": alt}
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "good_deal"
    assert result.alternative is None


async def test_alternative_missing_product_name_keeps_verdict_with_no_alternative(
    mock_create, extraction, identity
):
    alt = _make_valid_alternative()
    del alt["product_name"]
    verdict = {**VALID_VERDICT, "alternative": alt}
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "good_deal"
    assert result.alternative is None


async def test_alternative_missing_source_url_keeps_verdict_with_no_alternative(
    mock_create, extraction, identity
):
    alt = _make_valid_alternative()
    del alt["source_url"]
    verdict = {**VALID_VERDICT, "alternative": alt}
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "good_deal"
    assert result.alternative is None


async def test_alternative_missing_reason_keeps_verdict_with_no_alternative(
    mock_create, extraction, identity
):
    alt = _make_valid_alternative()
    del alt["reason"]
    verdict = {**VALID_VERDICT, "alternative": alt}
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "good_deal"
    assert result.alternative is None


async def test_invalid_evidence_still_falls_back_despite_valid_alternative(
    mock_create, extraction, identity
):
    verdict = {
        **VALID_VERDICT,
        "evidence": [
            {"text": "Only one.", "source_url": "https://example.com/product"}
        ],
        "alternative": _make_valid_alternative(),
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    result = await run_research_agent("https://example.com/product", extraction, identity)
    assert result.verdict.value == "insufficient_data"


# --- Offer candidate ranking and best-alternative validation tests ---


def _make_fetched_page(url: str, title: str, price: str | None) -> FetchedPage:
    return FetchedPage(url=url, title=title, content="...", price_guess=price)


def _make_alternative(source_url: str, price: str) -> dict:
    return {
        "product_name": "Test Product",
        "price": price,
        "reason": "Cheaper option",
        "source_url": source_url,
        "is_cheaper": True,
        "is_better_reviewed": False,
    }


def _verdict_with_evidence_and_alternative(alternative: dict | None, urls: list[str]) -> dict:
    evidence = [
        {"text": f"Price confirmed at source {i + 1}.", "source_url": url}
        for i, url in enumerate(urls[:3])
    ]
    return {
        **VALID_VERDICT,
        "evidence": evidence,
        "alternative": alternative,
    }


@pytest.mark.asyncio
async def test_walmart_alternative_rejected_when_amazon_is_cheaper(mock_create, extraction):
    amazon_url = "https://www.amazon.com/cleanser"
    walmart_url = "https://www.walmart.com/cleanser"
    submitted_url = "https://example.com/product"
    identity = ProductIdentity(value="Great Widget", source="product_name")

    amazon_page = _make_fetched_page(amazon_url, "Great Widget", "$7.95")
    walmart_page = _make_fetched_page(walmart_url, "Great Widget", "$13.59")

    evidence_urls = [amazon_url, walmart_url, submitted_url]
    verdict_walmart = _verdict_with_evidence_and_alternative(
        _make_alternative(walmart_url, "$13.59"), evidence_urls
    )
    verdict_amazon = _verdict_with_evidence_and_alternative(
        _make_alternative(amazon_url, "$7.95"), evidence_urls
    )

    mock_create.side_effect = [
        make_response(make_tool_block("fetch_page", {"url": amazon_url}, "b1")),
        make_response(make_tool_block("fetch_page", {"url": walmart_url}, "b2")),
        make_response(make_tool_block("submit_verdict", verdict_walmart, "b3")),
        make_response(make_tool_block("submit_verdict", verdict_amazon, "b4")),
    ]

    def fetch_side_effect(url):
        return amazon_page if "amazon" in url else walmart_page

    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = fetch_side_effect
        result = await run_research_agent(submitted_url, extraction, identity)

    assert result.alternative is not None
    assert "amazon" in str(result.alternative.source_url).lower()


@pytest.mark.asyncio
async def test_wrong_size_candidate_is_not_eligible(mock_create, extraction):
    wrong_size_url = "https://www.amazon.com/cleanser-200ml"
    submitted_url = "https://example.com/product"
    identity = ProductIdentity(value="Great Widget 100ml", source="product_name")

    wrong_size_page = _make_fetched_page(wrong_size_url, "Great Widget 200ml", "$7.95")

    # With wrong-size candidate not eligible, no offer table is shown.
    # The verdict without an alternative should be accepted on the first call.
    verdict_no_alt = _verdict_with_evidence_and_alternative(None, [submitted_url] * 3)

    mock_create.side_effect = [
        make_response(make_tool_block("fetch_page", {"url": wrong_size_url}, "b1")),
        make_response(make_tool_block("submit_verdict", verdict_no_alt, "b2")),
    ]

    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = wrong_size_page
        result = await run_research_agent(submitted_url, extraction, identity)

    assert result.verdict.value == "good_deal"
    # Two total LLM calls: fetch response + submit response (no table round-trip needed).
    assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_only_eligible_candidate_can_be_selected(mock_create, extraction):
    walmart_url = "https://www.walmart.com/cleanser"
    submitted_url = "https://example.com/product"
    identity = ProductIdentity(value="Great Widget", source="product_name")

    walmart_page = _make_fetched_page(walmart_url, "Great Widget", "$13.59")
    evidence_urls = [walmart_url, submitted_url, submitted_url]
    verdict_walmart = _verdict_with_evidence_and_alternative(
        _make_alternative(walmart_url, "$13.59"), evidence_urls
    )

    # First submit shows the offer table; second submit accepts.
    mock_create.side_effect = [
        make_response(make_tool_block("fetch_page", {"url": walmart_url}, "b1")),
        make_response(make_tool_block("submit_verdict", verdict_walmart, "b2")),
        make_response(make_tool_block("submit_verdict", verdict_walmart, "b3")),
    ]

    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = walmart_page
        result = await run_research_agent(submitted_url, extraction, identity)

    assert result.alternative is not None
    assert "walmart" in str(result.alternative.source_url).lower()


@pytest.mark.asyncio
async def test_no_alternative_required_when_no_eligible_cheaper_offer(
    mock_create, extraction, identity
):
    submitted_url = "https://example.com/product"
    verdict_no_alt = _verdict_with_evidence_and_alternative(None, [submitted_url] * 3)

    # No fetch_page calls; no candidates; verdict accepted immediately.
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict_no_alt))
    result = await run_research_agent(submitted_url, extraction, identity)

    assert result.verdict.value == "good_deal"
    assert result.alternative is None
    assert mock_create.call_count == 1


async def test_no_alternative_rejected_when_eligible_candidates_exist(mock_create, extraction):
    amazon_url = "https://www.amazon.com/cleanser"
    submitted_url = "https://example.com/product"
    identity = ProductIdentity(value="Great Widget", source="product_name")

    amazon_page = _make_fetched_page(amazon_url, "Great Widget", "$7.95")
    evidence_urls = [amazon_url, submitted_url, submitted_url]

    verdict_no_alt = _verdict_with_evidence_and_alternative(None, evidence_urls)
    verdict_amazon = _verdict_with_evidence_and_alternative(
        _make_alternative(amazon_url, "$7.95"), evidence_urls
    )

    mock_create.side_effect = [
        make_response(make_tool_block("fetch_page", {"url": amazon_url}, "b1")),
        make_response(make_tool_block("submit_verdict", verdict_no_alt, "b2")),
        make_response(make_tool_block("submit_verdict", verdict_amazon, "b3")),
    ]

    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = amazon_page
        result = await run_research_agent(submitted_url, extraction, identity)

    assert result.alternative is not None
    assert "amazon" in str(result.alternative.source_url).lower()

    messages = mock_create.call_args_list[2].kwargs["messages"]
    rejection_tool_result = messages[-2]["content"][0]["content"]
    assert "Please include the lowest eligible comparable offer" in rejection_tool_result
    assert "www.amazon.com" in rejection_tool_result
    assert "$7.95" in rejection_tool_result


async def test_alternative_url_rejected_when_wrong_source_url(mock_create, extraction):
    amazon_url = "https://www.amazon.com/cleanser"
    stylevana_url = "https://www.stylevana.com/cleanser"
    submitted_url = "https://example.com/product"
    identity = ProductIdentity(value="Great Widget", source="product_name")

    amazon_page = _make_fetched_page(amazon_url, "Great Widget", "$7.95")
    stylevana_page = _make_fetched_page(stylevana_url, "Great Widget", "$11.90")
    evidence_urls = [amazon_url, stylevana_url, submitted_url]

    # Amazon is cheapest, but agent submits Stylevana URL with Amazon price.
    wrong_url_verdict = _verdict_with_evidence_and_alternative(
        _make_alternative(stylevana_url, "$7.95"), evidence_urls
    )
    verdict_amazon = _verdict_with_evidence_and_alternative(
        _make_alternative(amazon_url, "$7.95"), evidence_urls
    )

    def fetch_side_effect(url):
        return amazon_page if "amazon" in url else stylevana_page

    mock_create.side_effect = [
        make_response(make_tool_block("fetch_page", {"url": amazon_url}, "b1")),
        make_response(make_tool_block("fetch_page", {"url": stylevana_url}, "b2")),
        make_response(make_tool_block("submit_verdict", wrong_url_verdict, "b3")),
        make_response(make_tool_block("submit_verdict", verdict_amazon, "b4")),
    ]

    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = fetch_side_effect
        result = await run_research_agent(submitted_url, extraction, identity)

    assert result.alternative is not None
    assert "amazon" in str(result.alternative.source_url).lower()

    messages = mock_create.call_args_list[3].kwargs["messages"]
    rejection_tool_result = messages[-2]["content"][0]["content"]
    assert "source URL does not match" in rejection_tool_result
    assert "www.amazon.com" in rejection_tool_result


@pytest.mark.asyncio
async def test_sold_out_alternative_is_rejected_with_unavailable_message(
    mock_create, extraction
):
    amazon_url = "https://www.amazon.com/cleanser"
    ebay_url = "https://www.ebay.com/cleanser"
    submitted_url = "https://example.com/product"
    identity = ProductIdentity(value="Pokémon Card", source="product_name")

    # Amazon is available and cheaper than submitted price.
    amazon_page = FetchedPage(
        url=amazon_url,
        title="Pokémon Card",
        content="In stock. Ships in 2 days.",
        price_guess="$7.95",
    )
    # eBay page has a price but is sold out.
    ebay_page = FetchedPage(
        url=ebay_url,
        title="Pokémon Card",
        content="Sold out. This listing has ended.",
        price_guess="$6.50",
    )

    evidence_urls = [amazon_url, submitted_url, submitted_url]
    verdict_with_soldout_alt = _verdict_with_evidence_and_alternative(
        _make_alternative(ebay_url, "$6.50"), evidence_urls
    )
    verdict_with_amazon_alt = _verdict_with_evidence_and_alternative(
        _make_alternative(amazon_url, "$7.95"), evidence_urls
    )

    mock_create.side_effect = [
        make_response(make_tool_block("fetch_page", {"url": amazon_url}, "b1")),
        make_response(make_tool_block("fetch_page", {"url": ebay_url}, "b2")),
        make_response(make_tool_block("submit_verdict", verdict_with_soldout_alt, "b3")),
        make_response(make_tool_block("submit_verdict", verdict_with_amazon_alt, "b4")),
    ]

    def fetch_side_effect(url):
        return amazon_page if "amazon" in url else ebay_page

    with patch("app.agent.fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = fetch_side_effect
        result = await run_research_agent(submitted_url, extraction, identity)

    assert result.alternative is not None
    assert "amazon" in str(result.alternative.source_url).lower()

    messages = mock_create.call_args_list[3].kwargs["messages"]
    rejection_tool_result = messages[-2]["content"][0]["content"]
    assert "sold out or unavailable" in rejection_tool_result


def test_validate_alternative_is_lowest_rejects_with_clear_error():
    from app.agent import _validate_alternative_is_lowest
    from app.tools.offer_candidates import OfferCandidate

    eligible = [
        OfferCandidate(
            merchant="www.amazon.com",
            source_url="https://www.amazon.com/p",
            product_name="Widget",
            price_text="$7.95",
            price_amount=7.95,
        ),
        OfferCandidate(
            merchant="www.walmart.com",
            source_url="https://www.walmart.com/p",
            product_name="Widget",
            price_text="$13.59",
            price_amount=13.59,
        ),
    ]
    alternative = {"price": "$13.59", "source_url": "https://www.walmart.com/p"}
    rejection = _validate_alternative_is_lowest(alternative, eligible)

    assert rejection is not None
    assert "www.amazon.com" in rejection
    assert "7.95" in rejection
    assert "13.59" in rejection


def test_validate_alternative_is_lowest_rejects_wrong_url():
    from app.agent import _validate_alternative_is_lowest
    from app.tools.offer_candidates import OfferCandidate

    eligible = [
        OfferCandidate(
            merchant="www.amazon.com",
            source_url="https://www.amazon.com/p",
            product_name="Widget",
            price_text="$7.95",
            price_amount=7.95,
        ),
    ]
    # Right price, wrong URL.
    alternative = {"price": "$7.95", "source_url": "https://www.stylevana.com/p"}
    rejection = _validate_alternative_is_lowest(alternative, eligible)

    assert rejection is not None
    assert "source URL does not match" in rejection
    assert "www.amazon.com" in rejection


def test_validate_alternative_is_lowest_accepts_correct_price_and_url():
    from app.agent import _validate_alternative_is_lowest
    from app.tools.offer_candidates import OfferCandidate

    eligible = [
        OfferCandidate(
            merchant="www.amazon.com",
            source_url="https://www.amazon.com/p",
            product_name="Widget",
            price_text="$7.95",
            price_amount=7.95,
        ),
    ]
    alternative = {"price": "$7.95", "source_url": "https://www.amazon.com/p"}
    assert _validate_alternative_is_lowest(alternative, eligible) is None


# --- System prompt tests ---


def test_system_prompt_includes_tool_names():
    assert "search_web" in SYSTEM_PROMPT
    assert "fetch_page" in SYSTEM_PROMPT
    assert "submit_verdict" in SYSTEM_PROMPT


def test_system_prompt_instructs_submit_verdict():
    assert "Call submit_verdict" in SYSTEM_PROMPT


def test_system_prompt_forbids_fabricated_evidence():
    assert "Do not invent" in SYSTEM_PROMPT


def test_system_prompt_treats_content_as_untrusted():
    assert "untrusted" in SYSTEM_PROMPT


def test_system_prompt_defines_all_verdict_values():
    assert "good_deal" in SYSTEM_PROMPT
    assert "fair" in SYSTEM_PROMPT
    assert "overpriced" in SYSTEM_PROMPT
    assert "insufficient_data" in SYSTEM_PROMPT


def test_system_prompt_defines_confidence_levels():
    assert "high" in SYSTEM_PROMPT
    assert "medium" in SYSTEM_PROMPT
    assert "low" in SYSTEM_PROMPT


def test_system_prompt_defines_pricing_thresholds():
    assert "percentile" in SYSTEM_PROMPT
    assert "median" in SYSTEM_PROMPT


def test_system_prompt_excludes_different_product_comparisons():
    assert "Do not compare" in SYSTEM_PROMPT


def test_system_prompt_defines_alternative_rules():
    assert "10%" in SYSTEM_PROMPT
    assert "is_cheaper" in SYSTEM_PROMPT or "cheaper" in SYSTEM_PROMPT
