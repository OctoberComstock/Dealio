from unittest.mock import AsyncMock, MagicMock, patch

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


VALID_VERDICT = {
    "product_name": "Great Widget",
    "merchant": "example.com",
    "listed_price": "$29.99",
    "verdict": "good_deal",
    "confidence": "high",
    "summary": "Competitively priced and well-reviewed.",
    "evidence": [
        {"text": "Lowest price found.", "source_url": "https://example.com/product"}
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
    assert result.confidence.value == "high"
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


async def test_tool_errors_are_returned_as_tool_results(mock_create, extraction, identity):
    mock_create.side_effect = [
        make_response(make_tool_block("search_web", {"query": "widget"}, "b1")),
        make_response(make_tool_block("submit_verdict", VALID_VERDICT, "b2")),
    ]
    with patch("app.agent.search_web", new_callable=AsyncMock) as mock_search:
        mock_search.side_effect = ValueError("Tavily API key is not configured")
        await run_research_agent("https://example.com/product", extraction, identity)
    assert mock_create.call_count == 2
    # messages[-2]: the list is mutated after the second call (assistant reply appended),
    # so the tool result user message is at index -2, not -1.
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


async def test_loop_guard_prevents_infinite_loops(mock_create, extraction, identity):
    mock_create.return_value = make_response(
        make_tool_block("search_web", {"query": "widget"}, "b1")
    )
    with patch("app.agent.search_web", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []
        with pytest.raises(ValueError, match="exceeded"):
            await run_research_agent("https://example.com/product", extraction, identity)


async def test_invalid_submit_verdict_payload_raises_error(mock_create, extraction, identity):
    mock_create.return_value = make_response(
        make_tool_block("submit_verdict", {"product_name": "Widget"})
    )
    with pytest.raises(ValueError, match="Invalid verdict payload"):
        await run_research_agent("https://example.com/product", extraction, identity)


async def test_evidence_url_not_in_seen_urls_raises_error(mock_create, extraction, identity):
    verdict = {
        **VALID_VERDICT,
        "evidence": [
            {"text": "Fake source.", "source_url": "https://fabricated.example.com/fake"}
        ],
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    with pytest.raises(ValueError, match="not observed in tool results"):
        await run_research_agent("https://example.com/product", extraction, identity)


async def test_alternative_url_not_in_seen_urls_raises_error(mock_create, extraction, identity):
    verdict = {
        **VALID_VERDICT,
        "alternative": {
            "product_name": "Widget Pro",
            "price": "$25.00",
            "reason": "Better value",
            "source_url": "https://fabricated.example.com/alt",
            "is_cheaper": True,
            "is_better_reviewed": False,
        },
    }
    mock_create.return_value = make_response(make_tool_block("submit_verdict", verdict))
    with pytest.raises(ValueError, match="not observed in tool results"):
        await run_research_agent("https://example.com/product", extraction, identity)


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
