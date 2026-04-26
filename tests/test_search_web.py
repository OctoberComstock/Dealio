from unittest.mock import AsyncMock, patch

import pytest
from tavily import BadRequestError, InvalidAPIKeyError, MissingAPIKeyError, UsageLimitExceededError

from app.tools.search_web import SearchResult, search_web

SAMPLE_RESPONSE = {
    "results": [
        {
            "title": "Best Widget Review",
            "url": "https://example.com/review",
            "content": "This widget is excellent value for money and highly rated by customers.",
            "score": 0.92,
        },
        {
            "title": "Widget Price Comparison",
            "url": "https://example.com/compare",
            "content": "Compare widget prices across major retailers.",
            "score": 0.85,
        },
    ]
}


@pytest.fixture
def mock_tavily():
    with patch("app.tools.search_web.AsyncTavilyClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


async def test_search_web_returns_search_results(mock_tavily):
    mock_tavily.search.return_value = SAMPLE_RESPONSE
    results = await search_web("best widget under $30")
    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)


async def test_search_web_result_fields(mock_tavily):
    mock_tavily.search.return_value = SAMPLE_RESPONSE
    results = await search_web("best widget under $30")
    first = results[0]
    assert first.title == "Best Widget Review"
    assert str(first.url) == "https://example.com/review"
    assert "excellent value" in first.snippet
    assert first.metadata["score"] == 0.92
    assert first.metadata["source"] == "tavily"


async def test_search_web_truncates_long_snippet(mock_tavily):
    long_content = "a" * 600
    mock_tavily.search.return_value = {
        "results": [
            {"title": "Test", "url": "https://example.com", "content": long_content, "score": 0.5}
        ]
    }
    results = await search_web("test query")
    assert len(results[0].snippet) == 500


async def test_search_web_skips_results_without_url(mock_tavily):
    mock_tavily.search.return_value = {
        "results": [
            {"title": "No URL result", "url": None, "content": "Some content", "score": 0.7},
            {
                "title": "Valid result",
                "url": "https://example.com",
                "content": "Content",
                "score": 0.8,
            },
        ]
    }
    results = await search_web("test query")
    assert len(results) == 1
    assert results[0].title == "Valid result"


async def test_search_web_empty_query_raises():
    with pytest.raises(ValueError, match="cannot be blank"):
        await search_web("   ")


async def test_search_web_missing_api_key_raises(mock_tavily):
    mock_tavily.search.side_effect = MissingAPIKeyError()
    with pytest.raises(ValueError, match="not configured"):
        await search_web("test query")


async def test_search_web_invalid_api_key_raises(mock_tavily):
    mock_tavily.search.side_effect = InvalidAPIKeyError("invalid key")
    with pytest.raises(ValueError, match="invalid"):
        await search_web("test query")


async def test_search_web_usage_limit_raises(mock_tavily):
    mock_tavily.search.side_effect = UsageLimitExceededError("limit exceeded")
    with pytest.raises(ValueError, match="usage limit"):
        await search_web("test query")


async def test_search_web_bad_request_raises(mock_tavily):
    mock_tavily.search.side_effect = BadRequestError("bad request")
    with pytest.raises(ValueError, match="rejected"):
        await search_web("test query")


async def test_search_web_empty_results(mock_tavily):
    mock_tavily.search.return_value = {"results": []}
    results = await search_web("obscure query with no results")
    assert results == []
