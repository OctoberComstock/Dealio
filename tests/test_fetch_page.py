from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.tools.fetch_page import FetchedPage, fetch_page

SAMPLE_HTML = """
<html>
<head><title>Great Widget</title></head>
<body>
<article>
<h1>Great Widget</h1>
<p>Price: $29.99</p>
<p>This widget is excellent for home use with premium materials and great durability.
Customer reviews consistently praise its performance. Ships in 2 business days with
a 1-year warranty included.</p>
</article>
</body>
</html>
"""


def make_mock_response(html: str, content_type: str = "text/html; charset=utf-8"):
    mock_response = MagicMock()
    mock_response.text = html
    mock_response.headers = {"content-type": content_type}
    mock_response.raise_for_status = MagicMock()
    return mock_response


@pytest.fixture
def mock_http_client():
    mock_client = AsyncMock()
    with patch("app.tools.fetch_page.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        yield mock_client


async def test_fetch_page_returns_fetched_page(mock_http_client):
    mock_http_client.get.return_value = make_mock_response(SAMPLE_HTML)
    result = await fetch_page("https://example.com/product")
    assert isinstance(result, FetchedPage)


async def test_fetch_page_extracts_title(mock_http_client):
    mock_http_client.get.return_value = make_mock_response(SAMPLE_HTML)
    result = await fetch_page("https://example.com/product")
    assert result.title is not None


async def test_fetch_page_extracts_content(mock_http_client):
    mock_http_client.get.return_value = make_mock_response(SAMPLE_HTML)
    result = await fetch_page("https://example.com/product")
    assert result.content != ""


async def test_fetch_page_extracts_price_guess(mock_http_client):
    mock_http_client.get.return_value = make_mock_response(SAMPLE_HTML)
    result = await fetch_page("https://example.com/product")
    assert result.price_guess == "$29.99"


async def test_fetch_page_uses_requested_url_when_no_canonical(mock_http_client):
    mock_http_client.get.return_value = make_mock_response(SAMPLE_HTML)
    result = await fetch_page("https://example.com/product")
    assert result.url == "https://example.com/product"


async def test_fetch_page_http_error_raises(mock_http_client):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_http_client.get.side_effect = httpx.HTTPStatusError(
        "Not Found", request=MagicMock(), response=mock_response
    )
    with pytest.raises(ValueError, match="HTTP 404"):
        await fetch_page("https://example.com/product")


async def test_fetch_page_network_error_raises(mock_http_client):
    mock_http_client.get.side_effect = httpx.RequestError(
        "Connection refused", request=MagicMock()
    )
    with pytest.raises(ValueError, match="Network error"):
        await fetch_page("https://example.com/product")


async def test_fetch_page_non_html_content_type_raises(mock_http_client):
    mock_http_client.get.return_value = make_mock_response(
        '{"price": 29.99}', content_type="application/json"
    )
    with pytest.raises(ValueError, match="Unsupported content type"):
        await fetch_page("https://example.com/api/product")


async def test_fetch_page_no_extractable_content_raises(mock_http_client):
    mock_http_client.get.return_value = make_mock_response("<html><body></body></html>")
    with pytest.raises(ValueError, match="No extractable content"):
        await fetch_page("https://example.com/empty")
