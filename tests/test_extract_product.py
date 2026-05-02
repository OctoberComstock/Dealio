from unittest.mock import AsyncMock, patch

import pytest

from app.tools.extract_product import (
    ProductExtractionResult,
    ProductPageExtraction,
    extract_product,
)
from app.tools.fetch_page import FetchedPage


def make_fetched_page(**overrides) -> FetchedPage:
    base = {
        "url": "https://example.com/product",
        "title": "Great Widget",
        "content": "This widget is excellent. Price: $29.99",
        "price_guess": "$29.99",
    }
    base.update(overrides)
    return FetchedPage(**base)


@pytest.fixture
def mock_fetch_page():
    with patch("app.tools.extract_product.fetch_page", new_callable=AsyncMock) as mock:
        yield mock


async def test_extract_product_returns_extraction_result(mock_fetch_page):
    mock_fetch_page.return_value = make_fetched_page()
    result = await extract_product("https://example.com/product")
    assert isinstance(result, ProductExtractionResult)


async def test_extract_product_extraction_field_is_product_page_extraction(mock_fetch_page):
    mock_fetch_page.return_value = make_fetched_page()
    result = await extract_product("https://example.com/product")
    assert isinstance(result.extraction, ProductPageExtraction)


async def test_extract_product_fetched_page_field_is_fetched_page(mock_fetch_page):
    mock_fetch_page.return_value = make_fetched_page()
    result = await extract_product("https://example.com/product")
    assert isinstance(result.fetched_page, FetchedPage)


async def test_extract_product_name_from_page_title(mock_fetch_page):
    mock_fetch_page.return_value = make_fetched_page(title="Great Widget")
    result = await extract_product("https://example.com/product")
    assert result.extraction.product_name == "Great Widget"


async def test_extract_listed_price_from_price_guess(mock_fetch_page):
    mock_fetch_page.return_value = make_fetched_page(price_guess="$29.99")
    result = await extract_product("https://example.com/product")
    assert result.extraction.listed_price == "$29.99"


async def test_extract_merchant_from_page_url(mock_fetch_page):
    mock_fetch_page.return_value = make_fetched_page(url="https://store.example.com/product")
    result = await extract_product("https://store.example.com/product")
    assert result.extraction.merchant == "store.example.com"


async def test_extract_merchant_from_source_url_when_page_url_missing(mock_fetch_page):
    mock_fetch_page.return_value = make_fetched_page(url="")
    result = await extract_product("https://example.com/product")
    assert result.extraction.merchant == "example.com"


async def test_extract_product_name_is_none_when_title_missing(mock_fetch_page):
    mock_fetch_page.return_value = make_fetched_page(title=None)
    result = await extract_product("https://example.com/product")
    assert result.extraction.product_name is None


async def test_extract_listed_price_is_none_when_no_price(mock_fetch_page):
    mock_fetch_page.return_value = make_fetched_page(price_guess=None)
    result = await extract_product("https://example.com/product")
    assert result.extraction.listed_price is None


async def test_fetch_failure_returns_partial_extraction(mock_fetch_page):
    mock_fetch_page.side_effect = ValueError("HTTP 404 fetching https://example.com/product")
    result = await extract_product("https://example.com/product")
    assert isinstance(result.extraction, ProductPageExtraction)
    assert result.extraction.product_name is None
    assert result.extraction.listed_price is None


async def test_fetch_failure_returns_none_fetched_page(mock_fetch_page):
    mock_fetch_page.side_effect = ValueError("Network error")
    result = await extract_product("https://example.com/product")
    assert result.fetched_page is None


async def test_fetch_failure_still_extracts_merchant(mock_fetch_page):
    mock_fetch_page.side_effect = ValueError("Network error")
    result = await extract_product("https://shop.example.co.uk/product")
    assert result.extraction.merchant == "shop.example.co.uk"


async def test_extract_merchant_full_hostname_preserved(mock_fetch_page):
    mock_fetch_page.return_value = make_fetched_page(url="https://store.example.co.uk/product")
    result = await extract_product("https://store.example.co.uk/product")
    assert result.extraction.merchant == "store.example.co.uk"


async def test_fetched_page_preserved_in_result(mock_fetch_page):
    page = make_fetched_page(title="Widget", price_guess="$19.99")
    mock_fetch_page.return_value = page
    result = await extract_product("https://example.com/product")
    assert result.fetched_page is page
