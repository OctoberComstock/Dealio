import socket
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.tools.fetch_page import (
    FetchedPage,
    _check_redirect_url_is_safe,
    _detect_condition_text,
    _detect_listing_count,
    _detect_lowest_price,
    _detect_shipping_text,
    _find_price,
    _is_marketplace_like,
    _needs_rendered_fallback,
    _StaticResult,
    _validate_fetch_url,
    _validate_resolved_addresses_are_safe,
    fetch_page,
)

# --- HTML fixtures ---

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

JSON_LD_HTML = """
<html>
<head>
<script type="application/ld+json">
{
  "@type": "Product",
  "name": "Test Headphones",
  "offers": {"@type": "Offer", "price": "199.99", "priceCurrency": "USD"}
}
</script>
</head>
<body><p>Some page content about headphones</p></body>
</html>
"""

JSON_LD_GBP_HTML = """
<html>
<head>
<script type="application/ld+json">
{
  "@type": "Product",
  "name": "UK Widget",
  "offers": {"@type": "Offer", "price": "51.77", "priceCurrency": "GBP"}
}
</script>
</head>
<body><p>Some page content</p></body>
</html>
"""

OG_HTML = """
<html>
<head>
<meta property="og:title" content="Test Headphones">
<meta property="product:price:amount" content="199.99">
<meta property="product:price:currency" content="USD">
</head>
<body><p>Some page content</p></body>
</html>
"""

NEXT_DATA_HTML = """
<html>
<head><title>TCG Product</title></head>
<body>
<script id="__NEXT_DATA__">
{"props": {"pageProps": {"product": {"title": "Card", "price": 9.99}}}}
</script>
</body>
</html>
"""

MARKETPLACE_HTML = """
<html>
<head><title>TCGPlayer Listing</title></head>
<body>
<p>10 listings available</p>
<p>as low as $9.00</p>
<p>Market price $13.33</p>
<p>+ $1.50 Shipping</p>
<p>Condition: Near Mint</p>
</body>
</html>
"""


def make_mock_response(html: str, content_type: str = "text/html; charset=utf-8"):
    mock_response = MagicMock()
    mock_response.text = html
    mock_response.headers = {"content-type": content_type}
    mock_response.raise_for_status = MagicMock()
    mock_response.status_code = 200
    return mock_response


@pytest.fixture(autouse=True)
def mock_dns_resolution():
    """Prevent real DNS lookups in fetch_page tests. Override per-test for DNS-specific tests."""
    fake_results = [(None, None, None, None, ("93.184.216.34", 0))]
    with patch("socket.getaddrinfo", return_value=fake_results):
        yield


@pytest.fixture
def mock_http_client():
    mock_client = AsyncMock()
    with patch("app.tools.fetch_page.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        yield mock_client


@pytest.fixture
def mock_no_shopify():
    with patch(
        "app.tools.fetch_page._fetch_shopify_product",
        new_callable=AsyncMock,
        return_value=None,
    ):
        yield


@pytest.fixture
def mock_no_render():
    from app.tools.fetch_page import _RenderedResult

    with patch(
        "app.tools.fetch_page._render_page",
        new_callable=AsyncMock,
        return_value=_RenderedResult(title=None, price_guess=None, content=""),
    ):
        yield


# --- Price and signal helpers ---

def test_find_price_dollar():
    assert _find_price("Price: $29.99") == "$29.99"


def test_find_price_pound():
    assert _find_price("Price: £51.77") == "£51.77"


def test_find_price_euro():
    assert _find_price("€19.99 per item") == "€19.99"


def test_find_price_yen():
    assert _find_price("¥1999 today") == "¥1999"


def test_find_price_suffix_usd():
    assert _find_price("19.99 USD") == "19.99USD"


def test_find_price_returns_none_when_absent():
    assert _find_price("no price here") is None


def test_detect_shipping_plus_currency():
    assert _detect_shipping_text("+ $1.50 Shipping") is not None


def test_detect_shipping_currency_only():
    assert _detect_shipping_text("$3.99 shipping available") is not None


def test_detect_shipping_free():
    result = _detect_shipping_text("Free Shipping on orders over $25")
    assert result is not None
    assert "free" in result.lower()


def test_detect_shipping_included():
    result = _detect_shipping_text("Shipping Included in price")
    assert result is not None
    assert "shipping included" in result.lower()


def test_detect_shipping_returns_none_when_absent():
    assert _detect_shipping_text("no delivery info here") is None


def test_detect_lowest_price_as_low_as():
    result = _detect_lowest_price("as low as $9.00 for this item")
    assert result == "$9.00"


def test_detect_lowest_price_starting_at():
    result = _detect_lowest_price("starting at $12.50")
    assert result == "$12.50"


def test_detect_lowest_price_returns_none_when_absent():
    assert _detect_lowest_price("price is $19.99") is None


def test_detect_listing_count():
    result = _detect_listing_count("23 listings available")
    assert result is not None
    assert "23" in result


def test_detect_listing_count_results():
    result = _detect_listing_count("1,200 results found")
    assert result is not None


def test_detect_listing_count_returns_none_when_absent():
    assert _detect_listing_count("no listing info") is None


def test_detect_condition_text():
    result = _detect_condition_text("Condition: Near Mint")
    assert result is not None
    assert "Near Mint" in result


def test_detect_condition_text_returns_none_when_absent():
    assert _detect_condition_text("no condition info") is None


# --- Marketplace detection ---

def test_is_marketplace_like_tcgplayer_domain():
    assert _is_marketplace_like("https://www.tcgplayer.com/product/123", "") is True


def test_is_marketplace_like_ebay_domain():
    assert _is_marketplace_like("https://www.ebay.com/itm/123", "") is True


def test_is_marketplace_like_html_signals():
    html = "<p>10 listings available. as low as $5.00</p>"
    assert _is_marketplace_like("https://example.com/item", html) is True


def test_is_marketplace_like_plain_product_page():
    html = "<p>Buy this item for $19.99</p>"
    assert _is_marketplace_like("https://example.com/product", html) is False


# --- _needs_rendered_fallback ---

def test_needs_rendered_fallback_when_no_price():
    static = _StaticResult(
        canonical_url="https://example.com/p",
        title=None, price_guess=None, content="", source="text_regex",
    )
    assert _needs_rendered_fallback(static, "", "https://example.com/p") is True


def test_no_rendered_fallback_for_json_ld_with_price():
    static = _StaticResult(
        canonical_url="https://example.com/p",
        title="Widget", price_guess="$19.99", content="...", source="json_ld",
    )
    assert _needs_rendered_fallback(static, "", "https://example.com/p") is False


def test_no_rendered_fallback_for_og_with_price():
    static = _StaticResult(
        canonical_url="https://example.com/p",
        title="Widget", price_guess="$19.99", content="...", source="open_graph",
    )
    assert _needs_rendered_fallback(static, "", "https://example.com/p") is False


def test_rendered_fallback_for_marketplace_even_with_static_price():
    static = _StaticResult(
        canonical_url="https://tcgplayer.com/product/1",
        title="Card", price_guess="$13.33", content="...", source="text_regex",
    )
    html = "<p>Market price shown</p>"
    assert _needs_rendered_fallback(static, html, "https://tcgplayer.com/product/1") is True


def test_no_rendered_fallback_for_simple_page_with_trafilatura_price():
    static = _StaticResult(
        canonical_url="https://shop.example.com/widget",
        title="Widget", price_guess="$19.99", content="...", source="text_regex",
    )
    html = "<p>Buy this widget for $19.99</p>"
    assert _needs_rendered_fallback(static, html, "https://shop.example.com/widget") is False


# --- FetchedPage model ---

def test_price_candidates_default_does_not_share_mutable_state():
    page_a = FetchedPage(url="https://a.com", title=None, content="a", price_guess=None)
    page_b = FetchedPage(url="https://b.com", title=None, content="b", price_guess=None)
    page_a.price_candidates.append("$1.00")
    assert page_b.price_candidates == []


# --- Existing fetch_page tests ---

async def test_fetch_page_returns_fetched_page(mock_http_client, mock_no_shopify, mock_no_render):
    mock_http_client.get.return_value = make_mock_response(SAMPLE_HTML)
    result = await fetch_page("https://example.com/product")
    assert isinstance(result, FetchedPage)


async def test_fetch_page_extracts_title(mock_http_client, mock_no_shopify, mock_no_render):
    mock_http_client.get.return_value = make_mock_response(SAMPLE_HTML)
    result = await fetch_page("https://example.com/product")
    assert result.title is not None


async def test_fetch_page_extracts_content(mock_http_client, mock_no_shopify, mock_no_render):
    mock_http_client.get.return_value = make_mock_response(SAMPLE_HTML)
    result = await fetch_page("https://example.com/product")
    assert result.content != ""


async def test_fetch_page_extracts_price_guess(mock_http_client, mock_no_shopify, mock_no_render):
    mock_http_client.get.return_value = make_mock_response(SAMPLE_HTML)
    result = await fetch_page("https://example.com/product")
    assert result.price_guess == "$29.99"


async def test_fetch_page_uses_requested_url_when_no_canonical(
    mock_http_client, mock_no_shopify, mock_no_render
):
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


async def test_fetch_page_no_extractable_content_raises(
    mock_http_client, mock_no_shopify, mock_no_render
):
    mock_http_client.get.return_value = make_mock_response("<html><body></body></html>")
    with pytest.raises(ValueError, match="No extractable content"):
        await fetch_page("https://example.com/empty")


# --- JSON-LD extraction ---

async def test_fetch_page_uses_json_ld_price(mock_http_client, mock_no_shopify, mock_no_render):
    mock_http_client.get.return_value = make_mock_response(JSON_LD_HTML)
    result = await fetch_page("https://example.com/product")
    assert result.price_guess == "$199.99"


async def test_fetch_page_json_ld_sets_extraction_source(
    mock_http_client, mock_no_shopify, mock_no_render
):
    mock_http_client.get.return_value = make_mock_response(JSON_LD_HTML)
    result = await fetch_page("https://example.com/product")
    assert result.extraction_source == "json_ld"


async def test_fetch_page_json_ld_preserves_gbp_currency(
    mock_http_client, mock_no_shopify, mock_no_render
):
    mock_http_client.get.return_value = make_mock_response(JSON_LD_GBP_HTML)
    result = await fetch_page("https://example.com/product")
    assert result.price_guess is not None
    assert "£" in result.price_guess
    assert "$" not in result.price_guess


async def test_fetch_page_json_ld_product_name(
    mock_http_client, mock_no_shopify, mock_no_render
):
    mock_http_client.get.return_value = make_mock_response(JSON_LD_HTML)
    result = await fetch_page("https://example.com/product")
    assert result.title == "Test Headphones"


async def test_fetch_page_json_ld_content_does_not_contain_raw_html(
    mock_http_client, mock_no_shopify, mock_no_render
):
    mock_http_client.get.return_value = make_mock_response(JSON_LD_HTML)
    result = await fetch_page("https://example.com/product")
    assert "<html>" not in result.content
    assert "<script" not in result.content


# --- Open Graph extraction ---

async def test_fetch_page_uses_open_graph_price(
    mock_http_client, mock_no_shopify, mock_no_render
):
    mock_http_client.get.return_value = make_mock_response(OG_HTML)
    result = await fetch_page("https://example.com/product")
    assert result.price_guess is not None
    assert "199.99" in result.price_guess


async def test_fetch_page_open_graph_sets_extraction_source(
    mock_http_client, mock_no_shopify, mock_no_render
):
    mock_http_client.get.return_value = make_mock_response(OG_HTML)
    result = await fetch_page("https://example.com/product")
    assert result.extraction_source == "open_graph"


# --- __NEXT_DATA__ extraction ---

async def test_fetch_page_uses_next_data_prices(
    mock_http_client, mock_no_shopify, mock_no_render
):
    mock_http_client.get.return_value = make_mock_response(NEXT_DATA_HTML)
    result = await fetch_page("https://example.com/product")
    assert result.price_guess is not None
    assert result.extraction_source == "next_data"


# --- Shopify extraction ---

async def test_fetch_page_shopify_js_format_converts_cents(
    mock_http_client, mock_no_render
):
    shopify_js_product = {
        "title": "Cool Shirt",
        "vendor": "TestCo",
        "variants": [{"price": 2999}],  # cents
    }
    with patch(
        "app.tools.fetch_page._fetch_shopify_product",
        new_callable=AsyncMock,
        return_value=(shopify_js_product, "js"),
    ):
        mock_http_client.get.return_value = make_mock_response(
            "<html><body>shop</body></html>"
        )
        result = await fetch_page("https://myshop.com/products/cool-shirt")
    assert result.price_guess == "$29.99"
    assert result.extraction_source == "shopify_product_json"


async def test_fetch_page_shopify_json_format_uses_decimal_price(
    mock_http_client, mock_no_render
):
    shopify_json_product = {
        "title": "Cool Shirt",
        "vendor": "TestCo",
        "variants": [{"price": "19.99"}],
    }
    with patch(
        "app.tools.fetch_page._fetch_shopify_product",
        new_callable=AsyncMock,
        return_value=(shopify_json_product, "json"),
    ):
        mock_http_client.get.return_value = make_mock_response(
            "<html><body>shop</body></html>"
        )
        result = await fetch_page("https://myshop.com/products/cool-shirt")
    assert result.price_guess == "$19.99"


# --- Rendered extraction ---

async def test_fetch_page_falls_back_to_render_when_no_static_price(mock_http_client):
    from app.tools.fetch_page import _RenderedResult

    mock_http_client.get.return_value = make_mock_response(
        "<html><body>No price data here</body></html>"
    )
    rendered = _RenderedResult(
        title="Rendered Title",
        price_guess="$49.99",
        content="[Rendered page content]\nPrices visible: $49.99",
        price_candidates=["$49.99"],
    )
    with (
        patch(
            "app.tools.fetch_page._fetch_shopify_product",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "app.tools.fetch_page._render_page",
            new_callable=AsyncMock,
            return_value=rendered,
        ),
    ):
        result = await fetch_page("https://example.com/product")
    assert result.price_guess == "$49.99"
    assert result.extraction_source == "rendered_visible_text"


async def test_fetch_page_marketplace_triggers_render_even_with_static_price(mock_http_client):
    from app.tools.fetch_page import _RenderedResult

    # MARKETPLACE_HTML has listing signals → _needs_rendered_fallback returns True
    # even though trafilatura may find a price
    mock_http_client.get.return_value = make_mock_response(MARKETPLACE_HTML)
    rendered = _RenderedResult(
        title="TCGPlayer Card",
        price_guess="$9.00",
        content="[Rendered page content]\nPrices visible: $9.00",
        price_candidates=["$9.00", "$13.33"],
        price_candidate_labels=["as low as", "market price"],
        lowest_visible_price="$9.00",
        shipping_text="+ $1.50 Shipping",
        listing_count="10 listings",
        condition_text="Condition: Near Mint",
    )
    with (
        patch(
            "app.tools.fetch_page._fetch_shopify_product",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "app.tools.fetch_page._render_page",
            new_callable=AsyncMock,
            return_value=rendered,
        ),
    ):
        result = await fetch_page("https://tcgplayer.com/product/1")
    assert result.extraction_source == "rendered_visible_text"
    assert result.lowest_visible_price == "$9.00"
    assert result.shipping_text == "+ $1.50 Shipping"
    assert result.listing_count == "10 listings"
    assert result.condition_text == "Condition: Near Mint"
    assert "$9.00" in result.price_candidates
    assert "as low as" in result.price_candidate_labels


async def test_fetch_page_render_failure_does_not_set_rendered_source(mock_http_client):
    from app.tools.fetch_page import _RenderedResult

    mock_http_client.get.return_value = make_mock_response(SAMPLE_HTML)
    empty_render = _RenderedResult(title=None, price_guess=None, content="")
    with (
        patch(
            "app.tools.fetch_page._fetch_shopify_product",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "app.tools.fetch_page._render_page",
            new_callable=AsyncMock,
            return_value=empty_render,
        ),
    ):
        result = await fetch_page("https://example.com/product")
    assert result.extraction_source != "rendered_visible_text"


async def test_fetch_page_rendered_content_does_not_contain_raw_html(mock_http_client):
    from app.tools.fetch_page import _RenderedResult

    mock_http_client.get.return_value = make_mock_response(
        "<html><body>No price</body></html>"
    )
    rendered = _RenderedResult(
        title="Page",
        price_guess="$9.00",
        content="[Rendered page content]\nSome visible text $9.00",
    )
    with (
        patch(
            "app.tools.fetch_page._fetch_shopify_product",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "app.tools.fetch_page._render_page",
            new_callable=AsyncMock,
            return_value=rendered,
        ),
    ):
        result = await fetch_page("https://example.com/product")
    assert "<html>" not in result.content
    assert "<script" not in result.content
    assert "<body>" not in result.content


async def test_fetch_page_static_json_ld_skips_render(mock_http_client, mock_no_shopify):
    from app.tools.fetch_page import _RenderedResult

    mock_http_client.get.return_value = make_mock_response(JSON_LD_HTML)
    mock_render = AsyncMock(
        return_value=_RenderedResult(title=None, price_guess=None, content="")
    )
    with patch("app.tools.fetch_page._render_page", mock_render):
        result = await fetch_page("https://example.com/product")
    mock_render.assert_not_called()
    assert result.extraction_source == "json_ld"


async def test_render_failure_is_logged(mock_http_client, caplog):
    import logging

    mock_http_client.get.return_value = make_mock_response(
        "<html><body>No price</body></html>"
    )
    with (
        patch(
            "app.tools.fetch_page._fetch_shopify_product",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "app.tools.fetch_page._render_page",
            new_callable=AsyncMock,
            side_effect=Exception("browser crash"),
        ),
        caplog.at_level(logging.WARNING, logger="app.tools.fetch_page"),
    ):
        with pytest.raises(Exception):
            await fetch_page("https://example.com/product")


# --- URL safety tests ---


def test_valid_http_url_passes_validation():
    _validate_fetch_url("http://example.com/product")


def test_valid_https_url_passes_validation():
    _validate_fetch_url("https://example.com/product")


def test_ftp_scheme_is_rejected():
    with pytest.raises(ValueError, match="http and https"):
        _validate_fetch_url("ftp://example.com/file")


def test_file_scheme_is_rejected():
    with pytest.raises(ValueError, match="http and https"):
        _validate_fetch_url("file:///etc/passwd")


def test_data_scheme_is_rejected():
    with pytest.raises(ValueError, match="http and https"):
        _validate_fetch_url("data:text/html,<script>alert(1)</script>")


def test_localhost_hostname_is_rejected():
    with pytest.raises(ValueError, match="localhost"):
        _validate_fetch_url("http://localhost/admin")


def test_localhost_with_port_is_rejected():
    with pytest.raises(ValueError, match="localhost"):
        _validate_fetch_url("http://localhost:8080/admin")


def test_loopback_ip_127_is_rejected():
    with pytest.raises(ValueError, match="private"):
        _validate_fetch_url("http://127.0.0.1/admin")


def test_unspecified_ip_0_0_0_0_is_rejected():
    with pytest.raises(ValueError, match="private"):
        _validate_fetch_url("http://0.0.0.0/")


def test_private_ip_10_block_is_rejected():
    with pytest.raises(ValueError, match="private"):
        _validate_fetch_url("http://10.0.0.1/internal")


def test_private_ip_172_16_block_is_rejected():
    with pytest.raises(ValueError, match="private"):
        _validate_fetch_url("http://172.16.0.1/internal")


def test_private_ip_192_168_block_is_rejected():
    with pytest.raises(ValueError, match="private"):
        _validate_fetch_url("http://192.168.1.1/internal")


def test_link_local_169_254_is_rejected():
    with pytest.raises(ValueError, match="private"):
        _validate_fetch_url("http://169.254.0.1/")


def test_cloud_metadata_ip_is_rejected():
    with pytest.raises(ValueError, match="private"):
        _validate_fetch_url("http://169.254.169.254/latest/meta-data/")


def test_ipv6_loopback_is_rejected():
    with pytest.raises(ValueError, match="private"):
        _validate_fetch_url("http://[::1]/")


def test_ipv6_link_local_is_rejected():
    with pytest.raises(ValueError, match="private"):
        _validate_fetch_url("http://[fe80::1]/")


def test_null_byte_in_url_is_rejected():
    with pytest.raises(ValueError, match="unsafe characters"):
        _validate_fetch_url("http://example.com/\x00path")


def test_newline_in_url_is_rejected():
    with pytest.raises(ValueError, match="unsafe characters"):
        _validate_fetch_url("http://example.com/path\ninjection")


def test_embedded_space_in_url_is_rejected():
    with pytest.raises(ValueError, match="unsafe characters"):
        _validate_fetch_url("https://example.com/prod uct")


def test_percent_encoded_space_in_url_is_allowed():
    _validate_fetch_url("https://example.com/product%20name")


def test_unicode_format_char_in_url_is_rejected():
    with pytest.raises(ValueError, match="unsafe characters"):
        _validate_fetch_url("http://example.com/path​")


def test_emoji_in_hostname_is_rejected():
    with pytest.raises(ValueError, match="non-ASCII"):
        _validate_fetch_url("https://exa\U0001f381mple.com/product")


def test_redirect_hook_blocks_unsafe_ip_target():
    request = httpx.Request("GET", "http://169.254.169.254/metadata")
    with pytest.raises(ValueError, match="private"):
        import asyncio
        asyncio.get_event_loop().run_until_complete(_check_redirect_url_is_safe(request))


def test_redirect_hook_blocks_localhost_target():
    request = httpx.Request("GET", "http://localhost/admin")
    with pytest.raises(ValueError, match="localhost"):
        import asyncio
        asyncio.get_event_loop().run_until_complete(_check_redirect_url_is_safe(request))


async def test_hostname_resolving_to_private_ip_is_rejected():
    fake_results = [(None, None, None, None, ("192.168.1.100", 0))]
    with patch("socket.getaddrinfo", return_value=fake_results):
        with pytest.raises(ValueError, match="private"):
            await _validate_resolved_addresses_are_safe("internal.corp")


async def test_hostname_resolving_to_public_ip_passes():
    fake_results = [(None, None, None, None, ("93.184.216.34", 0))]
    with patch("socket.getaddrinfo", return_value=fake_results):
        await _validate_resolved_addresses_are_safe("example.com")


async def test_fetch_page_rejects_private_ip_before_any_request():
    with pytest.raises(ValueError, match="private"):
        await fetch_page("http://192.168.1.1/product")


async def test_fetch_page_rejects_localhost_before_any_request():
    with pytest.raises(ValueError, match="localhost"):
        await fetch_page("http://localhost/admin")


async def test_fetch_page_rejects_disallowed_scheme_before_any_request():
    with pytest.raises(ValueError, match="http and https"):
        await fetch_page("ftp://example.com/file")
