import pytest

from app.tools.normalize_url import fetch_page_cache_key, normalize_url


def test_empty_string_raises():
    with pytest.raises(ValueError, match="blank"):
        normalize_url("")


def test_whitespace_only_raises():
    with pytest.raises(ValueError, match="blank"):
        normalize_url("   ")


def test_missing_scheme_raises():
    with pytest.raises(ValueError, match="http or https"):
        normalize_url("example.com/product")


def test_non_http_scheme_raises():
    with pytest.raises(ValueError, match="http or https"):
        normalize_url("ftp://example.com/product")


def test_missing_host_raises():
    with pytest.raises(ValueError, match="missing a host"):
        normalize_url("https:///no-host")


def test_scheme_is_lowercased():
    result = normalize_url("HTTPS://example.com/product")
    assert result.startswith("https://")


def test_host_is_lowercased():
    result = normalize_url("https://Example.COM/product")
    assert "example.com" in result


def test_http_default_port_removed():
    result = normalize_url("http://example.com:80/product")
    assert ":80" not in result


def test_https_default_port_removed():
    result = normalize_url("https://example.com:443/product")
    assert ":443" not in result


def test_non_default_port_preserved():
    result = normalize_url("https://example.com:8080/product")
    assert ":8080" in result


def test_http_port_443_not_removed():
    result = normalize_url("http://example.com:443/product")
    assert ":443" in result


def test_https_port_80_not_removed():
    result = normalize_url("https://example.com:80/product")
    assert ":80" in result


def test_trailing_slash_removed():
    result = normalize_url("https://example.com/product/")
    assert not result.endswith("/")


def test_multiple_trailing_slashes_removed():
    result = normalize_url("https://example.com/product///")
    assert not result.endswith("/")


def test_root_path_preserved():
    result = normalize_url("https://example.com/")
    assert result == "https://example.com/"


def test_fragment_removed():
    result = normalize_url("https://example.com/product#reviews")
    assert "#" not in result


def test_utm_params_removed():
    result = normalize_url(
        "https://example.com/product?utm_source=google&utm_medium=cpc&utm_campaign=sale"
    )
    assert "utm_" not in result


def test_fbclid_removed():
    result = normalize_url("https://example.com/product?fbclid=abc123")
    assert "fbclid" not in result


def test_gclid_removed():
    result = normalize_url("https://example.com/product?gclid=abc123")
    assert "gclid" not in result


def test_tracking_param_matching_is_case_insensitive():
    result = normalize_url("https://example.com/product?UTM_SOURCE=google")
    assert "UTM_SOURCE" not in result


def test_non_tracking_params_preserved():
    result = normalize_url("https://example.com/search?q=widget&color=red")
    assert "q=widget" in result
    assert "color=red" in result


def test_query_params_sorted():
    result = normalize_url("https://example.com/product?z=last&a=first")
    assert result.index("a=first") < result.index("z=last")


def test_product_slug_preserved():
    result = normalize_url("https://amazon.com/dp/B09XYZ1234/ref=sr_1_1?utm_source=google")
    assert "B09XYZ1234" in result


def test_same_logical_url_normalizes_identically():
    url_a = normalize_url("https://Example.com/product/?utm_source=google&fbclid=123#section")
    url_b = normalize_url("HTTPS://example.com/product/?fbclid=123&utm_source=google#top")
    assert url_a == url_b


def test_idempotent():
    url = "https://example.com/product?color=red&size=large"
    assert normalize_url(url) == normalize_url(normalize_url(url))


# --- fetch_page_cache_key ---


def test_fetch_page_cache_key_amazon_dp_path_returns_asin_key():
    url = "https://www.amazon.com/Product-Name/dp/B08W2HG4WP?th=1"
    assert fetch_page_cache_key(url) == "https://www.amazon.com/dp/B08W2HG4WP"


def test_fetch_page_cache_key_amazon_with_tracking_params_matches_clean_url():
    submitted = (
        "https://www.amazon.com/Product-Name/dp/B08W2HG4WP"
        "?_encoding=UTF8&pd_rd_r=abc&pd_rd_w=def&pf_rd_p=ghi&pf_rd_r=jkl"
    )
    agent_requested = "https://www.amazon.com/Haruharu/dp/B08W2HG4WP?th=1"
    assert fetch_page_cache_key(submitted) == fetch_page_cache_key(agent_requested)


def test_fetch_page_cache_key_amazon_gp_product_path():
    url = "https://www.amazon.com/gp/product/B08W2HG4WP?pf_rd_r=abc"
    assert fetch_page_cache_key(url) == "https://www.amazon.com/dp/B08W2HG4WP"


def test_fetch_page_cache_key_amazon_different_asins_do_not_match():
    url_a = "https://www.amazon.com/dp/B08W2HG4WP"
    url_b = "https://www.amazon.com/dp/B09XYZ12345"
    assert fetch_page_cache_key(url_a) != fetch_page_cache_key(url_b)


def test_fetch_page_cache_key_non_amazon_uses_normalize_url():
    url = "https://example.com/product?utm_source=google"
    assert fetch_page_cache_key(url) == normalize_url(url)


def test_fetch_page_cache_key_invalid_url_returns_raw():
    assert fetch_page_cache_key("not-a-url") == "not-a-url"
