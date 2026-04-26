from app.tools.extract_product import ProductPageExtraction
from app.tools.product_identity import ProductIdentity, infer_product_identity


def make_extraction(**overrides) -> ProductPageExtraction:
    base = {"product_name": None, "listed_price": None, "merchant": None}
    base.update(overrides)
    return ProductPageExtraction(**base)


def test_returns_product_name_when_present():
    extraction = make_extraction(product_name="Sony WH-1000XM5")
    result = infer_product_identity(extraction, "https://example.com/product")
    assert result.value == "Sony WH-1000XM5"
    assert result.source == "product_name"


def test_source_is_product_name_when_product_name_used():
    extraction = make_extraction(product_name="Sony WH-1000XM5")
    result = infer_product_identity(extraction, "https://example.com/product")
    assert result.source == "product_name"
    assert result.value == "Sony WH-1000XM5"


def test_falls_back_to_url_slug_when_product_name_missing():
    extraction = make_extraction()
    result = infer_product_identity(extraction, "https://example.com/great-widget-pro")
    assert result.value == "great widget pro"
    assert result.source == "url_slug"


def test_source_is_url_slug_when_slug_used():
    extraction = make_extraction()
    result = infer_product_identity(extraction, "https://example.com/great-widget-pro")
    assert result.source == "url_slug"
    assert result.value == "great widget pro"


def test_slug_fallback_ignores_numeric_only_segments():
    extraction = make_extraction()
    result = infer_product_identity(extraction, "https://example.com/12345/great-widget")
    assert "12345" not in result.value
    assert result.value == "great widget"
    assert result.source == "url_slug"


def test_slug_fallback_replaces_hyphens_with_spaces():
    extraction = make_extraction()
    result = infer_product_identity(extraction, "https://example.com/great-widget-pro")
    assert "-" not in result.value
    assert result.source == "url_slug"


def test_slug_fallback_replaces_underscores_with_spaces():
    extraction = make_extraction()
    result = infer_product_identity(extraction, "https://example.com/great_widget_pro")
    assert result.value == "great widget pro"
    assert result.source == "url_slug"


def test_falls_back_to_domain_path_when_no_usable_slug():
    extraction = make_extraction()
    result = infer_product_identity(extraction, "https://example.com/dp/12345")
    assert result.value == "example.com/dp/12345"
    assert result.source == "domain_path"


def test_source_is_domain_path_when_domain_path_used():
    extraction = make_extraction()
    result = infer_product_identity(extraction, "https://example.com/dp/12345")
    assert result.source == "domain_path"
    assert result.value == "example.com/dp/12345"


def test_returns_insufficient_data_when_nothing_usable():
    # No hostname, path contains only a numeric segment — nothing usable to infer identity from
    extraction = make_extraction()
    result = infer_product_identity(extraction, "https:///123")
    assert result.value == "insufficient_data"
    assert result.source == "insufficient_data"


def test_source_is_insufficient_data_in_final_fallback():
    extraction = make_extraction()
    result = infer_product_identity(extraction, "https:///123")
    assert result.source == "insufficient_data"
    assert result.value == "insufficient_data"


def test_result_is_product_identity_model():
    extraction = make_extraction(product_name="Widget")
    result = infer_product_identity(extraction, "https://example.com/product")
    assert isinstance(result, ProductIdentity)
