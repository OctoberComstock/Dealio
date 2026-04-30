from app.prompts import build_initial_prompt, build_seeded_retailer_queries
from app.tools.extract_product import ProductPageExtraction
from app.tools.product_identity import ProductIdentity

# --- build_seeded_retailer_queries ---


def test_seeded_queries_include_all_four_retailers():
    queries = build_seeded_retailer_queries("Haruharu Wonder Cleanser", "www.kiyoko.com")
    sites = ["amazon.com", "walmart.com", "target.com", "ebay.com"]
    for site in sites:
        assert any(f"site:{site}" in q for q in queries)


def test_seeded_queries_contain_product_name():
    product = "Haruharu Wonder Black Rice Moisture 5.5 Soft Cleansing Gel"
    queries = build_seeded_retailer_queries(product, "www.kiyoko.com")
    for query in queries:
        assert product in query


def test_seeded_queries_no_brand_site_for_non_retailer_submission():
    queries = build_seeded_retailer_queries("Haruharu Wonder Cleanser", "www.kiyoko.com")
    assert not any("official site" in q for q in queries)


def test_seeded_queries_adds_brand_site_when_submitted_from_amazon():
    queries = build_seeded_retailer_queries("Haruharu Wonder Cleanser", "www.amazon.com")
    assert any("official site" in q for q in queries)


def test_seeded_queries_adds_brand_site_when_submitted_from_walmart():
    queries = build_seeded_retailer_queries("Haruharu Wonder Cleanser", "www.walmart.com")
    assert any("official site" in q for q in queries)


def test_seeded_queries_adds_brand_site_when_submitted_from_target():
    queries = build_seeded_retailer_queries("Haruharu Wonder Cleanser", "www.target.com")
    assert any("official site" in q for q in queries)


def test_seeded_queries_adds_brand_site_when_submitted_from_ebay():
    queries = build_seeded_retailer_queries("Haruharu Wonder Cleanser", "www.ebay.com")
    assert any("official site" in q for q in queries)


def test_seeded_queries_returns_four_items_for_non_retailer_url():
    queries = build_seeded_retailer_queries("Some Product", "www.brandsite.com")
    assert len(queries) == 4


def test_seeded_queries_returns_five_items_for_major_retailer_url():
    queries = build_seeded_retailer_queries("Some Product", "amazon.com")
    assert len(queries) == 5


# --- build_initial_prompt ---


def _make_identity(value: str, source: str) -> ProductIdentity:
    return ProductIdentity(value=value, source=source)


def _make_extraction(name: str | None = None, price: str | None = None) -> ProductPageExtraction:
    return ProductPageExtraction(product_name=name, listed_price=price, merchant=None)


def test_initial_prompt_includes_seeded_retailer_queries():
    identity = _make_identity("Haruharu Wonder Cleanser 100ml", "product_name")
    extraction = _make_extraction("Haruharu Wonder Cleanser 100ml", "$18.99")
    prompt = build_initial_prompt(
        "https://www.kiyoko.com/product/cleanser",
        extraction,
        identity,
    )
    assert "site:amazon.com" in prompt
    assert "site:walmart.com" in prompt
    assert "site:target.com" in prompt
    assert "site:ebay.com" in prompt


def test_initial_prompt_seeded_queries_use_identity_value():
    identity = _make_identity("Haruharu Wonder Cleanser 100ml", "product_name")
    extraction = _make_extraction("Different Extracted Name", "$18.99")
    prompt = build_initial_prompt(
        "https://www.kiyoko.com/product/cleanser",
        extraction,
        identity,
    )
    assert "Haruharu Wonder Cleanser 100ml" in prompt


def test_initial_prompt_falls_back_to_extraction_name_when_identity_insufficient():
    identity = _make_identity("insufficient_data", "insufficient_data")
    extraction = _make_extraction("Fallback Product Name", "$18.99")
    prompt = build_initial_prompt(
        "https://www.kiyoko.com/product/cleanser",
        extraction,
        identity,
    )
    assert "site:amazon.com Fallback Product Name" in prompt


def test_initial_prompt_no_seeded_queries_when_no_product_name():
    identity = _make_identity("insufficient_data", "insufficient_data")
    extraction = _make_extraction(name=None)
    prompt = build_initial_prompt(
        "https://www.kiyoko.com/product/cleanser",
        extraction,
        identity,
    )
    assert "site:amazon.com" not in prompt


def test_initial_prompt_includes_brand_site_when_submitted_from_amazon():
    identity = _make_identity("Haruharu Wonder Cleanser 100ml", "product_name")
    extraction = _make_extraction("Haruharu Wonder Cleanser 100ml", "$18.99")
    prompt = build_initial_prompt(
        "https://www.amazon.com/dp/B08W2HG4WP",
        extraction,
        identity,
    )
    assert "official site" in prompt


def test_initial_prompt_no_brand_site_when_submitted_from_brand():
    identity = _make_identity("Haruharu Wonder Cleanser 100ml", "product_name")
    extraction = _make_extraction("Haruharu Wonder Cleanser 100ml", "$18.99")
    prompt = build_initial_prompt(
        "https://www.haruharu.com/products/cleanser",
        extraction,
        identity,
    )
    assert "official site" not in prompt
