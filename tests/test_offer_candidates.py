from app.tools.fetch_page import FetchedPage
from app.tools.offer_candidates import (
    OfferCandidate,
    candidate_from_page,
    candidate_from_search_result,
    extract_offer_price_text,
    format_offer_table,
    is_same_size,
    parse_price_amount,
)
from app.tools.search_web import SearchResult


# --- parse_price_amount ---


def test_parse_price_amount_with_dollar_sign():
    assert parse_price_amount("$7.95") == 7.95


def test_parse_price_amount_without_dollar_sign():
    assert parse_price_amount("7.95") == 7.95


def test_parse_price_amount_with_commas():
    assert parse_price_amount("$1,299.99") == 1299.99


def test_parse_price_amount_whole_number():
    assert parse_price_amount("$7") == 7.0


def test_parse_price_amount_embedded_in_text():
    assert parse_price_amount("Price: $13.59") == 13.59


def test_parse_price_amount_none_input():
    assert parse_price_amount(None) is None


def test_parse_price_amount_empty_string():
    assert parse_price_amount("") is None


def test_parse_price_amount_no_numeric_content():
    assert parse_price_amount("no price here") is None


# --- extract_offer_price_text ---


def test_extract_offer_price_text_requires_currency_signal():
    assert extract_offer_price_text("Haruharu cleanser 100ml") is None


def test_extract_offer_price_text_extracts_dollar_price():
    assert extract_offer_price_text("Available now for $7.95 on Amazon") == "$7.95"


def test_extract_offer_price_text_extracts_usd_suffix():
    assert extract_offer_price_text("Current price 7.95 USD") == "7.95 USD"


def test_extract_offer_price_text_ignores_size_without_currency():
    assert extract_offer_price_text("Black Rice Moisture 5.5 Soft Cleansing Gel 100ml") is None


# --- is_same_size ---


def test_is_same_size_matching_ml():
    assert is_same_size("Haruharu Cleanser 100ml", "Wonder Cleanser 100ml") is True


def test_is_same_size_different_ml():
    assert is_same_size("Haruharu Cleanser 100ml", "Wonder Cleanser 200ml") is False


def test_is_same_size_no_size_in_submitted_is_permissive():
    assert is_same_size("Haruharu Cleanser", "Wonder Cleanser 100ml") is True


def test_is_same_size_no_candidate_name_is_permissive():
    assert is_same_size("Haruharu Cleanser 100ml", None) is True


def test_is_same_size_matching_oz():
    assert is_same_size("Moisturizer 2oz", "Daily Moisturizer 2oz") is True


def test_is_same_size_different_oz():
    assert is_same_size("Moisturizer 2oz", "Daily Moisturizer 4oz") is False


def test_is_same_size_both_no_size_is_permissive():
    assert is_same_size("Haruharu Cleanser", "Some Other Cleanser") is True


# --- candidate_from_page ---


def test_candidate_from_page_extracts_merchant_from_hostname():
    page = FetchedPage(
        url="https://www.amazon.com/product",
        title="Cleanser 100ml",
        content="...",
        price_guess="$7.95",
    )
    candidate = candidate_from_page(page)
    assert candidate.merchant == "www.amazon.com"


def test_candidate_from_page_extracts_price_amount():
    page = FetchedPage(
        url="https://www.amazon.com/product",
        title="Cleanser 100ml",
        content="...",
        price_guess="$7.95",
    )
    candidate = candidate_from_page(page)
    assert candidate.price_amount == 7.95


def test_candidate_from_page_price_amount_none_when_no_price():
    page = FetchedPage(
        url="https://www.amazon.com/product",
        title="Cleanser 100ml",
        content="...",
        price_guess=None,
    )
    candidate = candidate_from_page(page)
    assert candidate.price_amount is None


def test_candidate_from_page_preserves_source_url():
    page = FetchedPage(
        url="https://www.amazon.com/product",
        title="Cleanser",
        content="...",
        price_guess="$7.95",
    )
    assert candidate_from_page(page).source_url == "https://www.amazon.com/product"


def test_candidate_from_page_marks_source_type_fetched_page():
    page = FetchedPage(
        url="https://www.amazon.com/product",
        title="Cleanser",
        content="...",
        price_guess="$7.95",
    )
    assert candidate_from_page(page).source_type == "fetched_page"


# --- candidate_from_search_result ---


def test_candidate_from_search_result_returns_candidate_when_snippet_contains_price():
    result = SearchResult(
        title="Amazon.com: Haruharu Wonder Black Rice Moisture 5.5 Soft Cleansing Gel",
        url="https://www.amazon.com/Haruharu-Moisture-Cleansing-Fermented-Cleanser/dp/B08W2HG4WP",
        snippet="Current price: $7.95 one-time purchase. 100ml / 3.4 fl oz.",
        metadata={"source": "tavily", "score": 0.9},
    )

    candidate = candidate_from_search_result(result)

    assert candidate is not None
    assert candidate.merchant == "www.amazon.com"
    assert candidate.source_url == str(result.url)
    assert candidate.product_name == result.title
    assert candidate.price_text == "$7.95"
    assert candidate.price_amount == 7.95
    assert candidate.source_type == "search_result"


def test_candidate_from_search_result_returns_candidate_when_title_contains_price():
    result = SearchResult(
        title="Haruharu Cleanser 100ml - $7.95",
        url="https://www.amazon.com/product",
        snippet="Same cleanser.",
        metadata={"source": "tavily", "score": 0.9},
    )

    candidate = candidate_from_search_result(result)

    assert candidate is not None
    assert candidate.price_text == "$7.95"
    assert candidate.price_amount == 7.95


def test_candidate_from_search_result_returns_none_when_only_size_has_number():
    result = SearchResult(
        title="Haruharu Wonder Black Rice Moisture 5.5 Soft Cleansing Gel 100ml",
        url="https://www.amazon.com/product",
        snippet="Same 100ml product. No current price shown.",
        metadata={"source": "tavily", "score": 0.9},
    )

    assert candidate_from_search_result(result) is None


def test_candidate_from_search_result_returns_none_without_parseable_price():
    result = SearchResult(
        title="Haruharu cleanser at Amazon",
        url="https://www.amazon.com/product",
        snippet="Same product, price not shown.",
        metadata={"source": "tavily", "score": 0.9},
    )

    assert candidate_from_search_result(result) is None


# --- format_offer_table ---


def test_format_offer_table_empty_candidates():
    assert format_offer_table([]) == ""


def test_format_offer_table_includes_merchant_and_price():
    candidates = [
        OfferCandidate(
            merchant="www.amazon.com",
            source_url="https://www.amazon.com/p",
            product_name="Cleanser 100ml",
            price_text="$7.95",
            price_amount=7.95,
        ),
    ]
    table = format_offer_table(candidates)
    assert "www.amazon.com" in table
    assert "$7.95" in table


def test_format_offer_table_sorted_order_preserved():
    candidates = [
        OfferCandidate(
            merchant="www.amazon.com",
            source_url="https://www.amazon.com/p",
            product_name="Cleanser",
            price_text="$7.95",
            price_amount=7.95,
        ),
        OfferCandidate(
            merchant="www.walmart.com",
            source_url="https://www.walmart.com/p",
            product_name="Cleanser",
            price_text="$13.59",
            price_amount=13.59,
        ),
    ]
    table = format_offer_table(candidates)
    assert table.index("amazon") < table.index("walmart")


def test_format_offer_table_includes_guidance_line():
    candidates = [
        OfferCandidate(
            merchant="www.amazon.com",
            source_url="https://www.amazon.com/p",
            product_name="Cleanser",
            price_text="$7.95",
            price_amount=7.95,
        ),
    ]
    table = format_offer_table(candidates)
    assert "lowest eligible" in table


def test_format_offer_table_includes_source_type():
    candidates = [
        OfferCandidate(
            merchant="www.amazon.com",
            source_url="https://www.amazon.com/p",
            product_name="Cleanser",
            price_text="$7.95",
            price_amount=7.95,
            source_type="search_result",
        ),
    ]
    table = format_offer_table(candidates)
    assert "search_result" in table