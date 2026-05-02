from app.tools.fetch_page import FetchedPage
from app.tools.offer_candidates import (
    OfferCandidate,
    candidate_from_page,
    candidate_from_search_result,
    extract_offer_price_text,
    format_offer_table,
    is_same_size,
    is_unavailable,
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


def test_is_same_size_fl_oz_equivalent_to_ml():
    assert is_same_size("Haruharu Cleanser 100ml", "Wonder Cleanser 3.4 fl oz") is True


def test_is_same_size_3_38_fl_oz_equivalent_to_100ml():
    assert is_same_size("Haruharu Cleanser 100ml", "Wonder Cleanser 3.38 fl oz") is True


def test_is_same_size_200ml_not_equivalent_to_100ml_fl_oz():
    assert is_same_size("Haruharu Cleanser 200ml", "Wonder Cleanser 3.4 fl oz") is False


def test_is_same_size_dot_fl_oz_format():
    assert is_same_size("Haruharu Cleanser 100ml", "Wonder Cleanser 3.4 fl.oz") is True


def test_is_same_size_uses_match_text_when_title_lacks_size():
    assert is_same_size(
        "Haruharu Cleanser 100ml",
        "Wonder Cleanser",
        match_text="100ml / 3.4 fl oz",
    ) is True


def test_is_same_size_match_text_with_fl_oz_equivalent():
    assert is_same_size(
        "Haruharu Cleanser 100ml",
        "Wonder Cleanser",
        match_text="3.4 fl oz cleanser",
    ) is True


def test_is_same_size_wrong_size_in_match_text():
    assert is_same_size(
        "Haruharu Cleanser 100ml",
        "Wonder Cleanser",
        match_text="200ml cleanser",
    ) is False


# --- is_unavailable ---


def test_is_unavailable_out_of_stock():
    candidate = OfferCandidate(
        merchant="www.walmart.com",
        source_url="https://www.walmart.com/p",
        product_name="Cleanser 100ml",
        price_text="$5.99",
        price_amount=5.99,
        match_text="This item is out of stock.",
    )
    assert is_unavailable(candidate) is True


def test_is_unavailable_sold_out():
    candidate = OfferCandidate(
        merchant="www.target.com",
        source_url="https://www.target.com/p",
        product_name="Cleanser 100ml",
        price_text="$6.99",
        price_amount=6.99,
        match_text="Sold out. Check back later.",
    )
    assert is_unavailable(candidate) is True


def test_is_unavailable_currently_unavailable():
    candidate = OfferCandidate(
        merchant="www.amazon.com",
        source_url="https://www.amazon.com/p",
        product_name="Cleanser 100ml",
        price_text="$7.95",
        price_amount=7.95,
        match_text="Currently unavailable.",
    )
    assert is_unavailable(candidate) is True


def test_is_unavailable_unavailable_signal_in_product_name():
    candidate = OfferCandidate(
        merchant="www.amazon.com",
        source_url="https://www.amazon.com/p",
        product_name="Cleanser 100ml - Unavailable",
        price_text="$7.95",
        price_amount=7.95,
        match_text="",
    )
    assert is_unavailable(candidate) is True


def test_is_unavailable_in_stock_candidate_returns_false():
    candidate = OfferCandidate(
        merchant="www.amazon.com",
        source_url="https://www.amazon.com/p",
        product_name="Cleanser 100ml",
        price_text="$7.95",
        price_amount=7.95,
        match_text="In stock. Ships within 2 days.",
    )
    assert is_unavailable(candidate) is False


def test_is_unavailable_no_longer_available():
    candidate = OfferCandidate(
        merchant="www.ebay.com",
        source_url="https://www.ebay.com/p",
        product_name="Pokémon Card",
        price_text="$6.50",
        price_amount=6.50,
        match_text="This item is no longer available.",
    )
    assert is_unavailable(candidate) is True


def test_is_unavailable_not_available():
    candidate = OfferCandidate(
        merchant="www.target.com",
        source_url="https://www.target.com/p",
        product_name="Pokémon Card",
        price_text="$8.00",
        price_amount=8.00,
        match_text="Not available in your region.",
    )
    assert is_unavailable(candidate) is True


def test_is_unavailable_temporarily_out_of_stock():
    candidate = OfferCandidate(
        merchant="www.walmart.com",
        source_url="https://www.walmart.com/p",
        product_name="Pokémon Card",
        price_text="$9.00",
        price_amount=9.00,
        match_text="Temporarily out of stock.",
    )
    assert is_unavailable(candidate) is True


def test_is_unavailable_item_ended():
    candidate = OfferCandidate(
        merchant="www.ebay.com",
        source_url="https://www.ebay.com/p",
        product_name="Pokémon Card",
        price_text="$6.50",
        price_amount=6.50,
        match_text="Item ended. This listing is no longer accepting bids.",
    )
    assert is_unavailable(candidate) is True


def test_is_unavailable_listing_ended():
    candidate = OfferCandidate(
        merchant="www.ebay.com",
        source_url="https://www.ebay.com/p",
        product_name="Pokémon Card",
        price_text="$6.50",
        price_amount=6.50,
        match_text="Listing ended.",
    )
    assert is_unavailable(candidate) is True


def test_is_unavailable_ended_standalone():
    candidate = OfferCandidate(
        merchant="www.ebay.com",
        source_url="https://www.ebay.com/p",
        product_name="Pokémon Card",
        price_text="$6.50",
        price_amount=6.50,
        match_text="This auction has ended.",
    )
    assert is_unavailable(candidate) is True


def test_is_unavailable_unknown_availability_returns_false():
    candidate = OfferCandidate(
        merchant="www.ebay.com",
        source_url="https://www.ebay.com/p",
        product_name="Pokémon Card",
        price_text="$6.50",
        price_amount=6.50,
        match_text="Great condition. Fast shipping.",
    )
    assert is_unavailable(candidate) is False


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


def test_candidate_from_page_populates_match_text_from_content():
    page = FetchedPage(
        url="https://www.amazon.com/product",
        title="Cleanser",
        content="100ml / 3.4 fl oz. In stock.",
        price_guess="$7.95",
    )
    assert "3.4 fl oz" in candidate_from_page(page).match_text


def test_candidate_from_page_match_text_truncates_long_content():
    page = FetchedPage(
        url="https://www.amazon.com/product",
        title="Cleanser",
        content="x" * 1000,
        price_guess="$7.95",
    )
    assert len(candidate_from_page(page).match_text) <= 500


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
    assert "3.4 fl oz" in candidate.match_text


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