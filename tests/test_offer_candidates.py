from app.tools.fetch_page import FetchedPage
from app.tools.offer_candidates import (
    OfferCandidate,
    assess_product_identity,
    candidate_from_page,
    candidate_from_search_result,
    extract_offer_price_text,
    format_offer_table,
    is_accessory_or_partial_listing,
    is_installment_price,
    is_purchasable_offer_candidate,
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


def test_is_same_size_non_volume_size_mismatch_returns_false():
    # Regression: non-volume sizes (ct, g, pack) must not call
    # _candidate_text_has_volume_equivalent with ml_value=None.
    assert is_same_size("Pokémon Card 10ct", "Pokémon Card 20ct") is False


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


# --- candidate identity and listing type ---


def test_assess_product_identity_accepts_same_product_with_model_identity():
    match = assess_product_identity(
        "Bissell PowerClean FurGuard 280W Cordless Vacuum 4039",
        "BISSELL PowerClean FurGuard Cordless Vacuum, Model 4039",
        "280W self-standing vacuum for pet hair.",
    )

    assert match.matches is True
    assert match.is_strong is True
    assert "4039" in match.reason


def test_assess_product_identity_rejects_unrelated_cheap_product():
    match = assess_product_identity(
        "Bissell PowerClean FurGuard 280W Cordless Vacuum 4039",
        "Generic Handheld Car Vacuum",
        "Compact rechargeable cleaner for $19.99.",
    )

    assert match.matches is False
    assert match.is_strong is False


def test_assess_product_identity_checks_products_without_recognized_size():
    match = assess_product_identity(
        "Bissell PowerClean FurGuard Cordless Vacuum",
        "Shark Navigator Lift-Away Upright Vacuum",
        "Full-size vacuum cleaner.",
    )

    assert match.matches is False


def test_is_accessory_or_partial_listing_detects_replacement_part():
    candidate = OfferCandidate(
        merchant="www.ebay.com",
        source_url="https://www.ebay.com/itm/123",
        product_name="Replacement Brush Roll for Bissell PowerClean FurGuard 4039",
        price_text="$19.99",
        price_amount=19.99,
        source_type="search_result",
    )

    assert is_accessory_or_partial_listing(candidate) is True


def test_is_accessory_or_partial_listing_allows_complete_product():
    candidate = OfferCandidate(
        merchant="www.ebay.com",
        source_url="https://www.ebay.com/itm/456",
        product_name="Bissell PowerClean FurGuard 4039 Cordless Vacuum",
        price_text="$159.99",
        price_amount=159.99,
        source_type="search_result",
        match_text="Complete vacuum with battery and attachments included.",
    )

    assert is_accessory_or_partial_listing(candidate) is False


def test_is_installment_price_detects_monthly_payment():
    candidate = OfferCandidate(
        merchant="www.example.com",
        source_url="https://www.example.com/vacuum",
        product_name="Bissell PowerClean FurGuard 4039",
        price_text="$19.99",
        price_amount=19.99,
        source_type="search_result",
        match_text="$19.99 per month for 8 months with financing.",
    )

    assert is_installment_price(candidate) is True


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


# --- is_purchasable_offer_candidate ---


def test_is_purchasable_offer_candidate_price_range_in_match_text_returns_false():
    candidate = OfferCandidate(
        merchant="www.beautyprices.com",
        source_url="https://www.beautyprices.com/cleanser",
        product_name="CleanSkin Cleanser Price Comparison",
        price_text="$17",
        price_amount=17.0,
        match_text="CleanSkin Gentle Foaming Cleanser 100ml sells for $17 to $21.",
    )
    assert is_purchasable_offer_candidate(candidate) is False


def test_is_purchasable_offer_candidate_price_range_with_en_dash_returns_false():
    candidate = OfferCandidate(
        merchant="www.pricecheck.com",
        source_url="https://www.pricecheck.com/cream",
        product_name="HydraBoost Cream Price Check",
        price_text="$11",
        price_amount=11.0,
        match_text="HydraBoost Cream is widely available for $11–$14.",
    )
    assert is_purchasable_offer_candidate(candidate) is False


def test_is_purchasable_offer_candidate_price_range_in_title_returns_false():
    candidate = OfferCandidate(
        merchant="www.pricecheck.com",
        source_url="https://www.pricecheck.com/cream",
        product_name="HydraBoost Cream $11 to $14",
        price_text="$11",
        price_amount=11.0,
        match_text="",
    )
    assert is_purchasable_offer_candidate(candidate) is False


def test_is_purchasable_offer_candidate_market_summary_without_positive_signals_returns_false():
    candidate = OfferCandidate(
        merchant="www.beautyblog.com",
        source_url="https://www.beautyblog.com/face-wash-review",
        product_name="DailyCare Gentle Face Wash Review",
        price_text="$17",
        price_amount=17.0,
        match_text="DailyCare Gentle Face Wash typically retails for around $17.",
    )
    assert is_purchasable_offer_candidate(candidate) is False


def test_is_purchasable_offer_candidate_market_summary_with_in_stock_returns_true():
    candidate = OfferCandidate(
        merchant="www.amazon.com",
        source_url="https://www.amazon.com/product",
        product_name="DailyCare Gentle Face Wash 200ml",
        price_text="$17.99",
        price_amount=17.99,
        match_text="Market price around $17.99. In stock. Ships from Amazon.",
    )
    assert is_purchasable_offer_candidate(candidate) is True


def test_is_purchasable_offer_candidate_specific_price_with_in_stock_returns_true():
    candidate = OfferCandidate(
        merchant="www.amazon.com",
        source_url="https://www.amazon.com/product",
        product_name="ReviveEye Peptide Eye Cream 30ml",
        price_text="$16.49",
        price_amount=16.49,
        match_text="ReviveEye Peptide Eye Cream 30ml. $16.49. In stock. Sold by Amazon.",
    )
    assert is_purchasable_offer_candidate(candidate) is True


def test_is_purchasable_offer_candidate_product_page_with_reviews_text_returns_true():
    candidate = OfferCandidate(
        merchant="www.amazon.com",
        source_url="https://www.amazon.com/product",
        product_name="HydraBoost Moisturizing Cream 16oz",
        price_text="$11.99",
        price_amount=11.99,
        match_text="$11.99. In stock. 4.7 stars, 50,000+ reviews. Ships same day.",
    )
    assert is_purchasable_offer_candidate(candidate) is True


def test_is_purchasable_offer_candidate_buying_guide_copy_with_in_stock_returns_true():
    candidate = OfferCandidate(
        merchant="www.walmart.com",
        source_url="https://www.walmart.com/product",
        product_name="HydraBoost Cream 16oz - Walmart.com",
        price_text="$12.49",
        price_amount=12.49,
        match_text="$12.49. In stock. Free pickup available. See our buying guide for skincare.",
    )
    assert is_purchasable_offer_candidate(candidate) is True


def test_is_purchasable_offer_candidate_price_range_is_hard_stop_even_with_positive_signals():
    candidate = OfferCandidate(
        merchant="www.somesite.com",
        source_url="https://www.somesite.com/product",
        product_name="Cleanser 100ml",
        price_text="$17",
        price_amount=17.0,
        match_text="Available for $17 to $21. In stock. Free shipping.",
    )
    assert is_purchasable_offer_candidate(candidate) is False


def test_is_purchasable_offer_candidate_no_signals_returns_true():
    candidate = OfferCandidate(
        merchant="www.target.com",
        source_url="https://www.target.com/product",
        product_name="CleanSkin Cleanser 100ml",
        price_text="$17.99",
        price_amount=17.99,
        match_text="CleanSkin Cleanser 100ml. $17.99.",
    )
    assert is_purchasable_offer_candidate(candidate) is True
