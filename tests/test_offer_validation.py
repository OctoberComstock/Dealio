from decimal import Decimal

from app.schemas import (
    AvailabilityStatus,
    MatchStatus,
    PriceClassification,
    VerificationStatus,
)
from app.services.offer_validation import (
    alternative_eligible_offers,
    current_market_offers,
    validate_offer_candidates,
)
from app.tools.offer_candidates import OfferCandidate

SUBMITTED_PRODUCT = "Brooks Ghost Max 3 Men's Running Shoes Size 10"


def make_candidate(**overrides) -> OfferCandidate:
    values = {
        "merchant": "www.example.com",
        "source_url": "https://www.example.com/product",
        "product_name": SUBMITTED_PRODUCT,
        "price_text": "$119.95",
        "price_amount": 119.95,
        "source_type": "fetched_page",
        "match_text": "In stock. Add to cart. Free shipping.",
        "verification_status": "verified",
    }
    values.update(overrides)
    return OfferCandidate(**values)


def test_validated_offer_calculates_free_shipping_delivered_price():
    offers = validate_offer_candidates([make_candidate()], SUBMITTED_PRODUCT)

    assert offers[0].current_item_price == Decimal("119.95")
    assert offers[0].shipping_price == Decimal("0.00")
    assert offers[0].delivered_price == Decimal("119.95")


def test_validated_offer_calculates_paid_shipping_delivered_price():
    candidate = make_candidate(match_text="In stock. Add to cart. + $7.99 shipping.")

    offers = validate_offer_candidates([candidate], SUBMITTED_PRODUCT)

    assert offers[0].shipping_price == Decimal("7.99")
    assert offers[0].delivered_price == Decimal("127.94")


def test_unknown_shipping_does_not_create_delivered_price():
    candidate = make_candidate(match_text="In stock. Add to cart.")

    offers = validate_offer_candidates([candidate], SUBMITTED_PRODUCT)

    assert offers[0].shipping_price is None
    assert offers[0].delivered_price is None


def test_msrp_and_previous_prices_are_excluded_from_current_market_offers():
    msrp = make_candidate(
        source_url="https://www.example.com/msrp",
        match_text="MSRP $160.00. Product details.",
        price_text="$160.00",
        price_amount=160.00,
    )
    previous = make_candidate(
        source_url="https://www.example.com/was",
        match_text="Was $160.00. Now discounted.",
        price_text="$160.00",
        price_amount=160.00,
    )
    current = make_candidate(source_url="https://www.example.com/current")

    offers = validate_offer_candidates([msrp, previous, current], SUBMITTED_PRODUCT)

    classifications = {offer.price_classification for offer in offers}
    assert PriceClassification.msrp in classifications
    assert PriceClassification.previous_price in classifications
    assert [offer.source_url.unicode_string() for offer in current_market_offers(offers)] == [
        "https://www.example.com/current"
    ]


def test_member_and_coupon_prices_are_labeled_and_excluded_from_market_range():
    candidate = make_candidate(match_text="VIP member price $107.88. In stock. Free shipping.")

    offers = validate_offer_candidates([candidate], SUBMITTED_PRODUCT)

    assert offers[0].pricing_restrictions == ["membership_or_coupon_required"]
    assert offers[0].price_classification == PriceClassification.current_restricted_offer
    assert current_market_offers(offers) == []


def test_duplicate_sources_keep_one_validated_offer():
    search_candidate = make_candidate(
        price_text="$119.95",
        price_amount=119.95,
        source_type="search_result",
        verification_status="unverified",
        match_text="In stock. Free shipping.",
    )
    fetched_candidate = make_candidate(match_text="In stock. Add to cart. Free shipping.")

    offers = validate_offer_candidates([search_candidate, fetched_candidate], SUBMITTED_PRODUCT)

    assert len(offers) == 1
    assert offers[0].verification_status == VerificationStatus.verified


def test_malformed_price_is_rejected():
    candidate = make_candidate(price_text="$10899", price_amount=10899.0)

    offers = validate_offer_candidates([candidate], SUBMITTED_PRODUCT)

    assert offers[0].price_classification == PriceClassification.malformed_price
    assert offers[0].verification_status == VerificationStatus.rejected
    assert current_market_offers(offers) == []


def test_variant_mismatch_is_excluded_from_current_market_offers():
    candidate = make_candidate(
        product_name="Brooks Ghost Max 3 Men's Running Shoes Size 8",
        match_text="Size 8. In stock. Free shipping.",
    )

    offers = validate_offer_candidates([candidate], SUBMITTED_PRODUCT)

    assert offers[0].variant_match_status == MatchStatus.mismatch
    assert current_market_offers(offers) == []


def test_unavailable_offer_is_excluded_from_current_market_offers():
    candidate = make_candidate(match_text="Sold out. Free shipping.")

    offers = validate_offer_candidates([candidate], SUBMITTED_PRODUCT)

    assert offers[0].availability == AvailabilityStatus.unavailable
    assert offers[0].price_classification == PriceClassification.unavailable_offer
    assert current_market_offers(offers) == []


def test_alternative_rejected_when_unknown_shipping_could_remove_savings():
    candidate = make_candidate(price_text="$108.99", price_amount=108.99, match_text="In stock.")
    offers = validate_offer_candidates([candidate], SUBMITTED_PRODUCT)

    eligible = alternative_eligible_offers(offers, Decimal("119.95"), 0.08)

    assert eligible == []


def test_alternative_eligible_when_delivered_savings_meet_threshold():
    candidate = make_candidate(
        price_text="$100.00",
        price_amount=100.00,
        match_text="In stock. Free shipping.",
    )
    offers = validate_offer_candidates([candidate], SUBMITTED_PRODUCT)

    eligible = alternative_eligible_offers(offers, Decimal("119.95"), 0.08)

    assert eligible == offers
