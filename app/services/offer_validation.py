import re
from decimal import Decimal, InvalidOperation

from app.schemas import (
    AvailabilityStatus,
    ComparableOffer,
    MatchStatus,
    PriceClassification,
    VerificationStatus,
)
from app.tools.normalize_url import normalize_url
from app.tools.offer_candidates import (
    OfferCandidate,
    assess_product_identity,
    is_accessory_or_partial_listing,
    is_installment_price,
    is_purchasable_offer_candidate,
    is_same_size,
    is_unavailable,
)

_FREE_SHIPPING_RE = re.compile(
    r"\b(?:free\s+shipping|free\s+delivery|shipping\s+included)\b",
    re.IGNORECASE,
)
_SHIPPING_PRICE_RE = re.compile(
    r"(?:\+?\s*)\$\s*([\d,]+(?:\.\d{1,2})?)\s*(?:shipping|delivery)",
    re.IGNORECASE,
)
_RESTRICTED_PRICE_RE = re.compile(
    r"\b(?:member|membership|vip|coupon|promo\s+code|with\s+code)\b",
    re.IGNORECASE,
)
_MSRP_RE = re.compile(r"\b(?:msrp|manufacturer'?s?\s+suggested|retail\s+price)\b", re.IGNORECASE)
_PREVIOUS_PRICE_RE = re.compile(
    r"\b(?:was|previous(?:ly)?|regular(?:ly)?|list\s+price|original(?:ly)?)\b",
    re.IGNORECASE,
)
_VARIANT_SIZE_RE = re.compile(r"\bsize\s+([a-z0-9]+(?:\.\d+)?)\b", re.IGNORECASE)


def validate_offer_candidates(
    candidates: list[OfferCandidate],
    submitted_product_name: str,
) -> list[ComparableOffer]:
    offers_by_url: dict[str, ComparableOffer] = {}

    for candidate in candidates:
        offer = validate_offer_candidate(candidate, submitted_product_name)
        normalized_url = _normalized_offer_url(candidate.source_url)
        existing_offer = offers_by_url.get(normalized_url)
        if existing_offer is None:
            offers_by_url[normalized_url] = offer
            continue

        if _offer_preference_score(offer) > _offer_preference_score(existing_offer):
            offers_by_url[normalized_url] = offer

    return list(offers_by_url.values())


def validate_offer_candidate(
    candidate: OfferCandidate,
    submitted_product_name: str,
) -> ComparableOffer:
    combined_text = f"{candidate.product_name or ''} {candidate.match_text}"
    product_match = assess_product_identity(
        submitted_product_name,
        candidate.product_name,
        candidate.match_text,
    )
    variant_matches = _variant_matches(
        submitted_product_name,
        candidate.product_name,
        candidate.match_text,
    )
    restrictions = _pricing_restrictions(combined_text)
    availability = _availability_status(candidate)
    item_price = _decimal_price(candidate.price_amount)
    shipping_price = _shipping_price(combined_text)
    price_classification = _price_classification(candidate, combined_text, restrictions)
    verification_status = _verification_status(candidate, price_classification)

    return ComparableOffer(
        product_name=candidate.product_name or "unknown product",
        merchant=candidate.merchant,
        source_url=candidate.source_url,
        current_item_price=item_price,
        shipping_price=shipping_price,
        currency="USD",
        availability=availability,
        product_match_status=MatchStatus.match if product_match.matches else MatchStatus.mismatch,
        variant_match_status=MatchStatus.match if variant_matches else MatchStatus.mismatch,
        condition=_condition_text(combined_text),
        verification_status=verification_status,
        pricing_restrictions=restrictions,
        source_type=candidate.source_type,
        price_classification=price_classification,
    )


def current_market_offers(offers: list[ComparableOffer]) -> list[ComparableOffer]:
    current_offers = []
    for offer in offers:
        if offer.price_classification != PriceClassification.current_unconditional_offer:
            continue
        if offer.availability != AvailabilityStatus.available:
            continue
        if offer.product_match_status != MatchStatus.match:
            continue
        if offer.variant_match_status != MatchStatus.match:
            continue
        if offer.current_item_price is None:
            continue
        if offer.verification_status == VerificationStatus.rejected:
            continue
        current_offers.append(offer)
    return current_offers


def alternative_eligible_offers(
    offers: list[ComparableOffer],
    submitted_price: Decimal,
    meaningful_savings_threshold: float,
) -> list[ComparableOffer]:
    market_offers = current_market_offers(offers)
    if not market_offers:
        return []

    threshold = Decimal(str(meaningful_savings_threshold))
    eligible = []

    for offer in market_offers:
        comparison_price = offer.delivered_price or offer.current_item_price
        if comparison_price is None:
            continue

        if offer.shipping_price is None and not _unknown_shipping_still_leaves_savings(
            submitted_price,
            comparison_price,
            threshold,
        ):
            continue

        savings = (submitted_price - comparison_price) / submitted_price
        if savings < threshold:
            continue

        eligible.append(offer)

    verified_eligible = [
        offer for offer in eligible
        if offer.verification_status == VerificationStatus.verified
    ]
    if verified_eligible:
        eligible = verified_eligible

    eligible.sort(
        key=lambda offer: offer.delivered_price or offer.current_item_price or Decimal("0")
    )
    return eligible


def _unknown_shipping_still_leaves_savings(
    submitted_price: Decimal,
    item_price: Decimal,
    threshold: Decimal,
) -> bool:
    minimum_savings = submitted_price * threshold
    conservative_shipping_buffer = Decimal("2.00")
    return submitted_price - item_price >= minimum_savings + conservative_shipping_buffer


def _price_classification(
    candidate: OfferCandidate,
    combined_text: str,
    restrictions: list[str],
) -> PriceClassification:
    if _has_malformed_price(candidate.price_text, candidate.price_amount):
        return PriceClassification.malformed_price
    if is_unavailable(candidate):
        return PriceClassification.unavailable_offer
    if is_installment_price(candidate):
        return PriceClassification.installment_amount
    if _MSRP_RE.search(combined_text):
        return PriceClassification.msrp
    if _PREVIOUS_PRICE_RE.search(combined_text):
        return PriceClassification.previous_price
    if restrictions:
        return PriceClassification.current_restricted_offer
    if is_accessory_or_partial_listing(candidate):
        return PriceClassification.unavailable_offer
    if not is_purchasable_offer_candidate(candidate):
        return PriceClassification.unavailable_offer
    return PriceClassification.current_unconditional_offer


def _availability_status(candidate: OfferCandidate) -> AvailabilityStatus:
    if is_unavailable(candidate):
        return AvailabilityStatus.unavailable
    return AvailabilityStatus.available


def _verification_status(
    candidate: OfferCandidate,
    price_classification: PriceClassification,
) -> VerificationStatus:
    if price_classification == PriceClassification.malformed_price:
        return VerificationStatus.rejected
    try:
        return VerificationStatus(candidate.verification_status)
    except ValueError:
        return VerificationStatus.unverified


def _shipping_price(text: str) -> Decimal | None:
    if _FREE_SHIPPING_RE.search(text):
        return Decimal("0.00")

    match = _SHIPPING_PRICE_RE.search(text)
    if match is None:
        return None

    return _decimal_from_text(match.group(1))


def _pricing_restrictions(text: str) -> list[str]:
    restrictions = []
    if _RESTRICTED_PRICE_RE.search(text):
        restrictions.append("membership_or_coupon_required")
    return restrictions


def _condition_text(text: str) -> str | None:
    for condition in ("new", "used", "refurbished", "open box"):
        if re.search(rf"\b{re.escape(condition)}\b", text, re.IGNORECASE):
            return condition
    return None


def _variant_matches(
    submitted_product_name: str,
    candidate_name: str | None,
    match_text: str,
) -> bool:
    if not is_same_size(submitted_product_name, candidate_name, match_text):
        return False

    submitted_sizes = set(_VARIANT_SIZE_RE.findall(submitted_product_name.lower()))
    if not submitted_sizes:
        return True

    candidate_text = f"{candidate_name or ''} {match_text}".lower()
    candidate_sizes = set(_VARIANT_SIZE_RE.findall(candidate_text))
    return bool(submitted_sizes & candidate_sizes)


def _decimal_price(value: float | None) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value)).quantize(Decimal("0.01"))


def _decimal_from_text(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def _has_malformed_price(price_text: str | None, price_amount: float | None) -> bool:
    if price_text is None or price_amount is None:
        return True

    normalized = price_text.replace(",", "").replace("$", "").strip()
    if "." not in normalized and price_amount >= 1000:
        return True

    return False


def _normalized_offer_url(source_url: str) -> str:
    try:
        return normalize_url(source_url)
    except ValueError:
        return source_url.strip()


def _offer_preference_score(offer: ComparableOffer) -> int:
    score = 0
    if offer.verification_status == VerificationStatus.verified:
        score += 4
    if offer.price_classification == PriceClassification.current_unconditional_offer:
        score += 3
    if offer.delivered_price is not None:
        score += 2
    if offer.product_match_status == MatchStatus.match:
        score += 1
    return score
