from datetime import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl, model_validator


class Verdict(str, Enum):
    good_deal = "good_deal"
    fair = "fair"
    overpriced = "overpriced"
    insufficient_data = "insufficient_data"
    failed = "failed"

    @property
    def display_label(self) -> str:
        verdict_display_labels = {
            Verdict.good_deal: "Buy",
            Verdict.fair: "Fair Price",
            Verdict.overpriced: "Don't Buy",
            Verdict.insufficient_data: "Not Enough Data",
            Verdict.failed: "Failed",
        }
        return verdict_display_labels[self]


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class EvidenceItem(BaseModel):
    text: str
    source_url: HttpUrl


class BetterAlternative(BaseModel):
    product_name: str
    price: str | None
    reason: str
    source_url: HttpUrl
    is_cheaper: bool
    is_better_reviewed: bool


class PriceClassification(str, Enum):
    current_unconditional_offer = "current_unconditional_offer"
    current_restricted_offer = "current_restricted_offer"
    msrp = "msrp"
    previous_price = "previous_price"
    installment_amount = "installment_amount"
    unavailable_offer = "unavailable_offer"
    malformed_price = "malformed_price"


class MatchStatus(str, Enum):
    match = "match"
    mismatch = "mismatch"
    unknown = "unknown"


class AvailabilityStatus(str, Enum):
    available = "available"
    unavailable = "unavailable"
    unknown = "unknown"


class VerificationStatus(str, Enum):
    verified = "verified"
    unverified = "unverified"
    fetch_failed = "fetch_failed"
    rejected = "rejected"


class ComparableOffer(BaseModel):
    product_name: str
    merchant: str
    source_url: HttpUrl
    current_item_price: Decimal | None
    shipping_price: Decimal | None = None
    delivered_price: Decimal | None = None
    currency: str = "USD"
    availability: AvailabilityStatus = AvailabilityStatus.unknown
    product_match_status: MatchStatus = MatchStatus.unknown
    variant_match_status: MatchStatus = MatchStatus.unknown
    condition: str | None = None
    verification_status: VerificationStatus = VerificationStatus.unverified
    pricing_restrictions: list[str] = Field(default_factory=list)
    source_type: str
    price_classification: PriceClassification

    @model_validator(mode="after")
    def calculate_delivered_price(self) -> "ComparableOffer":
        if self.delivered_price is None and self.current_item_price is not None:
            if self.shipping_price is not None:
                self.delivered_price = self.current_item_price + self.shipping_price
        return self


class ResearchResult(BaseModel):
    product_name: str
    merchant: str
    listed_price: str | None
    verdict: Verdict
    confidence: Confidence
    summary: str
    evidence: list[EvidenceItem]
    alternative: BetterAlternative | None
    comparable_offers: list[ComparableOffer] = Field(default_factory=list)
    last_checked: datetime
