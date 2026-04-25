from datetime import datetime
from enum import Enum

from pydantic import BaseModel, HttpUrl


class Verdict(str, Enum):
    good_deal = "good_deal"
    fair = "fair"
    overpriced = "overpriced"
    insufficient_data = "insufficient_data"


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


class ResearchResult(BaseModel):
    product_name: str
    merchant: str
    listed_price: str | None
    verdict: Verdict
    confidence: Confidence
    summary: str
    evidence: list[EvidenceItem]
    alternative: BetterAlternative | None
    last_checked: datetime
