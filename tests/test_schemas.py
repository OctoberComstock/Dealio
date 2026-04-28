from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas import (
    BetterAlternative,
    Confidence,
    EvidenceItem,
    ResearchResult,
    Verdict,
)


def make_result(**overrides) -> dict:
    base = {
        "product_name": "Sony WH-1000XM5",
        "merchant": "amazon.com",
        "listed_price": "$279.99",
        "verdict": Verdict.good_deal,
        "confidence": Confidence.high,
        "summary": "Competitively priced with strong reviews.",
        "evidence": [
            {"text": "Lowest price found across major retailers.", "source_url": "https://example.com/1"},
        ],
        "alternative": None,
        "last_checked": datetime.now(timezone.utc),
    }
    base.update(overrides)
    return base


def test_valid_result_parses():
    result = ResearchResult(**make_result())
    assert result.verdict == Verdict.good_deal
    assert result.confidence == Confidence.high


def test_all_verdict_values_are_accepted():
    for verdict in Verdict:
        result = ResearchResult(**make_result(verdict=verdict))
        assert result.verdict == verdict


def test_all_confidence_values_are_accepted():
    for confidence in Confidence:
        result = ResearchResult(**make_result(confidence=confidence))
        assert result.confidence == confidence


def test_invalid_verdict_raises():
    with pytest.raises(ValidationError):
        ResearchResult(**make_result(verdict="bad_value"))


def test_invalid_confidence_raises():
    with pytest.raises(ValidationError):
        ResearchResult(**make_result(confidence="ultra"))


def test_listed_price_is_optional():
    result = ResearchResult(**make_result(listed_price=None))
    assert result.listed_price is None


def test_alternative_is_optional():
    result = ResearchResult(**make_result(alternative=None))
    assert result.alternative is None


def test_alternative_populates_correctly():
    alt = {
        "product_name": "Sony WH-1000XM4",
        "price": "$199.99",
        "reason": "Same sound quality, 28% cheaper.",
        "source_url": "https://example.com/alt",
        "is_cheaper": True,
        "is_better_reviewed": False,
    }
    result = ResearchResult(**make_result(alternative=alt))
    assert result.alternative.is_cheaper is True


def test_evidence_source_url_must_be_valid():
    with pytest.raises(ValidationError):
        EvidenceItem(text="Some evidence", source_url="not-a-url")


def test_alternative_price_is_required():
    with pytest.raises(ValidationError):
        BetterAlternative(
            product_name="Widget Pro",
            reason="Cheaper option",
            source_url="https://example.com/alt",
            is_cheaper=True,
            is_better_reviewed=False,
        )
