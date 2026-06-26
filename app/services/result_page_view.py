from dataclasses import dataclass
from decimal import Decimal

from app.schemas import ResearchResult, Verdict
from app.services.offer_validation import current_market_offers


@dataclass(frozen=True)
class MarketPriceSnapshot:
    listed_price: str | None
    lowest_comparable_price: str | None
    typical_comparable_price: str | None
    highest_comparable_price: str | None
    comparable_price_count: int

    @property
    def has_comparable_prices(self) -> bool:
        return self.comparable_price_count >= 2


@dataclass(frozen=True)
class ResultPageView:
    result: ResearchResult
    market_snapshot: MarketPriceSnapshot

    @property
    def show_market_modules(self) -> bool:
        if self.result.verdict in {Verdict.insufficient_data, Verdict.failed}:
            return False
        return True


def build_result_page_view(result: ResearchResult) -> ResultPageView:
    market_snapshot = _build_market_snapshot(result)
    return ResultPageView(
        result=result,
        market_snapshot=market_snapshot,
    )


def _build_market_snapshot(result: ResearchResult) -> MarketPriceSnapshot:
    comparable_prices = sorted(_current_delivered_prices(result))
    has_sufficient_sample = len(comparable_prices) >= 2

    return MarketPriceSnapshot(
        listed_price=result.listed_price,
        lowest_comparable_price=_format_price(comparable_prices[0])
        if has_sufficient_sample
        else None,
        typical_comparable_price=_format_price(_median_price(comparable_prices))
        if has_sufficient_sample
        else None,
        highest_comparable_price=_format_price(comparable_prices[-1])
        if has_sufficient_sample
        else None,
        comparable_price_count=len(comparable_prices),
    )


def _current_delivered_prices(result: ResearchResult) -> list[Decimal]:
    prices = []
    for offer in current_market_offers(result.comparable_offers):
        if offer.delivered_price is None:
            continue
        prices.append(offer.delivered_price)
    return prices


def _median_price(prices: list[Decimal]) -> Decimal:
    midpoint = len(prices) // 2
    if len(prices) % 2 == 1:
        return prices[midpoint]
    return (prices[midpoint - 1] + prices[midpoint]) / Decimal("2")


def _format_price(price: Decimal) -> str:
    return f"${price.quantize(Decimal('0.01'))}"
