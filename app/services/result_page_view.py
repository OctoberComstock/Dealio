import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from app.schemas import ResearchResult, Verdict

_PRICE_PATTERN = re.compile(r"\$\s?([0-9]+(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)")


@dataclass(frozen=True)
class MarketPriceSnapshot:
    listed_price: str | None
    lowest_comparable_price: str | None
    typical_comparable_price: str | None
    highest_comparable_price: str | None
    comparable_price_count: int

    @property
    def has_comparable_prices(self) -> bool:
        return self.comparable_price_count > 0


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
    comparable_prices: list[Decimal] = []
    for item in result.evidence:
        comparable_prices.extend(_extract_prices_from_text(item.text))

    comparable_prices = sorted(comparable_prices)

    return MarketPriceSnapshot(
        listed_price=result.listed_price,
        lowest_comparable_price=_format_price(comparable_prices[0]) if comparable_prices else None,
        typical_comparable_price=_format_price(_median_price(comparable_prices))
        if comparable_prices
        else None,
        highest_comparable_price=_format_price(comparable_prices[-1])
        if comparable_prices
        else None,
        comparable_price_count=len(comparable_prices),
    )


def _extract_prices_from_text(text: str) -> list[Decimal]:
    prices: list[Decimal] = []
    for match in _PRICE_PATTERN.finditer(text):
        normalized_price = match.group(1).replace(",", "")
        try:
            prices.append(Decimal(normalized_price))
        except InvalidOperation:
            continue
    return prices


def _median_price(prices: list[Decimal]) -> Decimal:
    midpoint = len(prices) // 2
    if len(prices) % 2 == 1:
        return prices[midpoint]
    return (prices[midpoint - 1] + prices[midpoint]) / Decimal("2")


def _format_price(price: Decimal) -> str:
    return f"${price.quantize(Decimal('0.01'))}"
