import re
from dataclasses import dataclass
from urllib.parse import urlparse

from app.tools.fetch_page import FetchedPage

_PRICE_RE = re.compile(r"\$?\s*([\d]+(?:\.\d{1,2})?)")

_SIZE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ml|g|oz|fl\s*oz|lb|kg|ct|count|pack)\b",
    re.IGNORECASE,
)


@dataclass
class OfferCandidate:
    merchant: str
    source_url: str
    product_name: str | None
    price_text: str | None
    price_amount: float | None


def parse_price_amount(price_text: str | None) -> float | None:
    if not price_text:
        return None
    cleaned = price_text.replace(",", "")
    match = _PRICE_RE.search(cleaned)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_sizes(text: str) -> set[str]:
    return {m.group().lower().replace(" ", "") for m in _SIZE_RE.finditer(text)}


def is_same_size(submitted_name: str, candidate_name: str | None) -> bool:
    """Conservative size check: if the submitted product specifies a size,
    the candidate must include that same size. Returns True when uncertain.
    """
    if not candidate_name:
        return True
    submitted_sizes = _extract_sizes(submitted_name)
    if not submitted_sizes:
        return True
    candidate_sizes = _extract_sizes(candidate_name)
    return bool(submitted_sizes & candidate_sizes)


def candidate_from_page(page: FetchedPage) -> OfferCandidate:
    merchant = urlparse(page.url).hostname or page.url
    return OfferCandidate(
        merchant=merchant,
        source_url=page.url,
        product_name=page.title,
        price_text=page.price_guess,
        price_amount=parse_price_amount(page.price_guess),
    )


def format_offer_table(candidates: list[OfferCandidate]) -> str:
    if not candidates:
        return ""
    lines = ["Comparable offers found (sorted by price):"]
    for i, candidate in enumerate(candidates, 1):
        if candidate.price_amount is not None:
            price_str = f"${candidate.price_amount:.2f}"
        else:
            price_str = candidate.price_text or "price unknown"
        name_str = candidate.product_name or "unknown product"
        lines.append(f"{i}. {candidate.merchant} — {price_str} — {name_str}")
    lines.append(
        "If recommending an alternative, use the lowest eligible offer from this list."
    )
    return "\n".join(lines)
