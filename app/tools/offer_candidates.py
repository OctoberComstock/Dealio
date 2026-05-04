import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from app.tools.fetch_page import FetchedPage

_PRICE_RE = re.compile(r"\$?\s*([\d]+(?:\.\d{1,2})?)")

# For noisy text like search snippets, require an actual currency signal.
_OFFER_PRICE_TEXT_RE = re.compile(
    r"(?:US\$|USD\s*)?\$\s*[\d,]+(?:\.\d{1,2})?"
    r"|[\d,]+(?:\.\d{1,2})?\s*(?:USD|CAD|AUD|GBP|EUR)\b",
    re.IGNORECASE,
)

_SIZE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ml|g|fl\.?\s*oz|oz|lb|kg|ct|count|pack)\b",
    re.IGNORECASE,
)

# Matches volume sizes (ml and fl oz variants) for cross-unit equivalence checks.
_VOLUME_SIZE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(ml|fl\.?\s*oz)\b",
    re.IGNORECASE,
)

_UNAVAILABLE_RE = re.compile(
    r"\b(?:"
    r"out\s+of\s+stock"
    r"|sold\s+out"
    r"|unavailable"
    r"|currently\s+unavailable"
    r"|no\s+longer\s+available"
    r"|not\s+available"
    r"|temporarily\s+out\s+of\s+stock"
    r"|item\s+ended"
    r"|listing\s+ended"
    r"|ended"
    r")\b",
    re.IGNORECASE,
)

_FL_OZ_TO_ML = 29.5735
_VOLUME_TOLERANCE = 0.02  # 2% to accommodate 3.38/3.4 fl oz rounding for 100ml

_MATCH_TEXT_MAX_CHARS = 500

# A price expressed as a range (e.g. "$17 to $21", "$11–$14") signals market-summary
# content rather than a single purchasable offer. Requires $ on both sides to avoid
# false matches on product codes or dimension strings.
_PRICE_RANGE_RE = re.compile(
    r"\$\s*[\d,]+(?:\.\d{1,2})?\s*(?:to|[-–—])\s*\$\s*[\d,]+(?:\.\d{1,2})?",
    re.IGNORECASE,
)

# Language that describes market-wide pricing rather than a single concrete offer.
_MARKET_SUMMARY_RE = re.compile(
    r"\b(?:"
    r"typically\s+retails?"
    r"|typically\s+sells?"
    r"|usually\s+sells?"
    r"|market\s+price"
    r"|standard\s+retail\s+price"
    r"|widely\s+available\s+for"
    r")\b",
    re.IGNORECASE,
)

# Concrete purchasing signals that indicate a specific available listing even when
# surrounding text contains comparison or editorial language.
_PURCHASABLE_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"in\s+stock"
    r"|ships?\s+from"
    r"|ships?\s+within"
    r"|add\s+to\s+cart"
    r"|buy\s+now"
    r"|sold\s+by"
    r"|free\s+shipping"
    r"|pickup"
    r"|free\s+delivery"
    r"|available\s+for\s+delivery"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class OfferCandidate:
    merchant: str
    source_url: str
    product_name: str | None
    price_text: str | None
    price_amount: float | None
    source_type: str = "fetched_page"
    match_text: str = field(default="")


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

def extract_offer_price_text(text: str | None) -> str | None:
    """Extract a price from noisy offer/search text.

    This intentionally requires a currency signal so sizes like "100ml" are not
    accidentally treated as prices.
    """
    if not text:
        return None
    match = _OFFER_PRICE_TEXT_RE.search(text.replace(",", ""))
    return match.group(0).strip() if match else None

def _extract_sizes(text: str) -> set[str]:
    return {m.group().lower().replace(" ", "") for m in _SIZE_RE.finditer(text)}

def _size_to_ml(size_str: str) -> float | None:
    """Convert a normalized size string (e.g. '100ml', '3.4floz') to ml.

    Returns None for non-volume units like oz (mass), g, ct, etc.
    """
    m = re.match(r"^(\d+(?:\.\d+)?)(ml|fl\.?oz)$", size_str, re.IGNORECASE)
    if not m:
        return None
    value = float(m.group(1))
    unit = re.sub(r"[\s.]", "", m.group(2)).lower()
    if unit == "ml":
        return value
    if unit == "floz":
        return value * _FL_OZ_TO_ML
    return None

def _candidate_text_has_volume_equivalent(ml_value: float, candidate_text: str) -> bool:
    """Return True if candidate_text contains a volume measurement within tolerance of ml_value."""
    for m in _VOLUME_SIZE_RE.finditer(candidate_text):
        candidate_value = float(m.group(1))
        unit = re.sub(r"[\s.]", "", m.group(2)).lower()
        if unit == "ml":
            candidate_ml = candidate_value
        elif unit == "floz":
            candidate_ml = candidate_value * _FL_OZ_TO_ML
        else:
            continue
        if abs(ml_value - candidate_ml) / ml_value <= _VOLUME_TOLERANCE:
            return True
    return False

def is_same_size(submitted_name: str, candidate_name: str | None, match_text: str = "") -> bool:
    """Conservative size check: if the submitted product specifies a size,
    the candidate must include that same size. Returns True when uncertain.

    Checks candidate_name and match_text combined. Treats fl oz and ml as
    equivalent when values are within rounding tolerance (e.g. 3.4 fl oz ≈ 100ml).
    """
    candidate_full_text = f"{candidate_name or ''} {match_text}".strip()
    if not candidate_full_text:
        return True

    submitted_sizes = _extract_sizes(submitted_name)
    if not submitted_sizes:
        return True

    candidate_sizes = _extract_sizes(candidate_full_text)

    if submitted_sizes & candidate_sizes:
        return True

    for size_str in submitted_sizes:
        ml_value = _size_to_ml(size_str)
        if ml_value is None:
            continue
        if _candidate_text_has_volume_equivalent(ml_value, candidate_full_text):
            return True

    return False

def is_unavailable(candidate: OfferCandidate) -> bool:
    """Return True if the candidate contains a clear unavailability signal."""
    combined_text = f"{candidate.product_name or ''} {candidate.match_text}"
    return bool(_UNAVAILABLE_RE.search(combined_text))


def is_purchasable_offer_candidate(candidate: OfferCandidate) -> bool:
    """Return True when the candidate represents a concrete purchasable listing.

    A candidate fails this check when its text contains a price range (e.g.
    "$17 to $21") or market-summary language without any concrete purchasing
    signals. This lets editorial and comparison pages remain as evidence while
    preventing them from being recommended as alternatives.

    Broad keywords like "review" or "blog" are intentionally not used here because
    legitimate ecommerce product pages frequently contain that language.
    """
    combined = f"{candidate.product_name or ''} {candidate.match_text}"

    if _PRICE_RANGE_RE.search(combined):
        return False

    if _MARKET_SUMMARY_RE.search(combined) and not _PURCHASABLE_SIGNAL_RE.search(combined):
        return False

    return True


def candidate_from_page(page: FetchedPage) -> OfferCandidate:
    merchant = urlparse(page.url).hostname or page.url
    return OfferCandidate(
        merchant=merchant,
        source_url=page.url,
        product_name=page.title,
        price_text=page.price_guess,
        price_amount=parse_price_amount(page.price_guess),
        source_type="fetched_page",
        match_text=(page.content or "")[:_MATCH_TEXT_MAX_CHARS],
    )

def candidate_from_search_result(result) -> OfferCandidate | None:
    """Build an offer candidate from a search result only when a price is explicit."""
    text = " ".join(
        part for part in (result.title, result.snippet) if part
    )
    price_text = extract_offer_price_text(text)
    if price_text is None:
        return None

    url = str(result.url)
    merchant = urlparse(url).hostname or url

    return OfferCandidate(
        merchant=merchant,
        source_url=url,
        product_name=result.title,
        price_text=price_text,
        price_amount=parse_price_amount(price_text),
        source_type="search_result",
        match_text=result.snippet or "",
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
        lines.append(
            f"{i}. {candidate.merchant} — {price_str} — {name_str} "
            f"({candidate.source_type})"
        )
    lines.append(
        "If recommending an alternative, use the lowest eligible offer from this list."
    )
    return "\n".join(lines)
