import json
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)


class FetchedPage(BaseModel):
    url: str
    title: str | None
    content: str
    price_guess: str | None
    extraction_source: str = "text_regex"
    price_candidates: list[str] = Field(default_factory=list)
    price_candidate_labels: list[str] = Field(default_factory=list)
    lowest_visible_price: str | None = None
    shipping_text: str | None = None
    listing_count: str | None = None
    condition_text: str | None = None


@dataclass
class _StaticResult:
    canonical_url: str
    title: str | None
    price_guess: str | None
    content: str
    source: str


@dataclass
class _RenderedResult:
    title: str | None
    price_guess: str | None
    content: str
    price_candidates: list[str] = field(default_factory=list)
    price_candidate_labels: list[str] = field(default_factory=list)
    lowest_visible_price: str | None = None
    shipping_text: str | None = None
    listing_count: str | None = None
    condition_text: str | None = None


# --- Price and signal patterns ---

_PRICE_PATTERN = re.compile(
    r"(?:"
    r"[$£€¥]\s*\d[\d,]*(?:\.\d{1,2})?"
    r"|\d[\d,]*(?:\.\d{1,2})?\s*(?:USD|GBP|EUR|JPY|CAD|AUD)\b"
    r")"
)

_SHIPPING_PATTERN = re.compile(
    r"(?:"
    r"free\s*shipping"
    r"|shipping\s+included"
    r"|[+]\s*[$£€¥]\s*[\d,.]+\s*shipping"
    r"|[$£€¥]\s*[\d,.]+\s*shipping"
    r")",
    re.IGNORECASE,
)

_LOWEST_PRICE_PATTERN = re.compile(
    r"(?:as low as|starting at|lowest(?:\s+price)?)[:\s]+"
    r"[$£€¥]\s*[\d,.]+",
    re.IGNORECASE,
)

_LISTING_COUNT_PATTERN = re.compile(
    r"\d[\d,]*\s*(?:listings?|results?|available)",
    re.IGNORECASE,
)

_CONDITION_PATTERN = re.compile(
    r"(?:condition|grade):\s*\w+(?:\s+\w+){0,2}",
    re.IGNORECASE,
)

_CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "$", "GBP": "£", "EUR": "€", "JPY": "¥",
    "CAD": "C$", "AUD": "A$",
}


def _find_price(text: str) -> str | None:
    match = _PRICE_PATTERN.search(text)
    return match.group(0).replace(" ", "") if match else None


def _find_prices(text: str) -> list[str]:
    return [m.group(0).replace(" ", "") for m in _PRICE_PATTERN.finditer(text)][:5]


def _detect_shipping_text(text: str) -> str | None:
    match = _SHIPPING_PATTERN.search(text)
    return match.group(0).strip() if match else None


def _detect_lowest_price(text: str) -> str | None:
    match = _LOWEST_PRICE_PATTERN.search(text)
    return _find_price(match.group(0)) if match else None


def _detect_listing_count(text: str) -> str | None:
    match = _LISTING_COUNT_PATTERN.search(text)
    return match.group(0).strip() if match else None


def _detect_condition_text(text: str) -> str | None:
    match = _CONDITION_PATTERN.search(text)
    return match.group(0).strip() if match else None


def _format_price(price: object, currency_code: str | None = None) -> str | None:
    if price is None:
        return None
    price_str = str(price)
    if not price_str:
        return None
    if currency_code:
        symbol = _CURRENCY_SYMBOLS.get(currency_code.upper())
        return f"{symbol}{price_str}" if symbol else f"{price_str} {currency_code}"
    return f"${price_str}"


# --- Marketplace detection ---

_MARKETPLACE_DOMAINS = frozenset({
    "tcgplayer.com", "ebay.com", "etsy.com", "amazon.com",
    "poshmark.com", "mercari.com", "stockx.com", "goat.com",
    "cardmarket.com",
})

_MARKETPLACE_HTML_SIGNALS = re.compile(
    r"\b(?:listings?|as low as|market price|add to cart|in stock)\b",
    re.IGNORECASE,
)


def _is_marketplace_like(url: str, html: str) -> bool:
    domain = urlparse(url).hostname or ""
    if any(domain == d or domain.endswith("." + d) for d in _MARKETPLACE_DOMAINS):
        return True
    return bool(_MARKETPLACE_HTML_SIGNALS.search(html[:3000]))


def _needs_rendered_fallback(static: _StaticResult, html: str, url: str) -> bool:
    if static.price_guess is None:
        return True
    if static.source in ("json_ld", "shopify_product_json", "open_graph"):
        return False
    return _is_marketplace_like(url, html)


# --- JSON-LD ---

def _extract_json_ld_product(html: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            if item.get("@type") == "Product":
                return item
            for graph_item in item.get("@graph", []):
                if isinstance(graph_item, dict) and graph_item.get("@type") == "Product":
                    return graph_item
    return None


def _price_from_offers(offers: object) -> str | None:
    if isinstance(offers, dict):
        price = offers.get("price") or offers.get("lowPrice")
        currency = offers.get("priceCurrency")
        return _format_price(price, currency)
    if isinstance(offers, list) and offers:
        return _price_from_offers(offers[0])
    return None


def _summarize_json_ld_product(product: dict) -> str:
    parts = ["[JSON-LD product data]"]
    if name := product.get("name"):
        parts.append(f"Product: {name}")
    if brand := product.get("brand"):
        brand_name = brand.get("name") if isinstance(brand, dict) else str(brand)
        if brand_name:
            parts.append(f"Brand: {brand_name}")
    if desc := product.get("description"):
        parts.append(f"Description: {str(desc)[:300]}")
    offers = product.get("offers")
    if isinstance(offers, list):
        prices = [
            p for o in offers
            if isinstance(o, dict) and o.get("price")
            and (p := _format_price(o["price"], o.get("priceCurrency")))
        ]
        if prices:
            parts.append(f"Prices: {', '.join(prices[:5])}")
    elif isinstance(offers, dict) and offers.get("price"):
        price = _format_price(offers["price"], offers.get("priceCurrency"))
        if price:
            parts.append(f"Price: {price}")
    return "\n".join(parts)


# --- Open Graph ---

def _extract_open_graph(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    og: dict[str, str] = {}
    for meta in soup.find_all("meta"):
        prop = meta.get("property") or meta.get("name") or ""
        content = meta.get("content") or ""
        if content and (prop.startswith("og:") or prop.startswith("product:")):
            og[prop] = content
    return og


def _price_from_open_graph(og: dict[str, str]) -> str | None:
    amount = og.get("product:price:amount") or og.get("og:price:amount")
    if not amount:
        return None
    currency = og.get("product:price:currency") or og.get("og:price:currency")
    return _format_price(amount, currency)


def _summarize_open_graph(og: dict[str, str]) -> str:
    parts = ["[Open Graph metadata]"]
    if title := og.get("og:title"):
        parts.append(f"Title: {title}")
    if desc := og.get("og:description"):
        parts.append(f"Description: {str(desc)[:300]}")
    if price := _price_from_open_graph(og):
        parts.append(f"Price: {price}")
    return "\n".join(parts)


# --- __NEXT_DATA__ ---

def _extract_next_data_script(html: str) -> dict | None:
    pattern = re.compile(
        r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(html)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None


def _find_prices_in_next_data(data: object, depth: int = 0) -> list[str]:
    if depth > 6 or not data:
        return []
    if isinstance(data, dict):
        results: list[str] = []
        for key, value in list(data.items())[:50]:
            if any(p in key.lower() for p in ("price", "cost", "amount", "msrp")):
                if isinstance(value, (int, float)) and value > 0:
                    results.append(f"${value:.2f}")
                elif isinstance(value, str) and re.search(r"\d", value):
                    results.append(value)
            elif isinstance(value, (dict, list)):
                results.extend(_find_prices_in_next_data(value, depth + 1))
        return results[:5]
    if isinstance(data, list):
        results = []
        for item in data[:5]:
            results.extend(_find_prices_in_next_data(item, depth + 1))
        return results[:5]
    return []


# --- Shopify ---

def _shopify_handle(url: str) -> tuple[str, str] | None:
    parsed = urlparse(url)
    match = re.search(r"/products/([^/?#]+)", parsed.path)
    if not match:
        return None
    return f"{parsed.scheme}://{parsed.netloc}", match.group(1)


def _price_from_shopify_js(product: dict) -> str | None:
    """`.js` format: prices in cents as integers."""
    variants = product.get("variants", [])
    if isinstance(variants, list) and variants:
        price = variants[0].get("price")
        if isinstance(price, (int, float)) and price > 0:
            return f"${price / 100:.2f}"
    return None


def _price_from_shopify_json(product: dict) -> str | None:
    """`.json` format: prices as decimal strings."""
    variants = product.get("variants", [])
    if isinstance(variants, list) and variants:
        price = variants[0].get("price")
        if isinstance(price, str) and price:
            return f"${price}"
        if isinstance(price, (int, float)) and price > 0:
            return f"${price:.2f}"
    return None


def _summarize_shopify_product(product: dict, price_fn) -> str:
    parts = ["[Shopify product data]"]
    if title := product.get("title"):
        parts.append(f"Product: {title}")
    if vendor := product.get("vendor"):
        parts.append(f"Vendor: {vendor}")
    body = product.get("body_html") or product.get("description") or ""
    if body:
        clean = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip()
        if clean:
            parts.append(f"Description: {clean[:300]}")
    variants = product.get("variants", [])
    if isinstance(variants, list):
        prices = list(dict.fromkeys(
            p for v in variants[:5]
            if (p := price_fn({"variants": [v]}))
        ))
        if prices:
            parts.append(f"Prices: {', '.join(prices)}")
    return "\n".join(parts)


async def _fetch_shopify_product(url: str) -> tuple[dict, str] | None:
    info = _shopify_handle(url)
    if not info:
        return None
    base, handle = info
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(f"{base}/products/{handle}.js")
            if resp.status_code == 200:
                return resp.json(), "js"
            resp = await client.get(f"{base}/products/{handle}.json")
            if resp.status_code == 200:
                product = resp.json().get("product")
                if product:
                    return product, "json"
    except (httpx.RequestError, ValueError, KeyError):
        pass
    return None


# --- Rendered extraction ---

# JS extracts price element candidates and compact visible text.
# All pattern-based detection (shipping, lowest price, listing count) runs in Python.
_RENDER_EXTRACT_JS = r"""
    () => {
        const sels = [
            '[itemprop="price"]',
            '[itemprop="lowPrice"]',
            '[data-price]',
            '[data-testid*="price" i]',
            '[class*="price" i]',
            '[id*="price" i]',
        ];
        const LABELS = [
            'as low as','market price','shipping','listing',
            'condition','seller','chart','add to cart',
        ];
        function nearbyLabel(el) {
            const ctx = (el.closest('[class]')?.innerText||'').toLowerCase();
            return LABELS.find(k => ctx.includes(k)) || null;
        }
        const seen = new Set(), candidates = [];
        for (const sel of sels) {
            for (const el of document.querySelectorAll(sel)) {
                const t = (el.innerText||el.textContent||'').trim();
                if (!t || t.length > 60 || seen.has(t)) continue;
                seen.add(t);
                candidates.push({text: t, label: nearbyLabel(el)});
                if (candidates.length >= 10) break;
            }
            if (candidates.length >= 10) break;
        }
        const clone = document.body?.cloneNode(true);
        if (clone) {
            const rm = ['script','style','nav','footer','header','iframe'];
            for (const tag of rm) {
                for (const e of clone.querySelectorAll(tag)) e.remove();
            }
        }
        const vt = (clone?.innerText||clone?.textContent||'').trim().slice(0,1500);
        return {candidates, visibleText: vt};
    }
"""


async def _render_page(url: str, timeout_ms: int) -> _RenderedResult:
    # NOTE: ROK-32 (URL safety guardrails) is not yet implemented.
    # Rendered extraction must not be used on arbitrary unsafe URLs in production.
    try:
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
        from playwright.async_api import async_playwright
    except ImportError:
        logger.warning("Playwright not installed; skipping rendered extraction for %s", url)
        return _RenderedResult(title=None, price_guess=None, content="")

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                idle_budget = min(timeout_ms // 4, 8000)
                try:
                    await page.wait_for_load_state("networkidle", timeout=idle_budget)
                except PlaywrightTimeoutError:
                    pass
                title = (await page.title()) or None
                data = await page.evaluate(_RENDER_EXTRACT_JS)
                candidates = data.get("candidates", [])
                visible_text: str = data.get("visibleText", "")
                price_texts = [c["text"] for c in candidates]
                price_labels = [c.get("label") or "" for c in candidates]
                price_guess = next(
                    (_find_price(t) for t in price_texts if _find_price(t)), None
                ) or _find_price(visible_text)
                lowest = _detect_lowest_price(visible_text)
                shipping = _detect_shipping_text(visible_text)
                listing = _detect_listing_count(visible_text)
                condition = _detect_condition_text(visible_text)
                content_parts = ["[Rendered page content]"]
                if price_texts:
                    content_parts.append(f"Prices visible: {', '.join(price_texts[:5])}")
                labeled = [
                    (lbl, txt) for lbl, txt in zip(price_labels, price_texts) if lbl
                ]
                if labeled:
                    content_parts.append(
                        "Labeled: " + "; ".join(f"{lbl}: {txt}" for lbl, txt in labeled[:3])
                    )
                if lowest:
                    content_parts.append(f"Lowest price: {lowest}")
                if shipping:
                    content_parts.append(f"Shipping: {shipping}")
                if listing:
                    content_parts.append(f"Listings: {listing}")
                if condition:
                    content_parts.append(f"Condition: {condition}")
                if visible_text:
                    content_parts.append(visible_text)
                return _RenderedResult(
                    title=title,
                    price_guess=price_guess,
                    content="\n".join(content_parts),
                    price_candidates=price_texts[:5],
                    price_candidate_labels=price_labels[:5],
                    lowest_visible_price=lowest,
                    shipping_text=shipping,
                    listing_count=listing,
                    condition_text=condition,
                )
            except PlaywrightTimeoutError:
                logger.warning("Playwright timed out rendering %s", url)
                return _RenderedResult(title=None, price_guess=None, content="")
            finally:
                await browser.close()
    except Exception as exc:
        logger.warning("Rendered extraction failed for %s: %s", url, exc)
        return _RenderedResult(title=None, price_guess=None, content="")


# --- Static extraction pipeline ---

async def _static_extract(url: str, html: str) -> _StaticResult:

    # 1. JSON-LD
    product = _extract_json_ld_product(html)
    if product:
        price = _price_from_offers(product.get("offers"))
        if price:
            return _StaticResult(
                canonical_url=url,
                title=product.get("name"),
                price_guess=price,
                content=_summarize_json_ld_product(product),
                source="json_ld",
            )

    # 2. Shopify product API
    shopify = await _fetch_shopify_product(url)
    if shopify:
        shopify_product, fmt = shopify
        price_fn = _price_from_shopify_js if fmt == "js" else _price_from_shopify_json
        price = price_fn(shopify_product)
        if price:
            return _StaticResult(
                canonical_url=url,
                title=shopify_product.get("title"),
                price_guess=price,
                content=_summarize_shopify_product(shopify_product, price_fn),
                source="shopify_product_json",
            )

    # 3. Open Graph
    og = _extract_open_graph(html)
    og_price = _price_from_open_graph(og)
    if og_price:
        return _StaticResult(
            canonical_url=url,
            title=og.get("og:title"),
            price_guess=og_price,
            content=_summarize_open_graph(og),
            source="open_graph",
        )

    # 4. __NEXT_DATA__
    next_data = _extract_next_data_script(html)
    if next_data:
        prices = _find_prices_in_next_data(next_data)
        if prices:
            title_tag = BeautifulSoup(html, "html.parser").find("title")
            nd_title = title_tag.get_text(strip=True) if title_tag else None
            return _StaticResult(
                canonical_url=url,
                title=nd_title,
                price_guess=prices[0],
                content=f"[Next.js page data]\nPrices found: {', '.join(prices)}",
                source="next_data",
            )

    # 5. trafilatura + price regex
    extracted = trafilatura.bare_extraction(
        html, url=url, with_metadata=True, include_comments=False
    )
    if extracted:
        doc = extracted.as_dict()
        text = doc.get("text") or ""
        if text:
            return _StaticResult(
                canonical_url=doc.get("url") or url,
                title=doc.get("title"),
                price_guess=_find_price(text),
                content=text,
                source="text_regex",
            )

    return _StaticResult(
        canonical_url=url, title=None, price_guess=None, content="", source="text_regex"
    )


# --- Main ---

async def fetch_page(url: str) -> FetchedPage:
    try:
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_seconds,
            follow_redirects=True,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ValueError(f"HTTP {exc.response.status_code} fetching {url}") from exc
    except httpx.RequestError as exc:
        raise ValueError(f"Network error fetching {url}: {exc}") from exc

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type:
        raise ValueError(f"Unsupported content type '{content_type}' for {url}")

    html = response.text
    static = await _static_extract(url, html)

    url_out = static.canonical_url
    title = static.title
    price_guess = static.price_guess
    content = static.content
    source = static.source
    price_candidates: list[str] = []
    price_candidate_labels: list[str] = []
    lowest_visible_price: str | None = None
    shipping_text: str | None = None
    listing_count: str | None = None
    condition_text: str | None = None

    if _needs_rendered_fallback(static, html, url):
        rendered = await _render_page(url, settings.render_page_timeout_seconds * 1000)
        rendered_has_data = bool(rendered.content or rendered.price_guess)
        if rendered_has_data:
            title = rendered.title or title
            price_guess = rendered.price_guess or price_guess
            content = rendered.content or content
            source = "rendered_visible_text"
            price_candidates = rendered.price_candidates
            price_candidate_labels = rendered.price_candidate_labels
            lowest_visible_price = rendered.lowest_visible_price
            shipping_text = rendered.shipping_text
            listing_count = rendered.listing_count
            condition_text = rendered.condition_text

    if not content:
        raise ValueError(f"No extractable content found at {url}")

    return FetchedPage(
        url=url_out,
        title=title,
        content=content,
        price_guess=price_guess,
        extraction_source=source,
        price_candidates=price_candidates,
        price_candidate_labels=price_candidate_labels,
        lowest_visible_price=lowest_visible_price,
        shipping_text=shipping_text,
        listing_count=listing_count,
        condition_text=condition_text,
    )
