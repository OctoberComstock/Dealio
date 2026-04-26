import re

import httpx
import trafilatura
from pydantic import BaseModel

from app.config import settings


class FetchedPage(BaseModel):
    url: str
    title: str | None
    content: str
    price_guess: str | None


_PRICE_PATTERN = re.compile(r"\$\s*\d[\d,]*(?:\.\d{2})?")


def _extract_price_guess(text: str) -> str | None:
    match = _PRICE_PATTERN.search(text)
    return match.group(0).replace(" ", "") if match else None


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
    extracted = trafilatura.bare_extraction(
        html,
        url=url,
        with_metadata=True,
        include_comments=False,
    )

    if extracted is None:
        raise ValueError(f"No extractable content found at {url}")

    doc = extracted.as_dict()
    content = doc.get("text") or ""

    if not content:
        raise ValueError(f"No extractable content found at {url}")

    title = doc.get("title")
    canonical_url = doc.get("url") or url
    price_guess = _extract_price_guess(content)

    return FetchedPage(url=canonical_url, title=title, content=content, price_guess=price_guess)
