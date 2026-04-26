import re
from urllib.parse import urlparse

from pydantic import BaseModel

from app.tools.extract_product import ProductPageExtraction

_NUMERIC_ONLY = re.compile(r"^\d+$")
_MIN_SLUG_LENGTH = 3


class ProductIdentity(BaseModel):
    value: str
    source: str


def _slug_from_url(url: str) -> str | None:
    path = urlparse(url).path
    segments = [
        s for s in path.split("/")
        if s and not _NUMERIC_ONLY.match(s) and len(s) >= _MIN_SLUG_LENGTH
    ]
    if not segments:
        return None
    best = max(segments, key=len)
    return best.replace("-", " ").replace("_", " ").strip()


def _domain_path_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if not parsed.hostname:
        return None
    path = parsed.path.rstrip("/")
    return f"{parsed.hostname}{path}" if path else parsed.hostname


def infer_product_identity(extraction: ProductPageExtraction, url: str) -> ProductIdentity:
    if extraction.product_name:
        return ProductIdentity(value=extraction.product_name, source="product_name")

    slug = _slug_from_url(url)
    if slug:
        return ProductIdentity(value=slug, source="url_slug")

    domain_path = _domain_path_from_url(url)
    if domain_path:
        return ProductIdentity(value=domain_path, source="domain_path")

    return ProductIdentity(value="insufficient_data", source="insufficient_data")
