from dataclasses import dataclass
from urllib.parse import urlparse

from pydantic import BaseModel

from app.tools.fetch_page import FetchedPage, fetch_page


class ProductPageExtraction(BaseModel):
    product_name: str | None
    listed_price: str | None
    merchant: str | None


@dataclass
class ProductExtractionResult:
    extraction: ProductPageExtraction
    fetched_page: FetchedPage | None


def _merchant_from_url(url: str) -> str | None:
    return urlparse(url).hostname


def _extraction_from_page(page: FetchedPage, source_url: str) -> ProductPageExtraction:
    return ProductPageExtraction(
        product_name=page.title,
        listed_price=page.price_guess,
        merchant=_merchant_from_url(page.url or source_url),
    )


async def extract_product(url: str) -> ProductExtractionResult:
    try:
        page = await fetch_page(url)
    except ValueError:
        return ProductExtractionResult(
            extraction=ProductPageExtraction(
                product_name=None,
                listed_price=None,
                merchant=_merchant_from_url(url),
            ),
            fetched_page=None,
        )
    return ProductExtractionResult(
        extraction=_extraction_from_page(page, url),
        fetched_page=page,
    )
