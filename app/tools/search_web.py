from pydantic import BaseModel, HttpUrl
from tavily import (
    AsyncTavilyClient,
    BadRequestError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)

from app.config import settings


class SearchResult(BaseModel):
    title: str
    url: HttpUrl
    snippet: str
    metadata: dict[str, str | float | None]


def _compact_snippet(content: str, max_length: int = 500) -> str:
    snippet = content.strip()
    return snippet[:max_length]


async def search_web(query: str) -> list[SearchResult]:
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Search query cannot be blank")

    client = AsyncTavilyClient(api_key=settings.tavily_api_key)

    try:
        response = await client.search(
            query=clean_query,
            max_results=settings.tavily_max_results_per_query,
            search_depth=settings.tavily_search_depth,
        )
    except MissingAPIKeyError as exc:
        raise ValueError("Tavily API key is not configured") from exc
    except InvalidAPIKeyError as exc:
        raise ValueError("Tavily API key is invalid") from exc
    except UsageLimitExceededError as exc:
        raise ValueError("Tavily usage limit exceeded") from exc
    except BadRequestError as exc:
        raise ValueError(f"Tavily rejected the search request: {exc}") from exc

    results = []

    for result in response.get("results", []):
        url = result.get("url")
        if not url:
            continue

        results.append(
            SearchResult(
                title=(result.get("title") or "").strip(),
                url=url,
                snippet=_compact_snippet(result.get("content") or ""),
                metadata={
                    "source": "tavily",
                    "score": result.get("score"),
                },
            )
        )

    return results