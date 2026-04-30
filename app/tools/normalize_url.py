import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# Matches /dp/<ASIN> or /gp/product/<ASIN> in an Amazon URL path.
_AMAZON_ASIN_RE = re.compile(r"/(?:dp|gp/product)/([A-Z0-9]{10})(?=/|$)")

_TRACKING_PARAMS = frozenset({
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "utm_source_platform",
    "utm_creative_format",
    "utm_marketing_tactic",
    "fbclid",
    "gclid",
    "gclsrc",
    "dclid",
    "msclkid",
    "ref",
    "referrer",
    "_ga",
    "_gid",
    "mc_cid",
    "mc_eid",
})


def normalize_url(raw_url: str) -> str:
    stripped = raw_url.strip()
    if not stripped:
        raise ValueError("URL cannot be blank")

    parsed = urlparse(stripped)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL must use http or https, got '{parsed.scheme or 'none'}'")

    if not parsed.netloc:
        raise ValueError("URL is missing a host")

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    path = parsed.path.rstrip("/") or "/"

    filtered_params = sorted(
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS
    )
    query = urlencode(filtered_params)

    return urlunparse((scheme, netloc, path, parsed.params, query, ""))


def fetch_page_cache_key(url: str) -> str:
    """Return a stable per-run cache key for the same-run fetch deduplication.

    For Amazon product URLs, the key is scheme+hostname+/dp/ASIN so that
    URL variants carrying different Amazon session/tracking params (pd_rd_r,
    pf_rd_p, _encoding, th=1, etc.) all resolve to the same key.
    For all other URLs the key is the result of normalize_url().
    """
    try:
        normalized = normalize_url(url)
    except ValueError:
        return url

    parsed = urlparse(normalized)

    if "amazon." in parsed.netloc:
        match = _AMAZON_ASIN_RE.search(parsed.path)
        if match:
            asin = match.group(1)
            return f"{parsed.scheme}://{parsed.netloc}/dp/{asin}"

    return normalized