from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

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