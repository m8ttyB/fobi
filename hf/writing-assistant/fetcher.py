# writing-assistant/fetcher.py
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import trafilatura


MAX_CHARS = 20_000
_TIMEOUT = 10.0
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB


@dataclass
class ExtractedContent:
    text: str
    title: str | None
    truncated: bool
    original_length: int


class FetchError(Exception):
    """Base class for all fetch/extract errors."""


class InvalidURLError(FetchError): ...
class FetchTimeoutError(FetchError): ...
class FetchHTTPError(FetchError): ...
class UnsupportedContentTypeError(FetchError): ...
class ExtractionEmptyError(FetchError): ...


def fetch_and_extract(url: str) -> ExtractedContent:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise InvalidURLError(url)

    try:
        response = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
    except httpx.TimeoutException:
        raise FetchTimeoutError(url)
    except httpx.HTTPError as e:
        raise FetchHTTPError(str(e))

    if response.status_code >= 400:
        raise FetchHTTPError(f"HTTP {response.status_code}")

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("text/html"):
        raise UnsupportedContentTypeError(content_type)

    if len(response.text.encode("utf-8")) > _MAX_BYTES:
        raise FetchHTTPError(f"Response too large (>{_MAX_BYTES} bytes)")

    extracted = trafilatura.extract(response.text)
    if not extracted:
        raise ExtractionEmptyError(url)

    original_length = len(extracted)
    truncated = original_length > MAX_CHARS
    text = extracted[:MAX_CHARS] if truncated else extracted

    metadata = trafilatura.extract_metadata(response.text)
    title = metadata.title if metadata else None

    return ExtractedContent(
        text=text,
        title=title,
        truncated=truncated,
        original_length=original_length,
    )
