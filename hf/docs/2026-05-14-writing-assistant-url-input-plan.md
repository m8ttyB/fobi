# Writing Assistant — URL Input Feature — Implementation Plan

**Goal:** Extend `writing-assistant` so the input field accepts either pasted text or a URL. When a URL is provided, fetch the page server-side, extract the main article content, show it to the user for review/edit, then run the chosen writing mode as today.

**Why now:** This adds a content-acquisition layer to the existing single-shot flow. It's the natural setup for the next project (`doc-qa`/RAG), where document acquisition becomes central — building the fetch-and-extract pipeline here, in a simpler context, gets the patterns right before chunking and vector search land on top.

**Tech stack additions:** `trafilatura` (article extraction). `httpx` is already in `requirements.txt`.

---

## Decisions

| Decision | Choice |
|---|---|
| Input UI | Smart single field — auto-detect URL vs. text via `^https?://` |
| URL scope (v1) | HTML pages only |
| Pre-generation flow | Fetch → show editable preview → user clicks Run |
| Length cap | Truncate to ~20K chars with visible warning |
| Extractor | `trafilatura` |
| Errors | Inline message below input; URL preserved so user can fix and retry |

---

## File Map

### New files

| File | Responsibility |
|---|---|
| `writing-assistant/fetcher.py` | `fetch_and_extract(url) -> ExtractedContent`. Uses `httpx` for the HTTP call and `trafilatura` for content extraction. Raises typed errors. |
| `writing-assistant/tests/test_fetcher.py` | Unit tests with `httpx` and `trafilatura` mocked. |
| `writing-assistant/tests/test_fetch_endpoint.py` | Integration tests for `POST /fetch` with `fetcher` mocked. |

### Modified files

| File | Change |
|---|---|
| `writing-assistant/main.py` | Add `POST /fetch` endpoint, `FetchRequest`/`FetchResponse` Pydantic models, exception → HTTP status mapping. |
| `writing-assistant/static/index.html` | Detect URL on Run click; call `/fetch`, render preview; second Run uses existing `/generate`. |
| `writing-assistant/requirements.txt` | Add `trafilatura`. |
| `writing-assistant/README.md` | Document the URL flow under a new "URL input" section. |
| `writing-assistant/CLAUDE.md` | Note the new `fetcher.py` module and `/fetch` endpoint. |

---

## Data Model

```python
# in fetcher.py
@dataclass
class ExtractedContent:
    text: str            # extracted article body, possibly truncated
    title: str | None    # article title if available
    truncated: bool      # True if original_length > MAX_CHARS
    original_length: int # length of extracted text before truncation
```

### `POST /fetch` shape

Request:
```json
{ "url": "https://example.com/article" }
```

Response (success):
```json
{
  "text": "Cleaned article text...",
  "title": "Article title",
  "truncated": false,
  "original_length": 8423
}
```

Response (error): non-2xx HTTP with `{"detail": "..."}` body (FastAPI default).

---

## Error Taxonomy

`fetcher.py` defines a small set of typed exceptions; `main.py` maps them to HTTP statuses so the frontend has a clean contract:

| Exception | HTTP | Message |
|---|---|---|
| `InvalidURLError` | 400 | "Invalid URL — must start with http:// or https://" |
| `FetchTimeoutError` | 504 | "Request timed out after 10s" |
| `FetchHTTPError` | 502 | "Could not fetch: HTTP <status>" |
| `UnsupportedContentTypeError` | 415 | "Unsupported content type: <type>" |
| `ExtractionEmptyError` | 422 | "No article content found at this URL" |

Guards in `fetch_and_extract`:
- 10s timeout on the HTTP request
- 5 MB max response body size
- `Content-Type` must start with `text/html`
- `MAX_CHARS = 20_000` enforced after extraction

---

## Frontend Behaviour

URL detection: client-side regex `^https?:\/\/`.

```
User pastes URL                  User pastes text
  ↓                                ↓
Click Run                        Click Run
  ↓                                ↓
Detect URL → POST /fetch         (skip fetch — same as today)
  ↓                                ↓
Preview shown:                   POST /generate → stream tokens
- editable textarea              (existing flow)
- "Source: <url>" dim line
- truncation warning chip
  if applicable
  ↓
Click Run (second time)
  ↓
POST /generate → stream tokens
```

After a successful fetch the textarea content is **replaced** with the extracted text and the button label changes from "Fetch" → "Run". A small dim "Source: \<url\>" line appears above the textarea. A "Start over" link resets the state.

On fetch error: red inline message under the input, URL preserved, button stays as "Fetch".

---

## Task 1: Add `trafilatura` dependency

- [ ] **Step 1: Update requirements.txt**

Add `trafilatura>=1.8.0` (alphabetical insertion after the existing deps).

- [ ] **Step 2: Install**

```bash
cd /Users/m8ttyb/workspace/fobi/hf/writing-assistant
source .venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 3: Smoke-import**

```bash
python -c "import trafilatura; print(trafilatura.__version__)"
```

---

## Task 2: `fetcher.py` with tests (TDD)

**Files:**
- Create: `writing-assistant/fetcher.py`
- Create: `writing-assistant/tests/test_fetcher.py`

- [ ] **Step 1: Write failing tests**

Cover: success path, invalid URL, timeout, HTTP error, non-HTML content-type, empty extraction, truncation.

```python
# writing-assistant/tests/test_fetcher.py
import pytest
import httpx
from unittest.mock import patch, MagicMock
import fetcher


def make_response(status=200, content_type="text/html; charset=utf-8", text="<html><body><article>Hi</article></body></html>"):
    r = MagicMock(spec=httpx.Response)
    r.status_code = status
    r.headers = {"content-type": content_type}
    r.text = text
    return r


def test_invalid_url_raises():
    with pytest.raises(fetcher.InvalidURLError):
        fetcher.fetch_and_extract("not a url")


def test_invalid_scheme_raises():
    with pytest.raises(fetcher.InvalidURLError):
        fetcher.fetch_and_extract("ftp://example.com")


def test_success_returns_extracted():
    with patch("fetcher.httpx.get", return_value=make_response()):
        with patch("fetcher.trafilatura.extract", return_value="Hi there."):
            result = fetcher.fetch_and_extract("https://example.com/a")
    assert result.text == "Hi there."
    assert result.truncated is False
    assert result.original_length == len("Hi there.")


def test_http_error_raises():
    with patch("fetcher.httpx.get", return_value=make_response(status=404)):
        with pytest.raises(fetcher.FetchHTTPError):
            fetcher.fetch_and_extract("https://example.com/a")


def test_timeout_raises():
    with patch("fetcher.httpx.get", side_effect=httpx.TimeoutException("slow")):
        with pytest.raises(fetcher.FetchTimeoutError):
            fetcher.fetch_and_extract("https://example.com/a")


def test_non_html_content_type_raises():
    with patch("fetcher.httpx.get", return_value=make_response(content_type="application/pdf")):
        with pytest.raises(fetcher.UnsupportedContentTypeError):
            fetcher.fetch_and_extract("https://example.com/a")


def test_empty_extraction_raises():
    with patch("fetcher.httpx.get", return_value=make_response()):
        with patch("fetcher.trafilatura.extract", return_value=None):
            with pytest.raises(fetcher.ExtractionEmptyError):
                fetcher.fetch_and_extract("https://example.com/a")


def test_long_content_is_truncated():
    long_text = "x" * 50_000
    with patch("fetcher.httpx.get", return_value=make_response()):
        with patch("fetcher.trafilatura.extract", return_value=long_text):
            result = fetcher.fetch_and_extract("https://example.com/a")
    assert result.truncated is True
    assert result.original_length == 50_000
    assert len(result.text) == fetcher.MAX_CHARS
```

- [ ] **Step 2: Implement fetcher.py**

```python
# writing-assistant/fetcher.py
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import trafilatura


MAX_CHARS = 20_000
TIMEOUT_SECONDS = 10.0
MAX_BYTES = 5 * 1024 * 1024  # 5 MB


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
        response = httpx.get(url, timeout=TIMEOUT_SECONDS, follow_redirects=True)
    except httpx.TimeoutException:
        raise FetchTimeoutError(url)
    except httpx.HTTPError as e:
        raise FetchHTTPError(str(e))

    if response.status_code >= 400:
        raise FetchHTTPError(f"HTTP {response.status_code}")

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("text/html"):
        raise UnsupportedContentTypeError(content_type)

    if len(response.text.encode("utf-8")) > MAX_BYTES:
        raise FetchHTTPError(f"Response too large (>{MAX_BYTES} bytes)")

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
```

- [ ] **Step 3: Run tests**

```bash
make test
```

Expected: all `test_fetcher.py` tests pass.

---

## Task 3: `POST /fetch` endpoint with tests (TDD)

**Files:**
- Modify: `writing-assistant/main.py`
- Create: `writing-assistant/tests/test_fetch_endpoint.py`

- [ ] **Step 1: Write failing tests**

```python
# writing-assistant/tests/test_fetch_endpoint.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import fetcher


@pytest.fixture
def client():
    with patch("main.load_model") as mock_load:
        mock_load.return_value = (MagicMock(), MagicMock())
        from main import app
        with TestClient(app) as c:
            yield c


def test_fetch_returns_extracted(client):
    fake = fetcher.ExtractedContent(text="Hi", title="Hello", truncated=False, original_length=2)
    with patch("main.fetch_and_extract", return_value=fake):
        r = client.post("/fetch", json={"url": "https://example.com/a"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "Hi"
    assert body["title"] == "Hello"
    assert body["truncated"] is False


def test_fetch_invalid_url_returns_400(client):
    with patch("main.fetch_and_extract", side_effect=fetcher.InvalidURLError("bad")):
        r = client.post("/fetch", json={"url": "not a url"})
    assert r.status_code == 400


def test_fetch_timeout_returns_504(client):
    with patch("main.fetch_and_extract", side_effect=fetcher.FetchTimeoutError("slow")):
        r = client.post("/fetch", json={"url": "https://example.com/a"})
    assert r.status_code == 504


def test_fetch_http_error_returns_502(client):
    with patch("main.fetch_and_extract", side_effect=fetcher.FetchHTTPError("HTTP 404")):
        r = client.post("/fetch", json={"url": "https://example.com/a"})
    assert r.status_code == 502


def test_fetch_unsupported_content_returns_415(client):
    with patch("main.fetch_and_extract", side_effect=fetcher.UnsupportedContentTypeError("application/pdf")):
        r = client.post("/fetch", json={"url": "https://example.com/a.pdf"})
    assert r.status_code == 415


def test_fetch_empty_extraction_returns_422(client):
    with patch("main.fetch_and_extract", side_effect=fetcher.ExtractionEmptyError("https://x.com")):
        r = client.post("/fetch", json={"url": "https://x.com"})
    assert r.status_code == 422
```

- [ ] **Step 2: Add the endpoint to main.py**

```python
# additions to main.py

from fastapi import HTTPException
from fetcher import (
    fetch_and_extract,
    InvalidURLError, FetchTimeoutError, FetchHTTPError,
    UnsupportedContentTypeError, ExtractionEmptyError,
)


class FetchRequest(BaseModel):
    url: str


class FetchResponse(BaseModel):
    text: str
    title: str | None
    truncated: bool
    original_length: int


_FETCH_ERROR_STATUS = {
    InvalidURLError: (400, "Invalid URL — must start with http:// or https://"),
    FetchTimeoutError: (504, "Request timed out"),
    FetchHTTPError: (502, "Could not fetch the URL"),
    UnsupportedContentTypeError: (415, "Unsupported content type"),
    ExtractionEmptyError: (422, "No article content found at this URL"),
}


@app.post("/fetch", response_model=FetchResponse)
async def fetch(request: FetchRequest):
    try:
        extracted = await asyncio.to_thread(fetch_and_extract, request.url)
    except tuple(_FETCH_ERROR_STATUS.keys()) as e:
        status, default_msg = _FETCH_ERROR_STATUS[type(e)]
        detail = str(e) if str(e) else default_msg
        raise HTTPException(status_code=status, detail=f"{default_msg}: {detail}")
    return FetchResponse(
        text=extracted.text,
        title=extracted.title,
        truncated=extracted.truncated,
        original_length=extracted.original_length,
    )
```

- [ ] **Step 3: Run tests**

```bash
make test
```

Expected: all new endpoint tests pass; existing 9 tests still pass.

---

## Task 4: Frontend two-stage flow

**Files:**
- Modify: `writing-assistant/static/index.html`

The state machine:

```
state = "idle"     → button: "Run"   → on click: if URL, fetch → "preview"; else generate
state = "fetching" → button: spinner / disabled
state = "preview"  → button: "Run"   → on click: generate using textarea content
state = "generating" → button: disabled
```

A "Start over" link, visible in `preview` state, resets to `idle` and clears the textarea.

- [ ] **Step 1: Update index.html**

Key changes:
- Add a `state` variable in JS
- Add a hidden source-url line `<div id="source"></div>`
- Add a hidden truncation warning `<div id="warning"></div>`
- Add a hidden "Start over" link
- In the click handler, check `state` and the URL regex; branch accordingly
- After `/fetch` success: populate textarea, show source line and (if `truncated`) warning, transition to `preview`

(Implementation details mirror the existing event-handling style — no framework, just `document.getElementById` and direct DOM mutation.)

- [ ] **Step 2: Manual smoke test**

```bash
make run
```

Open `http://127.0.0.1:8000` and verify:
- Plain text path still works
- Wikipedia URL → preview appears
- 404 URL → inline error
- PDF URL → "Unsupported content type" error
- Long article → truncation warning visible

---

## Task 5: Documentation

**Files:**
- Modify: `writing-assistant/README.md`
- Modify: `writing-assistant/CLAUDE.md`

- [ ] **Step 1: Add a "URL input" section to README.md**

Cover: the smart-field UX, the fetch-then-preview-then-run flow, supported content (HTML only in v1), the 20K-char truncation cap, error handling.

- [ ] **Step 2: Update CLAUDE.md**

- Add `fetcher.py` to the module table
- Add `POST /fetch` to the endpoint description
- Note the typed-exception → HTTP-status mapping pattern

---

## Task 6: Final test run and commit

- [ ] **Step 1: Full test suite**

```bash
make test
```

Expected: previous 9 tests + ~7 fetcher tests + ~6 endpoint tests = ~22 passing.

- [ ] **Step 2: Lint**

```bash
make lint
```

- [ ] **Step 3: Commit**

```bash
git add writing-assistant/ docs/2026-05-14-writing-assistant-url-input-plan.md
git commit -m "feat: add URL input to writing-assistant"
```

---

## Out of Scope (deliberate)

- PDF / non-HTML content
- JavaScript-rendered pages (no headless browser)
- Caching fetched content
- Authentication / cookies for paywalled URLs
- Sending the source URL through to the model alongside the text
- Per-mode prompt tweaking based on whether input came from a URL
