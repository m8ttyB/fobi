import json
import re

from pydantic import ValidationError

import config
from model import generate
from schema import ExtractedDocument

_SCHEMA_EXAMPLE = """{
  "title": "string or null",
  "topic": "string (required)",
  "people": [
    {"name": "string", "role": "string or null", "context": "string or null"}
  ],
  "places": [
    {"name": "string", "context": "string or null"}
  ],
  "dates": [
    {"date": "string as written in the text", "event": "string or null"}
  ],
  "summary": "string (required)"
}"""

_SYSTEM_PROMPT = f"""You are a precise information extraction assistant.

Extract structured information from the document and return it as a single JSON object matching this exact schema:

{_SCHEMA_EXAMPLE}

Rules:
- Return ONLY the JSON object. No explanation. No markdown code fences. No preamble.
- Use null for fields where the information is not present in the document.
- Use an empty list [] if no items of that type are found.
- Preserve dates exactly as written in the text (e.g. "mid-1930s", "early July", "1945").
- topic and summary are required and must not be null."""


class ExtractionError(Exception):
    """Raised when extraction fails after all retries are exhausted."""


def _strip_fences(text: str) -> str:
    """Remove markdown code fences and surrounding whitespace from model output."""
    text = text.strip()
    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def extract(
    text: str,
    model,
    tokenizer,
    max_retries: int = config.MAX_RETRIES,
) -> ExtractedDocument:
    """Extract structured entities from document text using the local model.

    Retries up to max_retries times, appending validation errors to the
    conversation so the model can correct its output.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Document:\n\n{text}"},
    ]

    last_error: str = ""
    raw_response: str = ""
    for attempt in range(max_retries):
        if attempt > 0:
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({
                "role": "user",
                "content": (
                    f"Your previous response failed validation. Error: {last_error}\n"
                    "Return only the corrected JSON object."
                ),
            })

        raw_response = generate(model, tokenizer, messages)
        cleaned = _strip_fences(raw_response)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON: {e}"
            continue

        try:
            return ExtractedDocument.model_validate(parsed)
        except ValidationError as e:
            last_error = str(e)
            continue

    raise ExtractionError(f"Extraction failed after {max_retries} attempts: {last_error}")
