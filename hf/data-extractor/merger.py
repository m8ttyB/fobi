import json

from pydantic import ValidationError

import config
from extractor import ExtractionError, _strip_fences
from model import generate
from schema import ExtractedDocument

_MERGE_SYSTEM_PROMPT = """\
You are a precise information extraction assistant performing a deduplication and merge task.

You will receive several partial extractions from consecutive overlapping chunks of the same document. \
Each partial extraction is a JSON object. Your task is to merge them into one canonical result.

Rules for merging:
- Deduplicate people, places, and dates by identity — not just by exact string match. \
"Einstein" and "Albert Einstein" are the same person; keep the most complete version.
- For people and places, prefer the entry with the most detail (name, role, context).
- For dates, keep all unique events. Deduplicate only when the same event appears in multiple chunks.
- For title: prefer non-null values; if multiple non-null titles exist, prefer the one from the earliest chunk.
- For topic and summary: write unified versions that describe the full document, not just one chunk.
- Return ONLY the JSON object. No explanation. No markdown code fences. No preamble.
- Use null for fields where no information is present. Use [] if no items of that type were found.

Output schema:
{
  "title": "string or null",
  "topic": "string (required)",
  "people": [{"name": "string", "role": "string or null", "context": "string or null"}],
  "places": [{"name": "string", "context": "string or null"}],
  "dates": [{"date": "string as written in source", "event": "string or null"}],
  "summary": "string (required)"
}"""


def merge(
    partials: list[ExtractedDocument],
    model,
    tokenizer,
    max_retries: int = config.MAX_RETRIES,
) -> ExtractedDocument:
    """Merge partial per-chunk extractions into a single deduplicated ExtractedDocument.

    Returns the single partial directly if only one is provided (no model call needed).
    Raises ValueError if partials is empty.
    Raises ExtractionError if all retries are exhausted.
    """
    if not partials:
        raise ValueError("Cannot merge empty list of partial extractions.")
    if len(partials) == 1:
        return partials[0]

    partials_json = json.dumps(
        [p.model_dump() for p in partials],
        indent=2,
    )

    messages = [
        {"role": "system", "content": _MERGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Partial extractions to merge:\n\n{partials_json}"},
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

    raise ExtractionError(f"Merge failed after {max_retries} attempts: {last_error}")
