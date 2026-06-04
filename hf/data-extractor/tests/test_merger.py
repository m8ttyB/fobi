import json
from unittest.mock import MagicMock, patch

import pytest

from extractor import ExtractionError
from merger import merge
from schema import ExtractedDocument, Person


def _doc(**kwargs) -> ExtractedDocument:
    defaults = dict(topic="Physics", summary="A document.", people=[], places=[], dates=[])
    defaults.update(kwargs)
    return ExtractedDocument(**defaults)


def _make_model():
    return MagicMock(), MagicMock()


MERGED_JSON = json.dumps({
    "title": "Einstein",
    "topic": "Physics",
    "people": [{"name": "Albert Einstein", "role": "physicist", "context": "developed relativity"}],
    "places": [{"name": "Berlin", "context": "where he worked"}],
    "dates": [{"date": "1905", "event": "annus mirabilis"}],
    "summary": "A document about Einstein.",
})


class TestMergeShortCircuits:
    def test_empty_list_raises(self):
        model, tokenizer = _make_model()
        with pytest.raises(ValueError, match="empty"):
            merge([], model, tokenizer)

    def test_single_partial_returned_directly(self):
        model, tokenizer = _make_model()
        doc = _doc(title="Only", topic="Solo", summary="One chunk.")
        with patch("merger.generate") as mock_gen:
            result = merge([doc], model, tokenizer)
        mock_gen.assert_not_called()
        assert result is doc


class TestMergeSuccess:
    def test_two_partials_calls_model(self):
        model, tokenizer = _make_model()
        partials = [_doc(), _doc()]
        with patch("merger.generate", return_value=MERGED_JSON):
            result = merge(partials, model, tokenizer)
        assert result.topic == "Physics"
        assert result.people[0].name == "Albert Einstein"

    def test_strips_fences(self):
        fenced = f"```json\n{MERGED_JSON}\n```"
        model, tokenizer = _make_model()
        with patch("merger.generate", return_value=fenced):
            result = merge([_doc(), _doc()], model, tokenizer)
        assert result.title == "Einstein"

    def test_deduplication_in_merge_prompt(self):
        p1 = _doc(people=[Person(name="Einstein", role="physicist")])
        p2 = _doc(people=[Person(name="Albert Einstein", role="theoretical physicist")])
        captured = []

        def fake_generate(model, tokenizer, messages):
            captured.append(messages)
            return MERGED_JSON

        model, tokenizer = _make_model()
        with patch("merger.generate", side_effect=fake_generate):
            merge([p1, p2], model, tokenizer)

        # Both partial extractions should appear in the merge prompt
        prompt_text = str(captured[0])
        assert "Einstein" in prompt_text
        assert "Albert Einstein" in prompt_text


class TestMergeRetry:
    def test_retries_on_invalid_json(self):
        model, tokenizer = _make_model()
        responses = ["not json", MERGED_JSON]
        with patch("merger.generate", side_effect=responses):
            result = merge([_doc(), _doc()], model, tokenizer, max_retries=2)
        assert result.topic == "Physics"

    def test_retries_on_validation_error(self):
        bad = json.dumps({"title": "X", "people": [], "places": [], "dates": []})  # missing topic+summary
        model, tokenizer = _make_model()
        responses = [bad, MERGED_JSON]
        with patch("merger.generate", side_effect=responses):
            result = merge([_doc(), _doc()], model, tokenizer, max_retries=2)
        assert result.summary == "A document about Einstein."

    def test_exhaustion_raises_extraction_error(self):
        model, tokenizer = _make_model()
        with patch("merger.generate", return_value="bad"):
            with pytest.raises(ExtractionError):
                merge([_doc(), _doc()], model, tokenizer, max_retries=2)

    def test_retry_appends_error_to_messages(self):
        captured = []

        def fake_generate(model, tokenizer, messages):
            captured.append(list(messages))
            if len(captured) == 1:
                return "not json"
            return MERGED_JSON

        model, tokenizer = _make_model()
        with patch("merger.generate", side_effect=fake_generate):
            merge([_doc(), _doc()], model, tokenizer, max_retries=2)

        assert len(captured[1]) > len(captured[0])
        assert "error" in captured[1][-1]["content"].lower() or "failed" in captured[1][-1]["content"].lower()
