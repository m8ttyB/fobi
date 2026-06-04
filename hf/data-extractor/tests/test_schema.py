import pytest
from pydantic import ValidationError

from schema import Date, ExtractedDocument, Person, Place


class TestPerson:
    def test_full(self):
        p = Person(name="Einstein", role="physicist", context="discussed relativity")
        assert p.name == "Einstein"
        assert p.role == "physicist"
        assert p.context == "discussed relativity"

    def test_name_only(self):
        p = Person(name="Einstein")
        assert p.role is None
        assert p.context is None

    def test_name_required(self):
        with pytest.raises(ValidationError):
            Person(role="physicist")


class TestPlace:
    def test_full(self):
        pl = Place(name="Los Alamos", context="primary weapons design facility")
        assert pl.name == "Los Alamos"
        assert pl.context == "primary weapons design facility"

    def test_name_only(self):
        pl = Place(name="Los Alamos")
        assert pl.context is None

    def test_name_required(self):
        with pytest.raises(ValidationError):
            Place(context="somewhere")


class TestDate:
    def test_full(self):
        d = Date(date="July 16, 1945", event="Trinity nuclear test")
        assert d.date == "July 16, 1945"
        assert d.event == "Trinity nuclear test"

    def test_approximate_date_preserved(self):
        d = Date(date="mid-1930s")
        assert d.date == "mid-1930s"
        assert d.event is None

    def test_date_required(self):
        with pytest.raises(ValidationError):
            Date(event="something happened")


class TestExtractedDocument:
    def test_minimal_valid(self):
        doc = ExtractedDocument(
            topic="Nuclear physics",
            summary="A document about nuclear physics.",
            people=[],
            places=[],
            dates=[],
        )
        assert doc.title is None
        assert doc.topic == "Nuclear physics"
        assert doc.people == []

    def test_full(self):
        doc = ExtractedDocument(
            title="The Oppenheimer Story",
            topic="Manhattan Project",
            people=[Person(name="Oppenheimer", role="physicist")],
            places=[Place(name="Los Alamos")],
            dates=[Date(date="July 16, 1945", event="Trinity test")],
            summary="An account of the bomb's development.",
        )
        assert doc.title == "The Oppenheimer Story"
        assert len(doc.people) == 1
        assert doc.people[0].name == "Oppenheimer"

    def test_topic_required(self):
        with pytest.raises(ValidationError):
            ExtractedDocument(summary="A summary.", people=[], places=[], dates=[])

    def test_summary_required(self):
        with pytest.raises(ValidationError):
            ExtractedDocument(topic="Something", people=[], places=[], dates=[])

    def test_list_fields_default_empty(self):
        doc = ExtractedDocument(topic="Something", summary="A summary.")
        assert doc.people == []
        assert doc.places == []
        assert doc.dates == []

    def test_model_validate_from_dict(self):
        data = {
            "title": "Test",
            "topic": "Testing",
            "people": [{"name": "Alice", "role": "engineer", "context": None}],
            "places": [],
            "dates": [{"date": "2024-01-01", "event": "Launch"}],
            "summary": "A test document.",
        }
        doc = ExtractedDocument.model_validate(data)
        assert doc.people[0].name == "Alice"
        assert doc.dates[0].event == "Launch"

    def test_invalid_people_entry(self):
        with pytest.raises(ValidationError):
            ExtractedDocument(
                topic="X",
                summary="Y",
                people=[{"role": "engineer"}],  # missing name
            )
