"""Tests for Pydantic models."""
from __future__ import annotations

from uuid import UUID

from signet.models.dreams import Dream, DreamReport, DreamType


class TestDream:
    def test_defaults(self):
        d = Dream(dream_type=DreamType.DIGEST, content="test")

        assert isinstance(d.id, UUID)
        assert d.entity_name == ""
        assert d.tags == []
        assert d.source_message_ids == []

    def test_entity_fact_fields(self):
        d = Dream(
            dream_type=DreamType.ENTITY_FACT,
            content="Pete: likes raw data",
            entity_name="Pete",
        )

        assert d.dream_type == DreamType.ENTITY_FACT
        assert d.entity_name == "Pete"


class TestDreamReport:
    def test_total_dreams(self):
        r = DreamReport(digests=2, entity_facts=3, reflections=1)

        assert r.total_dreams == 6

    def test_empty_report(self):
        r = DreamReport()

        assert r.total_dreams == 0
        assert r.messages_processed == 0
        assert r.sessions_processed == 0


class TestDreamType:
    def test_values(self):
        assert DreamType.DIGEST.value == "digest"
        assert DreamType.ENTITY_FACT.value == "entity_fact"
        assert DreamType.REFLECTION.value == "reflection"

    def test_roundtrip(self):
        assert DreamType("digest") == DreamType.DIGEST
        assert DreamType("entity_fact") == DreamType.ENTITY_FACT
