"""Tests for research models."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from signet.models.research import (
    ResearchArtifact,
    ResearchReport,
    ResearchResult,
    ResearchSection,
    ResearchStatus,
)


class TestResearchStatus:
    def test_values(self):
        assert ResearchStatus.QUEUED.value == "queued"
        assert ResearchStatus.IN_PROGRESS.value == "in_progress"
        assert ResearchStatus.COMPLETED.value == "completed"
        assert ResearchStatus.INTERRUPTED.value == "interrupted"
        assert ResearchStatus.FAILED.value == "failed"

    def test_roundtrip(self):
        assert ResearchStatus("queued") == ResearchStatus.QUEUED
        assert ResearchStatus("in_progress") == ResearchStatus.IN_PROGRESS
        assert ResearchStatus("completed") == ResearchStatus.COMPLETED
        assert ResearchStatus("interrupted") == ResearchStatus.INTERRUPTED
        assert ResearchStatus("failed") == ResearchStatus.FAILED


class TestResearchSection:
    def test_defaults(self):
        s = ResearchSection(question="What is X?", findings="X is Y.")

        assert s.question == "What is X?"
        assert s.findings == "X is Y."
        assert s.confidence == "medium"
        assert s.sources == []

    def test_with_sources(self):
        s = ResearchSection(
            question="How does BRCA1 interact with RAD51?",
            findings="Through the BRCT domain.",
            confidence="high",
            sources=["PMID:12345", "PMID:67890"],
        )

        assert s.confidence == "high"
        assert len(s.sources) == 2


class TestResearchArtifact:
    def test_defaults(self):
        a = ResearchArtifact(topic="somatic variant calling")

        assert isinstance(a.id, UUID)
        assert a.topic == "somatic variant calling"
        assert a.angle == ""
        assert a.status == ResearchStatus.QUEUED
        assert a.plan == ""
        assert a.sections == []
        assert a.synthesis == ""
        assert a.confidence == ""
        assert a.open_questions == []
        assert a.suggested_next == []
        assert a.source_wiki_slugs == []
        assert a.source_dream_ids == []
        assert a.model_used == ""
        assert a.token_count == 0
        assert a.tags == []
        assert isinstance(a.started_at, datetime)
        assert a.completed_at is None

    def test_field_types(self):
        dream_id = uuid4()
        a = ResearchArtifact(
            topic="scRNA-seq deconvolution",
            angle="benchmarking CIBERSORTx vs MuSiC",
            status=ResearchStatus.COMPLETED,
            sections=[
                ResearchSection(question="q1", findings="f1"),
                ResearchSection(question="q2", findings="f2"),
            ],
            source_wiki_slugs=["scrna-seq-overview"],
            source_dream_ids=[dream_id],
            model_used="claude-sonnet-4-6",
            token_count=4500,
            tags=["scrna", "deconvolution"],
            completed_at=datetime.now(timezone.utc),
        )

        assert a.status == ResearchStatus.COMPLETED
        assert len(a.sections) == 2
        assert a.sections[0].question == "q1"
        assert a.source_dream_ids[0] == dream_id
        assert a.token_count == 4500
        assert a.completed_at is not None

    def test_status_transitions(self):
        a = ResearchArtifact(topic="test")

        assert a.status == ResearchStatus.QUEUED

        a.status = ResearchStatus.IN_PROGRESS
        assert a.status == ResearchStatus.IN_PROGRESS

        a.status = ResearchStatus.COMPLETED
        assert a.status == ResearchStatus.COMPLETED

    def test_status_from_string(self):
        a = ResearchArtifact(topic="test", status="failed")

        assert a.status == ResearchStatus.FAILED


class TestResearchResult:
    def test_wrapping(self):
        artifact = ResearchArtifact(topic="proteomics QC")
        result = ResearchResult(artifact=artifact, similarity=0.87)

        assert result.artifact.topic == "proteomics QC"
        assert result.similarity == 0.87

    def test_default_similarity(self):
        artifact = ResearchArtifact(topic="whatever")
        result = ResearchResult(artifact=artifact)

        assert result.similarity == 0.0


class TestResearchReport:
    def test_defaults(self):
        r = ResearchReport()

        assert r.topic == ""
        assert r.sections_completed == 0
        assert r.total_tokens == 0
        assert r.duration_seconds == 0.0
        assert r.status == ResearchStatus.COMPLETED

    def test_with_values(self):
        r = ResearchReport(
            topic="cfDNA fragmentomics",
            sections_completed=4,
            total_tokens=8200,
            duration_seconds=45.3,
            status=ResearchStatus.COMPLETED,
        )

        assert r.sections_completed == 4
        assert r.total_tokens == 8200
        assert r.duration_seconds == 45.3

    def test_failed_report(self):
        r = ResearchReport(
            topic="oops",
            status=ResearchStatus.FAILED,
        )

        assert r.status == ResearchStatus.FAILED
        assert r.sections_completed == 0
