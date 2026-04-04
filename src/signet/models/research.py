"""Pydantic models for nightshift research artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ResearchStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class ResearchSection(BaseModel):
    """One sub-question and its findings from a research deep-dive."""
    question: str
    findings: str
    confidence: str = "medium"  # high, medium, low, speculative
    sources: list[str] = Field(default_factory=list)


class ResearchArtifact(BaseModel):
    """A complete research artifact from a nightshift session."""
    id: UUID = Field(default_factory=uuid4)
    topic: str
    angle: str = ""
    status: ResearchStatus = ResearchStatus.QUEUED
    plan: str = ""                    # Step 1 output: sub-questions
    sections: list[ResearchSection] = Field(default_factory=list)
    synthesis: str = ""               # Step 3 output: final writeup
    confidence: str = ""              # overall confidence
    open_questions: list[str] = Field(default_factory=list)
    suggested_next: list[str] = Field(default_factory=list)
    source_wiki_slugs: list[str] = Field(default_factory=list)
    source_dream_ids: list[UUID] = Field(default_factory=list)
    model_used: str = ""
    token_count: int = 0
    tags: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


class ResearchResult(BaseModel):
    """Search result wrapping a research artifact with similarity score."""
    artifact: ResearchArtifact
    similarity: float = 0.0


class ResearchReport(BaseModel):
    """Summary of a nightshift research run."""
    topic: str = ""
    sections_completed: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    status: ResearchStatus = ResearchStatus.COMPLETED
