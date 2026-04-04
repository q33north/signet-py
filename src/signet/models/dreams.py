"""Dream / memory consolidation models."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DreamType(str, Enum):
    DIGEST = "digest"
    ENTITY_FACT = "entity_fact"
    REFLECTION = "reflection"


class Dream(BaseModel):
    """A single consolidated memory artifact."""

    id: UUID = Field(default_factory=uuid4)
    dream_type: DreamType
    content: str
    source_message_ids: list[UUID] = Field(default_factory=list)
    entity_name: str = ""
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class DreamResult(BaseModel):
    """A dream returned from semantic search."""

    dream: Dream
    similarity: float


class DreamReport(BaseModel):
    """Summary of a dream run."""

    sessions_processed: int = 0
    messages_processed: int = 0
    digests: int = 0
    entity_facts: int = 0
    reflections: int = 0

    @property
    def total_dreams(self) -> int:
        return self.digests + self.entity_facts + self.reflections
