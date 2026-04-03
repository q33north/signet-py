"""Memory and session models."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single message in a conversation."""

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID | None = None
    role: MessageRole
    content: str
    platform: str = ""
    channel_id: str = ""
    author_id: str = ""
    author_name: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)


class Session(BaseModel):
    """A conversation session."""

    id: UUID = Field(default_factory=uuid4)
    platform: str = ""
    channel_id: str = ""
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    message_count: int = 0
    consolidated: bool = False


class MemoryResult(BaseModel):
    """A message returned from semantic search, with relevance score."""

    message: Message
    similarity: float
