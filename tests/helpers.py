"""Shared test helpers and constants."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from signet.models.memory import Message, MessageRole


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def make_message(
    content: str = "test message",
    *,
    role: MessageRole = MessageRole.USER,
    channel_id: str = "chan-1",
    author_name: str = "TestUser",
    timestamp: datetime | None = None,
) -> Message:
    """Create a Message with sensible defaults."""
    return Message(
        id=uuid4(),
        role=role,
        content=content,
        platform="discord",
        channel_id=channel_id,
        author_id="user-1",
        author_name=author_name,
        timestamp=timestamp or _utcnow(),
    )


def make_conversation(
    n: int = 5,
    channel_id: str = "chan-1",
    start: datetime | None = None,
    gap_minutes: int = 2,
) -> list[Message]:
    """Create a series of messages simulating a conversation."""
    start = start or _utcnow() - timedelta(hours=3)
    messages = []
    for i in range(n):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        author = "TestUser" if role == MessageRole.USER else "Signet"
        messages.append(
            make_message(
                content=f"message {i}",
                role=role,
                channel_id=channel_id,
                author_name=author,
                timestamp=start + timedelta(minutes=i * gap_minutes),
            )
        )
    return messages


VALID_LLM_RESPONSE = json.dumps(
    {
        "digest": "User discussed KRAS resistance mechanisms. Concluded that MRTX849 shows promise but needs combination therapy.",
        "entity_facts": [
            {"entity": "Pete", "fact": "Working on KRAS G12C resistance in lung adenocarcinoma"},
            {"entity": "Pete", "fact": "Prefers to see raw data before conclusions"},
        ],
        "reflections": [
            "KRAS resistance keeps coming up across multiple conversations",
        ],
    }
)

VALID_LLM_RESPONSE_FENCED = f"```json\n{VALID_LLM_RESPONSE}\n```"

EMPTY_LLM_RESPONSE = json.dumps(
    {"digest": None, "entity_facts": [], "reflections": []}
)

GARBAGE_LLM_RESPONSE = "I couldn't parse that properly, here are some thoughts..."
