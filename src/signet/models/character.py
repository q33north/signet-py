"""Character definition models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class MessageTurn(BaseModel):
    """A single message in an example conversation."""

    name: str
    text: str


class ConversationExample(BaseModel):
    """A full example conversation thread."""

    messages: list[MessageTurn]


class StyleDirectives(BaseModel):
    """Style rules applied to prompt generation."""

    all: list[str] = Field(default_factory=list)
    chat: list[str] = Field(default_factory=list)
    post: list[str] = Field(default_factory=list)


class Character(BaseModel):
    """Complete character definition. Loaded from YAML."""

    name: str
    system: str
    bio: list[str]
    adjectives: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    message_examples: list[ConversationExample] = Field(default_factory=list)
    style: StyleDirectives = Field(default_factory=StyleDirectives)
    bio_sample_size: int = 10
    example_sample_size: int = 5
