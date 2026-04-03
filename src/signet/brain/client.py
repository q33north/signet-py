"""Thin wrapper around the Anthropic SDK."""
from __future__ import annotations

import structlog
from anthropic import Anthropic

from signet.config import settings
from signet.models.memory import Message, MessageRole

log = structlog.get_logger()


class Brain:
    """Claude API client with model routing."""

    def __init__(self) -> None:
        self._client = Anthropic(api_key=settings.anthropic_api_key)

    def chat(
        self,
        system: str,
        messages: list[Message],
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Send a conversation to Claude and return the response text."""
        model = model or settings.model_heavy

        api_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
            if msg.role != MessageRole.SYSTEM
        ]

        log.debug("brain.chat", model=model, message_count=len(api_messages))

        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=api_messages,
        )

        text = response.content[0].text
        log.debug(
            "brain.response",
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return text

    def quick(self, prompt: str, system: str = "") -> str:
        """Single-shot prompt using the lightweight model. For evaluators etc."""
        messages = [Message(role=MessageRole.USER, content=prompt)]
        return self.chat(
            system=system,
            messages=messages,
            model=settings.model_light,
            max_tokens=1024,
        )
