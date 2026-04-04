"""Thin wrapper around the Anthropic SDK with tool use support."""
from __future__ import annotations

import structlog
from anthropic import Anthropic

from signet.config import settings
from signet.models.memory import Message, MessageRole

log = structlog.get_logger()


class Brain:
    """Claude API client with model routing and tool use."""

    def __init__(self) -> None:
        self._client = Anthropic(api_key=settings.anthropic_api_key)

    def chat(
        self,
        system: str,
        messages: list[Message],
        model: str | None = None,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        tool_executor: object | None = None,
    ) -> str:
        """Send a conversation to Claude and return the response text.

        If tools and tool_executor are provided, handles the tool use loop:
        Claude may request tool calls, which are executed via tool_executor,
        and results are fed back until Claude produces a final text response.

        tool_executor should be a callable: (name: str, input: dict) -> str
        """
        model = model or settings.model_heavy

        api_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
            if msg.role != MessageRole.SYSTEM
        ]

        log.debug("brain.chat", model=model, message_count=len(api_messages))

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = tools

        total_input_tokens = 0
        total_output_tokens = 0

        for iteration in range(settings.tool_max_iterations):
            response = self._client.messages.create(**kwargs)

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            # If no tool use or no executor, return the text
            if response.stop_reason != "tool_use" or not tool_executor:
                text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text += block.text
                log.debug(
                    "brain.response",
                    model=model,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    tool_iterations=iteration,
                )
                return text

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    log.info(
                        "brain.tool_use",
                        tool=block.name,
                        input=str(block.input)[:200],
                    )
                    result = tool_executor(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Append assistant response and tool results to messages
            api_messages.append({"role": "assistant", "content": response.content})
            api_messages.append({"role": "user", "content": tool_results})
            kwargs["messages"] = api_messages

        # Exhausted iterations
        log.warning("brain.tool_iterations_exhausted", iterations=settings.tool_max_iterations)
        return "I got stuck in a tool loop. Let me try a different approach."

    def quick(self, prompt: str, system: str = "") -> str:
        """Single-shot prompt using the lightweight model. For evaluators etc."""
        messages = [Message(role=MessageRole.USER, content=prompt)]
        return self.chat(
            system=system,
            messages=messages,
            model=settings.model_light,
            max_tokens=1024,
        )
