"""autoDream memory consolidation engine.

Groups unconsolidated messages into conversation bundles, sends each through
an LLM to extract digests, entity facts, and reflections, then stores the
dream artifacts for semantic retrieval during live conversation.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from uuid import UUID

import structlog

from signet.brain.client import Brain
from signet.memory.store import MemoryStore
from signet.models.dreams import Dream, DreamReport, DreamType
from signet.models.memory import Message
from signet.nightshift.prompts import CONSOLIDATION_PROMPT, CONSOLIDATION_SYSTEM
from signet.nightshift.store import DreamStore

log = structlog.get_logger()

# Messages in the same channel within this gap are grouped together
CONVERSATION_GAP = timedelta(minutes=30)


@dataclass
class ConversationBundle:
    """A group of related messages to consolidate together."""

    channel_id: str
    messages: list[Message] = field(default_factory=list)

    @property
    def message_ids(self) -> list[UUID]:
        return [m.id for m in self.messages]


class Dreamer:
    """Orchestrates the autoDream consolidation pipeline."""

    def __init__(
        self,
        memory: MemoryStore,
        dreams: DreamStore,
        brain: Brain,
    ) -> None:
        self._memory = memory
        self._dreams = dreams
        self._brain = brain

    async def dream(self, *, max_messages: int = 500) -> DreamReport:
        """Run a full dream cycle: gather, consolidate, store."""
        messages = await self._memory.get_unconsolidated_messages(limit=max_messages)
        if not messages:
            log.info("dream.nothing_to_process")
            return DreamReport()

        bundles = self._group_into_bundles(messages)
        log.info("dream.starting", messages=len(messages), bundles=len(bundles))

        report = DreamReport(messages_processed=len(messages))

        for bundle in bundles:
            try:
                dreams = await self._consolidate_bundle(bundle)
                for d in dreams:
                    await self._dreams.store_dream(d)

                report.digests += sum(1 for d in dreams if d.dream_type == DreamType.DIGEST)
                report.entity_facts += sum(1 for d in dreams if d.dream_type == DreamType.ENTITY_FACT)
                report.reflections += sum(1 for d in dreams if d.dream_type == DreamType.REFLECTION)
                report.sessions_processed += 1

            except Exception:
                log.exception("dream.bundle_error", channel=bundle.channel_id)
                continue

        # Mark all messages as consolidated (even if their bundle errored,
        # we don't want to reprocess them endlessly)
        all_ids = [m.id for m in messages]
        await self._memory.mark_messages_consolidated(all_ids)

        log.info(
            "dream.complete",
            messages=report.messages_processed,
            bundles=report.sessions_processed,
            digests=report.digests,
            entity_facts=report.entity_facts,
            reflections=report.reflections,
        )
        return report

    def _group_into_bundles(self, messages: list[Message]) -> list[ConversationBundle]:
        """Group messages into conversation bundles by channel and time gap."""
        by_channel: dict[str, list[Message]] = defaultdict(list)
        for msg in messages:
            by_channel[msg.channel_id].append(msg)

        bundles: list[ConversationBundle] = []

        for channel_id, channel_msgs in by_channel.items():
            channel_msgs.sort(key=lambda m: m.timestamp)

            current = ConversationBundle(channel_id=channel_id)
            current.messages.append(channel_msgs[0])

            for msg in channel_msgs[1:]:
                prev = current.messages[-1]
                gap = msg.timestamp - prev.timestamp
                if gap > CONVERSATION_GAP:
                    if len(current.messages) >= 2:
                        bundles.append(current)
                    current = ConversationBundle(channel_id=channel_id)
                current.messages.append(msg)

            if len(current.messages) >= 2:
                bundles.append(current)

        bundles.sort(key=lambda b: b.messages[0].timestamp)
        return bundles

    async def _consolidate_bundle(self, bundle: ConversationBundle) -> list[Dream]:
        """Send a conversation bundle through the LLM for consolidation."""
        formatted = self._format_messages(bundle.messages)
        channel_info = f"channel {bundle.channel_id}"

        prompt = CONSOLIDATION_PROMPT.format(
            message_count=len(bundle.messages),
            channel_info=channel_info,
            messages=formatted,
        )

        raw = self._brain.quick(prompt, system=CONSOLIDATION_SYSTEM)
        return self._parse_dreams(raw, bundle.message_ids)

    def _format_messages(self, messages: list[Message]) -> str:
        """Format messages for the consolidation prompt."""
        lines = []
        for msg in messages:
            ts = msg.timestamp.strftime("%Y-%m-%d %H:%M")
            author = msg.author_name or msg.role.value
            lines.append(f"[{ts}] {author}: {msg.content}")
        return "\n".join(lines)

    def _parse_dreams(self, raw_response: str, source_ids: list[UUID]) -> list[Dream]:
        """Parse LLM JSON response into Dream objects."""
        # Strip markdown fences if present
        text = raw_response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            log.warning("dream.json_parse_failed", response=raw_response[:200])
            # Fallback: store the whole response as a digest
            return [
                Dream(
                    dream_type=DreamType.DIGEST,
                    content=raw_response,
                    source_message_ids=source_ids,
                )
            ]

        dreams: list[Dream] = []

        # Digest
        digest = data.get("digest")
        if digest:
            dreams.append(
                Dream(
                    dream_type=DreamType.DIGEST,
                    content=digest,
                    source_message_ids=source_ids,
                )
            )

        # Entity facts
        for fact in data.get("entity_facts") or []:
            entity = fact.get("entity", "")
            fact_text = fact.get("fact", "")
            if entity and fact_text:
                dreams.append(
                    Dream(
                        dream_type=DreamType.ENTITY_FACT,
                        content=f"{entity}: {fact_text}",
                        entity_name=entity,
                        source_message_ids=source_ids,
                    )
                )

        # Reflections
        for reflection in data.get("reflections") or []:
            if reflection:
                dreams.append(
                    Dream(
                        dream_type=DreamType.REFLECTION,
                        content=reflection,
                        source_message_ids=source_ids,
                    )
                )

        return dreams
