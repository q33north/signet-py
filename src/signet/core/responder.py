"""Interface-agnostic response pipeline.

Handles context gathering, prompt assembly, tool-enabled brain calls,
and response storage. Used by Discord, Slack, CLI, or any other interface.
"""
from __future__ import annotations

from datetime import datetime, timezone

import structlog

from signet.brain.client import Brain
from signet.character.prompt import PromptAssembler
from signet.config import settings
from signet.knowledge.store import WikiStore
from signet.memory.store import MemoryStore
from signet.models.dreams import DreamResult
from signet.models.knowledge import WikiSearchResult
from signet.models.memory import MemoryResult, Message, MessageRole
from signet.models.research import ResearchResult
from signet.nightshift.research_store import ResearchStore
from signet.nightshift.store import DreamStore
from signet.tools import TOOL_DEFINITIONS, execute_tool

log = structlog.get_logger()


class Responder:
    """Core response pipeline shared across all interfaces."""

    def __init__(
        self,
        assembler: PromptAssembler,
        brain: Brain,
        memory: MemoryStore,
        wiki: WikiStore,
        dreams: DreamStore,
        research: ResearchStore,
    ) -> None:
        self._assembler = assembler
        self._brain = brain
        self._memory = memory
        self._wiki = wiki
        self._dreams = dreams
        self._research = research

    async def respond(
        self,
        content: str,
        *,
        channel_id: str,
        author_name: str,
        author_id: str,
        platform: str,
        history: list[Message] | None = None,
    ) -> str:
        """Generate a response to a message.

        Gathers all context (memory, dreams, research, wiki),
        builds the system prompt, calls the brain with tools enabled,
        and returns the response text.
        """
        # Get history if not provided
        if history is None:
            history = await self._memory.get_session_history(
                channel_id=channel_id,
                limit=20,
            )

        # Gather context in parallel (all are independent)
        memories = await self._memory.recall(
            query=content,
            limit=settings.memory_recall_limit,
        )

        wiki_results = await self._wiki.search(
            query=content,
            limit=settings.wiki_recall_limit,
            min_similarity=settings.wiki_min_similarity,
        )

        dream_results = await self._dreams.recall(
            query=content,
            limit=settings.dream_recall_limit,
        )

        research_results = await self._research.recall(
            query=content,
            limit=settings.research_recall_limit,
        )

        # Format context
        memory_context = format_memories(memories) if memories else ""
        dream_context = format_dream_context(dream_results) if dream_results else ""
        research_context = format_research_context(research_results) if research_results else ""
        wiki_context = format_wiki_context(wiki_results) if wiki_results else ""

        # Build system prompt
        system = self._assembler.build_system_prompt(
            platform=platform,
            memory_context=memory_context,
            dream_context=dream_context,
            research_context=research_context,
            wiki_context=wiki_context,
        )

        # Call brain with tools
        tools = TOOL_DEFINITIONS if settings.tools_enabled else None
        tool_executor = execute_tool if settings.tools_enabled else None

        response_text = self._brain.chat(
            system=system,
            messages=history,
            tools=tools,
            tool_executor=tool_executor,
        )

        # Store the response
        response_msg = Message(
            role=MessageRole.ASSISTANT,
            content=response_text,
            platform=platform,
            channel_id=channel_id,
        )
        await self._memory.store_message(response_msg)

        return response_text


# ── Context formatters ─────────────────────────────────────
# These are shared across all interfaces.


def format_memories(memories: list[MemoryResult]) -> str:
    """Format recalled memories as context for the system prompt."""
    now = datetime.now(timezone.utc)
    lines = []
    for mem in memories:
        age = now - mem.message.timestamp
        if age.days > 0:
            time_str = f"{age.days}d ago"
        elif age.seconds > 3600:
            time_str = f"{age.seconds // 3600}h ago"
        else:
            time_str = "recently"

        author = mem.message.author_name or mem.message.role.value
        content = mem.message.content[:200]
        lines.append(f"- [{time_str}] {author}: {content}")

    header = "Relevant things from past conversations (use naturally, don't force references):"
    return header + "\n" + "\n".join(lines)


def format_dream_context(results: list[DreamResult]) -> str:
    """Format dream artifacts as context for the system prompt."""
    prefixes = {
        "digest": "Past conversation",
        "entity_fact": "You know",
        "reflection": "You've noticed",
    }
    lines = []
    for r in results:
        prefix = prefixes.get(r.dream.dream_type.value, "Memory")
        lines.append(f"- [{prefix}] {r.dream.content[:300]}")

    header = "Things you've internalized from past experience (use naturally, these are YOUR thoughts):"
    return header + "\n" + "\n".join(lines)


def format_research_context(results: list[ResearchResult]) -> str:
    """Format research artifacts as context for the system prompt."""
    lines = []
    for r in results:
        a = r.artifact
        lines.append(f"- [Research: {a.topic}] {a.synthesis[:300]}")
        if a.open_questions:
            lines.append(f"  Open questions: {'; '.join(a.open_questions[:3])}")

    header = "Research you've done recently (reference naturally when relevant):"
    return header + "\n" + "\n".join(lines)


def format_wiki_context(results: list[WikiSearchResult]) -> str:
    """Format wiki articles as context for the system prompt."""
    sections = []
    for result in results:
        fm = result.article.frontmatter
        excerpt = " ".join(result.article.body.split()[:300])
        parts = [f"### {fm.title}"]
        if fm.summary:
            parts.append(fm.summary)
        parts.append(excerpt)
        if fm.tags:
            parts.append(f"Tags: {', '.join(fm.tags)}")
        sections.append("\n".join(parts))

    header = "Relevant knowledge from your research wiki (reference naturally when relevant):"
    return header + "\n\n" + "\n\n".join(sections)
