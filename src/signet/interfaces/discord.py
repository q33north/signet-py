"""Discord bot interface using discord.py."""
from __future__ import annotations

from datetime import datetime, timezone

import discord
import structlog

from signet.brain.client import Brain
from signet.character.prompt import PromptAssembler
from signet.config import settings
from signet.knowledge.store import WikiStore
from signet.memory.store import MemoryStore
from signet.models.dreams import DreamResult
from signet.models.knowledge import WikiSearchResult
from signet.models.memory import MemoryResult, Message, MessageRole
from signet.nightshift.store import DreamStore

log = structlog.get_logger()

MAX_HISTORY = 20
CONVERSATION_TIMEOUT = 120  # seconds - stay engaged for 2 min after last response


class SignetBot(discord.Client):
    """Discord bot that routes messages through Signet's personality system."""

    def __init__(
        self,
        assembler: PromptAssembler,
        brain: Brain,
        memory: MemoryStore,
        wiki: WikiStore,
        dreams: DreamStore,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(intents=intents)

        self._assembler = assembler
        self._brain = brain
        self._memory = memory
        self._wiki = wiki
        self._dreams = dreams
        self._last_response: dict[int, datetime] = {}  # channel_id -> timestamp

    async def setup_hook(self) -> None:
        """Called by discord.py after login, before events. Async init goes here."""
        await self._memory.connect()
        await self._memory.initialize_schema()
        log.info("discord.memory_ready")

        await self._wiki.connect()
        await self._wiki.initialize_schema()
        await self._wiki.sync()
        log.info("discord.wiki_ready")

        await self._dreams.connect()
        await self._dreams.initialize_schema()
        log.info("discord.dreams_ready")

    async def close(self) -> None:
        await self._dreams.close()
        await self._wiki.close()
        await self._memory.close()
        await super().close()

    async def on_ready(self) -> None:
        log.info("discord.ready", user=str(self.user), guilds=len(self.guilds))

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return

        content = message.content
        if self.user:
            content = content.replace(f"<@{self.user.id}>", "").strip()

        # Read text file attachments (pasted files, .txt, .tsv, .csv, etc.)
        for att in message.attachments:
            if att.content_type and att.content_type.startswith("text/"):
                try:
                    file_bytes = await att.read()
                    file_text = file_bytes.decode("utf-8", errors="replace")
                    label = f"\n[attached file: {att.filename}]\n{file_text}"
                    content = f"{content}\n{label}" if content else label
                    log.info("discord.attachment_read", filename=att.filename, size=len(file_text))
                except Exception:
                    log.exception("discord.attachment_error", filename=att.filename)

        if not content:
            return

        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self.user in message.mentions if self.user else False
        role_mentioned = any(
            r.name.lower() == self._assembler._char.name.lower()
            for r in message.role_mentions
        )
        name_mentioned = self._assembler._char.name.lower() in message.content.lower()

        channel_id = message.channel.id
        author_name = str(message.author.display_name)

        log.info(
            "discord.message",
            author=author_name,
            channel=channel_id,
            content=content[:100],
        )

        # Store every message she sees
        incoming = Message(
            role=MessageRole.USER,
            content=f"{author_name}: {content}",
            platform="discord",
            channel_id=str(channel_id),
            author_id=str(message.author.id),
            author_name=author_name,
        )
        await self._memory.store_message(incoming)

        # Is she already in conversation with this channel?
        in_conversation = False
        if channel_id in self._last_response:
            age = (datetime.now(timezone.utc) - self._last_response[channel_id]).total_seconds()
            in_conversation = age < CONVERSATION_TIMEOUT

        # Decide whether to respond
        should_respond = is_dm or is_mentioned or role_mentioned or name_mentioned or in_conversation
        if not should_respond:
            log.debug("discord.listening", channel=channel_id, author=author_name)
            return

        # Recent history from this channel
        history = await self._memory.get_session_history(
            channel_id=str(channel_id),
            limit=MAX_HISTORY,
        )

        # Semantic recall across all channels
        memories = await self._memory.recall(
            query=content,
            limit=settings.memory_recall_limit,
        )

        # Wiki knowledge search
        wiki_results = await self._wiki.search(
            query=content,
            limit=settings.wiki_recall_limit,
            min_similarity=settings.wiki_min_similarity,
        )

        # Dream recall (consolidated insights from past experience)
        dream_results = await self._dreams.recall(
            query=content,
            limit=settings.dream_recall_limit,
        )

        memory_context = _format_memories(memories) if memories else ""
        dream_context = _format_dream_context(dream_results) if dream_results else ""
        wiki_context = _format_wiki_context(wiki_results) if wiki_results else ""

        system = self._assembler.build_system_prompt(
            platform="discord",
            memory_context=memory_context,
            dream_context=dream_context,
            wiki_context=wiki_context,
        )

        try:
            async with message.channel.typing():
                response_text = self._brain.chat(
                    system=system,
                    messages=history,
                )
        except Exception:
            log.exception("discord.brain_error")
            await message.channel.send(
                "shit, something broke on my end. try again in a sec."
            )
            return

        # Store response
        response_msg = Message(
            role=MessageRole.ASSISTANT,
            content=response_text,
            platform="discord",
            channel_id=str(channel_id),
        )
        await self._memory.store_message(response_msg)

        for chunk in _split_message(response_text):
            await message.channel.send(chunk)

        self._last_response[channel_id] = datetime.now(timezone.utc)
        log.info("discord.response", channel=channel_id, length=len(response_text))


def _format_memories(memories: list[MemoryResult]) -> str:
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


def _format_dream_context(results: list[DreamResult]) -> str:
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


def _format_wiki_context(results: list[WikiSearchResult]) -> str:
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


def _split_message(text: str, limit: int = 2000) -> list[str]:
    """Split a message into chunks that fit Discord's character limit."""
    if len(text) <= limit:
        return [text]

    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = text.rfind(" ", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    return chunks


def run_discord_bot(
    assembler: PromptAssembler,
    brain: Brain,
    memory: MemoryStore,
    wiki: WikiStore,
    dreams: DreamStore,
) -> None:
    """Start the Discord bot. Blocks until shutdown."""
    bot = SignetBot(assembler, brain, memory, wiki, dreams)
    bot.run(settings.discord_token, log_handler=None)
