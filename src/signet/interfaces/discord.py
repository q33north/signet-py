"""Discord bot interface using discord.py."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import discord
import structlog

from signet.brain.client import Brain
from signet.character.prompt import PromptAssembler
from signet.config import settings
from signet.core.responder import Responder
from signet.knowledge.store import WikiStore
from signet.memory.store import MemoryStore
from signet.models.memory import Message, MessageRole
from signet.nightshift.research_store import ResearchStore
from signet.nightshift.researcher import Researcher, format_research_for_discord
from signet.nightshift.store import DreamStore

log = structlog.get_logger()

MAX_HISTORY = 20
CONVERSATION_TIMEOUT = 120  # seconds - stay engaged for 2 min after last response


class SignetBot(discord.Client):
    """Discord bot that routes messages through Signet's core pipeline."""

    def __init__(
        self,
        assembler: PromptAssembler,
        brain: Brain,
        memory: MemoryStore,
        wiki: WikiStore,
        dreams: DreamStore,
        research: ResearchStore,
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
        self._research = research
        self._responder = Responder(assembler, brain, memory, wiki, dreams, research)
        self._last_response: dict[int, datetime] = {}  # channel_id -> timestamp
        self._last_activity: datetime = datetime.now(timezone.utc)
        self._researcher: Researcher | None = None
        self._nightshift_task: asyncio.Task | None = None

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

        await self._research.connect()
        await self._research.initialize_schema()
        log.info("discord.research_ready")

        # Start nightshift background loop if enabled
        if settings.nightshift_enabled and settings.nightshift_channel_id:
            self._researcher = Researcher(
                brain=self._brain,
                memory=self._memory,
                wiki=self._wiki,
                dreams=self._dreams,
                research=self._research,
            )
            self._nightshift_task = asyncio.create_task(self._nightshift_loop())
            log.info("discord.nightshift_started")

    async def close(self) -> None:
        if self._nightshift_task and not self._nightshift_task.done():
            self._nightshift_task.cancel()
        await self._research.close()
        await self._dreams.close()
        await self._wiki.close()
        await self._memory.close()
        await super().close()

    async def on_ready(self) -> None:
        log.info("discord.ready", user=str(self.user), guilds=len(self.guilds))

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return

        self._last_activity = datetime.now(timezone.utc)

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

        # Interrupt nightshift research if user comes back
        if self._researcher and not self._researcher.interrupted:
            self._researcher.interrupt()

        try:
            async with message.channel.typing():
                response_text = await self._responder.respond(
                    content,
                    channel_id=str(channel_id),
                    author_name=author_name,
                    author_id=str(message.author.id),
                    platform="discord",
                )
        except Exception:
            log.exception("discord.brain_error")
            await message.channel.send(
                "shit, something broke on my end. try again in a sec."
            )
            return

        for chunk in _split_message(response_text):
            await message.channel.send(chunk)

        self._last_response[channel_id] = datetime.now(timezone.utc)
        log.info("discord.response", channel=channel_id, length=len(response_text))

    async def _nightshift_loop(self) -> None:
        """Background loop that triggers research during quiet periods."""
        interval = settings.nightshift_check_interval_minutes * 60
        quiet_threshold = settings.nightshift_quiet_minutes * 60

        while True:
            try:
                await asyncio.sleep(interval)

                age = (datetime.now(timezone.utc) - self._last_activity).total_seconds()
                if age < quiet_threshold:
                    continue

                log.info("nightshift.quiet_detected", idle_minutes=age / 60)

                report = await self._researcher.run()

                if report.status.value == "completed":
                    channel = self.get_channel(int(settings.nightshift_channel_id))
                    if channel:
                        recent = await self._research.recent(limit=1)
                        if recent:
                            text = format_research_for_discord(recent[0])
                            for chunk in _split_message(text):
                                await channel.send(chunk)
                            log.info("nightshift.posted", topic=report.topic)

            except asyncio.CancelledError:
                log.info("nightshift.cancelled")
                break
            except Exception:
                log.exception("nightshift.loop_error")
                await asyncio.sleep(60)


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
    research: ResearchStore,
) -> None:
    """Start the Discord bot. Blocks until shutdown."""
    bot = SignetBot(assembler, brain, memory, wiki, dreams, research)
    bot.run(settings.discord_token, log_handler=None)
