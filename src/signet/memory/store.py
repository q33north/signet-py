"""Persistent memory store backed by PostgreSQL + pgvector."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import UUID

import asyncpg
import structlog
from pgvector.asyncpg import register_vector

from signet.memory.embeddings import EmbeddingService
from signet.models.memory import MemoryResult, Message, MessageRole, Session

log = structlog.get_logger()

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class MemoryStore:
    """Async persistence layer for conversation memory.

    Manages message storage, embedding generation, and semantic retrieval.
    """

    def __init__(self, database_url: str, embedder: EmbeddingService) -> None:
        self._database_url = database_url
        self._embedder = embedder
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._database_url,
            min_size=2,
            max_size=10,
            init=register_vector,
        )
        log.info("memory.connected", database=self._database_url.split("@")[-1])

    async def initialize_schema(self) -> None:
        sql = SCHEMA_PATH.read_text()
        async with self._pool.acquire() as conn:
            await conn.execute(sql)
        log.info("memory.schema_initialized")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            log.info("memory.disconnected")

    # ── Messages ────────────────────────────────────────────────

    async def store_message(self, message: Message) -> Message:
        """Store a message and compute its embedding."""
        embedding = await self._embedder.embed(message.content)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO messages (id, session_id, role, content, platform,
                                     channel_id, author_id, author_name, timestamp, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                message.id,
                message.session_id,
                message.role.value,
                message.content,
                message.platform,
                message.channel_id,
                message.author_id,
                message.author_name,
                message.timestamp,
                embedding,
            )

        log.debug("memory.stored", message_id=str(message.id), role=message.role.value)
        return message

    async def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        channel_id: str | None = None,
        platform: str | None = None,
        before: datetime | None = None,
    ) -> list[MemoryResult]:
        """Semantic search across stored messages."""
        query_embedding = await self._embedder.embed(query)

        conditions = ["embedding IS NOT NULL"]
        params: list = [query_embedding, limit]
        param_idx = 3

        if channel_id is not None:
            conditions.append(f"channel_id = ${param_idx}")
            params.append(channel_id)
            param_idx += 1

        if platform is not None:
            conditions.append(f"platform = ${param_idx}")
            params.append(platform)
            param_idx += 1

        if before is not None:
            conditions.append(f"timestamp < ${param_idx}")
            params.append(before)
            param_idx += 1

        where = " AND ".join(conditions)

        sql = f"""
            SELECT id, session_id, role, content, platform, channel_id,
                   author_id, author_name, timestamp,
                   1 - (embedding <=> $1) AS similarity
            FROM messages
            WHERE {where}
            ORDER BY embedding <=> $1
            LIMIT $2
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        results = []
        for row in rows:
            msg = Message(
                id=row["id"],
                session_id=row["session_id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                platform=row["platform"],
                channel_id=row["channel_id"],
                author_id=row["author_id"],
                author_name=row["author_name"],
                timestamp=row["timestamp"],
            )
            results.append(MemoryResult(message=msg, similarity=row["similarity"]))

        log.debug("memory.recall", query=query[:50], results=len(results))
        return results

    async def get_session_history(
        self,
        channel_id: str,
        limit: int = 20,
    ) -> list[Message]:
        """Get recent messages for a channel, chronological order."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, session_id, role, content, platform, channel_id,
                       author_id, author_name, timestamp
                FROM messages
                WHERE channel_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """,
                channel_id,
                limit,
            )

        return [
            Message(
                id=row["id"],
                session_id=row["session_id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                platform=row["platform"],
                channel_id=row["channel_id"],
                author_id=row["author_id"],
                author_name=row["author_name"],
                timestamp=row["timestamp"],
            )
            for row in reversed(rows)
        ]

    # ── Sessions ────────────────────────────────────────────────

    async def create_session(
        self,
        platform: str = "",
        channel_id: str = "",
    ) -> Session:
        session = Session(platform=platform, channel_id=channel_id)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions (id, platform, channel_id, started_at)
                VALUES ($1, $2, $3, $4)
                """,
                session.id,
                session.platform,
                session.channel_id,
                session.started_at,
            )

        log.info("memory.session_created", session_id=str(session.id), channel=channel_id)
        return session

    async def end_session(self, session_id: UUID) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE sessions
                SET ended_at = now(),
                    message_count = (
                        SELECT COUNT(*) FROM messages WHERE session_id = $1
                    )
                WHERE id = $1
                """,
                session_id,
            )
        log.info("memory.session_ended", session_id=str(session_id))
