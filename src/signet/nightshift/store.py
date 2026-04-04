"""Persistence layer for dream artifacts (autoDream consolidation output)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import UUID

import asyncpg
import structlog
from pgvector.asyncpg import register_vector

from signet.memory.embeddings import EmbeddingService
from signet.models.dreams import Dream, DreamResult, DreamType

log = structlog.get_logger()

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class DreamStore:
    """Async persistence for consolidated dream artifacts."""

    def __init__(self, database_url: str, embedder: EmbeddingService) -> None:
        self._database_url = database_url
        self._embedder = embedder
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._database_url,
            min_size=1,
            max_size=5,
            init=register_vector,
        )
        log.info("dreams.connected")

    async def initialize_schema(self) -> None:
        sql = SCHEMA_PATH.read_text()
        async with self._pool.acquire() as conn:
            await conn.execute(sql)
        log.info("dreams.schema_initialized")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def store_dream(self, dream: Dream) -> None:
        """Store a dream artifact with its embedding."""
        embedding = await self._embedder.embed(dream.content)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO dreams (id, dream_type, content, source_message_ids,
                                    entity_name, tags, created_at, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                dream.id,
                dream.dream_type.value,
                dream.content,
                dream.source_message_ids,
                dream.entity_name,
                dream.tags,
                dream.created_at,
                embedding,
            )

        log.debug("dreams.stored", dream_id=str(dream.id), type=dream.dream_type.value)

    async def recall(
        self,
        query: str,
        *,
        limit: int = 3,
        dream_type: DreamType | None = None,
    ) -> list[DreamResult]:
        """Semantic search across dream artifacts."""
        query_embedding = await self._embedder.embed(query)

        conditions = ["embedding IS NOT NULL"]
        params: list = [query_embedding, limit]
        param_idx = 3

        if dream_type is not None:
            conditions.append(f"dream_type = ${param_idx}")
            params.append(dream_type.value)
            param_idx += 1

        where = " AND ".join(conditions)

        sql = f"""
            SELECT id, dream_type, content, source_message_ids,
                   entity_name, tags, created_at,
                   1 - (embedding <=> $1) AS similarity
            FROM dreams
            WHERE {where}
            ORDER BY embedding <=> $1
            LIMIT $2
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        results = []
        for row in rows:
            dream = Dream(
                id=row["id"],
                dream_type=DreamType(row["dream_type"]),
                content=row["content"],
                source_message_ids=row["source_message_ids"] or [],
                entity_name=row["entity_name"],
                tags=row["tags"] or [],
                created_at=row["created_at"],
            )
            results.append(DreamResult(dream=dream, similarity=row["similarity"]))

        log.debug("dreams.recall", query=query[:50], results=len(results))
        return results

    async def get_entity_facts(self, entity_name: str) -> list[Dream]:
        """Fetch all known facts about a specific entity."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, dream_type, content, source_message_ids,
                       entity_name, tags, created_at
                FROM dreams
                WHERE dream_type = 'entity_fact'
                  AND lower(entity_name) = lower($1)
                ORDER BY created_at DESC
                """,
                entity_name,
            )

        return [
            Dream(
                id=row["id"],
                dream_type=DreamType(row["dream_type"]),
                content=row["content"],
                source_message_ids=row["source_message_ids"] or [],
                entity_name=row["entity_name"],
                tags=row["tags"] or [],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def recent(
        self,
        limit: int = 20,
        dream_type: DreamType | None = None,
    ) -> list[Dream]:
        """Fetch recent dreams, newest first."""
        conditions = []
        params: list = [limit]
        param_idx = 2

        if dream_type is not None:
            conditions.append(f"dream_type = ${param_idx}")
            params.append(dream_type.value)
            param_idx += 1

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        sql = f"""
            SELECT id, dream_type, content, source_message_ids,
                   entity_name, tags, created_at
            FROM dreams
            {where}
            ORDER BY created_at DESC
            LIMIT $1
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [
            Dream(
                id=row["id"],
                dream_type=DreamType(row["dream_type"]),
                content=row["content"],
                source_message_ids=row["source_message_ids"] or [],
                entity_name=row["entity_name"],
                tags=row["tags"] or [],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def last_dream_time(self) -> datetime | None:
        """When was the last dream run?"""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT MAX(created_at) AS last FROM dreams"
            )
        return row["last"] if row else None

    async def count_by_type(self) -> dict[str, int]:
        """Count dreams grouped by type."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT dream_type, COUNT(*) AS cnt
                FROM dreams
                GROUP BY dream_type
                """
            )
        return {row["dream_type"]: row["cnt"] for row in rows}
