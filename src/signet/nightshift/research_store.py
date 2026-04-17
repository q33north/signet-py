"""Persistence layer for nightshift research artifacts."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
import structlog
from pgvector.asyncpg import register_vector

from signet.memory.embeddings import EmbeddingService
from signet.models.research import (
    ResearchArtifact,
    ResearchResult,
    ResearchSection,
    ResearchStatus,
)

log = structlog.get_logger()

SCHEMA_PATH = Path(__file__).parent / "research_schema.sql"


class ResearchStore:
    """Async persistence for nightshift research artifacts."""

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
        log.info("research.connected")

    async def initialize_schema(self) -> None:
        sql = SCHEMA_PATH.read_text()
        async with self._pool.acquire() as conn:
            await conn.execute(sql)
        log.info("research.schema_initialized")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    # ── CRUD ───────────────────────────────────────────────

    async def save(self, artifact: ResearchArtifact) -> None:
        """Insert or update a research artifact. Embeds the synthesis if present."""
        embedding = None
        if artifact.synthesis:
            embedding = await self._embedder.embed(artifact.synthesis)

        sections_json = json.dumps(
            [s.model_dump() for s in artifact.sections]
        )

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO research (
                    id, topic, angle, status, plan, sections, synthesis,
                    confidence, open_questions, suggested_next,
                    source_wiki_slugs, source_dream_ids, model_used,
                    token_count, tags, started_at, completed_at, embedding
                ) VALUES (
                    $1, $2, $3, $4, $5, $6::jsonb, $7,
                    $8, $9, $10, $11, $12, $13,
                    $14, $15, $16, $17, $18
                )
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    plan = EXCLUDED.plan,
                    sections = EXCLUDED.sections,
                    synthesis = EXCLUDED.synthesis,
                    confidence = EXCLUDED.confidence,
                    open_questions = EXCLUDED.open_questions,
                    suggested_next = EXCLUDED.suggested_next,
                    source_wiki_slugs = EXCLUDED.source_wiki_slugs,
                    source_dream_ids = EXCLUDED.source_dream_ids,
                    model_used = EXCLUDED.model_used,
                    token_count = EXCLUDED.token_count,
                    tags = EXCLUDED.tags,
                    completed_at = EXCLUDED.completed_at,
                    embedding = EXCLUDED.embedding
                """,
                artifact.id,
                artifact.topic,
                artifact.angle,
                artifact.status.value,
                artifact.plan,
                sections_json,
                artifact.synthesis,
                artifact.confidence,
                artifact.open_questions,
                artifact.suggested_next,
                artifact.source_wiki_slugs,
                [str(d) for d in artifact.source_dream_ids],
                artifact.model_used,
                artifact.token_count,
                artifact.tags,
                artifact.started_at,
                artifact.completed_at,
                embedding,
            )

        log.debug("research.saved", id=str(artifact.id), status=artifact.status.value)

    async def get(self, artifact_id: UUID) -> ResearchArtifact | None:
        """Fetch a single research artifact by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM research WHERE id = $1", artifact_id
            )
        if not row:
            return None
        return _row_to_artifact(row)

    # ── Recall / search ────────────────────────────────────

    async def recall(
        self,
        query: str,
        *,
        limit: int = 3,
        status: ResearchStatus | None = ResearchStatus.COMPLETED,
    ) -> list[ResearchResult]:
        """Semantic search across research syntheses."""
        query_embedding = await self._embedder.embed(query)

        conditions = ["embedding IS NOT NULL"]
        params: list = [query_embedding, limit]
        param_idx = 3

        if status is not None:
            conditions.append(f"status = ${param_idx}")
            params.append(status.value)
            param_idx += 1

        where = " AND ".join(conditions)

        sql = f"""
            SELECT *, 1 - (embedding <=> $1) AS similarity
            FROM research
            WHERE {where}
            ORDER BY embedding <=> $1
            LIMIT $2
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        results = [
            ResearchResult(artifact=_row_to_artifact(row), similarity=row["similarity"])
            for row in rows
        ]
        log.debug("research.recall", query=query[:50], results=len(results))
        return results

    async def recent(
        self,
        limit: int = 10,
        status: ResearchStatus | None = None,
    ) -> list[ResearchArtifact]:
        """Fetch recent research artifacts, newest first."""
        conditions = []
        params: list = [limit]
        param_idx = 2

        if status is not None:
            conditions.append(f"status = ${param_idx}")
            params.append(status.value)
            param_idx += 1

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        sql = f"""
            SELECT * FROM research
            {where}
            ORDER BY started_at DESC
            LIMIT $1
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [_row_to_artifact(row) for row in rows]

    # ── Queue management ───────────────────────────────────

    async def enqueue(self, topic: str, requested_by: str = "") -> UUID:
        """Add a topic to the research queue. Returns queue item ID."""
        item_id = uuid4()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO research_queue (id, topic, requested_by)
                VALUES ($1, $2, $3)
                """,
                item_id,
                topic,
                requested_by,
            )
        log.info("research.queued", topic=topic, id=str(item_id))
        return item_id

    async def next_queued(self) -> tuple[UUID, str] | None:
        """Pop the next unconsumed queue item. Returns (id, topic) or None."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, topic FROM research_queue
                WHERE consumed = FALSE
                ORDER BY requested_at ASC
                LIMIT 1
                """
            )
        if not row:
            return None
        return row["id"], row["topic"]

    async def consume_queue_item(self, queue_id: UUID) -> None:
        """Mark a queue item as consumed."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE research_queue
                SET consumed = TRUE, consumed_at = now()
                WHERE id = $1
                """,
                queue_id,
            )

    async def queue_length(self) -> int:
        """Count pending queue items."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) AS cnt FROM research_queue WHERE consumed = FALSE"
            )
        return row["cnt"]

    # ── Stats ──────────────────────────────────────────────

    async def count_by_status(self) -> dict[str, int]:
        """Count research artifacts grouped by status."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT status, COUNT(*) AS cnt FROM research GROUP BY status"
            )
        return {row["status"]: row["cnt"] for row in rows}

    async def total_tokens_today(self) -> int:
        """Sum of tokens used by research runs started today."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COALESCE(SUM(token_count), 0) AS total
                FROM research
                WHERE started_at >= CURRENT_DATE
                """
            )
        return row["total"]

    async def count_sessions_today(self) -> int:
        """Count research sessions started today (any status)."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COUNT(*) AS cnt
                FROM research
                WHERE started_at >= CURRENT_DATE
                """
            )
        return row["cnt"]


def _row_to_artifact(row) -> ResearchArtifact:
    """Convert a database row to a ResearchArtifact model."""
    sections_data = row["sections"] if row["sections"] else []
    if isinstance(sections_data, str):
        sections_data = json.loads(sections_data)

    sections = [ResearchSection(**s) for s in sections_data]

    return ResearchArtifact(
        id=row["id"],
        topic=row["topic"],
        angle=row["angle"],
        status=ResearchStatus(row["status"]),
        plan=row["plan"],
        sections=sections,
        synthesis=row["synthesis"],
        confidence=row["confidence"],
        open_questions=list(row["open_questions"] or []),
        suggested_next=list(row["suggested_next"] or []),
        source_wiki_slugs=list(row["source_wiki_slugs"] or []),
        source_dream_ids=[UUID(d) for d in (row["source_dream_ids"] or [])],
        model_used=row["model_used"],
        token_count=row["token_count"],
        tags=list(row["tags"] or []),
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )
