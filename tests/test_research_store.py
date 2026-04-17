"""Tests for ResearchStore persistence layer.

Mocks asyncpg at the pool boundary, same pattern as test_stores.py.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from signet.models.research import (
    ResearchArtifact,
    ResearchResult,
    ResearchSection,
    ResearchStatus,
)
from signet.nightshift.research_store import ResearchStore


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        pass


def _fake_pool():
    conn = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value = _FakeAcquire(conn)
    return pool, conn


def _make_row(**overrides):
    """Build a fake DB row dict for a research artifact."""
    defaults = {
        "id": uuid4(),
        "topic": "KRAS G12C resistance",
        "angle": "patient-derived models vs cell lines",
        "status": "completed",
        "plan": "sub-questions here",
        "sections": json.dumps([
            {"question": "q1", "findings": "f1", "confidence": "high", "sources": []}
        ]),
        "synthesis": "KRAS G12C shows differential resistance...",
        "confidence": "medium",
        "open_questions": ["does this apply to NSCLC?"],
        "suggested_next": ["check COSMIC for KRAS variants"],
        "source_wiki_slugs": ["kras-overview"],
        "source_dream_ids": [str(uuid4())],
        "model_used": "claude-sonnet-4-6",
        "token_count": 5000,
        "tags": ["genomics", "resistance"],
        "started_at": datetime.now(timezone.utc),
        "completed_at": datetime.now(timezone.utc),
        "embedding": [0.1] * 384,
    }
    defaults.update(overrides)
    return defaults


class TestResearchStoreSave:
    @pytest.mark.asyncio
    async def test_save_embeds_synthesis_and_inserts(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()
        embedder.embed.return_value = [0.5] * 384

        store = ResearchStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        artifact = ResearchArtifact(
            topic="KRAS resistance",
            angle="PDX vs cell line",
            status=ResearchStatus.COMPLETED,
            synthesis="Key findings about KRAS...",
            sections=[ResearchSection(question="q1", findings="f1")],
        )

        await store.save(artifact)

        embedder.embed.assert_called_once_with("Key findings about KRAS...")
        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "INSERT INTO research" in sql
        assert "ON CONFLICT (id) DO UPDATE" in sql

    @pytest.mark.asyncio
    async def test_save_without_synthesis_skips_embedding(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()

        store = ResearchStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        artifact = ResearchArtifact(
            topic="KRAS resistance",
            status=ResearchStatus.IN_PROGRESS,
        )

        await store.save(artifact)

        embedder.embed.assert_not_called()
        conn.execute.assert_called_once()


class TestResearchStoreRecall:
    @pytest.mark.asyncio
    async def test_recall_maps_rows_to_results(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()
        embedder.embed.return_value = [0.1] * 384

        row = _make_row(similarity=0.92)
        conn.fetch.return_value = [row]

        store = ResearchStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        results = await store.recall("KRAS resistance", limit=3)

        assert len(results) == 1
        assert results[0].similarity == 0.92
        assert results[0].artifact.topic == "KRAS G12C resistance"
        assert len(results[0].artifact.sections) == 1

    @pytest.mark.asyncio
    async def test_recall_with_status_filter(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()
        embedder.embed.return_value = [0.1] * 384
        conn.fetch.return_value = []

        store = ResearchStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        await store.recall("test", status=ResearchStatus.COMPLETED)

        sql = conn.fetch.call_args[0][0]
        assert "status = $3" in sql

    @pytest.mark.asyncio
    async def test_recall_without_status_filter(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()
        embedder.embed.return_value = [0.1] * 384
        conn.fetch.return_value = []

        store = ResearchStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        await store.recall("test", status=None)

        sql = conn.fetch.call_args[0][0]
        assert "status =" not in sql


class TestResearchStoreRecent:
    @pytest.mark.asyncio
    async def test_recent_returns_artifacts(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()

        conn.fetch.return_value = [_make_row(), _make_row()]

        store = ResearchStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        results = await store.recent(limit=5)

        assert len(results) == 2
        assert all(isinstance(r, ResearchArtifact) for r in results)

    @pytest.mark.asyncio
    async def test_recent_with_status_filter(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()
        conn.fetch.return_value = []

        store = ResearchStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        await store.recent(status=ResearchStatus.COMPLETED)

        sql = conn.fetch.call_args[0][0]
        assert "status = $2" in sql


class TestResearchStoreQueue:
    @pytest.mark.asyncio
    async def test_enqueue_inserts_topic(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()

        store = ResearchStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        item_id = await store.enqueue("ARID1A synthetic lethality", requested_by="Pete")

        assert item_id is not None
        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "INSERT INTO research_queue" in sql

    @pytest.mark.asyncio
    async def test_next_queued_returns_oldest(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()

        qid = uuid4()
        conn.fetchrow.return_value = {"id": qid, "topic": "BRCA2 variants", "wiki_folder": "cancer_genomics"}

        store = ResearchStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        result = await store.next_queued()

        assert result == (qid, "BRCA2 variants", "cancer_genomics")

    @pytest.mark.asyncio
    async def test_next_queued_returns_none_when_empty(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()
        conn.fetchrow.return_value = None

        store = ResearchStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        result = await store.next_queued()

        assert result is None

    @pytest.mark.asyncio
    async def test_queue_length(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()
        conn.fetchrow.return_value = {"cnt": 3}

        store = ResearchStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        count = await store.queue_length()

        assert count == 3


class TestResearchStoreStats:
    @pytest.mark.asyncio
    async def test_count_by_status(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()
        conn.fetch.return_value = [
            {"status": "completed", "cnt": 5},
            {"status": "failed", "cnt": 1},
        ]

        store = ResearchStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        counts = await store.count_by_status()

        assert counts == {"completed": 5, "failed": 1}

    @pytest.mark.asyncio
    async def test_total_tokens_today(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()
        conn.fetchrow.return_value = {"total": 42000}

        store = ResearchStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        total = await store.total_tokens_today()

        assert total == 42000
