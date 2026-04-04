"""Tests for DreamStore and MemoryStore dream support.

These test the row-to-model mapping and query construction by mocking asyncpg.
The actual SQL execution is an integration concern tested against a real DB.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from signet.models.dreams import Dream, DreamType
from signet.nightshift.store import DreamStore


class _FakeAcquire:
    """Mock for asyncpg pool.acquire() async context manager."""

    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        pass


def _fake_pool():
    """Create a mock asyncpg pool with context manager support."""
    conn = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value = _FakeAcquire(conn)
    return pool, conn


class TestDreamStoreRecall:
    """Test that recall() builds correct queries and maps rows to models."""

    @pytest.mark.asyncio
    async def test_recall_maps_rows_to_dream_results(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()
        embedder.embed.return_value = [0.1] * 384

        dream_id = uuid4()
        source_id = uuid4()
        now = datetime.now(timezone.utc)

        conn.fetch.return_value = [
            {
                "id": dream_id,
                "dream_type": "entity_fact",
                "content": "Pete: likes scatterplots",
                "source_message_ids": [source_id],
                "entity_name": "Pete",
                "tags": ["preference"],
                "created_at": now,
                "similarity": 0.85,
            }
        ]

        store = DreamStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        results = await store.recall("scatterplots", limit=3)

        assert len(results) == 1
        assert results[0].dream.dream_type == DreamType.ENTITY_FACT
        assert results[0].dream.entity_name == "Pete"
        assert results[0].similarity == 0.85
        assert results[0].dream.source_message_ids == [source_id]

    @pytest.mark.asyncio
    async def test_recall_with_type_filter(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()
        embedder.embed.return_value = [0.1] * 384
        conn.fetch.return_value = []

        store = DreamStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        await store.recall("test", dream_type=DreamType.REFLECTION)

        # Verify the SQL includes the type filter
        call_args = conn.fetch.call_args
        sql = call_args[0][0]
        assert "dream_type = $3" in sql

    @pytest.mark.asyncio
    async def test_recall_without_type_filter(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()
        embedder.embed.return_value = [0.1] * 384
        conn.fetch.return_value = []

        store = DreamStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        await store.recall("test")

        call_args = conn.fetch.call_args
        sql = call_args[0][0]
        # dream_type appears in SELECT but should NOT appear in WHERE
        assert "dream_type =" not in sql


class TestDreamStoreEntityFacts:
    @pytest.mark.asyncio
    async def test_get_entity_facts_returns_dreams(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()

        conn.fetch.return_value = [
            {
                "id": uuid4(),
                "dream_type": "entity_fact",
                "content": "Pete: prefers raw data",
                "source_message_ids": [],
                "entity_name": "Pete",
                "tags": [],
                "created_at": datetime.now(timezone.utc),
            }
        ]

        store = DreamStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        facts = await store.get_entity_facts("Pete")

        assert len(facts) == 1
        assert facts[0].entity_name == "Pete"
        assert facts[0].dream_type == DreamType.ENTITY_FACT


class TestDreamStoreStoreDream:
    @pytest.mark.asyncio
    async def test_store_dream_embeds_and_inserts(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()
        embedder.embed.return_value = [0.5] * 384

        store = DreamStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        dream = Dream(
            dream_type=DreamType.DIGEST,
            content="Discussed KRAS resistance",
            source_message_ids=[uuid4()],
        )

        await store.store_dream(dream)

        embedder.embed.assert_called_once_with("Discussed KRAS resistance")
        conn.execute.assert_called_once()
        # Verify the INSERT SQL
        sql = conn.execute.call_args[0][0]
        assert "INSERT INTO dreams" in sql


class TestDreamStoreCountAndRecent:
    @pytest.mark.asyncio
    async def test_count_by_type(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()

        conn.fetch.return_value = [
            {"dream_type": "digest", "cnt": 5},
            {"dream_type": "entity_fact", "cnt": 12},
            {"dream_type": "reflection", "cnt": 3},
        ]

        store = DreamStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        counts = await store.count_by_type()

        assert counts == {"digest": 5, "entity_fact": 12, "reflection": 3}

    @pytest.mark.asyncio
    async def test_last_dream_time_returns_none_when_empty(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()

        conn.fetchrow.return_value = {"last": None}

        store = DreamStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        result = await store.last_dream_time()

        assert result is None

    @pytest.mark.asyncio
    async def test_last_dream_time_returns_datetime(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()
        now = datetime.now(timezone.utc)

        conn.fetchrow.return_value = {"last": now}

        store = DreamStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        result = await store.last_dream_time()

        assert result == now

    @pytest.mark.asyncio
    async def test_recent_with_type_filter(self):
        pool, conn = _fake_pool()
        embedder = AsyncMock()
        conn.fetch.return_value = []

        store = DreamStore(database_url="fake://", embedder=embedder)
        store._pool = pool

        await store.recent(limit=10, dream_type=DreamType.DIGEST)

        sql = conn.fetch.call_args[0][0]
        assert "dream_type = $2" in sql
