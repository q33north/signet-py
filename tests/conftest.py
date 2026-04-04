"""Shared test fixtures."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.helpers import VALID_LLM_RESPONSE


@pytest.fixture
def mock_brain() -> MagicMock:
    """Brain mock that returns valid consolidation JSON."""
    brain = MagicMock()
    brain.quick.return_value = VALID_LLM_RESPONSE
    return brain


@pytest.fixture
def mock_memory() -> AsyncMock:
    """MemoryStore mock."""
    memory = AsyncMock()
    memory.get_unconsolidated_messages.return_value = []
    memory.mark_messages_consolidated.return_value = 0
    return memory


@pytest.fixture
def mock_dream_store() -> AsyncMock:
    """DreamStore mock."""
    store = AsyncMock()
    store.store_dream.return_value = None
    return store
