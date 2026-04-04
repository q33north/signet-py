"""Tests for the autoDream consolidation engine."""
from __future__ import annotations

import json
from datetime import timedelta

import pytest

from signet.models.dreams import DreamType
from signet.nightshift.dreamer import Dreamer
from tests.helpers import (
    EMPTY_LLM_RESPONSE,
    GARBAGE_LLM_RESPONSE,
    VALID_LLM_RESPONSE,
    VALID_LLM_RESPONSE_FENCED,
    make_conversation,
    make_message,
)


# ── Bundle grouping ──────────────────────────────────────────


class TestGroupIntoBundles:
    """Tests for Dreamer._group_into_bundles."""

    def _dreamer(self, mock_memory, mock_dream_store, mock_brain):
        return Dreamer(memory=mock_memory, dreams=mock_dream_store, brain=mock_brain)

    def test_single_conversation_one_bundle(self, mock_memory, mock_dream_store, mock_brain):
        dreamer = self._dreamer(mock_memory, mock_dream_store, mock_brain)
        msgs = make_conversation(n=5, channel_id="chan-1", gap_minutes=2)

        bundles = dreamer._group_into_bundles(msgs)

        assert len(bundles) == 1
        assert len(bundles[0].messages) == 5
        assert bundles[0].channel_id == "chan-1"

    def test_gap_splits_into_two_bundles(self, mock_memory, mock_dream_store, mock_brain):
        dreamer = self._dreamer(mock_memory, mock_dream_store, mock_brain)
        conv1 = make_conversation(n=3, channel_id="chan-1", gap_minutes=2)
        # Second conversation starts 45 min after first ends (> 30 min gap)
        start2 = conv1[-1].timestamp + timedelta(minutes=45)
        conv2 = make_conversation(n=3, channel_id="chan-1", start=start2, gap_minutes=2)

        bundles = dreamer._group_into_bundles(conv1 + conv2)

        assert len(bundles) == 2
        assert len(bundles[0].messages) == 3
        assert len(bundles[1].messages) == 3

    def test_different_channels_separate_bundles(self, mock_memory, mock_dream_store, mock_brain):
        dreamer = self._dreamer(mock_memory, mock_dream_store, mock_brain)
        msgs_a = make_conversation(n=3, channel_id="chan-a")
        msgs_b = make_conversation(n=4, channel_id="chan-b")

        bundles = dreamer._group_into_bundles(msgs_a + msgs_b)

        assert len(bundles) == 2
        channels = {b.channel_id for b in bundles}
        assert channels == {"chan-a", "chan-b"}

    def test_single_message_dropped(self, mock_memory, mock_dream_store, mock_brain):
        """A 'conversation' with only 1 message isn't worth consolidating."""
        dreamer = self._dreamer(mock_memory, mock_dream_store, mock_brain)
        msgs = [make_message(content="lone wolf", channel_id="chan-1")]

        bundles = dreamer._group_into_bundles(msgs)

        assert len(bundles) == 0

    def test_bundles_sorted_chronologically(self, mock_memory, mock_dream_store, mock_brain):
        dreamer = self._dreamer(mock_memory, mock_dream_store, mock_brain)
        # Channel B conversation happens before channel A
        msgs_b = make_conversation(n=3, channel_id="chan-b")
        start_a = msgs_b[-1].timestamp + timedelta(hours=1)
        msgs_a = make_conversation(n=3, channel_id="chan-a", start=start_a)

        bundles = dreamer._group_into_bundles(msgs_b + msgs_a)

        assert bundles[0].channel_id == "chan-b"
        assert bundles[1].channel_id == "chan-a"

    def test_empty_input_returns_empty(self, mock_memory, mock_dream_store, mock_brain):
        dreamer = self._dreamer(mock_memory, mock_dream_store, mock_brain)

        bundles = dreamer._group_into_bundles([])

        assert bundles == []


# ── JSON parsing ─────────────────────────────────────────────


class TestParseDreams:
    """Tests for Dreamer._parse_dreams."""

    def _dreamer(self, mock_memory, mock_dream_store, mock_brain):
        return Dreamer(memory=mock_memory, dreams=mock_dream_store, brain=mock_brain)

    def test_valid_json_produces_all_types(self, mock_memory, mock_dream_store, mock_brain):
        dreamer = self._dreamer(mock_memory, mock_dream_store, mock_brain)
        source_ids = [make_message().id, make_message().id]

        dreams = dreamer._parse_dreams(VALID_LLM_RESPONSE, source_ids)

        types = [d.dream_type for d in dreams]
        assert DreamType.DIGEST in types
        assert DreamType.ENTITY_FACT in types
        assert DreamType.REFLECTION in types
        assert len(dreams) == 4  # 1 digest + 2 entity facts + 1 reflection

    def test_fenced_json_stripped(self, mock_memory, mock_dream_store, mock_brain):
        dreamer = self._dreamer(mock_memory, mock_dream_store, mock_brain)
        source_ids = [make_message().id]

        dreams = dreamer._parse_dreams(VALID_LLM_RESPONSE_FENCED, source_ids)

        assert len(dreams) == 4
        assert dreams[0].dream_type == DreamType.DIGEST

    def test_empty_response_produces_nothing(self, mock_memory, mock_dream_store, mock_brain):
        dreamer = self._dreamer(mock_memory, mock_dream_store, mock_brain)
        source_ids = [make_message().id]

        dreams = dreamer._parse_dreams(EMPTY_LLM_RESPONSE, source_ids)

        assert len(dreams) == 0

    def test_garbage_response_fallback_digest(self, mock_memory, mock_dream_store, mock_brain):
        dreamer = self._dreamer(mock_memory, mock_dream_store, mock_brain)
        source_ids = [make_message().id]

        dreams = dreamer._parse_dreams(GARBAGE_LLM_RESPONSE, source_ids)

        assert len(dreams) == 1
        assert dreams[0].dream_type == DreamType.DIGEST
        assert dreams[0].content == GARBAGE_LLM_RESPONSE

    def test_entity_facts_include_entity_name(self, mock_memory, mock_dream_store, mock_brain):
        dreamer = self._dreamer(mock_memory, mock_dream_store, mock_brain)
        source_ids = [make_message().id]

        dreams = dreamer._parse_dreams(VALID_LLM_RESPONSE, source_ids)
        entity_facts = [d for d in dreams if d.dream_type == DreamType.ENTITY_FACT]

        assert all(d.entity_name == "Pete" for d in entity_facts)
        assert all("Pete:" in d.content for d in entity_facts)

    def test_source_ids_propagated(self, mock_memory, mock_dream_store, mock_brain):
        dreamer = self._dreamer(mock_memory, mock_dream_store, mock_brain)
        source_ids = [make_message().id, make_message().id]

        dreams = dreamer._parse_dreams(VALID_LLM_RESPONSE, source_ids)

        for d in dreams:
            assert d.source_message_ids == source_ids

    def test_empty_entity_or_fact_skipped(self, mock_memory, mock_dream_store, mock_brain):
        """Entity facts with missing entity or fact text should be dropped."""
        import json

        dreamer = self._dreamer(mock_memory, mock_dream_store, mock_brain)
        response = json.dumps(
            {
                "digest": "test",
                "entity_facts": [
                    {"entity": "", "fact": "some fact"},
                    {"entity": "Pete", "fact": ""},
                    {"entity": "Pete", "fact": "valid fact"},
                ],
                "reflections": [],
            }
        )

        dreams = dreamer._parse_dreams(response, [make_message().id])
        entity_facts = [d for d in dreams if d.dream_type == DreamType.ENTITY_FACT]

        assert len(entity_facts) == 1
        assert "valid fact" in entity_facts[0].content


# ── Full pipeline ────────────────────────────────────────────


class TestDreamPipeline:
    """Integration tests for the full dream() method with mocked deps."""

    @pytest.mark.asyncio
    async def test_no_messages_returns_empty_report(self, mock_memory, mock_dream_store, mock_brain):
        mock_memory.get_unconsolidated_messages.return_value = []
        dreamer = Dreamer(memory=mock_memory, dreams=mock_dream_store, brain=mock_brain)

        report = await dreamer.dream()

        assert report.total_dreams == 0
        assert report.messages_processed == 0
        mock_dream_store.store_dream.assert_not_called()
        mock_memory.mark_messages_consolidated.assert_not_called()

    @pytest.mark.asyncio
    async def test_full_cycle_stores_dreams_and_marks_consolidated(
        self, mock_memory, mock_dream_store, mock_brain
    ):
        msgs = make_conversation(n=6, channel_id="chan-1")
        mock_memory.get_unconsolidated_messages.return_value = msgs
        mock_brain.quick.return_value = VALID_LLM_RESPONSE
        dreamer = Dreamer(memory=mock_memory, dreams=mock_dream_store, brain=mock_brain)

        report = await dreamer.dream()

        assert report.messages_processed == 6
        assert report.digests == 1
        assert report.entity_facts == 2
        assert report.reflections == 1
        assert report.total_dreams == 4
        assert mock_dream_store.store_dream.call_count == 4
        mock_memory.mark_messages_consolidated.assert_called_once()
        marked_ids = mock_memory.mark_messages_consolidated.call_args[0][0]
        assert len(marked_ids) == 6

    @pytest.mark.asyncio
    async def test_bundle_error_doesnt_crash_pipeline(
        self, mock_memory, mock_dream_store, mock_brain
    ):
        msgs = make_conversation(n=4, channel_id="chan-1")
        mock_memory.get_unconsolidated_messages.return_value = msgs
        mock_brain.quick.side_effect = RuntimeError("API down")
        dreamer = Dreamer(memory=mock_memory, dreams=mock_dream_store, brain=mock_brain)

        report = await dreamer.dream()

        # Pipeline completes, marks messages consolidated, but produces no dreams
        assert report.total_dreams == 0
        assert report.messages_processed == 4
        mock_memory.mark_messages_consolidated.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_bundles_processed(self, mock_memory, mock_dream_store, mock_brain):
        conv1 = make_conversation(n=3, channel_id="chan-a")
        conv2 = make_conversation(n=3, channel_id="chan-b")
        mock_memory.get_unconsolidated_messages.return_value = conv1 + conv2
        mock_brain.quick.return_value = VALID_LLM_RESPONSE
        dreamer = Dreamer(memory=mock_memory, dreams=mock_dream_store, brain=mock_brain)

        report = await dreamer.dream()

        assert report.sessions_processed == 2
        assert mock_brain.quick.call_count == 2
        # 4 dreams per bundle * 2 bundles
        assert mock_dream_store.store_dream.call_count == 8


# ── Message formatting ───────────────────────────────────────


class TestFormatMessages:
    def test_format_includes_timestamp_and_author(self, mock_memory, mock_dream_store, mock_brain):
        dreamer = Dreamer(memory=mock_memory, dreams=mock_dream_store, brain=mock_brain)
        msgs = make_conversation(n=2, channel_id="chan-1")

        formatted = dreamer._format_messages(msgs)

        assert "TestUser" in formatted
        assert "Signet" in formatted
        assert "message 0" in formatted
        assert "message 1" in formatted
        # Timestamps present
        assert "]" in formatted
