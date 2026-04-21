"""Tests for the Discord dream scheduler loop.

Exercises the _dream_loop decision logic (skip below threshold, run above)
without spinning up an actual discord.Client connection. Uses __new__ to
build a bare SignetBot and injects mocked dependencies.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from signet.interfaces.discord import SignetBot, _format_dream_receipt
from signet.models.dreams import DreamReport


def _bare_bot() -> SignetBot:
    """Construct a SignetBot without running discord.Client.__init__."""
    bot = SignetBot.__new__(SignetBot)
    bot._memory = AsyncMock()
    bot._dreamer = AsyncMock()
    bot._dream_task = None
    return bot


class TestFormatDreamReceipt:
    def test_receipt_includes_counts(self):
        report = DreamReport(
            messages_processed=42,
            sessions_processed=3,
            digests=2,
            entity_facts=5,
            reflections=1,
        )
        text = _format_dream_receipt(report)
        assert "💭" in text
        assert "42" in text
        assert "3 conversation" in text
        assert "2 digests" in text
        assert "5 entity facts" in text
        assert "1 reflections" in text


class TestDreamLoop:
    @pytest.mark.asyncio
    async def test_skips_when_below_threshold(self):
        bot = _bare_bot()
        bot._memory.unconsolidated_count.return_value = 5  # below default 20

        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)
            if len(sleep_calls) >= 2:
                raise asyncio.CancelledError

        with patch("signet.interfaces.discord.settings") as mock_settings:
            mock_settings.dream_interval_minutes = 1
            mock_settings.dream_min_messages = 20
            mock_settings.dream_max_messages_per_run = 500
            mock_settings.dream_channel_id = ""
            with patch("asyncio.sleep", fake_sleep):
                await bot._dream_loop()

        bot._dreamer.dream.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_when_above_threshold(self):
        bot = _bare_bot()
        bot._memory.unconsolidated_count.return_value = 50
        bot._dreamer.dream.return_value = DreamReport(
            messages_processed=50, sessions_processed=2, digests=1
        )

        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)
            if len(sleep_calls) >= 2:
                raise asyncio.CancelledError

        with patch("signet.interfaces.discord.settings") as mock_settings:
            mock_settings.dream_interval_minutes = 1
            mock_settings.dream_min_messages = 20
            mock_settings.dream_max_messages_per_run = 500
            mock_settings.dream_channel_id = ""
            with patch("asyncio.sleep", fake_sleep):
                await bot._dream_loop()

        bot._dreamer.dream.assert_called_once_with(max_messages=500)

    @pytest.mark.asyncio
    async def test_no_dreamer_skips_run(self):
        """If dreamer is None (scheduler not initialized), loop stays alive but no-ops."""
        bot = _bare_bot()
        bot._dreamer = None
        bot._memory.unconsolidated_count = AsyncMock(return_value=100)

        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)
            if len(sleep_calls) >= 2:
                raise asyncio.CancelledError

        with patch("signet.interfaces.discord.settings") as mock_settings:
            mock_settings.dream_interval_minutes = 1
            mock_settings.dream_min_messages = 20
            mock_settings.dream_max_messages_per_run = 500
            mock_settings.dream_channel_id = ""
            with patch("asyncio.sleep", fake_sleep):
                await bot._dream_loop()

        bot._memory.unconsolidated_count.assert_not_called()

    @pytest.mark.asyncio
    async def test_receipt_posted_when_channel_configured(self):
        bot = _bare_bot()
        bot._memory.unconsolidated_count.return_value = 50
        bot._dreamer.dream.return_value = DreamReport(
            messages_processed=50, sessions_processed=2, digests=1
        )

        channel = AsyncMock()
        bot.get_channel = MagicMock(return_value=channel)

        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)
            if len(sleep_calls) >= 2:
                raise asyncio.CancelledError

        with patch("signet.interfaces.discord.settings") as mock_settings:
            mock_settings.dream_interval_minutes = 1
            mock_settings.dream_min_messages = 20
            mock_settings.dream_max_messages_per_run = 500
            mock_settings.dream_channel_id = "1234567890"
            with patch("asyncio.sleep", fake_sleep):
                await bot._dream_loop()

        channel.send.assert_called_once()
        posted_text = channel.send.call_args[0][0]
        assert "💭" in posted_text
        assert "50" in posted_text

    @pytest.mark.asyncio
    async def test_no_receipt_when_nothing_processed(self):
        """Empty dream reports don't spam the admin channel."""
        bot = _bare_bot()
        bot._memory.unconsolidated_count.return_value = 50
        bot._dreamer.dream.return_value = DreamReport(messages_processed=0)

        channel = AsyncMock()
        bot.get_channel = MagicMock(return_value=channel)

        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)
            if len(sleep_calls) >= 2:
                raise asyncio.CancelledError

        with patch("signet.interfaces.discord.settings") as mock_settings:
            mock_settings.dream_interval_minutes = 1
            mock_settings.dream_min_messages = 20
            mock_settings.dream_max_messages_per_run = 500
            mock_settings.dream_channel_id = "1234567890"
            with patch("asyncio.sleep", fake_sleep):
                await bot._dream_loop()

        channel.send.assert_not_called()
