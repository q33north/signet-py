"""Tests for the Researcher orchestrator.

Mocks Brain, stores, and providers at their boundaries.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from signet.models.dreams import Dream, DreamResult, DreamType
from signet.models.knowledge import WikiArticle, WikiFrontmatter, WikiSearchResult
from signet.models.research import (
    ResearchArtifact,
    ResearchResult,
    ResearchSection,
    ResearchStatus,
)
from signet.nightshift.researcher import Researcher, format_research_for_discord, _parse_json


def _mock_brain(quick_response: str = "", chat_response: str = ""):
    """Create a mock Brain with configurable responses."""
    brain = MagicMock()
    brain.quick.return_value = quick_response
    brain.chat.return_value = chat_response
    return brain


def _mock_stores():
    """Create mock store instances."""
    memory = AsyncMock()
    wiki = AsyncMock()
    dreams = AsyncMock()
    research = AsyncMock()

    # Defaults
    wiki.search.return_value = []
    dreams.recall.return_value = []
    research.recall.return_value = []
    research.total_tokens_today.return_value = 0
    research.count_sessions_today.return_value = 0
    research.next_queued.return_value = None

    return memory, wiki, dreams, research


class TestPickTopic:
    @pytest.mark.asyncio
    async def test_queued_topic_takes_priority(self):
        brain = _mock_brain(
            quick_response=json.dumps({"sub_questions": ["what about it?"]}),
        )
        memory, wiki, dreams, research = _mock_stores()
        queue_id = uuid4()
        research.next_queued.return_value = (queue_id, "BRCA2 variants")

        researcher = Researcher(brain, memory, wiki, dreams, research)
        topic, angle = await researcher._pick_topic()

        assert topic == "BRCA2 variants"
        research.consume_queue_item.assert_called_once_with(queue_id)

    @pytest.mark.asyncio
    async def test_llm_selection_from_candidates(self):
        brain = _mock_brain(
            quick_response=json.dumps({
                "topic": "KRAS G12C",
                "angle": "resistance mechanisms",
                "why": "active area",
            }),
        )
        memory, wiki, dreams, research = _mock_stores()

        dream = Dream(
            dream_type=DreamType.ENTITY_FACT,
            content="KRAS G12C resistance in NSCLC",
            entity_name="KRAS",
        )
        dreams.recall.return_value = [DreamResult(dream=dream, similarity=0.8)]

        researcher = Researcher(brain, memory, wiki, dreams, research)
        topic, angle = await researcher._pick_topic()

        assert topic == "KRAS G12C"
        assert angle == "resistance mechanisms"

    @pytest.mark.asyncio
    async def test_no_candidates_returns_empty(self):
        brain = _mock_brain()
        memory, wiki, dreams, research = _mock_stores()

        researcher = Researcher(brain, memory, wiki, dreams, research)
        topic, angle = await researcher._pick_topic()

        assert topic == ""
        assert angle == ""

    @pytest.mark.asyncio
    async def test_already_researched_topics_excluded_from_candidates(self):
        """Topics with completed research should not appear as candidates."""
        brain = _mock_brain(
            quick_response=json.dumps({
                "topic": "TP53",
                "angle": "gain of function",
                "why": "only remaining candidate",
            }),
        )
        memory, wiki, dreams, research = _mock_stores()

        # Two dream candidates: KRAS (already done) and TP53 (new)
        dreams.recall.return_value = [
            DreamResult(
                dream=Dream(
                    dream_type=DreamType.ENTITY_FACT,
                    content="KRAS G12C resistance",
                    entity_name="KRAS",
                ),
                similarity=0.9,
            ),
            DreamResult(
                dream=Dream(
                    dream_type=DreamType.ENTITY_FACT,
                    content="TP53 gain of function",
                    entity_name="TP53",
                ),
                similarity=0.8,
            ),
        ]

        # KRAS was already researched
        already_done = ResearchArtifact(
            topic="KRAS",
            status=ResearchStatus.COMPLETED,
        )
        research.recent.return_value = [already_done]

        researcher = Researcher(brain, memory, wiki, dreams, research)
        topic, angle = await researcher._pick_topic()

        # KRAS should have been filtered out, only TP53 remains
        assert topic == "TP53"
        # Verify the prompt sent to the LLM only contains TP53
        call_args = brain.quick.call_args
        prompt_text = call_args[0][0]
        assert "TP53" in prompt_text
        assert "KRAS" not in prompt_text.split("Already researched")[0]


class TestPlan:
    @pytest.mark.asyncio
    async def test_plan_creates_sections(self):
        brain = _mock_brain(
            quick_response=json.dumps({
                "sub_questions": ["q1?", "q2?", "q3?"]
            }),
        )
        memory, wiki, dreams, research = _mock_stores()

        researcher = Researcher(brain, memory, wiki, dreams, research)

        artifact = ResearchArtifact(topic="KRAS", angle="resistance")
        result = await researcher._plan(artifact)

        assert len(result.sections) == 3
        assert result.sections[0].question == "q1?"
        assert "q1?" in result.plan

    @pytest.mark.asyncio
    async def test_plan_caps_at_max_sections(self):
        brain = _mock_brain(
            quick_response=json.dumps({
                "sub_questions": [f"q{i}?" for i in range(20)]
            }),
        )
        memory, wiki, dreams, research = _mock_stores()

        researcher = Researcher(brain, memory, wiki, dreams, research)

        artifact = ResearchArtifact(topic="KRAS", angle="resistance")
        with patch("signet.nightshift.researcher.settings") as mock_settings:
            mock_settings.nightshift_max_sections = 5
            mock_settings.model_heavy = "claude-sonnet-4-6"
            result = await researcher._plan(artifact)

        assert len(result.sections) == 5

    @pytest.mark.asyncio
    async def test_plan_fallback_on_parse_error(self):
        brain = _mock_brain(quick_response="not json at all")
        memory, wiki, dreams, research = _mock_stores()

        researcher = Researcher(brain, memory, wiki, dreams, research)

        artifact = ResearchArtifact(topic="KRAS", angle="resistance")
        result = await researcher._plan(artifact)

        # Falls back to using the angle as the only question
        assert len(result.sections) == 1
        assert result.sections[0].question == "resistance"


class TestSynthesize:
    @pytest.mark.asyncio
    async def test_synthesis_parses_labeled_response(self):
        synthesis_response = (
            "===SYNTHESIS===\n"
            "KRAS G12C shows differential resistance patterns...\n\n"
            "===CONFIDENCE===\n"
            "high\n\n"
            "===OPEN_QUESTIONS===\n"
            "- does this extend to NSCLC?\n\n"
            "===NEXT_STEPS===\n"
            "- check COSMIC database\n"
        )
        brain = _mock_brain(chat_response=synthesis_response)
        memory, wiki, dreams, research = _mock_stores()

        researcher = Researcher(brain, memory, wiki, dreams, research)

        artifact = ResearchArtifact(
            topic="KRAS",
            angle="resistance",
            sections=[
                ResearchSection(question="q1", findings="finding 1"),
            ],
        )
        result = await researcher._synthesize(artifact)

        assert "differential resistance" in result.synthesis
        assert result.confidence == "high"
        assert len(result.open_questions) == 1
        assert len(result.suggested_next) == 1

    @pytest.mark.asyncio
    async def test_synthesis_fallback_on_parse_error(self):
        brain = _mock_brain(chat_response="raw synthesis text here")
        memory, wiki, dreams, research = _mock_stores()

        researcher = Researcher(brain, memory, wiki, dreams, research)

        artifact = ResearchArtifact(
            topic="KRAS",
            sections=[ResearchSection(question="q1", findings="f1")],
        )
        result = await researcher._synthesize(artifact)

        assert result.synthesis == "raw synthesis text here"
        assert result.confidence == "medium"

    @pytest.mark.asyncio
    async def test_synthesis_parses_json_when_model_ignores_markers(self):
        json_response = json.dumps({
            "synthesis": "## ALK+ NSCLC\n\nHost-response proteins dominate.",
            "confidence": "medium",
            "open_questions": ["tumor-specific signal?"],
            "suggested_next": ["expand reference cohort"],
        })
        brain = _mock_brain(chat_response=json_response)
        memory, wiki, dreams, research = _mock_stores()

        researcher = Researcher(brain, memory, wiki, dreams, research)

        artifact = ResearchArtifact(
            topic="ALK proteomics",
            sections=[ResearchSection(question="q1", findings="f1")],
        )
        result = await researcher._synthesize(artifact)

        assert result.synthesis.startswith("## ALK+ NSCLC")
        assert "{" not in result.synthesis
        assert result.confidence == "medium"
        assert result.open_questions == ["tumor-specific signal?"]
        assert result.suggested_next == ["expand reference cohort"]


class TestInterruption:
    @pytest.mark.asyncio
    async def test_interrupt_stops_after_current_step(self):
        brain = _mock_brain(
            quick_response=json.dumps({"sub_questions": ["q1?", "q2?"]}),
            chat_response="findings",
        )
        memory, wiki, dreams, research = _mock_stores()
        research.next_queued.return_value = (uuid4(), "test topic")

        researcher = Researcher(brain, memory, wiki, dreams, research)

        # Interrupt after planning
        original_plan = researcher._plan

        async def plan_then_interrupt(artifact):
            result = await original_plan(artifact)
            researcher.interrupt()
            return result

        researcher._plan = plan_then_interrupt

        report = await researcher.run()

        assert report.status == ResearchStatus.INTERRUPTED


class TestBudget:
    @pytest.mark.asyncio
    async def test_budget_exceeded_skips_research(self):
        brain = _mock_brain()
        memory, wiki, dreams, research = _mock_stores()
        research.total_tokens_today.return_value = 999_999
        research.count_sessions_today.return_value = 0

        researcher = Researcher(brain, memory, wiki, dreams, research)
        report = await researcher.run()

        assert report.status == ResearchStatus.FAILED
        research.save.assert_not_called()


class TestSessionLimit:
    @pytest.mark.asyncio
    async def test_session_limit_reached_skips_research(self):
        brain = _mock_brain()
        memory, wiki, dreams, research = _mock_stores()
        research.count_sessions_today.return_value = 3

        researcher = Researcher(brain, memory, wiki, dreams, research)
        with patch("signet.nightshift.researcher.settings") as mock_settings:
            mock_settings.nightshift_max_sessions = 3
            mock_settings.nightshift_daily_token_budget = 100_000
            report = await researcher.run()

        assert report.status == ResearchStatus.FAILED
        research.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_under_session_limit_allows_research(self):
        brain = _mock_brain(
            quick_response=json.dumps({
                "topic": "TP53",
                "angle": "gain of function",
                "why": "testing",
                "sub_questions": ["q1?"],
            }),
            chat_response=(
                "===SYNTHESIS===\nfindings here\n\n"
                "===CONFIDENCE===\nmedium\n\n"
                "===OPEN_QUESTIONS===\n\n"
                "===NEXT_STEPS===\n"
            ),
        )
        memory, wiki, dreams, research = _mock_stores()
        research.count_sessions_today.return_value = 1
        research.next_queued.return_value = (uuid4(), "TP53")

        researcher = Researcher(brain, memory, wiki, dreams, research)
        report = await researcher.run()

        assert report.status == ResearchStatus.COMPLETED


class TestFormatDiscord:
    def test_format_completed_research(self):
        artifact = ResearchArtifact(
            topic="KRAS G12C",
            synthesis="resistance is complex",
            confidence="medium",
            open_questions=["q1?"],
            suggested_next=["check COSMIC"],
            token_count=5000,
            completed_at=datetime.now(timezone.utc),
        )

        text = format_research_for_discord(artifact)

        assert "KRAS G12C" in text
        assert "resistance is complex" in text
        assert "medium" in text
        assert "q1?" in text
        assert "check COSMIC" in text
        assert "5,000" in text


class TestParseJson:
    def test_plain_json(self):
        result = _parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_fenced_json(self):
        result = _parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json("not json")
