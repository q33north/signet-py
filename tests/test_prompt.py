"""Tests for prompt assembly with dream context."""
from __future__ import annotations

from signet.character.prompt import PromptAssembler
from signet.interfaces.discord import _format_dream_context
from signet.models.character import Character, StyleDirectives
from signet.models.dreams import Dream, DreamResult, DreamType


def _minimal_character() -> Character:
    """Create a minimal character for testing prompt assembly."""
    return Character(
        name="TestBot",
        system="You are a test bot.",
        bio=["Test bio trait"],
        adjectives=["calm"],
        topics=["testing"],
        style=StyleDirectives(all=["be concise"], chat=[], post=[]),
        message_examples=[],
    )


class TestPromptAssemblerDreamContext:
    def test_dream_context_included_when_provided(self):
        char = _minimal_character()
        assembler = PromptAssembler(char)

        prompt = assembler.build_system_prompt(
            dream_context="[dream] Pete likes raw data",
        )

        assert "[dream] Pete likes raw data" in prompt

    def test_dream_context_absent_when_empty(self):
        char = _minimal_character()
        assembler = PromptAssembler(char)

        prompt = assembler.build_system_prompt(dream_context="")

        assert "internalized" not in prompt

    def test_context_ordering_memory_then_dream_then_wiki(self):
        char = _minimal_character()
        assembler = PromptAssembler(char)

        prompt = assembler.build_system_prompt(
            memory_context="MEMORY_MARKER",
            dream_context="DREAM_MARKER",
            wiki_context="WIKI_MARKER",
        )

        mem_pos = prompt.index("MEMORY_MARKER")
        dream_pos = prompt.index("DREAM_MARKER")
        wiki_pos = prompt.index("WIKI_MARKER")
        assert mem_pos < dream_pos < wiki_pos


class TestFormatDreamContext:
    def test_entity_fact_prefix(self):
        results = [
            DreamResult(
                dream=Dream(
                    dream_type=DreamType.ENTITY_FACT,
                    content="Pete: prefers scatterplots",
                    entity_name="Pete",
                ),
                similarity=0.85,
            )
        ]

        text = _format_dream_context(results)

        assert "[You know]" in text
        assert "Pete: prefers scatterplots" in text

    def test_reflection_prefix(self):
        results = [
            DreamResult(
                dream=Dream(
                    dream_type=DreamType.REFLECTION,
                    content="KRAS keeps coming up",
                ),
                similarity=0.7,
            )
        ]

        text = _format_dream_context(results)

        assert "[You've noticed]" in text

    def test_digest_prefix(self):
        results = [
            DreamResult(
                dream=Dream(
                    dream_type=DreamType.DIGEST,
                    content="Discussed resistance mechanisms",
                ),
                similarity=0.75,
            )
        ]

        text = _format_dream_context(results)

        assert "[Past conversation]" in text

    def test_header_present(self):
        results = [
            DreamResult(
                dream=Dream(dream_type=DreamType.DIGEST, content="test"),
                similarity=0.5,
            )
        ]

        text = _format_dream_context(results)

        assert "internalized from past experience" in text

    def test_content_truncated_at_300(self):
        long_content = "x" * 500
        results = [
            DreamResult(
                dream=Dream(dream_type=DreamType.DIGEST, content=long_content),
                similarity=0.5,
            )
        ]

        text = _format_dream_context(results)

        # The content portion should be at most 300 chars
        for line in text.split("\n"):
            if "[Past conversation]" in line:
                # prefix + 300 chars max
                assert len(long_content) > 300
                assert "x" * 301 not in line
