"""Prompt assembly pipeline. Composes system prompt from character + providers."""
from __future__ import annotations

from signet.character.sampler import CharacterSampler
from signet.models.character import Character, ConversationExample


class PromptAssembler:
    """Composes the final system prompt from character sampling."""

    def __init__(self, character: Character) -> None:
        self._char = character
        self._sampler = CharacterSampler(character)

    def build_system_prompt(
        self,
        platform: str = "discord",
        memory_context: str = "",
        dream_context: str = "",
        research_context: str = "",
        wiki_context: str = "",
    ) -> str:
        """Build a system prompt with randomized personality slices.

        Each call produces a slightly different prompt due to bio/example sampling.
        """
        parts: list[str] = []

        # 1. Core identity (never randomized)
        parts.append(self._char.system)

        # 2. Bio (randomized sample)
        bio_items = self._sampler.sample_bio()
        parts.append("About you:\n" + " ".join(bio_items))

        # 3. Adjective + Topic (variety injection)
        adj = self._sampler.sample_adjective()
        topic = self._sampler.sample_topic()
        if adj:
            parts.append(f"Your current vibe: {adj}")
        if topic:
            parts.append(f"Something on your mind lately: {topic}")

        # 4. Memory context (relevant past conversations)
        if memory_context:
            parts.append(memory_context)

        # 4b. Dream context (consolidated insights from past experience)
        if dream_context:
            parts.append(dream_context)

        # 4c. Research context (nightshift research findings)
        if research_context:
            parts.append(research_context)

        # 4d. Wiki knowledge context (relevant domain knowledge)
        if wiki_context:
            parts.append(wiki_context)

        # 5. Style directives (deterministic per platform)
        style_lines = list(self._char.style.all)
        if platform in ("discord", "cli"):
            style_lines.extend(self._char.style.chat)
        elif platform == "twitter":
            style_lines.extend(self._char.style.post)
        if style_lines:
            parts.append("Style:\n" + "\n".join(f"- {s}" for s in style_lines))

        # 6. Message examples (randomized sample)
        examples = self._sampler.sample_examples()
        if examples:
            parts.append("Example conversations:\n" + _format_examples(examples))

        return "\n\n".join(parts)


def _format_examples(examples: list[ConversationExample]) -> str:
    """Format conversation examples as readable text."""
    blocks = []
    for ex in examples:
        lines = [f"{t.name}: {t.text}" for t in ex.messages]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
