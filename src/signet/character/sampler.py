"""Randomized personality sampling. Each prompt gets a different slice."""
from __future__ import annotations

import random
import string

from signet.models.character import Character, ConversationExample


class CharacterSampler:
    """Produces randomized personality slices per prompt."""

    def __init__(self, character: Character) -> None:
        self._char = character

    def sample_bio(self) -> list[str]:
        """Shuffle bio, take N items. Different every call."""
        pool = list(self._char.bio)
        random.shuffle(pool)
        return pool[: self._char.bio_sample_size]

    def sample_examples(self) -> list[ConversationExample]:
        """Shuffle examples, take N. Replace name placeholders."""
        pool = list(self._char.message_examples)
        random.shuffle(pool)
        selected = pool[: self._char.example_sample_size]

        result = []
        for ex in selected:
            name = _random_name()
            messages = []
            for turn in ex.messages:
                messages.append(turn.model_copy(
                    update={"name": turn.name.replace("{{name1}}", name)}
                ))
            result.append(ConversationExample(messages=messages))
        return result

    def sample_adjective(self) -> str:
        """Random single adjective for this prompt."""
        if not self._char.adjectives:
            return ""
        return random.choice(self._char.adjectives)

    def sample_topic(self) -> str:
        """Random single topic for this prompt."""
        if not self._char.topics:
            return ""
        return random.choice(self._char.topics)


def _random_name() -> str:
    """Generate a random short name for example placeholders."""
    names = [
        "Alex", "Jordan", "Sam", "Riley", "Morgan",
        "Casey", "Drew", "Quinn", "Kai", "Avery",
    ]
    return random.choice(names)
