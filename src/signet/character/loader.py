"""Load and validate character YAML."""
from __future__ import annotations

from pathlib import Path

import yaml

from signet.models.character import Character, ConversationExample, MessageTurn


def load_character(path: Path) -> Character:
    """Load a character definition from YAML."""
    raw = yaml.safe_load(path.read_text())

    # Convert message_examples from YAML-friendly format to pydantic models
    examples = []
    for ex in raw.get("message_examples", []):
        turns = [MessageTurn(name=t["name"], text=t["text"]) for t in ex]
        examples.append(ConversationExample(messages=turns))

    return Character(
        name=raw["name"],
        system=raw["system"],
        bio=raw["bio"],
        adjectives=raw.get("adjectives", []),
        topics=raw.get("topics", []),
        message_examples=examples,
        style=raw.get("style", {}),
        bio_sample_size=raw.get("bio_sample_size", 10),
        example_sample_size=raw.get("example_sample_size", 5),
    )
