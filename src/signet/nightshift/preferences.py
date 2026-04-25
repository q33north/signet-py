"""Topic preferences for nightshift research.

Loaded from the character YAML's `research_preferences` block:

    research_preferences:
      block:  ["single-cell foundation models"]   # hard reject (substring, case-insensitive)
      avoid:  ["benchmarking studies"]             # soft hint to the LLM
      prefer: ["foundation models", "variant calling"]  # soft hint to the LLM

`block` is enforced post-LLM. `avoid`/`prefer` are injected into the prompt only.
"""
from __future__ import annotations

from pathlib import Path

import structlog
import yaml
from pydantic import BaseModel, Field

log = structlog.get_logger()


class ResearchPreferences(BaseModel):
    block: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    prefer: list[str] = Field(default_factory=list)

    def is_blocked(self, text: str) -> str | None:
        """Return the matching block phrase if `text` hits the blocklist, else None."""
        if not text:
            return None
        haystack = text.lower()
        for phrase in self.block:
            if phrase and phrase.lower() in haystack:
                return phrase
        return None

    def prompt_section(self) -> str:
        parts: list[str] = []
        if self.block:
            parts.append(
                "\n\nBLOCKED topics (NEVER pick these or close variants):\n"
                + "\n".join(f"- {p}" for p in self.block)
            )
        if self.avoid:
            parts.append(
                "\n\nDe-prioritize (pick only if nothing better fits):\n"
                + "\n".join(f"- {p}" for p in self.avoid)
            )
        if self.prefer:
            parts.append(
                "\n\nPreferred areas (favor these when candidates compete):\n"
                + "\n".join(f"- {p}" for p in self.prefer)
            )
        return "".join(parts)


def load_preferences(character_path: Path) -> ResearchPreferences:
    """Load `research_preferences` from the character YAML. Empty on any failure."""
    try:
        raw = yaml.safe_load(character_path.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        log.warning("preferences.load_failed", error=str(e))
        return ResearchPreferences()
    data = raw.get("research_preferences") or {}
    try:
        return ResearchPreferences(**data)
    except Exception as e:
        log.warning("preferences.invalid", error=str(e))
        return ResearchPreferences()
