"""Tests for nightshift research preferences (block/avoid/prefer)."""
from __future__ import annotations

from pathlib import Path

from signet.nightshift.preferences import ResearchPreferences, load_preferences


class TestResearchPreferences:
    def test_is_blocked_substring_case_insensitive(self):
        prefs = ResearchPreferences(block=["single-cell foundation models"])
        assert prefs.is_blocked("Single-Cell Foundation Models in cancer") == "single-cell foundation models"
        assert prefs.is_blocked("scRNA-seq atlases") is None

    def test_is_blocked_empty_text_returns_none(self):
        prefs = ResearchPreferences(block=["x"])
        assert prefs.is_blocked("") is None

    def test_is_blocked_no_block_list(self):
        prefs = ResearchPreferences()
        assert prefs.is_blocked("anything") is None

    def test_prompt_section_renders_all_three(self):
        prefs = ResearchPreferences(
            block=["a"], avoid=["b"], prefer=["c"]
        )
        section = prefs.prompt_section()
        assert "BLOCKED" in section and "- a" in section
        assert "De-prioritize" in section and "- b" in section
        assert "Preferred" in section and "- c" in section

    def test_prompt_section_empty(self):
        assert ResearchPreferences().prompt_section() == ""


class TestLoadPreferences:
    def test_loads_from_yaml(self, tmp_path: Path):
        f = tmp_path / "char.yaml"
        f.write_text(
            "name: x\nsystem: x\nbio: []\n"
            "research_preferences:\n"
            "  block: [\"foo\"]\n"
            "  prefer: [\"bar\"]\n"
        )
        prefs = load_preferences(f)
        assert prefs.block == ["foo"]
        assert prefs.prefer == ["bar"]
        assert prefs.avoid == []

    def test_missing_section_returns_empty(self, tmp_path: Path):
        f = tmp_path / "char.yaml"
        f.write_text("name: x\nsystem: x\nbio: []\n")
        prefs = load_preferences(f)
        assert prefs.block == [] and prefs.avoid == [] and prefs.prefer == []

    def test_missing_file_returns_empty(self, tmp_path: Path):
        prefs = load_preferences(tmp_path / "nope.yaml")
        assert prefs.block == []

    def test_invalid_yaml_returns_empty(self, tmp_path: Path):
        f = tmp_path / "char.yaml"
        f.write_text("name: x\nresearch_preferences: [not, a, dict]\n")
        prefs = load_preferences(f)
        assert prefs.block == []
