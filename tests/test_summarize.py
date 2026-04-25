"""Tests for knowledge.summarize: migration, skip rules, orchestration.

Does not exercise the real Brain — summarize_body is stubbed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from signet.knowledge import summarize
from signet.knowledge.parser import scan_articles


DOCLING_RAW = """---
title: "Some Paper"
tags:
  - foundation_models
summary: ""
created: 2026-04-01
updated: 2026-04-01
source: docling
---

# Some Paper

Long full-text body of the paper goes here. Methods. Results. Discussion.
"""


HAND_AUTHORED = """---
title: "A Note"
tags:
  - misc
summary: "short"
created: 2026-04-01
updated: 2026-04-01
source: ""
---

A hand-authored note that should NOT be migrated.
"""


@pytest.fixture
def wiki_dir(tmp_path: Path) -> Path:
    (tmp_path / "foundation_models").mkdir()
    (tmp_path / "misc").mkdir()
    return tmp_path


def _fake_brain(digest_text: str = "## Claim\nStub digest."):
    brain = MagicMock()
    # Bypass summarize_body by monkey-patching in the caller side
    return brain, digest_text


def test_scan_skips_raw_md(wiki_dir: Path) -> None:
    (wiki_dir / "foundation_models" / "paper.raw.md").write_text(DOCLING_RAW)
    (wiki_dir / "foundation_models" / "paper.md").write_text(HAND_AUTHORED)
    articles = scan_articles(wiki_dir)
    slugs = [a.slug for a in articles]
    assert "paper" in slugs
    assert not any(s.endswith(".raw") for s in slugs)
    assert len(articles) == 1


def test_migrate_inline_docling_renames_to_raw(wiki_dir: Path) -> None:
    md_path = wiki_dir / "foundation_models" / "legacy.md"
    md_path.write_text(DOCLING_RAW)

    result = summarize._migrate_inline_raw(md_path)

    raw_path = wiki_dir / "foundation_models" / "legacy.raw.md"
    assert result == raw_path
    assert raw_path.exists()
    assert not md_path.exists()


def test_migrate_leaves_hand_authored_alone(wiki_dir: Path) -> None:
    md_path = wiki_dir / "misc" / "note.md"
    md_path.write_text(HAND_AUTHORED)

    result = summarize._migrate_inline_raw(md_path)

    assert result is None
    assert md_path.exists()


def test_migrate_noop_when_raw_sibling_exists(wiki_dir: Path) -> None:
    md_path = wiki_dir / "foundation_models" / "paper.md"
    raw_path = wiki_dir / "foundation_models" / "paper.raw.md"
    md_path.write_text("---\nsource: summary\n---\nthe digest")
    raw_path.write_text(DOCLING_RAW)

    result = summarize._migrate_inline_raw(md_path)

    assert result == raw_path
    assert md_path.exists()  # not touched
    assert raw_path.exists()


def test_summarize_all_produces_summary(
    wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path = wiki_dir / "foundation_models" / "paper.raw.md"
    raw_path.write_text(DOCLING_RAW)

    def fake_summarize_body(brain, title, body, *, max_tokens=4000):
        assert "Some Paper" in title
        assert "full-text body" in body
        return "## Claim\nStubbed digest.\n\n## Source\nStub, 2026."

    monkeypatch.setattr(summarize, "summarize_body", fake_summarize_body)

    result = summarize.summarize_all(wiki_dir, brain=MagicMock())

    assert result["summarized"] == 1
    assert result["errored"] == 0
    summary_path = wiki_dir / "foundation_models" / "paper.md"
    assert summary_path.exists()
    text = summary_path.read_text()
    assert "source: summary" in text
    assert "source_raw: paper.raw.md" in text
    assert "## Claim" in text


def test_summarize_all_skips_existing_without_force(
    wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (wiki_dir / "foundation_models" / "paper.raw.md").write_text(DOCLING_RAW)
    (wiki_dir / "foundation_models" / "paper.md").write_text("existing summary")

    monkeypatch.setattr(summarize, "summarize_body", lambda *a, **kw: "should not run")

    result = summarize.summarize_all(wiki_dir, brain=MagicMock(), force=False)

    assert result["summarized"] == 0
    assert result["skipped"] == 1
    assert (
        (wiki_dir / "foundation_models" / "paper.md").read_text() == "existing summary"
    )


def test_summarize_all_force_overwrites(
    wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (wiki_dir / "foundation_models" / "paper.raw.md").write_text(DOCLING_RAW)
    (wiki_dir / "foundation_models" / "paper.md").write_text("stale summary")

    monkeypatch.setattr(
        summarize, "summarize_body", lambda *a, **kw: "## Claim\nFresh."
    )

    result = summarize.summarize_all(wiki_dir, brain=MagicMock(), force=True)

    assert result["summarized"] == 1
    assert "Fresh" in (wiki_dir / "foundation_models" / "paper.md").read_text()


def test_summarize_all_migrates_and_summarizes(
    wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # legacy layout: inline docling at .md, no .raw.md sibling
    (wiki_dir / "foundation_models" / "legacy.md").write_text(DOCLING_RAW)

    monkeypatch.setattr(
        summarize, "summarize_body", lambda *a, **kw: "## Claim\nMigrated."
    )

    result = summarize.summarize_all(wiki_dir, brain=MagicMock())

    assert result["migrated"] == 1
    assert result["summarized"] == 1
    assert (wiki_dir / "foundation_models" / "legacy.raw.md").exists()
    summary = (wiki_dir / "foundation_models" / "legacy.md").read_text()
    assert "source: summary" in summary
    assert "Migrated" in summary


def test_summarize_all_only_slug_filter(
    wiki_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (wiki_dir / "foundation_models" / "wanted.raw.md").write_text(DOCLING_RAW)
    (wiki_dir / "foundation_models" / "other.raw.md").write_text(DOCLING_RAW)

    monkeypatch.setattr(
        summarize, "summarize_body", lambda *a, **kw: "## Claim\nOnly one."
    )

    result = summarize.summarize_all(
        wiki_dir, brain=MagicMock(), only_slug="wanted"
    )

    assert result["summarized"] == 1
    assert (wiki_dir / "foundation_models" / "wanted.md").exists()
    assert not (wiki_dir / "foundation_models" / "other.md").exists()
