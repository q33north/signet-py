"""Tests for the nightshift wiki writer.

Verifies that research artifacts are correctly written as wiki markdown files,
frontmatter is generated properly, and topic indexes are maintained.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from signet.models.research import (
    ResearchArtifact,
    ResearchSection,
    ResearchStatus,
)
from signet.nightshift.wiki_writer import (
    _build_summary,
    _parse_frontmatter_quick,
    build_article_body,
    build_frontmatter,
    slugify,
    topic_dir_name,
    update_topic_index,
    write_artifact_to_wiki,
)


class TestSlugify:
    def test_basic_slug(self):
        assert slugify("KRAS G12C Resistance") == "kras-g12c-resistance"

    def test_special_chars_removed(self):
        assert slugify("what's up? (nothing)") == "whats-up-nothing"

    def test_multiple_spaces_collapsed(self):
        assert slugify("too   many   spaces") == "too-many-spaces"

    def test_strips_leading_trailing(self):
        assert slugify("  hello  ") == "hello"


class TestTopicDirName:
    def test_simple_topic(self):
        assert topic_dir_name("cancer genomics") == "cancer-genomics"

    def test_complex_topic(self):
        assert topic_dir_name("KRAS G12C Resistance Mechanisms") == "kras-g12c-resistance-mechanisms"


class TestBuildFrontmatter:
    def test_basic_frontmatter(self):
        fm = build_frontmatter(
            title="Test Article",
            tags=["cancer", "genomics"],
            summary="A test summary",
            confidence="high",
        )
        assert 'title: "Test Article"' in fm
        assert "  - cancer" in fm
        assert "  - genomics" in fm
        assert 'summary: "A test summary"' in fm
        assert "confidence: high" in fm
        assert "source: nightshift" in fm
        assert fm.startswith("---\n")
        assert "---\n\n" in fm

    def test_preserves_created_date(self):
        fm = build_frontmatter(
            title="Test",
            tags=["test"],
            summary="",
            confidence="medium",
            created=date(2026, 1, 15),
        )
        assert "created: 2026-01-15" in fm

    def test_escapes_quotes_in_summary(self):
        fm = build_frontmatter(
            title="Test",
            tags=["test"],
            summary='this "quoted" summary',
            confidence="medium",
        )
        assert 'summary: "this \\"quoted\\" summary"' in fm

    def test_default_tags_when_empty(self):
        fm = build_frontmatter(
            title="Test",
            tags=[],
            summary="",
            confidence="medium",
        )
        assert "  - research" in fm


class TestBuildArticleBody:
    def test_includes_synthesis(self):
        artifact = ResearchArtifact(
            topic="KRAS",
            synthesis="Main findings here",
        )
        body = build_article_body(artifact)
        assert "Main findings here" in body

    def test_includes_sections(self):
        artifact = ResearchArtifact(
            topic="KRAS",
            synthesis="Summary",
            sections=[
                ResearchSection(
                    question="What about resistance?",
                    findings="Resistance is complex",
                    sources=["PMC12345"],
                ),
            ],
        )
        body = build_article_body(artifact)
        assert "## Research Details" in body
        assert "### What about resistance?" in body
        assert "Resistance is complex" in body
        assert "PMC12345" in body

    def test_includes_open_questions(self):
        artifact = ResearchArtifact(
            topic="KRAS",
            synthesis="Summary",
            open_questions=["Does this apply to NSCLC?"],
        )
        body = build_article_body(artifact)
        assert "## Open Questions" in body
        assert "- Does this apply to NSCLC?" in body

    def test_includes_next_steps(self):
        artifact = ResearchArtifact(
            topic="KRAS",
            synthesis="Summary",
            suggested_next=["Check COSMIC database"],
        )
        body = build_article_body(artifact)
        assert "## Next Steps" in body
        assert "- Check COSMIC database" in body

    def test_metadata_footer(self):
        artifact = ResearchArtifact(
            topic="KRAS",
            synthesis="Summary",
            model_used="claude-sonnet-4-6",
            confidence="high",
            started_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
        )
        body = build_article_body(artifact)
        assert "Signet nightshift" in body
        assert "claude-sonnet-4-6" in body
        assert "2026-04-15" in body


class TestWriteArtifactToWiki:
    def test_creates_new_article(self, tmp_path: Path):
        artifact = ResearchArtifact(
            topic="Cancer Genomics",
            angle="KRAS G12C resistance patterns",
            synthesis="Key findings about resistance",
            confidence="high",
            tags=["cancer", "kras"],
            model_used="claude-sonnet-4-6",
            status=ResearchStatus.COMPLETED,
        )

        result = write_artifact_to_wiki(artifact, tmp_path)

        assert result.exists()
        assert result.parent.name == "cancer-genomics"
        assert result.name == "kras-g12c-resistance-patterns.md"

        content = result.read_text()
        assert "---" in content
        assert 'title: "Cancer Genomics: KRAS G12C resistance patterns"' in content
        assert "Key findings about resistance" in content
        assert "nightshift" in content

    def test_creates_topic_directory(self, tmp_path: Path):
        artifact = ResearchArtifact(
            topic="Brand New Topic",
            synthesis="Some findings",
        )

        result = write_artifact_to_wiki(artifact, tmp_path)

        assert (tmp_path / "brand-new-topic").is_dir()
        assert result.exists()

    def test_updates_existing_article_preserves_created(self, tmp_path: Path):
        # Write initial article
        topic_dir = tmp_path / "cancer-genomics"
        topic_dir.mkdir()
        existing = topic_dir / "kras-resistance.md"
        existing.write_text(
            '---\ntitle: "Old Title"\ncreated: 2026-01-01\nupdated: 2026-01-01\n---\n\nOld body\n'
        )

        artifact = ResearchArtifact(
            topic="Cancer Genomics",
            angle="KRAS resistance",
            synthesis="Updated findings",
            confidence="high",
        )

        result = write_artifact_to_wiki(artifact, tmp_path)

        content = result.read_text()
        # Should preserve original created date
        assert "created: 2026-01-01" in content
        # But have new content
        assert "Updated findings" in content

    def test_truncates_long_slugs(self, tmp_path: Path):
        artifact = ResearchArtifact(
            topic="Topic",
            angle="a" * 200,
            synthesis="Findings",
        )

        result = write_artifact_to_wiki(artifact, tmp_path)

        assert len(result.stem) <= 80

    def test_generates_index(self, tmp_path: Path):
        artifact = ResearchArtifact(
            topic="Cancer Genomics",
            angle="KRAS resistance",
            synthesis="Findings",
            confidence="medium",
        )

        write_artifact_to_wiki(artifact, tmp_path)

        index_path = tmp_path / "cancer-genomics" / "_index.md"
        assert index_path.exists()
        content = index_path.read_text()
        assert "Cancer Genomics" in content
        assert "1 articles" in content

    def test_uses_topic_as_fallback_when_no_angle(self, tmp_path: Path):
        artifact = ResearchArtifact(
            topic="Proteomics",
            angle="",
            synthesis="Findings",
        )

        result = write_artifact_to_wiki(artifact, tmp_path)

        assert result.name == "proteomics.md"

    def test_adds_nightshift_tag(self, tmp_path: Path):
        artifact = ResearchArtifact(
            topic="Test",
            synthesis="Findings",
            tags=["cancer"],
        )

        result = write_artifact_to_wiki(artifact, tmp_path)
        content = result.read_text()

        assert "  - cancer" in content
        assert "  - nightshift" in content

    def test_wiki_folder_overrides_topic_slug(self, tmp_path: Path):
        artifact = ResearchArtifact(
            topic="KRAS G12C Resistance",
            angle="sotorasib bypass mechanisms",
            synthesis="Findings about bypass",
            wiki_folder="cancer_genomics",
        )

        result = write_artifact_to_wiki(artifact, tmp_path)

        # Should use the explicit folder, not the slugified topic
        assert result.parent.name == "cancer_genomics"
        assert (tmp_path / "cancer_genomics").is_dir()

    def test_strips_null_bytes_from_synthesis(self, tmp_path: Path):
        artifact = ResearchArtifact(
            topic="Cancer Genomics",
            angle="null byte test",
            synthesis="findings with\x00embedded null\x00bytes",
        )

        result = write_artifact_to_wiki(artifact, tmp_path)

        content = result.read_text()
        assert "\x00" not in content
        assert "findings withembedded nullbytes" in content

    def test_empty_wiki_folder_falls_back_to_topic(self, tmp_path: Path):
        artifact = ResearchArtifact(
            topic="Proteomics Methods",
            synthesis="Findings",
            wiki_folder="",
        )

        result = write_artifact_to_wiki(artifact, tmp_path)

        assert result.parent.name == "proteomics-methods"


class TestUpdateTopicIndex:
    def test_creates_index_from_articles(self, tmp_path: Path):
        # Create a couple of articles
        (tmp_path / "article-one.md").write_text(
            '---\ntitle: "First Article"\nsummary: "About something"\nupdated: 2026-04-01\nsource: nightshift\n---\n\nBody\n'
        )
        (tmp_path / "article-two.md").write_text(
            '---\ntitle: "Second Article"\nsummary: "About another thing"\nconfidence: high\nupdated: 2026-04-02\nsource: manual\n---\n\nBody\n'
        )

        update_topic_index(tmp_path)

        index = tmp_path / "_index.md"
        assert index.exists()
        content = index.read_text()
        assert "2 articles" in content
        assert "First Article" in content
        assert "Second Article" in content
        assert "[high]" in content

    def test_skips_underscore_prefixed_files(self, tmp_path: Path):
        (tmp_path / "_index.md").write_text("old index")
        (tmp_path / "real-article.md").write_text(
            '---\ntitle: "Real"\n---\n\nBody\n'
        )

        update_topic_index(tmp_path)

        content = (tmp_path / "_index.md").read_text()
        assert "1 articles" in content
        assert "Real" in content

    def test_handles_articles_without_frontmatter(self, tmp_path: Path):
        (tmp_path / "no-fm.md").write_text("Just plain text, no frontmatter here")

        update_topic_index(tmp_path)

        content = (tmp_path / "_index.md").read_text()
        assert "1 articles" in content
        assert "no-fm" in content  # falls back to filename


class TestBuildSummary:
    def _artifact(self, synthesis: str, topic: str = "T", angle: str = "a") -> ResearchArtifact:
        return ResearchArtifact(
            topic=topic,
            angle=angle,
            synthesis=synthesis,
            status=ResearchStatus.COMPLETED,
        )

    def test_skips_horizontal_rule(self):
        synthesis = "# Heading\n\n---\n\nActual first sentence of the writeup."
        assert _build_summary(self._artifact(synthesis)) == "Actual first sentence of the writeup."

    def test_skips_multiple_separator_styles(self):
        synthesis = "---\n===\n***\n\nReal content here."
        assert _build_summary(self._artifact(synthesis)) == "Real content here."

    def test_falls_back_to_angle_when_only_separators(self):
        synthesis = "# A\n## B\n---\n===\n"
        out = _build_summary(self._artifact(synthesis, angle="fallback angle"))
        assert out == "fallback angle"

    def test_truncates_long_line(self):
        long = "x" * 400
        out = _build_summary(self._artifact(long))
        assert len(out) <= 200
        assert out.endswith("...")


class TestParseFrontmatterQuick:
    def test_handles_triple_dash_in_quoted_value(self):
        """A summary value of '---' must not trick the frontmatter splitter."""
        text = (
            '---\n'
            'title: "Some title"\n'
            'summary: "---"\n'
            'confidence: low\n'
            '---\n\n'
            'Body content here.\n'
        )
        fm = _parse_frontmatter_quick(text)
        assert fm.get("title") == "Some title"
        assert fm.get("summary") == "---"
        assert fm.get("confidence") == "low"

    def test_missing_frontmatter_returns_empty(self):
        assert _parse_frontmatter_quick("just a body") == {}

    def test_malformed_yaml_returns_empty(self):
        text = '---\ntitle: "unterminated\n---\n\nbody\n'
        assert _parse_frontmatter_quick(text) == {}


class TestParserHandlesTripleDashValue:
    """The top-level knowledge parser must also tolerate '---' inside scalars."""

    def test_parse_article_with_triple_dash_summary(self, tmp_path: Path):
        from signet.knowledge.parser import parse_article

        content = (
            '---\n'
            'title: "A paper"\n'
            'tags:\n  - tag1\n'
            'summary: "---"\n'
            'source: nightshift\n'
            '---\n\n'
            '# Body\n\nReal content.\n'
        )
        f = tmp_path / "article.md"
        f.write_text(content)

        a = parse_article(f, tmp_path)
        assert a.frontmatter.title == "A paper"
        assert a.frontmatter.summary == "---"
        assert "Real content." in a.body

    def test_parse_article_strips_null_bytes(self, tmp_path: Path):
        from signet.knowledge.parser import parse_article

        content = (
            '---\ntitle: "x"\nsummary: ""\nsource: nightshift\n---\n\n'
            'body with\x00null\x00bytes\n'
        )
        f = tmp_path / "article.md"
        f.write_bytes(content.encode("utf-8"))

        a = parse_article(f, tmp_path)
        assert "\x00" not in a.body
        assert "body withnullbytes" in a.body


class TestResearcherWikiIntegration:
    """Test that the researcher calls wiki writeback after synthesis."""

    @pytest.mark.asyncio
    async def test_write_to_wiki_called_on_completed_research(self, tmp_path: Path):
        """Researcher._write_to_wiki should write artifact and trigger sync."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from signet.nightshift.researcher import Researcher

        brain = MagicMock()
        memory = AsyncMock()
        wiki = AsyncMock()
        wiki.sync.return_value = {"added": 1, "updated": 0, "removed": 0}
        dreams = AsyncMock()
        research = AsyncMock()

        researcher = Researcher(brain, memory, wiki, dreams, research)

        artifact = ResearchArtifact(
            topic="Test Topic",
            angle="test angle",
            synthesis="Test findings",
            confidence="medium",
            status=ResearchStatus.COMPLETED,
        )

        with patch("signet.nightshift.researcher.settings") as mock_settings:
            mock_settings.wikis_path = tmp_path
            await researcher._write_to_wiki(artifact)

        # Article should be written to disk
        written_files = list(tmp_path.glob("**/*.md"))
        # Filter out _index.md
        article_files = [f for f in written_files if not f.name.startswith("_")]
        assert len(article_files) == 1

        # Wiki sync should be triggered
        wiki.sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_to_wiki_handles_errors_gracefully(self, tmp_path: Path):
        """Wiki writeback failure should not crash the researcher."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from signet.nightshift.researcher import Researcher

        brain = MagicMock()
        memory = AsyncMock()
        wiki = AsyncMock()
        wiki.sync.side_effect = Exception("DB connection failed")
        dreams = AsyncMock()
        research = AsyncMock()

        researcher = Researcher(brain, memory, wiki, dreams, research)

        artifact = ResearchArtifact(
            topic="Test",
            synthesis="Findings",
            status=ResearchStatus.COMPLETED,
        )

        with patch("signet.nightshift.researcher.settings") as mock_settings:
            mock_settings.wikis_path = tmp_path
            # Should not raise
            await researcher._write_to_wiki(artifact)
