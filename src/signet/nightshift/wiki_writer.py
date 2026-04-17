"""Write completed research artifacts back to the wiki as markdown files.

Closes the Karpathy loop: research -> wiki -> DB -> future research context.
Each completed research artifact becomes a wiki article with proper frontmatter,
and per-topic index.md files are maintained automatically.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import structlog

from signet.models.research import ResearchArtifact

log = structlog.get_logger()


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def topic_dir_name(topic: str) -> str:
    """Derive a directory name from a research topic.

    Maps broad topics to existing wiki directories where possible,
    otherwise creates a slugified directory name.
    """
    return slugify(topic)


def build_frontmatter(
    title: str,
    tags: list[str],
    summary: str,
    confidence: str,
    *,
    created: date | None = None,
    updated: date | None = None,
    source: str = "nightshift",
) -> str:
    """Build YAML frontmatter block for a wiki article."""
    today = date.today().isoformat()
    created_str = created.isoformat() if created else today
    updated_str = updated.isoformat() if updated else today

    tag_lines = "\n".join(f"  - {t}" for t in tags) if tags else "  - research"
    # Escape quotes in summary
    safe_summary = summary.replace('"', '\\"')

    return (
        f"---\n"
        f'title: "{title}"\n'
        f"tags:\n"
        f"{tag_lines}\n"
        f'summary: "{safe_summary}"\n'
        f"confidence: {confidence}\n"
        f"created: {created_str}\n"
        f"updated: {updated_str}\n"
        f"source: {source}\n"
        f"---\n\n"
    )


def build_article_body(artifact: ResearchArtifact) -> str:
    """Build the markdown body from a research artifact."""
    parts: list[str] = []

    # Main synthesis
    if artifact.synthesis:
        parts.append(artifact.synthesis.strip())

    # Section details (collapsed under headings)
    if artifact.sections:
        parts.append("\n## Research Details\n")
        for i, section in enumerate(artifact.sections, 1):
            if section.findings:
                parts.append(f"### {section.question}\n")
                parts.append(section.findings.strip())
                if section.sources:
                    parts.append("\n**Sources:** " + ", ".join(section.sources))
                parts.append("")

    # Open questions
    if artifact.open_questions:
        parts.append("\n## Open Questions\n")
        for q in artifact.open_questions:
            parts.append(f"- {q}")

    # Suggested next steps
    if artifact.suggested_next:
        parts.append("\n## Next Steps\n")
        for s in artifact.suggested_next:
            parts.append(f"- {s}")

    # Metadata footer
    parts.append("\n---\n")
    parts.append(f"*Research conducted by Signet nightshift*")
    parts.append(f"*Model: {artifact.model_used}*")
    parts.append(f"*Confidence: {artifact.confidence}*")
    if artifact.started_at:
        parts.append(f"*Date: {artifact.started_at.strftime('%Y-%m-%d')}*")

    return "\n".join(parts) + "\n"


def _build_summary(artifact: ResearchArtifact) -> str:
    """Extract a one-line summary from the synthesis."""
    if not artifact.synthesis:
        return artifact.angle or artifact.topic
    # Take the first non-empty line of the synthesis
    for line in artifact.synthesis.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            # Truncate to ~200 chars
            if len(stripped) > 200:
                return stripped[:197] + "..."
            return stripped
    return artifact.angle or artifact.topic


def write_artifact_to_wiki(
    artifact: ResearchArtifact,
    wikis_path: Path,
) -> Path:
    """Write a completed research artifact as a wiki markdown file.

    Creates the topic directory if needed. Returns the path to the written file.
    """
    # Determine topic directory
    topic_slug = topic_dir_name(artifact.topic)
    topic_path = wikis_path / topic_slug
    topic_path.mkdir(parents=True, exist_ok=True)

    # Build the filename from angle (more specific) or topic
    angle_text = artifact.angle if artifact.angle else artifact.topic
    file_slug = slugify(angle_text)
    # Truncate slug to avoid absurd filenames
    if len(file_slug) > 80:
        file_slug = file_slug[:80].rstrip("-")
    file_path = topic_path / f"{file_slug}.md"

    # Build tags from the artifact
    tags = list(artifact.tags) if artifact.tags else [topic_slug]
    if "nightshift" not in tags:
        tags.append("nightshift")

    summary = _build_summary(artifact)

    # Check if file already exists (update vs create)
    created_date = None
    if file_path.exists():
        # Preserve the original created date from existing frontmatter
        existing = file_path.read_text(encoding="utf-8")
        created_date = _extract_created_date(existing)
        log.info("wiki_writer.updating", path=str(file_path))
    else:
        log.info("wiki_writer.creating", path=str(file_path))

    frontmatter = build_frontmatter(
        title=artifact.topic if not artifact.angle else f"{artifact.topic}: {artifact.angle}",
        tags=tags,
        summary=summary,
        confidence=artifact.confidence or "medium",
        created=created_date,
    )

    body = build_article_body(artifact)
    file_path.write_text(frontmatter + body, encoding="utf-8")

    log.info(
        "wiki_writer.wrote",
        path=str(file_path),
        chars=len(body),
        topic=artifact.topic,
    )

    # Update the topic index
    update_topic_index(topic_path)

    return file_path


def _extract_created_date(text: str) -> date | None:
    """Pull the created date from existing frontmatter."""
    match = re.search(r"^created:\s*(.+)$", text, re.MULTILINE)
    if match:
        try:
            return datetime.fromisoformat(match.group(1).strip()).date()
        except ValueError:
            try:
                return date.fromisoformat(match.group(1).strip())
            except ValueError:
                pass
    return None


def update_topic_index(topic_path: Path) -> None:
    """Regenerate the _index.md file for a topic directory.

    Lists all articles with titles, summaries, and last-updated dates.
    Prefixed with _ so the wiki parser skips it (it's metadata, not content).
    """
    import yaml

    articles: list[dict[str, str]] = []

    for md_file in sorted(topic_path.glob("*.md")):
        if md_file.name.startswith("_"):
            continue

        text = md_file.read_text(encoding="utf-8")
        fm = _parse_frontmatter_quick(text)
        articles.append({
            "file": md_file.name,
            "title": fm.get("title", md_file.stem),
            "summary": fm.get("summary", ""),
            "confidence": fm.get("confidence", ""),
            "updated": fm.get("updated", ""),
            "source": fm.get("source", ""),
        })

    topic_name = topic_path.name.replace("-", " ").replace("_", " ").title()
    lines = [
        f"# {topic_name}\n",
        f"*{len(articles)} articles*\n",
    ]

    for a in articles:
        title = a["title"]
        fname = a["file"]
        summary = a["summary"]
        conf = f" [{a['confidence']}]" if a["confidence"] else ""
        updated = f" (updated {a['updated']})" if a["updated"] else ""
        source_tag = f" `{a['source']}`" if a["source"] else ""

        lines.append(f"- **[{title}]({fname})**{conf}{updated}{source_tag}")
        if summary:
            lines.append(f"  {summary}")

    index_path = topic_path / "_index.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.debug("wiki_writer.index_updated", topic=topic_path.name, articles=len(articles))


def _parse_frontmatter_quick(text: str) -> dict:
    """Quick frontmatter parse without the full WikiArticle model."""
    import yaml

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                return yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                return {}
    return {}
