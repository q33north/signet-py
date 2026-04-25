"""Summarize raw wiki articles (e.g. Docling-converted PDFs) into structured digests.

Layout convention:
  - {slug}.raw.md  — full source text (Docling output, frontmatter source: docling)
  - {slug}.md      — structured digest produced here (frontmatter source: summary)

Digests are biomedical/bioinformatics-focused, epistemically honest (sections
not supported by the source are marked "not reported" rather than inferred).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import structlog

from signet.brain.client import Brain
from signet.config import settings
from signet.knowledge.parser import _split_frontmatter

log = structlog.get_logger()


SUMMARY_SYSTEM = """You are a careful biomedical research reader producing a structured digest of a single source document for an autonomous research agent. Write in precise, neutral prose. Never infer or invent: if the source does not support a section, write "Not reported." Do not hedge unnecessarily when the source is clear.

Your output is consumed by a machine reader that cannot see the original PDF, so faithfulness matters more than style."""


SUMMARY_USER_TEMPLATE = """Produce a digest of the document below using EXACTLY these sections, in this order, each as an H2 markdown heading:

## Claim
One to three sentences stating what the paper argues or demonstrates.

## Cohort
Study population: n, disease / condition, inclusion / exclusion, case-control vs cohort, demographics if given. "Not reported" if absent or N/A.

## Methods — biology
Assays, sample types, experimental design, wet-lab techniques. Be specific (platform names, antibody panels, sequencing chemistries, etc.) when given.

## Methods — bioinformatics
Pipelines, tools, reference genomes / databases, feature-selection and ML / statistical methods, software versions, QC filters, chosen parameters. This section is load-bearing for our use — be thorough and specific. "Not reported" only if truly absent.

## Key numbers
Effect sizes, AUCs, sensitivities / specificities, p-values, sample sizes, cohort splits, runtimes. One bullet per number with its context.

## Limitations
What the authors acknowledge, plus obvious unstated caveats (small n, single center, no external validation, etc.) only if directly supportable from the text.

## Why it matters
One to three sentences on the significance for cancer genomics / biomarker development / clinical translation.

## Source
Title, authors (first + et al. is fine), venue, year, DOI / preprint ID if present in the document.

Document title: {title}

Document text follows below the line. Everything up to end-of-message is the source.
---
{body}
"""


def summarize_body(brain: Brain, title: str, body: str, *, max_tokens: int = 4000) -> str:
    """Call Sonnet to produce a structured digest of a single raw document."""
    from signet.models.memory import Message, MessageRole

    prompt = SUMMARY_USER_TEMPLATE.format(title=title, body=body)
    return brain.chat(
        system=SUMMARY_SYSTEM,
        messages=[Message(role=MessageRole.USER, content=prompt)],
        model=settings.model_heavy,
        max_tokens=max_tokens,
    )


def _make_summary_frontmatter(title: str, tags: list[str], source_raw: str) -> str:
    tag_str = "\n".join(f"  - {t}" for t in tags)
    today = date.today().isoformat()
    return (
        "---\n"
        f'title: "{title}"\n'
        "tags:\n"
        f"{tag_str}\n"
        'summary: ""\n'
        f"created: {today}\n"
        f"updated: {today}\n"
        "source: summary\n"
        f"source_raw: {source_raw}\n"
        "---\n\n"
    )


def _migrate_inline_raw(md_path: Path) -> Path | None:
    """If {slug}.md is a docling-produced raw file with no {slug}.raw.md sibling,
    rename it to {slug}.raw.md so we can produce a summary at {slug}.md.
    Returns the new .raw.md path, or None if no migration was needed.
    """
    if not md_path.exists() or md_path.name.endswith(".raw.md"):
        return None

    raw_sibling = md_path.parent / f"{md_path.stem}.raw.md"
    if raw_sibling.exists():
        return raw_sibling  # already split; .md here is presumed a summary

    text = md_path.read_text(encoding="utf-8")
    fm, _ = _split_frontmatter(text)
    if fm.source != "docling":
        return None  # hand-authored, leave alone

    md_path.rename(raw_sibling)
    log.info("summarize.migrate", slug=md_path.stem, renamed_to=raw_sibling.name)
    return raw_sibling


def summarize_all(
    wikis_path: Path,
    brain: Brain,
    *,
    force: bool = False,
    only_slug: str | None = None,
) -> dict[str, int]:
    """Walk wikis_path, summarize every {slug}.raw.md that lacks a fresh {slug}.md.

    Also migrates legacy inline-raw {slug}.md files (source: docling) to .raw.md.
    """
    summarized = 0
    skipped = 0
    errored = 0
    migrated = 0

    if not wikis_path.exists():
        return {"summarized": 0, "skipped": 0, "errored": 0, "migrated": 0}

    # Pass 1: migrate legacy inline docling files to .raw.md
    for md_file in sorted(wikis_path.glob("**/*.md")):
        if md_file.name.startswith("_") or md_file.name.endswith(".raw.md"):
            continue
        if only_slug and md_file.stem != only_slug:
            continue
        if _migrate_inline_raw(md_file) is not None:
            # only count as migrated if the file actually moved
            if not md_file.exists():
                migrated += 1

    # Pass 2: summarize every .raw.md
    for raw_file in sorted(wikis_path.glob("**/*.raw.md")):
        slug = raw_file.name[: -len(".raw.md")]
        if only_slug and slug != only_slug:
            continue

        summary_path = raw_file.parent / f"{slug}.md"
        if summary_path.exists() and not force:
            log.debug("summarize.skip", slug=slug, reason="summary exists")
            skipped += 1
            continue

        log.info("summarize.running", slug=slug, chars=raw_file.stat().st_size)

        try:
            text = raw_file.read_text(encoding="utf-8")
            fm, body = _split_frontmatter(text)
            title = fm.title or slug.replace("-", " ").replace("_", " ").title()
            tags = fm.tags or [raw_file.parent.name]

            digest = summarize_body(brain, title, body)
            frontmatter = _make_summary_frontmatter(title, tags, raw_file.name)
            summary_path.write_text(frontmatter + digest.strip() + "\n", encoding="utf-8")

            log.info("summarize.wrote", slug=slug, chars=len(digest))
            summarized += 1
        except Exception:
            log.exception("summarize.error", slug=slug)
            errored += 1

    return {
        "summarized": summarized,
        "skipped": skipped,
        "errored": errored,
        "migrated": migrated,
    }
