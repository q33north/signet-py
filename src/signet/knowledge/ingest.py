"""Ingest raw PDFs into wiki markdown articles via Docling."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import structlog

log = structlog.get_logger()


def _slugify(name: str) -> str:
    """Convert a filename to a URL-friendly slug."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_]+", "-", name)
    return re.sub(r"-+", "-", name).strip("-")


def _make_frontmatter(title: str, tags: list[str]) -> str:
    tag_str = "\n".join(f"  - {t}" for t in tags)
    today = date.today().isoformat()
    return f"""---
title: "{title}"
tags:
{tag_str}
summary: ""
created: {today}
updated: {today}
source: docling
---

"""


def _convert_pdf(pdf_path: Path) -> str:
    """Convert a single PDF to markdown using Docling."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


def ingest_raw(wikis_path: Path, *, force: bool = False) -> dict[str, int]:
    """Scan all raw/ subdirectories for PDFs and convert to markdown.

    Returns counts of converted, skipped, and errored files.
    """
    converted = 0
    skipped = 0
    errored = 0

    for topic_dir in sorted(wikis_path.iterdir()):
        if not topic_dir.is_dir():
            continue

        raw_dir = topic_dir / "raw"
        if not raw_dir.is_dir():
            continue

        topic = topic_dir.name
        pdfs = sorted(raw_dir.glob("*.pdf"))

        if not pdfs:
            continue

        log.info("ingest.scanning", topic=topic, pdfs=len(pdfs))

        for pdf_path in pdfs:
            slug = _slugify(pdf_path.stem)
            output_path = topic_dir / f"{slug}.md"

            if output_path.exists() and not force:
                log.debug("ingest.skip", slug=slug, reason="already exists")
                skipped += 1
                continue

            log.info("ingest.converting", pdf=pdf_path.name, topic=topic)

            try:
                body = _convert_pdf(pdf_path)
            except Exception:
                log.exception("ingest.error", pdf=pdf_path.name)
                errored += 1
                continue

            title = pdf_path.stem.replace("_", " ").replace("-", " ").title()
            frontmatter = _make_frontmatter(title, tags=[topic])
            output_path.write_text(frontmatter + body, encoding="utf-8")

            log.info("ingest.wrote", output=str(output_path), chars=len(body))
            converted += 1

    return {"converted": converted, "skipped": skipped, "errored": errored}
