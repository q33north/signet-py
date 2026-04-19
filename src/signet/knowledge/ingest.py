"""Ingest raw documents (PDF, PPTX, DOCX) into wiki markdown articles via Docling."""
from __future__ import annotations

import re
import tempfile
from datetime import date
from pathlib import Path

import structlog

log = structlog.get_logger()


def convert_bytes_to_markdown(content: bytes, *, suffix: str) -> str:
    """Convert raw document bytes to markdown via Docling.

    Writes to a temp file since Docling's API takes a path. suffix must start
    with '.' (e.g. '.pdf', '.docx', '.pptx') so Docling picks the right parser.
    """
    from docling.document_converter import DocumentConverter

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        converter = DocumentConverter()
        result = converter.convert(tmp.name)
        return result.document.export_to_markdown()


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


SUPPORTED_EXTENSIONS = ("*.pdf", "*.pptx", "*.docx")


def _convert_document(file_path: Path) -> str:
    """Convert a single document to markdown using Docling."""
    return convert_bytes_to_markdown(file_path.read_bytes(), suffix=file_path.suffix)


def ingest_raw(wikis_path: Path, *, force: bool = False) -> dict[str, int]:
    """Scan all raw/ subdirectories for supported documents and convert to markdown.

    Returns counts of converted, skipped, and errored files.
    """
    converted = 0
    skipped = 0
    errored = 0

    if not wikis_path.exists():
        return {"converted": 0, "skipped": 0, "errored": 0}

    for topic_dir in sorted(wikis_path.iterdir()):
        if not topic_dir.is_dir():
            continue

        raw_dir = topic_dir / "raw"
        if not raw_dir.is_dir():
            continue

        topic = topic_dir.name
        files = sorted(
            f for ext in SUPPORTED_EXTENSIONS for f in raw_dir.glob(ext)
        )

        if not files:
            continue

        log.info("ingest.scanning", topic=topic, files=len(files))

        for file_path in files:
            slug = _slugify(file_path.stem)
            output_path = topic_dir / f"{slug}.md"

            if output_path.exists() and not force:
                log.debug("ingest.skip", slug=slug, reason="already exists")
                skipped += 1
                continue

            log.info("ingest.converting", file=file_path.name, topic=topic)

            try:
                body = _convert_document(file_path)
            except Exception:
                log.exception("ingest.error", file=file_path.name)
                errored += 1
                continue

            title = file_path.stem.replace("_", " ").replace("-", " ").title()
            frontmatter = _make_frontmatter(title, tags=[topic])
            output_path.write_text(frontmatter + body, encoding="utf-8")

            log.info("ingest.wrote", output=str(output_path), chars=len(body))
            converted += 1

    return {"converted": converted, "skipped": skipped, "errored": errored}
