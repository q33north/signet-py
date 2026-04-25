"""Parse wiki markdown files with YAML frontmatter."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from signet.models.knowledge import WikiArticle, WikiFrontmatter

_FRONTMATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)\Z", re.DOTALL)


def parse_article(file_path: Path, wikis_root: Path) -> WikiArticle:
    """Parse a single .md file into a WikiArticle."""
    raw_bytes = file_path.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    # Postgres TEXT rejects 0x00; strip defensively so sync never blows up.
    text = raw_bytes.decode("utf-8").replace("\x00", "")

    frontmatter, body = _split_frontmatter(text)
    slug = file_path.stem
    rel_path = str(file_path.relative_to(wikis_root))

    return WikiArticle(
        slug=slug,
        path=rel_path,
        frontmatter=frontmatter,
        body=body.strip(),
        content_hash=content_hash,
    )


def scan_articles(wikis_path: Path) -> list[WikiArticle]:
    """Scan all .md files in wikis_path, skipping _-prefixed files."""
    if not wikis_path.exists():
        return []
    articles = []
    for md_file in sorted(wikis_path.glob("**/*.md")):
        if md_file.name.startswith("_"):
            continue
        if md_file.name.endswith(".raw.md"):
            continue
        articles.append(parse_article(md_file, wikis_path))
    return articles


def _split_frontmatter(text: str) -> tuple[WikiFrontmatter, str]:
    """Split ---frontmatter--- from body.

    The delimiter is '---' on its own line. Splitting naively on '---' breaks
    on YAML values that contain '---' (e.g. summaries that started life as
    horizontal rules).
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return WikiFrontmatter(), text

    fm_raw = yaml.safe_load(match.group(1)) or {}
    body = match.group(2)
    return WikiFrontmatter(**fm_raw), body
