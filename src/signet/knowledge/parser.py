"""Parse wiki markdown files with YAML frontmatter."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from signet.models.knowledge import WikiArticle, WikiFrontmatter


def parse_article(file_path: Path, wikis_root: Path) -> WikiArticle:
    """Parse a single .md file into a WikiArticle."""
    raw_bytes = file_path.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    text = raw_bytes.decode("utf-8")

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
        articles.append(parse_article(md_file, wikis_path))
    return articles


def _split_frontmatter(text: str) -> tuple[WikiFrontmatter, str]:
    """Split ---frontmatter--- from body."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm_raw = yaml.safe_load(parts[1]) or {}
            body = parts[2]
            return WikiFrontmatter(**fm_raw), body
    return WikiFrontmatter(), text
