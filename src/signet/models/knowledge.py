"""Wiki knowledge models."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class WikiFrontmatter(BaseModel):
    """YAML frontmatter parsed from a wiki article."""

    title: str = "Untitled"
    tags: list[str] = Field(default_factory=list)
    summary: str = ""
    created: datetime | None = None
    updated: datetime | None = None
    source: str = ""


class WikiArticle(BaseModel):
    """A parsed wiki article from disk."""

    slug: str
    path: str
    frontmatter: WikiFrontmatter
    body: str
    content_hash: str


class WikiSearchResult(BaseModel):
    """A wiki article returned from semantic search."""

    article: WikiArticle
    similarity: float
    chunk_index: int = 0
