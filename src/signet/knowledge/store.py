"""Wiki knowledge store backed by PostgreSQL + pgvector."""
from __future__ import annotations

from pathlib import Path

import asyncpg
import structlog
from pgvector.asyncpg import register_vector

from signet.knowledge.parser import scan_articles
from signet.memory.embeddings import EmbeddingService
from signet.models.knowledge import WikiArticle, WikiFrontmatter, WikiSearchResult

log = structlog.get_logger()

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class WikiStore:
    """Async persistence layer for wiki knowledge articles."""

    def __init__(
        self,
        wikis_path: Path,
        database_url: str,
        embedder: EmbeddingService,
    ) -> None:
        self._wikis_path = wikis_path
        self._database_url = database_url
        self._embedder = embedder
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._database_url,
            min_size=1,
            max_size=5,
            init=register_vector,
        )
        log.info("wiki.connected")

    async def initialize_schema(self) -> None:
        sql = SCHEMA_PATH.read_text()
        async with self._pool.acquire() as conn:
            await conn.execute(sql)
        log.info("wiki.schema_initialized")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def sync(self) -> dict[str, int]:
        """Sync .md files from disk into DB. Re-embeds changed files only."""
        articles = scan_articles(self._wikis_path)
        disk_slugs = {a.slug for a in articles}

        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT slug, content_hash FROM wiki_articles")
        db_state = {r["slug"]: r["content_hash"] for r in rows}

        added = 0
        updated = 0

        for article in articles:
            if article.slug not in db_state:
                await self._upsert_article(article)
                added += 1
            elif db_state[article.slug] != article.content_hash:
                await self._upsert_article(article)
                updated += 1

        removed_slugs = set(db_state.keys()) - disk_slugs
        removed = 0
        if removed_slugs:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM wiki_articles WHERE slug = ANY($1)",
                    list(removed_slugs),
                )
            removed = len(removed_slugs)

        log.info("wiki.synced", added=added, updated=updated, removed=removed)
        return {"added": added, "updated": updated, "removed": removed}

    async def search(
        self,
        query: str,
        *,
        limit: int = 3,
        min_similarity: float = 0.3,
        tags: list[str] | None = None,
    ) -> list[WikiSearchResult]:
        """Semantic search across wiki articles."""
        query_embedding = await self._embedder.embed(query)

        conditions = ["embedding IS NOT NULL"]
        params: list = [query_embedding, limit, min_similarity]
        param_idx = 4

        if tags:
            conditions.append(f"tags && ${param_idx}")
            params.append(tags)
            param_idx += 1

        where = " AND ".join(conditions)

        sql = f"""
            SELECT slug, title, summary, tags, body, content_hash,
                   path, source, created_at, updated_at, chunk_index,
                   1 - (embedding <=> $1) AS similarity
            FROM wiki_articles
            WHERE {where}
              AND 1 - (embedding <=> $1) >= $3
            ORDER BY embedding <=> $1
            LIMIT $2
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        results = []
        for row in rows:
            article = WikiArticle(
                slug=row["slug"],
                path=row["path"],
                frontmatter=WikiFrontmatter(
                    title=row["title"],
                    tags=row["tags"] or [],
                    summary=row["summary"] or "",
                    created=row["created_at"],
                    updated=row["updated_at"],
                    source=row["source"] or "",
                ),
                body=row["body"],
                content_hash=row["content_hash"],
            )
            results.append(
                WikiSearchResult(
                    article=article,
                    similarity=row["similarity"],
                    chunk_index=row["chunk_index"],
                )
            )

        log.debug("wiki.search", query=query[:50], results=len(results))
        return results

    async def list_articles(self) -> list[dict]:
        """List all indexed articles."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT slug, title, tags, summary, updated_at
                   FROM wiki_articles
                   ORDER BY updated_at DESC NULLS LAST"""
            )
        return [dict(r) for r in rows]

    async def _upsert_article(self, article: WikiArticle) -> None:
        embed_text = self._build_embed_text(article)
        embedding = await self._embedder.embed(embed_text)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO wiki_articles
                    (slug, title, summary, tags, body, content_hash,
                     path, source, created_at, updated_at, chunk_index, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (slug, chunk_index) DO UPDATE SET
                    title = EXCLUDED.title,
                    summary = EXCLUDED.summary,
                    tags = EXCLUDED.tags,
                    body = EXCLUDED.body,
                    content_hash = EXCLUDED.content_hash,
                    path = EXCLUDED.path,
                    source = EXCLUDED.source,
                    updated_at = EXCLUDED.updated_at,
                    embedding = EXCLUDED.embedding
                """,
                article.slug,
                article.frontmatter.title,
                article.frontmatter.summary,
                article.frontmatter.tags,
                article.body,
                article.content_hash,
                article.path,
                article.frontmatter.source,
                article.frontmatter.created,
                article.frontmatter.updated,
                0,
                embedding,
            )

    @staticmethod
    def _build_embed_text(article: WikiArticle) -> str:
        """Build text to embed: title + summary + truncated body."""
        parts = [article.frontmatter.title]
        if article.frontmatter.summary:
            parts.append(article.frontmatter.summary)
        words = article.body.split()
        parts.append(" ".join(words[:500]))
        return "\n".join(parts)
