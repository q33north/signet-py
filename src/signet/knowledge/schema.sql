-- Wiki knowledge schema. Idempotent: safe to run on every startup.

CREATE TABLE IF NOT EXISTS wiki_articles (
    slug            TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL DEFAULT 0,
    title           TEXT NOT NULL,
    summary         TEXT NOT NULL DEFAULT '',
    tags            TEXT[] NOT NULL DEFAULT '{}',
    body            TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    path            TEXT NOT NULL DEFAULT '',
    source          TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ,
    embedding       vector(384),
    PRIMARY KEY (slug, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_wiki_articles_tags
    ON wiki_articles USING GIN (tags);
