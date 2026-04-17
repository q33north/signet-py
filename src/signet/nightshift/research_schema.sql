-- Nightshift research schema. Idempotent.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS research (
    id                  UUID PRIMARY KEY,
    topic               TEXT NOT NULL,
    angle               TEXT NOT NULL DEFAULT '',
    status              TEXT NOT NULL DEFAULT 'queued',
    plan                TEXT NOT NULL DEFAULT '',
    sections            JSONB NOT NULL DEFAULT '[]',
    synthesis           TEXT NOT NULL DEFAULT '',
    confidence          TEXT NOT NULL DEFAULT '',
    open_questions      TEXT[] NOT NULL DEFAULT '{}',
    suggested_next      TEXT[] NOT NULL DEFAULT '{}',
    source_wiki_slugs   TEXT[] NOT NULL DEFAULT '{}',
    source_dream_ids    UUID[] NOT NULL DEFAULT '{}',
    model_used          TEXT NOT NULL DEFAULT '',
    token_count         INTEGER NOT NULL DEFAULT 0,
    tags                TEXT[] NOT NULL DEFAULT '{}',
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at        TIMESTAMPTZ,
    embedding           vector(384)
);

CREATE INDEX IF NOT EXISTS idx_research_status
    ON research (status, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_completed
    ON research (completed_at DESC) WHERE status = 'completed';

CREATE INDEX IF NOT EXISTS idx_research_embedding
    ON research USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

CREATE TABLE IF NOT EXISTS research_queue (
    id                  UUID PRIMARY KEY,
    topic               TEXT NOT NULL,
    requested_by        TEXT NOT NULL DEFAULT '',
    requested_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    consumed            BOOLEAN NOT NULL DEFAULT FALSE,
    consumed_at         TIMESTAMPTZ,
    research_id         UUID REFERENCES research(id),
    wiki_folder         TEXT NOT NULL DEFAULT ''
);

-- Idempotent column addition for existing databases
DO $$ BEGIN
    ALTER TABLE research_queue ADD COLUMN wiki_folder TEXT NOT NULL DEFAULT '';
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;

CREATE INDEX IF NOT EXISTS idx_research_queue_pending
    ON research_queue (requested_at ASC) WHERE consumed = FALSE;
