-- autoDream schema. Idempotent: safe to run on every startup.

CREATE EXTENSION IF NOT EXISTS vector;

-- Track which messages have been consolidated
ALTER TABLE messages ADD COLUMN IF NOT EXISTS consolidated BOOLEAN NOT NULL DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS idx_messages_unconsolidated
    ON messages (consolidated, timestamp ASC) WHERE consolidated = FALSE;

-- Dream artifacts produced by consolidation
CREATE TABLE IF NOT EXISTS dreams (
    id                  UUID PRIMARY KEY,
    dream_type          TEXT NOT NULL,
    content             TEXT NOT NULL,
    source_message_ids  UUID[] NOT NULL DEFAULT '{}',
    entity_name         TEXT NOT NULL DEFAULT '',
    tags                TEXT[] NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding           vector(384)
);

CREATE INDEX IF NOT EXISTS idx_dreams_type
    ON dreams (dream_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_dreams_entity
    ON dreams (entity_name) WHERE entity_name != '';
