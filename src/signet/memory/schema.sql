-- Signet memory schema. Idempotent: safe to run on every startup.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY,
    platform        TEXT NOT NULL DEFAULT '',
    channel_id      TEXT NOT NULL DEFAULT '',
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at        TIMESTAMPTZ,
    message_count   INTEGER NOT NULL DEFAULT 0,
    consolidated    BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS messages (
    id              UUID PRIMARY KEY,
    session_id      UUID REFERENCES sessions(id),
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    platform        TEXT NOT NULL DEFAULT '',
    channel_id      TEXT NOT NULL DEFAULT '',
    author_id       TEXT NOT NULL DEFAULT '',
    author_name     TEXT NOT NULL DEFAULT '',
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding       vector(384)
);

CREATE INDEX IF NOT EXISTS idx_messages_channel_ts
    ON messages (channel_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages (session_id, timestamp ASC);

CREATE INDEX IF NOT EXISTS idx_sessions_channel
    ON sessions (channel_id, started_at DESC);
