# Changelog

## 2026-04-03

### Phase 1: Core chat (fdf8c61)
- Built full Discord bot with discord.py: DM, @mention, role mention, name-in-message triggers
- Anthropic SDK wrapper (Brain) with model routing: Haiku (light), Sonnet (heavy), Opus (deep)
- Character system: YAML definition, personality sampler with randomized bio/adjectives/topics/examples per prompt
- PromptAssembler composes system prompt from character + context layers
- typer CLI with `run`, `check` commands
- structlog for structured logging throughout

### Phase 2: Memory persistence (fdf8c61)
- PostgreSQL + pgvector store for all messages with 384d embeddings
- sentence-transformers (all-MiniLM-L6-v2) running locally, lazy-loaded, GPU if available
- Semantic recall across all channels (cosine similarity search)
- Session-scoped channel history (last 20 messages, chronological)
- Async connection pooling via asyncpg
- `db-init` CLI command for idempotent schema setup

### Phase 4: Wiki/knowledge system (ffdbfbb + a151d7d)
- Markdown articles with YAML frontmatter, stored in PostgreSQL with embeddings
- Content-hash-based change detection for sync (skip unchanged files)
- Semantic search across wiki articles with tag filtering
- PDF-to-markdown ingestion via Docling with auto-generated frontmatter
- CLI: `wiki sync`, `wiki list`, `wiki search`, `wiki ingest`
- 6 cancer genomics articles ingested from PDFs

**Note:** Phase 4 was done before Phase 3 because wiki search was higher priority for live conversations.

## 2026-04-03 (later session)

### Conversation window
- Added 2-minute conversation timeout so Signet stays engaged without requiring her name every message
- Tracks `_last_response` per channel, checks age on each incoming message

### Diagnosed wiki search triggering issue
- Wiki search was running on the wrong text when user's question and bot trigger were in separate messages
- Root cause: `should_respond` and wiki search both key off `message.content`, so the question must be in the triggering message

## 2026-04-04

### Text file attachment support
- Discord sends pasted files as `message.attachments`, not `message.content`
- Bot now reads any `text/*` attachment, decodes UTF-8, prepends `[attached file: name]`
- Handles .txt, .tsv, .csv and similar

### Phase 3: autoDream memory consolidation
- New `dreams` table in PostgreSQL with embeddings for semantic search
- `consolidated` boolean column added to `messages` table (idempotent ALTER)
- Three dream artifact types: digests (conversation summaries), entity_facts (learned user knowledge), reflections (cross-conversation patterns)
- Dreamer orchestrator: groups unconsolidated messages into conversation bundles by channel + 30-min time gap, sends each through Haiku for structured JSON extraction
- DreamStore: full CRUD + semantic recall + entity lookup + chronological listing
- Dreams recalled alongside raw memories in live prompts (between memory and wiki context layers)
- Prompt framing: "Things you've internalized from past experience (use naturally, these are YOUR thoughts)"
- CLI: `dream run`, `dream status`, `dream list`
- Fallback: if LLM returns invalid JSON, raw response stored as single digest (no crash)

**What worked:** Keeping dream consolidation on Haiku keeps costs negligible. Grouping by channel + time gap produces natural conversation boundaries without requiring the session system (which was never wired into the Discord bot).

**What to watch:** The `consolidated` flag marks ALL messages from a run, even if their specific bundle errored. This prevents infinite reprocessing but means some messages might not get properly consolidated if there's a transient LLM error. Acceptable tradeoff for now.

### Missing: Tests, CLAUDE.md, CHANGELOG.md
- Realized the tests/ directory was completely empty despite "write tests alongside implementation" being a stated preference
- No project-level CLAUDE.md or CHANGELOG.md existed
- Adding all three now
