# Signet

Persistent AI research agent. Runs 24/7 on Discord, does autonomous overnight research in bioinformatics/genomics/AI 
and whatever else is fun and useful.


## Stack

- Python 3.12, uv, hatchling
- discord.py, anthropic SDK, pydantic v2, typer, structlog
- PostgreSQL + pgvector for memory and knowledge storage
- sentence-transformers (all-MiniLM-L6-v2, 384d) for local embeddings
- Docling for PDF-to-markdown ingestion
- Character definition in `characters/signet.yaml`

## Architecture

```
src/signet/
  brain/          # Anthropic SDK wrapper (chat, quick)
  character/      # YAML loader, personality sampler, prompt assembler
  memory/         # PostgreSQL + pgvector message store with semantic recall
  knowledge/      # Wiki articles + PDF ingestion pipeline
  nightshift/     # autoDream consolidation + autonomous research + wiki writeback
  interfaces/     # Discord bot
  models/         # Pydantic models (memory, knowledge, dreams)
  providers/      # (placeholder) external service integrations
  evaluators/     # (placeholder) quality assessment
  cli.py          # typer CLI entry point
  config.py       # pydantic-settings from .env
```

## Running

```bash
uv run signet run              # Start Discord bot
uv run signet check            # Validate character YAML
uv run signet db-init          # Initialize all DB schemas
uv run signet wiki sync        # Sync wiki articles to DB
uv run signet wiki ingest      # Convert PDFs to markdown
uv run signet dream run        # Run autoDream consolidation
uv run signet dream status     # Show consolidation status
uv run signet dream list       # Show recent dream artifacts
uv run signet nightshift run   # Manual research trigger
uv run signet nightshift list  # Show recent research artifacts
uv run signet nightshift queue # Add topic to research queue
```

## Phases

1. ~~Core chat~~ (done)
2. ~~Memory persistence~~ (done)
3. ~~autoDream memory consolidation~~ (done)
4. ~~Wiki/knowledge system + PDF ingestion~~ (done)
5. ~~Nightshift autonomous research~~ (done)
6. ~~Research-to-wiki writeback (Karpathy loop)~~ (done)

## Success Criteria

- Chat: Signet responds on DM, @mention, name mention, or within 2-min conversation window
- Memory: all messages stored with embeddings, semantic recall works cross-channel
- Wiki: articles sync from disk, PDF ingestion produces valid markdown, semantic search returns relevant results
- autoDream: consolidation produces digests, entity facts, and reflections; dreams recalled in live prompts
- Nightshift: autonomous research during quiet periods, writes findings back to wiki as markdown, auto-syncs to DB for future context
- Tests: all modules have test coverage, `uv run pytest` passes before every commit

## Constraints

- Signet NEVER guesses or fabricates. Epistemic discipline is non-negotiable.
- All DB schemas are idempotent (CREATE IF NOT EXISTS, ALTER ADD IF NOT EXISTS)
- autoDream uses Haiku (model_light) to keep consolidation costs low
- Embedding model runs locally on CPU/GPU, not via API
- Character personality is defined in YAML, not code
- Wiki lives on Google Drive (`WIKIS_PATH` env var), mounted via rclone at `~/gdrive/Mina-blade18/signet-wiki/`
- Research -> wiki -> DB -> future research: the knowledge loop must stay closed

## Git Rules

- Work on feature branches for non-trivial changes
- Run `uv run pytest` before committing
- Descriptive commit messages (what and why)
- Don't commit .env or credentials

## Testing

```bash
uv run pytest                  # Run all tests
uv run pytest tests/test_dreamer.py -v  # Run specific test file
```

Tests use pytest + pytest-asyncio. DB-dependent tests mock asyncpg; unit tests mock Brain responses. No live DB required for test suite.
