# Signet

Persistent AI research agent. Runs 24/7 on Discord, does autonomous overnight research in bioinformatics, genomics, proteomics, AI, and whatever else is interesting. Writes findings back to a compounding knowledge wiki.

Built by [Q33 North](https://github.com/q33north).

## How it works

Signet is a Discord bot with a personality, long-term memory, and an autonomous research pipeline that runs during quiet periods. The core loop:

1. **Chat** - responds to DMs, @mentions, and name mentions with context-aware conversation
2. **Memory** - stores all messages with semantic embeddings for cross-channel recall
3. **autoDream** - consolidates conversations into digests, entity facts, and reflections
4. **Wiki** - maintains a persistent markdown knowledge base with YAML frontmatter
5. **Nightshift** - autonomous multi-step research: topic selection, planning, deep dives, synthesis
6. **Writeback** - research findings are written back to the wiki as markdown, synced to DB with embeddings, and available as context for future research

Research compounds over time. Each session builds on what came before.

## Stack

- Python 3.12, [uv](https://github.com/astral-sh/uv)
- discord.py, Anthropic SDK, Pydantic v2, Typer, structlog
- PostgreSQL + pgvector for memory, knowledge, and research storage
- sentence-transformers (all-MiniLM-L6-v2, 384d) for local embeddings
- Docling for PDF/PPTX/DOCX to markdown ingestion
- Character personality defined in YAML (`characters/signet.yaml`)

## Install

```bash
# Clone
git clone https://github.com/q33north/signet-py.git
cd signet-py

# Install with uv (recommended)
uv sync --extra db --extra ingest --extra dev

# Or with pip
pip install -e ".[db,ingest,dev]"
```

### Prerequisites

- Python 3.11+
- PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) extension
- Discord bot token ([guide](https://discordpy.readthedocs.io/en/stable/discord.html))
- Anthropic API key

### Configuration

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
DISCORD_TOKEN=your-bot-token
DISCORD_APPLICATION_ID=your-app-id
DATABASE_URL=postgresql://signet:signet@localhost:5432/signet

# Nightshift (optional)
NIGHTSHIFT_ENABLED=true
NIGHTSHIFT_CHANNEL_ID=your-channel-id
NIGHTSHIFT_QUIET_MINUTES=30
NIGHTSHIFT_MAX_SESSIONS=3

# Wiki location (optional, defaults to ./wikis/)
WIKIS_PATH=/path/to/your/wiki/directory
```

### Database setup

```bash
# Create the database
createdb signet

# Initialize all schemas (idempotent, safe to re-run)
uv run signet db-init
```

## Commands

### Core

| Command | Description |
|---------|-------------|
| `signet run` | Start the Discord bot |
| `signet check` | Validate character YAML and config |
| `signet db-init` | Initialize database schemas |

### Wiki

The wiki is a collection of markdown files with YAML frontmatter, organized by topic into subdirectories. Files are synced to PostgreSQL with embeddings for semantic search.

| Command | Description |
|---------|-------------|
| `signet wiki sync` | Sync markdown files from disk to database, re-embed changed files |
| `signet wiki list` | List all indexed articles |
| `signet wiki search QUERY` | Semantic search across articles (e.g. `signet wiki search "KRAS resistance"`) |
| `signet wiki ingest [--force]` | Convert PDFs/PPTX/DOCX in `raw/` subdirectories to markdown via Docling |

Wiki directory structure:
```
wikis/
  cancer_genomics/
    kras-g12c-resistance.md
    raw/                      # Source documents for ingestion
      paper.pdf
  proteomics/
    demichev-dia-nn.md
```

### autoDream

Memory consolidation system. Groups recent conversations into bundles and extracts structured knowledge: digests (conversation summaries), entity facts (learned information), and reflections (cross-conversation patterns).

| Command | Description |
|---------|-------------|
| `signet dream run` | Run consolidation on unconsolidated messages |
| `signet dream status` | Show consolidation stats (total messages, pending, last run) |
| `signet dream list` | List recent dream artifacts |

### Nightshift

Autonomous research engine. Runs during quiet Discord periods or on-demand. Selects topics, generates sub-questions, does deep dives with context from wiki/dreams/prior research, and synthesizes findings. Results are posted to Discord and written back to the wiki.

| Command | Description |
|---------|-------------|
| `signet nightshift run` | Manually trigger a research session |
| `signet nightshift status` | Show daily token budget, session count, queue status |
| `signet nightshift list` | List recent research artifacts with UUIDs |
| `signet nightshift queue TOPIC [-f FOLDER]` | Add topic to research queue, optionally specifying wiki folder |
| `signet nightshift repost [--id UUID] [--topic STR] [--channel ID]` | Re-post research to Discord |

Examples:
```bash
# Queue research into an existing wiki folder
signet nightshift queue "KRAS G12C resistance mechanisms" -f cancer_genomics

# Queue research (creates new wiki folder from topic name)
signet nightshift queue "transformer architectures for protein structure"

# Repost by topic substring
signet nightshift repost --topic proteomics

# Repost to a specific channel
signet nightshift repost --id abc123 --channel 1234567890
```

## Wiki on Google Drive

The wiki directory can be pointed at any filesystem path via the `WIKIS_PATH` environment variable. To make the wiki accessible from multiple machines, mount Google Drive with [rclone](https://rclone.org/):

```bash
# Mount Google Drive
rclone mount gdrive: ~/gdrive --vfs-cache-mode full --daemon

# Set wiki path in .env
WIKIS_PATH=/home/you/gdrive/signet-wiki
```

Nightshift writes research directly to the mounted path. Changes sync to Drive automatically.

## Testing

```bash
uv run pytest           # Run all tests
uv run pytest -v        # Verbose output
uv run pytest tests/test_wiki_writer.py  # Specific test file
```

Tests use pytest + pytest-asyncio. All database interactions are mocked. No live DB required.

## License

MIT
