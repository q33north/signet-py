# Signet

**Prototype lab research agent.** An experimental Discord-based AI agent that does autonomous overnight research in bioinformatics, 
cancer genomics, proteomics, and AI. Writes findings back to a compounding markdown knowledge wiki. Has a personality.

> ⚠️ **Prototype, not production.** This is a personal research project exploring the "LLM + persistent wiki" pattern for compounding 
> knowledge. It's under active development, APIs and schemas will change, and it makes no guarantees about data durability, cost controls, 
> or multi-user safety. Use it for hacking and learning, not for anything you can't afford to lose.

Built by [Q33 North](https://github.com/q33north).

## The idea

If you've got compute sitting idle overnight and projects with well-defined success criteria, every night the agents aren't 
working is potential progress left on the table. Signet is an experiment in *research while you sleep*: an autonomous agent that 
picks up threads from the day's conversations, investigates them overnight, and leaves structured notes in a shared wiki that a 
whole lab can query.

The pattern sits between two existing ideas: Anthropic's KAIROS (daemon-mode Claude) handles autonomous memory consolidation, but 
targets coding, and Karpathy's [LLM-wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), which builds compounding 
knowledge, but only when directed. Neither quite fits a lab where researchers want an agent that surfaces relevant preprints 
overnight and maintains a queryable knowledge base without a bioinformatician manually compiling it.

Signet takes a stab at that gap with a four-tier knowledge layer:

1. **Raw memory** (episodic). Every message embedded in Postgres + pgvector for semantic recall.
2. **Consolidated dreams** (semantic). autoDream extracts digests, entity facts, and reflections from conversations.
3. **Curated knowledge** (reference). Markdown wiki articles plus PDF/DOCX/PPTX ingestion via Docling.
4. **Research artifacts** (generated). Nightshift deep dives with provenance, confidence levels, and open questions that feed the next run.

A few design commitments that should make the output trustworthy:

- **Epistemic discipline as architecture**, not just prompting. Confidence ratings, attributed facts, and explicit "I don't know" live 
in the data models, not just the system prompt.
- **Character as config**. Personality is defined in YAML, tunable without touching code. (Implements ideas from ElizaOS)
- **Interface-agnostic responder**. Discord as of now, but CLI, Slack, or a web UI later without much re-plumbing of the agent.

Each nightshift session sees existing wiki folders, prior research, and recent conversations, so topics compound over time instead of 
scattering into unrelated one-offs.

For the long version, read the launch post: [Hello Signet](https://q33north.substack.com/p/hello-signet).

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
| `signet nightshift queue TOPIC [-f FOLDER] [-b BRIEF]` | Add topic to research queue with optional wiki folder and research brief |
| `signet nightshift repost [--id UUID] [--topic STR] [--channel ID]` | Re-post research to Discord |

Examples:
```bash
# Queue research into an existing wiki folder
signet nightshift queue "KRAS G12C resistance mechanisms" -f cancer_genomics

# Queue with a detailed research brief (markdown file with goals, references, constraints)
signet nightshift queue "Agentic AI for cancer labs" -f agentic-cancer-lab -b brief.md

# Queue research (creates new wiki folder from topic name)
signet nightshift queue "transformer architectures for protein structure"

# Repost by topic substring
signet nightshift repost --topic proteomics

# Repost to a specific channel
signet nightshift repost --id abc123 --channel 1234567890
```

## Workflow: starting a new topic

The typical flow for seeding a new research area with source papers and letting Signet dig in overnight:

```bash
# 1. Create the topic folder in your wiki
mkdir -p $WIKIS_PATH/spatial-transcriptomics/raw

# 2. Drop source documents (PDF, PPTX, or DOCX) into raw/
cp ~/Downloads/stahl-2016-visium.pdf $WIKIS_PATH/spatial-transcriptomics/raw/
cp ~/Downloads/vickovic-hd.pdf $WIKIS_PATH/spatial-transcriptomics/raw/

# 3. Ingest — Docling converts raw/ docs to markdown and syncs to the DB with embeddings
uv run signet wiki ingest

# 4. Queue a nightshift run into that folder
uv run signet nightshift queue "Spatial transcriptomics resolution limits" -f spatial-transcriptomics

# 5a. Wait for the autonomous trigger (quiet Discord + NIGHTSHIFT_ENABLED), or:
# 5b. Fire it manually:
uv run signet nightshift run
```

When nightshift runs, it pulls the ingested papers as context for planning and deep dives, then writes the synthesis back into that same folder as a sibling `.md` file. Subsequent runs see the synthesis too, so the topic compounds.

### Optional: a research brief for richer direction

For non-trivial topics, pass a markdown brief with `-b`. It gets injected as the highest-priority context, overriding generic topic framing:

```bash
cat > brief.md <<'EOF'
# Goal
Understand whether Visium HD's 2μm bins actually deliver single-cell resolution.

# Key questions
- How does diffusion limit effective resolution?
- What's the real cell-capture rate vs paired scRNA-seq?

# Constraints
- Focus on lung tissue if possible
- Skip brain/mouse-only papers

# References
- Stahl 2016, Vickovic HD 2024
EOF

uv run signet nightshift queue "Visium HD resolution in practice" -f spatial-transcriptomics -b brief.md
```

### Gotchas

- `wiki ingest` is idempotent: already-converted files are skipped unless you pass `--force`.
- The `-f` folder name must match the directory you created exactly (lowercase, kebab-case). A typo creates a new sibling folder.
- Only one queued item gets picked up per autonomous trigger. Queue several and they'll process on successive quiet periods.

> 💡 This protocol is a lot of typing. Driving it from Discord or Slack directly (e.g. "@signet, research this folder and these papers") is on the roadmap.

### Tools

In live conversation Signet can invoke tools via the Anthropic tool-use API when she decides they're needed. No user syntax required — just mention a URL, paper, or file and she reaches for the right one.

| Tool | What it does |
|------|--------------|
| `fetch_url` | Fetch a web page and extract readable text. GitHub URLs (`blob/...` and bare repo URLs) are rewritten to `raw.githubusercontent.com` so she reads source/README, not the JS-rendered UI. |
| `pubmed_search` | Search PubMed and return titles, abstracts, authors, DOIs, PMC IDs. |
| `biorxiv_search` | Keyword search over recent bioRxiv preprints (last N days). |
| `read_file` / `list_directory` / `search_files` / `file_info` | Filesystem access, sandboxed to `ALLOWED_PATHS`. |

Toggle tools globally with `TOOLS_ENABLED=false` in `.env`. Web tools require no API keys.

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

## Status and limitations

Signet is a working prototype, not a polished product. A few things to know:

- **Single-user by design right now.** Memory and wiki are scoped to one operator; there's no multi-user isolation.
- **Cost controls are basic.** Daily token budgets for nightshift exist (`NIGHTSHIFT_DAILY_TOKEN_BUDGET`) but there's no hard circuit breaker per run.
- **No fact-checking layer.** The character prompt instructs the agent not to fabricate and to flag low confidence, and nightshift cites sources from PubMed/bioRxiv/wiki context, but the agent can still be wrong. Treat output as a research lead, not a citation.
- **Wiki maintenance is manual.** Periodic sweeps for contradictions, stale claims, or folder consolidation are on the roadmap, not yet built.
- **Schemas will change.** Database migrations are idempotent `CREATE IF NOT EXISTS` / `ALTER ADD IF NOT EXISTS`, but there's no formal migration story yet.
- **Discord-specific.** The chat surface is Discord; there's no web UI. A companion project for browsing the wiki is in early work.

Contributions, issues, and forks are welcome.

## License

MIT. See [LICENSE](LICENSE).
