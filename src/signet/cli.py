"""CLI entry point."""
from __future__ import annotations

import structlog
import typer
from rich.console import Console
from rich.table import Table

from signet.config import settings

app = typer.Typer(help="Signet: persistent AI research agent")
wiki_app = typer.Typer(help="Wiki knowledge management")
dream_app = typer.Typer(help="autoDream memory consolidation")
nightshift_app = typer.Typer(help="Nightshift autonomous research")
app.add_typer(wiki_app, name="wiki")
app.add_typer(dream_app, name="dream")
app.add_typer(nightshift_app, name="nightshift")
console = Console()


@app.command()
def run() -> None:
    """Start Signet and connect to Discord."""
    structlog.configure(
        processors=[
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),
    )
    log = structlog.get_logger()

    if not settings.anthropic_api_key:
        console.print("[red]ANTHROPIC_API_KEY not set in .env[/red]")
        raise typer.Exit(1)

    if not settings.discord_token:
        console.print("[red]DISCORD_TOKEN not set in .env[/red]")
        raise typer.Exit(1)

    # Late imports to avoid loading everything for --help
    from signet.brain.client import Brain
    from signet.character.loader import load_character
    from signet.character.prompt import PromptAssembler
    from signet.interfaces.discord import run_discord_bot
    from signet.knowledge.store import WikiStore
    from signet.memory.embeddings import EmbeddingService
    from signet.memory.store import MemoryStore
    from signet.nightshift.research_store import ResearchStore
    from signet.nightshift.store import DreamStore

    log.info("signet.loading_character", path=str(settings.character_path))
    character = load_character(settings.character_path)
    log.info("signet.character_loaded", name=character.name, bio_count=len(character.bio))

    assembler = PromptAssembler(character)
    brain = Brain()
    embedder = EmbeddingService(model_name=settings.embedding_model)
    memory = MemoryStore(database_url=settings.database_url, embedder=embedder)
    wiki = WikiStore(
        wikis_path=settings.wikis_path,
        database_url=settings.database_url,
        embedder=embedder,
    )
    dreams = DreamStore(database_url=settings.database_url, embedder=embedder)
    research = ResearchStore(database_url=settings.database_url, embedder=embedder)

    console.print(f"[bold green]Starting {character.name}...[/bold green]")
    run_discord_bot(assembler, brain, memory, wiki, dreams, research)


@app.command()
def check() -> None:
    """Validate character YAML and config without starting."""
    from signet.character.loader import load_character

    try:
        character = load_character(settings.character_path)
        console.print(f"[green]Character:[/green] {character.name}")
        console.print(f"[green]Bio traits:[/green] {len(character.bio)}")
        console.print(f"[green]Examples:[/green] {len(character.message_examples)}")
        console.print(f"[green]Style.all:[/green] {len(character.style.all)}")
        console.print(f"[green]Style.chat:[/green] {len(character.style.chat)}")
        console.print(f"[green]Adjectives:[/green] {len(character.adjectives)}")
        console.print(f"[green]Topics:[/green] {len(character.topics)}")
        console.print("\n[bold green]Character valid.[/bold green]")
    except Exception as e:
        console.print(f"[red]Error loading character:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def db_init() -> None:
    """Initialize the database schema. Idempotent, safe to run multiple times."""
    import asyncio

    from signet.knowledge.store import WikiStore
    from signet.memory.embeddings import EmbeddingService
    from signet.memory.store import MemoryStore
    from signet.nightshift.research_store import ResearchStore
    from signet.nightshift.store import DreamStore

    async def _init() -> None:
        embedder = EmbeddingService(model_name=settings.embedding_model)
        memory = MemoryStore(database_url=settings.database_url, embedder=embedder)
        wiki = WikiStore(
            wikis_path=settings.wikis_path,
            database_url=settings.database_url,
            embedder=embedder,
        )
        dreams = DreamStore(database_url=settings.database_url, embedder=embedder)
        research = ResearchStore(database_url=settings.database_url, embedder=embedder)
        await memory.connect()
        await memory.initialize_schema()
        await wiki.connect()
        await wiki.initialize_schema()
        await dreams.connect()
        await dreams.initialize_schema()
        await research.connect()
        await research.initialize_schema()
        await research.close()
        await dreams.close()
        await wiki.close()
        await memory.close()

    console.print(f"[bold]Connecting to:[/bold] {settings.database_url.split('@')[-1]}")
    asyncio.run(_init())
    console.print("[bold green]Database schema initialized (memory + wiki + dreams + research).[/bold green]")


# ── Wiki subcommands ────────────────────────────────────────


@wiki_app.command()
def sync() -> None:
    """Sync wiki articles from disk to database. Re-embeds changed files."""
    import asyncio

    from signet.knowledge.store import WikiStore
    from signet.memory.embeddings import EmbeddingService

    async def _sync() -> dict[str, int]:
        embedder = EmbeddingService(model_name=settings.embedding_model)
        store = WikiStore(
            wikis_path=settings.wikis_path,
            database_url=settings.database_url,
            embedder=embedder,
        )
        await store.connect()
        await store.initialize_schema()
        result = await store.sync()
        await store.close()
        return result

    console.print(f"[bold]Scanning:[/bold] {settings.wikis_path}")
    result = asyncio.run(_sync())
    console.print(
        f"[green]added={result['added']} updated={result['updated']} "
        f"removed={result['removed']}[/green]"
    )


@wiki_app.command("list")
def list_articles() -> None:
    """List all indexed wiki articles."""
    import asyncio

    from signet.knowledge.store import WikiStore
    from signet.memory.embeddings import EmbeddingService

    async def _list() -> list[dict]:
        embedder = EmbeddingService(model_name=settings.embedding_model)
        store = WikiStore(
            wikis_path=settings.wikis_path,
            database_url=settings.database_url,
            embedder=embedder,
        )
        await store.connect()
        articles = await store.list_articles()
        await store.close()
        return articles

    articles = asyncio.run(_list())
    if not articles:
        console.print("[dim]No wiki articles indexed. Add .md files to wikis/ and run: signet wiki sync[/dim]")
        return

    table = Table(title="Wiki Articles")
    table.add_column("Slug", style="cyan")
    table.add_column("Title")
    table.add_column("Tags", style="dim")
    table.add_column("Updated")

    for a in articles:
        tags = ", ".join(a["tags"]) if a["tags"] else ""
        updated = str(a["updated_at"].date()) if a["updated_at"] else ""
        table.add_row(a["slug"], a["title"], tags, updated)

    console.print(table)


@wiki_app.command()
def search(query: str, limit: int = 5) -> None:
    """Semantic search across wiki articles."""
    import asyncio

    from signet.knowledge.store import WikiStore
    from signet.memory.embeddings import EmbeddingService

    async def _search() -> list:
        embedder = EmbeddingService(model_name=settings.embedding_model)
        store = WikiStore(
            wikis_path=settings.wikis_path,
            database_url=settings.database_url,
            embedder=embedder,
        )
        await store.connect()
        results = await store.search(query, limit=limit, min_similarity=0.0)
        await store.close()
        return results

    results = asyncio.run(_search())
    if not results:
        console.print("[dim]No results.[/dim]")
        return

    for r in results:
        sim = f"{r.similarity:.3f}"
        console.print(f"[cyan]{r.article.slug}[/cyan] ({sim}) - {r.article.frontmatter.title}")
        if r.article.frontmatter.summary:
            console.print(f"  [dim]{r.article.frontmatter.summary}[/dim]")


@wiki_app.command()
def ingest(force: bool = typer.Option(False, "--force", help="Re-convert even if .md exists")) -> None:
    """Convert PDFs in raw/ directories to markdown via Docling."""
    import asyncio

    from signet.knowledge.ingest import ingest_raw
    from signet.knowledge.store import WikiStore
    from signet.memory.embeddings import EmbeddingService

    console.print(f"[bold]Ingesting from:[/bold] {settings.wikis_path}/*/raw/")
    result = ingest_raw(settings.wikis_path, force=force)
    console.print(
        f"[green]converted={result['converted']} skipped={result['skipped']} "
        f"errored={result['errored']}[/green]"
    )

    if result["converted"] > 0:
        console.print("[bold]Syncing new articles to database...[/bold]")

        async def _sync() -> dict[str, int]:
            embedder = EmbeddingService(model_name=settings.embedding_model)
            store = WikiStore(
                wikis_path=settings.wikis_path,
                database_url=settings.database_url,
                embedder=embedder,
            )
            await store.connect()
            await store.initialize_schema()
            sync_result = await store.sync()
            await store.close()
            return sync_result

        sync_result = asyncio.run(_sync())
        console.print(
            f"[green]added={sync_result['added']} updated={sync_result['updated']} "
            f"removed={sync_result['removed']}[/green]"
        )


# ── Dream subcommands ──────────────────────────────────────


@dream_app.command("run")
def dream_run(
    max_messages: int = typer.Option(500, help="Max messages to process per run"),
) -> None:
    """Run autoDream memory consolidation now."""
    import asyncio

    structlog.configure(
        processors=[structlog.dev.ConsoleRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(20),
    )

    from signet.brain.client import Brain
    from signet.memory.embeddings import EmbeddingService
    from signet.memory.store import MemoryStore
    from signet.nightshift.dreamer import Dreamer
    from signet.nightshift.store import DreamStore

    async def _dream():
        embedder = EmbeddingService(model_name=settings.embedding_model)
        memory = MemoryStore(database_url=settings.database_url, embedder=embedder)
        dreams = DreamStore(database_url=settings.database_url, embedder=embedder)
        brain = Brain()

        await memory.connect()
        await dreams.connect()
        await dreams.initialize_schema()

        dreamer = Dreamer(memory=memory, dreams=dreams, brain=brain)
        report = await dreamer.dream(max_messages=max_messages)

        await dreams.close()
        await memory.close()
        return report

    console.print("[bold]Starting autoDream consolidation...[/bold]")
    report = asyncio.run(_dream())

    if report.total_dreams == 0:
        console.print("[dim]Nothing to consolidate.[/dim]")
    else:
        console.print(
            f"[green]Processed {report.messages_processed} messages "
            f"across {report.sessions_processed} conversations[/green]"
        )
        console.print(
            f"[green]Produced: {report.digests} digests, "
            f"{report.entity_facts} entity facts, "
            f"{report.reflections} reflections[/green]"
        )


@dream_app.command("status")
def dream_status() -> None:
    """Show dream consolidation status."""
    import asyncio

    from signet.memory.embeddings import EmbeddingService
    from signet.memory.store import MemoryStore
    from signet.nightshift.store import DreamStore

    async def _status():
        embedder = EmbeddingService(model_name=settings.embedding_model)
        memory = MemoryStore(database_url=settings.database_url, embedder=embedder)
        dreams = DreamStore(database_url=settings.database_url, embedder=embedder)
        await memory.connect()
        await dreams.connect()
        await dreams.initialize_schema()

        pending = await memory.unconsolidated_count()
        last = await dreams.last_dream_time()
        counts = await dreams.count_by_type()

        await dreams.close()
        await memory.close()
        return pending, last, counts

    pending, last, counts = asyncio.run(_status())

    console.print(f"[bold]Messages awaiting consolidation:[/bold] {pending}")
    if last:
        console.print(f"[bold]Last dream run:[/bold] {last.strftime('%Y-%m-%d %H:%M UTC')}")
    else:
        console.print("[dim]No dreams yet.[/dim]")

    if counts:
        total = sum(counts.values())
        console.print(f"[bold]Total dreams:[/bold] {total}")
        for dtype, cnt in sorted(counts.items()):
            console.print(f"  {dtype}: {cnt}")


@dream_app.command("list")
def dream_list(
    limit: int = typer.Option(20, help="Max dreams to show"),
    dtype: str = typer.Option("", "--type", help="Filter by type: digest, entity_fact, reflection"),
) -> None:
    """List recent dream artifacts."""
    import asyncio

    from signet.models.dreams import DreamType
    from signet.nightshift.store import DreamStore
    from signet.memory.embeddings import EmbeddingService

    dream_type = DreamType(dtype) if dtype else None

    async def _list():
        embedder = EmbeddingService(model_name=settings.embedding_model)
        store = DreamStore(database_url=settings.database_url, embedder=embedder)
        await store.connect()
        await store.initialize_schema()
        results = await store.recent(limit=limit, dream_type=dream_type)
        await store.close()
        return results

    dreams = asyncio.run(_list())
    if not dreams:
        console.print("[dim]No dreams yet.[/dim]")
        return

    table = Table(title="Recent Dreams")
    table.add_column("Type", style="cyan", width=12)
    table.add_column("Content", max_width=80)
    table.add_column("Entity", style="dim", width=15)
    table.add_column("Created", width=16)

    for d in dreams:
        content = d.content[:80] + "..." if len(d.content) > 80 else d.content
        created = d.created_at.strftime("%Y-%m-%d %H:%M")
        table.add_row(d.dream_type.value, content, d.entity_name, created)

    console.print(table)


# ── Nightshift subcommands ─────────────────────────────────


@nightshift_app.command("run")
def nightshift_run() -> None:
    """Manually trigger a nightshift research session."""
    import asyncio

    structlog.configure(
        processors=[structlog.dev.ConsoleRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(20),
    )

    from signet.brain.client import Brain
    from signet.knowledge.store import WikiStore
    from signet.memory.embeddings import EmbeddingService
    from signet.memory.store import MemoryStore
    from signet.nightshift.research_store import ResearchStore
    from signet.nightshift.researcher import Researcher
    from signet.nightshift.store import DreamStore

    async def _run():
        embedder = EmbeddingService(model_name=settings.embedding_model)
        memory = MemoryStore(database_url=settings.database_url, embedder=embedder)
        wiki = WikiStore(
            wikis_path=settings.wikis_path,
            database_url=settings.database_url,
            embedder=embedder,
        )
        dreams = DreamStore(database_url=settings.database_url, embedder=embedder)
        research = ResearchStore(database_url=settings.database_url, embedder=embedder)
        brain = Brain()

        await memory.connect()
        await wiki.connect()
        await dreams.connect()
        await research.connect()
        await research.initialize_schema()

        researcher = Researcher(brain, memory, wiki, dreams, research)
        report = await researcher.run()

        await research.close()
        await dreams.close()
        await wiki.close()
        await memory.close()
        return report

    console.print("[bold]Starting nightshift research...[/bold]")
    report = asyncio.run(_run())

    if report.status.value == "completed":
        console.print(
            f"[green]Researched: {report.topic}[/green]\n"
            f"[green]Sections: {report.sections_completed} | "
            f"Tokens: {report.total_tokens:,} | "
            f"Time: {report.duration_seconds:.0f}s[/green]"
        )
    else:
        console.print(f"[yellow]Status: {report.status.value}[/yellow]")
        if report.topic:
            console.print(f"[dim]Topic: {report.topic}[/dim]")


@nightshift_app.command("status")
def nightshift_status() -> None:
    """Show nightshift research status."""
    import asyncio

    from signet.memory.embeddings import EmbeddingService
    from signet.nightshift.research_store import ResearchStore

    async def _status():
        embedder = EmbeddingService(model_name=settings.embedding_model)
        store = ResearchStore(database_url=settings.database_url, embedder=embedder)
        await store.connect()
        await store.initialize_schema()

        counts = await store.count_by_status()
        tokens = await store.total_tokens_today()
        queue_len = await store.queue_length()

        await store.close()
        return counts, tokens, queue_len

    counts, tokens, queue_len = asyncio.run(_status())

    console.print(f"[bold]Nightshift enabled:[/bold] {settings.nightshift_enabled}")
    console.print(f"[bold]Channel:[/bold] {settings.nightshift_channel_id or '(not set)'}")
    console.print(f"[bold]Queue:[/bold] {queue_len} pending")
    console.print(f"[bold]Tokens today:[/bold] {tokens:,} / {settings.nightshift_daily_token_budget:,}")

    if counts:
        console.print("[bold]Research artifacts:[/bold]")
        for status, cnt in sorted(counts.items()):
            console.print(f"  {status}: {cnt}")


@nightshift_app.command("list")
def nightshift_list(
    limit: int = typer.Option(10, help="Max results to show"),
) -> None:
    """List recent research artifacts."""
    import asyncio

    from signet.memory.embeddings import EmbeddingService
    from signet.nightshift.research_store import ResearchStore

    async def _list():
        embedder = EmbeddingService(model_name=settings.embedding_model)
        store = ResearchStore(database_url=settings.database_url, embedder=embedder)
        await store.connect()
        await store.initialize_schema()
        results = await store.recent(limit=limit)
        await store.close()
        return results

    artifacts = asyncio.run(_list())
    if not artifacts:
        console.print("[dim]No research yet.[/dim]")
        return

    table = Table(title="Recent Research")
    table.add_column("Topic", max_width=40)
    table.add_column("Status", style="cyan", width=12)
    table.add_column("Sections", width=8)
    table.add_column("Tokens", width=10)
    table.add_column("Started", width=16)

    for a in artifacts:
        started = a.started_at.strftime("%Y-%m-%d %H:%M")
        table.add_row(
            a.topic[:40],
            a.status.value,
            str(len(a.sections)),
            f"{a.token_count:,}",
            started,
        )

    console.print(table)


@nightshift_app.command("queue")
def nightshift_queue(
    topic: str = typer.Argument(..., help="Topic to research"),
) -> None:
    """Add a topic to the research queue."""
    import asyncio

    from signet.memory.embeddings import EmbeddingService
    from signet.nightshift.research_store import ResearchStore

    async def _queue():
        embedder = EmbeddingService(model_name=settings.embedding_model)
        store = ResearchStore(database_url=settings.database_url, embedder=embedder)
        await store.connect()
        await store.initialize_schema()
        item_id = await store.enqueue(topic, requested_by="cli")
        await store.close()
        return item_id

    item_id = asyncio.run(_queue())
    console.print(f"[green]Queued:[/green] {topic}")
    console.print(f"[dim]ID: {item_id}[/dim]")


if __name__ == "__main__":
    app()
