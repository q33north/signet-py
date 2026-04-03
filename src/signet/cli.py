"""CLI entry point."""
from __future__ import annotations

import structlog
import typer
from rich.console import Console
from rich.table import Table

from signet.config import settings

app = typer.Typer(help="Signet: persistent AI research agent")
wiki_app = typer.Typer(help="Wiki knowledge management")
app.add_typer(wiki_app, name="wiki")
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

    console.print(f"[bold green]Starting {character.name}...[/bold green]")
    run_discord_bot(assembler, brain, memory, wiki)


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

    async def _init() -> None:
        embedder = EmbeddingService(model_name=settings.embedding_model)
        memory = MemoryStore(database_url=settings.database_url, embedder=embedder)
        wiki = WikiStore(
            wikis_path=settings.wikis_path,
            database_url=settings.database_url,
            embedder=embedder,
        )
        await memory.connect()
        await memory.initialize_schema()
        await wiki.connect()
        await wiki.initialize_schema()
        await wiki.close()
        await memory.close()

    console.print(f"[bold]Connecting to:[/bold] {settings.database_url.split('@')[-1]}")
    asyncio.run(_init())
    console.print("[bold green]Database schema initialized (memory + wiki).[/bold green]")


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


if __name__ == "__main__":
    app()
