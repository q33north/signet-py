"""CLI entry point."""
from __future__ import annotations

import structlog
import typer
from rich.console import Console

from signet.config import settings

app = typer.Typer(help="Signet: persistent AI research agent")
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
    from signet.memory.embeddings import EmbeddingService
    from signet.memory.store import MemoryStore

    log.info("signet.loading_character", path=str(settings.character_path))
    character = load_character(settings.character_path)
    log.info("signet.character_loaded", name=character.name, bio_count=len(character.bio))

    assembler = PromptAssembler(character)
    brain = Brain()
    embedder = EmbeddingService(model_name=settings.embedding_model)
    memory = MemoryStore(database_url=settings.database_url, embedder=embedder)

    console.print(f"[bold green]Starting {character.name}...[/bold green]")
    run_discord_bot(assembler, brain, memory)


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

    from signet.memory.embeddings import EmbeddingService
    from signet.memory.store import MemoryStore

    async def _init() -> None:
        embedder = EmbeddingService(model_name=settings.embedding_model)
        store = MemoryStore(database_url=settings.database_url, embedder=embedder)
        await store.connect()
        await store.initialize_schema()
        await store.close()

    console.print(f"[bold]Connecting to:[/bold] {settings.database_url.split('@')[-1]}")
    asyncio.run(_init())
    console.print("[bold green]Database schema initialized.[/bold green]")


if __name__ == "__main__":
    app()
