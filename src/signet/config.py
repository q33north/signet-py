"""Configuration and paths."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App settings, loaded from env vars and .env file."""

    anthropic_api_key: str = ""
    discord_token: str = ""
    discord_application_id: str = ""

    # Model routing
    model_light: str = "claude-sonnet-4-6-20250514"
    model_heavy: str = "claude-sonnet-4-6"
    model_deep: str = "claude-opus-4-6"  # requires user confirmation

    # Database
    database_url: str = "postgresql://signet:signet@localhost:5432/signet"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Memory
    memory_recall_limit: int = 5

    # Wiki
    wiki_recall_limit: int = 3
    wiki_min_similarity: float = 0.3

    # Dreams / autoDream
    dream_recall_limit: int = 3

    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    character_path: Path = Path(__file__).parent.parent.parent / "characters" / "signet.yaml"
    wikis_path: Path = Path(__file__).parent.parent.parent / "wikis"
    audit_path: Path = Path(__file__).parent.parent.parent / "audit"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
