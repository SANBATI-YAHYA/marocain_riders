"""
Centralised application configuration.

Reads from environment variables (loaded from .env via python-dotenv).
All paths resolve relative to the project root.
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

# ── locate project root (directory containing .env) ──────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """Immutable-ish bag of application settings."""

    # Ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma2:9b")

    # Embedding
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Paths (resolved relative to project root)
    sqlite_db_path: Path = PROJECT_ROOT / os.getenv("SQLITE_DB_PATH", "app/data/morocco_moto.db")
    vector_store_path: Path = PROJECT_ROOT / os.getenv("VECTOR_STORE_PATH", "app/vector_store/faiss_index")
    chunk_metadata_path: Path = PROJECT_ROOT / os.getenv("CHUNK_METADATA_PATH", "app/vector_store/chunk_metadata.json")

    # Data sources
    json_kb_path: Path = PROJECT_ROOT / os.getenv("JSON_KB_PATH", "app/data/json/knowledge_base.json")
    text_docs_dir: Path = PROJECT_ROOT / os.getenv("TEXT_DOCS_DIR", "app/data/docs")
    gpx_data_dir: Path = PROJECT_ROOT / os.getenv("GPX_DATA_DIR", "app/data/gpx")

    # Server
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    return Settings()
