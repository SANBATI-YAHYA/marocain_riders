"""
FastAPI application entry point.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.database import close_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    settings = get_settings()
    logger.info("Morocco Moto Recommendation API starting…")
    logger.info("  Ollama: %s (model: %s)", settings.ollama_base_url, settings.ollama_model)
    logger.info("  SQLite: %s", settings.sqlite_db_path)
    logger.info("  FAISS:  %s", settings.vector_store_path)
    yield
    close_connection()
    logger.info("API shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Morocco Motorcycle Travel Recommender",
        description=(
            "AI-powered motorcycle trip recommendation engine for Morocco. "
            "Combines structured data, vector search, GPX routes, and LLM generation."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — open for local dev / future React frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
