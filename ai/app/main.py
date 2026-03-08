"""
FastAPI application entry point.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router, run_auto_ingest
from app.core.config import get_settings
from app.core.database import close_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Frontend directory
_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    settings = get_settings()
    logger.info("Morocco Moto Recommendation API starting…")
    logger.info("  Gemini model: %s", settings.gemini_model)
    logger.info("  SQLite: %s", settings.sqlite_db_path)
    logger.info("  FAISS:  %s", settings.vector_store_path)

    # Auto-ingest data on first startup
    try:
        run_auto_ingest()
    except Exception as e:
        logger.error("Auto-ingest failed: %s", e)

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

    # CORS — open for local dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(router)

    # Serve frontend static files
    if _FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_frontend():
            return FileResponse(str(_FRONTEND_DIR / "index.html"))

    return app


app = create_app()
