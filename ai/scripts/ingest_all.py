"""
Quick ingestion script — runs all three ingestion steps in sequence.

Usage:
    python -m scripts.ingest_all
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.api.dependencies import (
    get_json_ingestion_service,
    get_text_ingestion_service,
    get_vector_store,
    get_gpx_parser_service,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ingest_all")


def main() -> None:
    settings = get_settings()

    # 1. JSON knowledge base → SQLite
    kb_path = settings.json_kb_path
    if kb_path.exists():
        logger.info("═══ Ingesting JSON knowledge base ═══")
        svc = get_json_ingestion_service()
        counts = svc.ingest(kb_path)
        logger.info("JSON result: %s", counts)
    else:
        logger.warning("JSON KB not found at %s — skipping", kb_path)

    # 2. Text / Markdown → chunks → FAISS
    docs_dir = settings.text_docs_dir
    if docs_dir.exists() and any(docs_dir.iterdir()):
        logger.info("═══ Ingesting text documents ═══")
        text_svc = get_text_ingestion_service()
        result = text_svc.ingest_directory(docs_dir)
        logger.info("Text result: %s", result)

        chunks = text_svc.get_all_chunks()
        if chunks:
            vs = get_vector_store()
            vs.add_chunks(
                chunk_ids=[c.chunk_id for c in chunks],
                texts=[c.chunk_text for c in chunks],
                metadata=[
                    {"title": c.title, "related_entity_id": c.related_entity_id, "source_file": c.source_file}
                    for c in chunks
                ],
            )
            logger.info("Vectors stored: %d", vs.total_vectors)
    else:
        logger.warning("Docs directory empty or missing: %s — skipping", docs_dir)

    # 3. GPX files → parsed + vector store enrichment
    gpx_dir = settings.gpx_data_dir
    if gpx_dir.exists() and any(gpx_dir.iterdir()):
        logger.info("═══ Ingesting GPX files ═══")
        parser = get_gpx_parser_service()
        parsed = parser.parse_directory(gpx_dir)
        for pf in parsed:
            logger.info(
                "  %s — %s points, %.1f km",
                pf.filename, pf.total_points, pf.total_distance_km or 0,
            )
    else:
        logger.warning("GPX directory empty or missing: %s — skipping", gpx_dir)

    logger.info("═══ All ingestion complete ═══")


if __name__ == "__main__":
    main()
