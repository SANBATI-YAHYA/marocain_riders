"""
FastAPI route handlers.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.api.dependencies import (
    get_gpx_parser_service,
    get_json_ingestion_service,
    get_ollama_client,
    get_recommendation_engine,
    get_text_ingestion_service,
    get_vector_store,
    _settings,
    get_db,
)
from app.schemas.request import RecommendationRequest
from app.schemas.response import (
    HealthResponse,
    IngestResponse,
    RecommendationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ═══════════════════════════════════════════════════════
#  Health
# ═══════════════════════════════════════════════════════

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """System health check."""
    settings = _settings()
    ollama = get_ollama_client()
    vs = get_vector_store()

    db_ready = False
    try:
        conn = get_db()
        conn.execute("SELECT 1")
        db_ready = True
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version="1.0.0",
        ollama_reachable=ollama.is_reachable(),
        db_ready=db_ready,
        vector_store_ready=vs.is_ready(),
    )


# ═══════════════════════════════════════════════════════
#  Ingestion endpoints
# ═══════════════════════════════════════════════════════

@router.post("/ingest/json", response_model=IngestResponse, tags=["ingestion"])
async def ingest_json():
    """Ingest the structured JSON knowledge base into SQLite."""
    settings = _settings()
    kb_path = settings.json_kb_path
    if not kb_path.exists():
        raise HTTPException(status_code=404, detail=f"Knowledge base not found at {kb_path}")

    svc = get_json_ingestion_service()
    counts = svc.ingest(kb_path)
    total = sum(counts.values())
    return IngestResponse(
        status="ok",
        message=f"Ingested: {counts}",
        records_processed=total,
    )


@router.post("/ingest/text", response_model=IngestResponse, tags=["ingestion"])
async def ingest_text():
    """Ingest text/markdown documents, chunk, embed, and store in FAISS."""
    settings = _settings()
    docs_dir = settings.text_docs_dir
    if not docs_dir.exists():
        raise HTTPException(status_code=404, detail=f"Docs directory not found: {docs_dir}")

    # Step 1: chunk & store text
    text_svc = get_text_ingestion_service()
    result = text_svc.ingest_directory(docs_dir)

    # Step 2: embed chunks and add to FAISS
    chunks = text_svc.get_all_chunks()
    if chunks:
        vs = get_vector_store()
        chunk_ids = [c.chunk_id for c in chunks]
        texts = [c.chunk_text for c in chunks]
        metadata = [
            {
                "title": c.title,
                "related_entity_id": c.related_entity_id,
                "source_file": c.source_file,
            }
            for c in chunks
        ]
        vs.add_chunks(chunk_ids, texts, metadata)

    return IngestResponse(
        status="ok",
        message=f"Files: {result['files']}, Chunks: {result['chunks']}, Vectors: {len(chunks)}",
        records_processed=result["chunks"],
    )


@router.post("/ingest/gpx", response_model=IngestResponse, tags=["ingestion"])
async def ingest_gpx():
    """Parse GPX files and add route descriptions to the vector store."""
    settings = _settings()
    gpx_dir = settings.gpx_data_dir
    if not gpx_dir.exists():
        raise HTTPException(status_code=404, detail=f"GPX directory not found: {gpx_dir}")

    parser = get_gpx_parser_service()
    parsed_files = parser.parse_directory(gpx_dir)

    # Build text descriptions from GPX metadata and add to vector store
    vs = get_vector_store()
    chunk_ids = []
    texts = []
    metadata_list = []

    for pf in parsed_files:
        desc_parts = [f"GPX Route: {pf.route_name or pf.filename}"]
        if pf.total_distance_km:
            desc_parts.append(f"Total distance: {pf.total_distance_km:.1f} km")
        desc_parts.append(f"Total track points: {pf.total_points}")
        if pf.waypoints:
            wp_names = [wp.name for wp in pf.waypoints if wp.name]
            desc_parts.append(f"Waypoints: {', '.join(wp_names[:20])}")
        if pf.nearest_place_ids:
            desc_parts.append(f"Nearest places: {', '.join(pf.nearest_place_ids[:5])}")

        text = "\n".join(desc_parts)
        cid = f"GPX-{pf.filename}"
        chunk_ids.append(cid)
        texts.append(text)
        metadata_list.append({
            "title": pf.route_name or pf.filename,
            "source_file": pf.filename,
        })

    if texts:
        vs.add_chunks(chunk_ids, texts, metadata_list)

    return IngestResponse(
        status="ok",
        message=f"Parsed {len(parsed_files)} GPX file(s), {sum(pf.total_points for pf in parsed_files)} total points",
        records_processed=len(parsed_files),
    )


# ═══════════════════════════════════════════════════════
#  Recommendation
# ═══════════════════════════════════════════════════════

@router.post("/recommend", response_model=RecommendationResponse, tags=["recommendation"])
async def recommend(request: RecommendationRequest):
    """
    Generate a personalised motorcycle trip recommendation.

    Accepts a rider profile and optional free-text preferences.
    Runs structured filtering, scoring, semantic retrieval, and
    LLM generation to produce a full itinerary.
    """
    engine = get_recommendation_engine()
    return engine.recommend(request)
