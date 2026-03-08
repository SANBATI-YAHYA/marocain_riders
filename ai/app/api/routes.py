"""
FastAPI route handlers.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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
from app.schemas.request import RecommendationRequest, UserProfile
from app.schemas.response import (
    HealthResponse,
    IngestResponse,
    RecommendationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Profile directory (PROJECT_ROOT / profiles) ───────
_PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"
_PROFILES_DIR.mkdir(exist_ok=True)


def _safe_filename(name: str) -> str:
    """Turn a profile name into a safe filesystem slug."""
    slug = re.sub(r"[^a-z0-9_\-]", "_", name.strip().lower())
    return slug or "default"


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


# ═══════════════════════════════════════════════════════
#  Profile CRUD (local JSON files)
# ═══════════════════════════════════════════════════════

class ProfilePayload(BaseModel):
    """Full profile data sent from the frontend."""
    full_name: str
    age: Optional[int] = None
    country_of_origin: Optional[str] = None
    rider_experience: str = "beginner"
    bike_type: str = "scooter"
    group_size: int = 1
    budget_per_day_eur: float = 60.0
    trip_duration_days: int = 7
    preferred_vibes: List[str] = ["mixed"]
    sleep_preference: str = "mid-range"
    daily_ride_km_tolerance: int = 150
    travel_month: Optional[str] = None
    # optional extras stored but not mapped 1-to-1 to UserProfile
    food_preferences: Optional[str] = None
    avoidances: Optional[str] = None
    road_surface_preference: Optional[str] = None
    parking_requirement: Optional[str] = None
    start_city: Optional[str] = None
    end_city: Optional[str] = None


@router.post("/profiles", tags=["profiles"])
async def save_profile(payload: ProfilePayload):
    """Save a rider profile as a local JSON file."""
    slug = _safe_filename(payload.full_name)
    path = _PROFILES_DIR / f"{slug}.json"
    data = payload.model_dump()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Profile saved → %s", path)
    return {"status": "ok", "profile_id": slug, "path": str(path)}


@router.get("/profiles", tags=["profiles"])
async def list_profiles():
    """List all saved profile JSON files."""
    files = sorted(_PROFILES_DIR.glob("*.json"))
    profiles = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            profiles.append({
                "profile_id": f.stem,
                "full_name": data.get("full_name", f.stem),
                "rider_experience": data.get("rider_experience", ""),
                "bike_type": data.get("bike_type", ""),
            })
        except Exception:
            pass
    return {"profiles": profiles}


@router.get("/profiles/{profile_id}", tags=["profiles"])
async def get_profile(profile_id: str):
    """Load a single saved profile."""
    path = _PROFILES_DIR / f"{profile_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


@router.delete("/profiles/{profile_id}", tags=["profiles"])
async def delete_profile(profile_id: str):
    """Delete a saved profile."""
    path = _PROFILES_DIR / f"{profile_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
    path.unlink()
    return {"status": "ok", "deleted": profile_id}


# ═══════════════════════════════════════════════════════
#  Route geometry endpoint
# ═══════════════════════════════════════════════════════

@router.get("/routes/{route_id}/geometry", tags=["routes"])
async def get_route_geometry(route_id: str):
    """Return simplified GPX polyline for a route."""
    from app.repositories.route_repo import RouteRepository
    conn = get_db()
    route_repo = RouteRepository(conn)

    routes = route_repo.get_all()
    route = next((r for r in routes if r.route_id == route_id), None)
    if not route:
        raise HTTPException(status_code=404, detail=f"Route '{route_id}' not found")
    if not route.linked_gpx_name:
        return {"route_id": route_id, "polyline": [], "message": "No linked GPX file"}

    parser = get_gpx_parser_service()
    geometry = parser.get_route_geometry(route.linked_gpx_name, max_points=800)
    if geometry is None:
        return {"route_id": route_id, "polyline": [], "message": "GPX file not found"}

    return {
        "route_id": route_id,
        "route_name": route.route_name,
        "polyline": geometry,
        "points_count": len(geometry),
    }


# ═══════════════════════════════════════════════════════
#  Combined: profile + trip description → recommendation
# ═══════════════════════════════════════════════════════

class TripRequest(BaseModel):
    """Frontend sends profile_id + free-text trip description."""
    profile_id: str
    trip_description: str = ""


@router.post("/trip/recommend", response_model=RecommendationResponse, tags=["recommendation"])
async def trip_recommend(req: TripRequest):
    """
    Load a saved profile, combine it with the trip description,
    and run the existing recommendation engine.
    """
    # 1. Load profile JSON
    path = _PROFILES_DIR / f"{req.profile_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Profile '{req.profile_id}' not found")

    data = json.loads(path.read_text(encoding="utf-8"))

    # 2. Build UserProfile from saved data
    profile = UserProfile(
        rider_experience=data.get("rider_experience", "beginner"),
        bike_type=data.get("bike_type", "scooter"),
        group_size=data.get("group_size", 1),
        budget_per_day_eur=data.get("budget_per_day_eur", 60),
        trip_duration_days=data.get("trip_duration_days", 7),
        preferred_vibes=data.get("preferred_vibes", ["mixed"]),
        sleep_preference=data.get("sleep_preference", "mid-range"),
        daily_ride_km_tolerance=data.get("daily_ride_km_tolerance", 150),
        travel_month=data.get("travel_month"),
        country_of_origin=data.get("country_of_origin"),
    )

    # 3. Build recommendation request — trip_description becomes additional_preferences
    rec_request = RecommendationRequest(
        user_profile=profile,
        additional_preferences=req.trip_description or None,
    )

    # 4. Run existing engine
    engine = get_recommendation_engine()
    return engine.recommend(rec_request)


# ═══════════════════════════════════════════════════════
#  Auto-ingest (called once at startup)
# ═══════════════════════════════════════════════════════

def run_auto_ingest():
    """Ingest JSON KB + text docs if the DB appears empty.

    Safe to call multiple times — ingestion is idempotent.
    """
    settings = _settings()
    conn = get_db()

    # Check if places table has data
    try:
        row = conn.execute("SELECT COUNT(*) FROM places").fetchone()
        if row and row[0] > 0:
            logger.info("Auto-ingest: DB already populated (%d places). Skipping JSON ingest.", row[0])
            json_done = True
        else:
            json_done = False
    except Exception:
        json_done = False

    if not json_done:
        kb_path = settings.json_kb_path
        if kb_path.exists():
            svc = get_json_ingestion_service()
            counts = svc.ingest(kb_path)
            logger.info("Auto-ingest JSON: %s", counts)
        else:
            logger.warning("Auto-ingest: KB file not found at %s", kb_path)

    # Check if vectors exist
    vs = get_vector_store()
    if vs.total_vectors > 0:
        logger.info("Auto-ingest: FAISS already has %d vectors. Skipping text ingest.", vs.total_vectors)
        return

    docs_dir = settings.text_docs_dir
    if docs_dir.exists():
        text_svc = get_text_ingestion_service()
        result = text_svc.ingest_directory(docs_dir)
        chunks = text_svc.get_all_chunks()
        if chunks:
            vs.clear()
            vs.add_chunks(
                chunk_ids=[c.chunk_id for c in chunks],
                texts=[c.chunk_text for c in chunks],
                metadata=[
                    {"title": c.title, "related_entity_id": c.related_entity_id,
                     "source_file": c.source_file}
                    for c in chunks
                ],
            )
            logger.info("Auto-ingest text: %d chunks, %d vectors", len(chunks), vs.total_vectors)
    else:
        logger.warning("Auto-ingest: Text docs dir not found at %s", docs_dir)
