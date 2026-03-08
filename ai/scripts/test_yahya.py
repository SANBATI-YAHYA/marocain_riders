"""
End-to-end recommendation test for user: Yahya Sanbati.

Profile:
  - 21 years old, from Morocco
  - Bike: CP7 (scooter class)
  - Loves beach / coastal vibes
  - Budget-friendly rider

This script:
  1. Ingests the JSON knowledge base into SQLite
  2. Ingests text docs and builds FAISS vector index
  3. Runs the scoring engine for Yahya's profile
  4. Plans stops deterministically (GPX-driven)
  5. Calls the LLM to explain the pre-built plan
  6. Prints the full grounded recommendation

Usage:
    python scripts/test_yahya.py
"""

import sys
import json
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.core.database import get_connection
from app.services.json_ingestion import JsonIngestionService
from app.services.text_ingestion import TextIngestionService
from app.services.recommendation_engine import RecommendationEngine, score_route
from app.services.ollama_client import OllamaClient
from app.vector_store.faiss_store import VectorStoreService
from app.repositories.place_repo import PlaceRepository
from app.repositories.route_repo import RouteRepository
from app.repositories.entity_repos import (
    RouteSegmentRepository, StayRepository, RestaurantRepository,
    GasStationRepository, WeatherRiskRepository,
)
from app.schemas.request import RecommendationRequest, UserProfile

# ══════════════════════════════════════════════════
#  Setup
# ══════════════════════════════════════════════════
settings = get_settings()
conn = get_connection(settings.sqlite_db_path)

print("=" * 60)
print("  MOROCCO MOTO RECOMMENDER v2 - Yahya Sanbati")
print("=" * 60)

# ── Step 1: Ingest JSON KB ───────────────────────
print("\n[1/5] Ingesting JSON knowledge base...")
place_repo = PlaceRepository(conn)
route_repo = RouteRepository(conn)
segment_repo = RouteSegmentRepository(conn)
stay_repo = StayRepository(conn)
restaurant_repo = RestaurantRepository(conn)
gas_station_repo = GasStationRepository(conn)
weather_risk_repo = WeatherRiskRepository(conn)

json_svc = JsonIngestionService(
    place_repo, route_repo, segment_repo,
    stay_repo, restaurant_repo, gas_station_repo, weather_risk_repo,
)
kb_path = settings.json_kb_path
if kb_path.exists():
    counts = json_svc.ingest(kb_path)
    print(f"  Ingested: {counts}")
else:
    print(f"  ERROR: KB not found at {kb_path}")
    sys.exit(1)

# ── Step 2: Ingest text & build vectors ──────────
print("\n[2/5] Ingesting text docs & building vector index...")
text_svc = TextIngestionService(conn=conn)
docs_dir = settings.text_docs_dir

if docs_dir.exists():
    result = text_svc.ingest_directory(docs_dir)
    print(f"  Files: {result['files']}, Chunks: {result['chunks']}")

    chunks = text_svc.get_all_chunks()
    if chunks:
        vs = VectorStoreService(
            index_path=settings.vector_store_path,
            metadata_path=settings.chunk_metadata_path,
            embedding_model=settings.embedding_model,
        )
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
        print(f"  Vectors in FAISS: {vs.total_vectors}")
else:
    print("  No docs directory found, skipping text ingestion")
    vs = VectorStoreService(
        index_path=settings.vector_store_path,
        metadata_path=settings.chunk_metadata_path,
    )

# ══════════════════════════════════════════════════
#  Yahya's Profile
# ══════════════════════════════════════════════════
print("\n[3/5] Building Yahya's rider profile...")

yahya_profile = UserProfile(
    rider_experience="beginner",
    bike_type="scooter",
    group_size=1,
    budget_per_day_eur=30,
    trip_duration_days=3,
    preferred_vibes=["coastal", "scenic"],
    sleep_preference="budget/auberge",
    daily_ride_km_tolerance=120,
    travel_month="April",
    country_of_origin="Morocco",
)

print(f"""
  Rider: Yahya Sanbati (21 years)
  Bike: CP7 (scooter)
  Experience: {yahya_profile.rider_experience}
  Budget: EUR{yahya_profile.budget_per_day_eur}/day
  Duration: {yahya_profile.trip_duration_days} days
  Vibes: {', '.join(yahya_profile.preferred_vibes)}
  Sleep: {yahya_profile.sleep_preference}
  Daily km: {yahya_profile.daily_ride_km_tolerance} km max
  Travel month: {yahya_profile.travel_month}
""")

# ══════════════════════════════════════════════════
#  Step 3: Score all routes
# ══════════════════════════════════════════════════
print("[3/5] Scoring all routes for Yahya's profile...\n")

all_places = place_repo.get_all()
all_routes = route_repo.get_all()

scored = []
for route in all_routes:
    total, breakdown = score_route(route, all_places, yahya_profile)
    scored.append((route, total, breakdown))

scored.sort(key=lambda x: x[1], reverse=True)

print("  Route Rankings:")
print("  " + "-" * 56)
for i, (r, score, bd) in enumerate(scored, 1):
    marker = " << BEST >>" if i == 1 else ""
    print(f"  #{i}  {r.route_name:<35} Score: {score:.3f}{marker}")
    print(f"       Diff: {bd['difficulty']:.2f}  Bike: {bd['bike_suit']:.2f}  "
          f"Vibe: {bd['vibe']:.2f}  Dur: {bd['duration']:.2f}  "
          f"Fuel: {bd['fuel_safety']:.2f}  Season: {bd['season']:.2f}")

# ══════════════════════════════════════════════════
#  Step 4: Ollama health check
# ══════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  [4/5] Checking Ollama & model availability...")
print(f"{'='*60}")

ollama = OllamaClient(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
)

ollama_ok = ollama.is_reachable()
model_ok = ollama.is_model_available() if ollama_ok else False
print(f"\n  Ollama reachable:  {ollama_ok}")
print(f"  Model available:   {model_ok} ({settings.ollama_model})")
if ollama_ok and not model_ok:
    available = ollama.list_models()
    print(f"  Available models:  {available}")
    print(f"  --> Run:  ollama pull {settings.ollama_model}")
if not ollama_ok:
    print("  (LLM explanation will be a fallback message)")

# ══════════════════════════════════════════════════
#  Step 5: Full recommendation pipeline
# ══════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  [5/5] Running full recommendation pipeline...")
print(f"{'='*60}")

request = RecommendationRequest(
    user_profile=yahya_profile,
    additional_preferences="I love the beach and coastal roads. Relaxed trip with good food.",
)

engine = RecommendationEngine(
    place_repo=place_repo,
    route_repo=route_repo,
    segment_repo=segment_repo,
    stay_repo=stay_repo,
    restaurant_repo=restaurant_repo,
    gas_station_repo=gas_station_repo,
    weather_risk_repo=weather_risk_repo,
    vector_store=vs,
    ollama=ollama,
)

response = engine.recommend(request)

# ══════════════════════════════════════════════════
#  Print Results
# ══════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  RECOMMENDATION RESULTS FOR YAHYA")
print(f"{'='*60}")

if response.recommended_route:
    rr = response.recommended_route
    print(f"\n  Recommended Route: {rr.route_name}")
    print(f"  Route ID:          {rr.route_id}")
    print(f"  Score:             {rr.total_score:.3f}")
    print(f"  Distance:          {rr.distance_km} km")
    print(f"  Difficulty:        {rr.difficulty_level}")
    print(f"  Days:              {rr.recommended_days}")
    print(f"  Why:               {rr.why_choose}")

if response.alternative_routes:
    print(f"\n  Alternative Routes:")
    for alt in response.alternative_routes:
        print(f"    - {alt.route_name} (score: {alt.total_score:.3f}, {alt.distance_km}km)")

# ── Vibe match notes (new) ───────────────────────
if response.vibe_match_notes:
    print(f"\n  Vibe Match Notes:")
    for note in response.vibe_match_notes:
        print(f"    [NOTE] {note}")

# ── Day plan ─────────────────────────────────────
if response.day_plan:
    print(f"\n  Day-by-Day Plan:")
    for day in response.day_plan:
        print(f"    Day {day.day_number}: {day.start_place} -> {day.end_place} ({day.distance_km}km)")
        for stop in day.stops:
            print(f"      [{stop.stop_type.upper()}] {stop.name} -- {stop.notes or ''}")
else:
    print("\n  Day plan: (none built)")

# ── Planned stop points (new, map-ready) ─────────
if response.stop_points:
    print(f"\n  Planned Stop Points ({len(response.stop_points)}):")
    for sp in response.stop_points:
        coords = f"({sp.latitude:.3f}, {sp.longitude:.3f})" if sp.latitude else "(no coords)"
        print(f"    [{sp.stop_type.upper():7s}] {sp.name:<30s} day {sp.estimated_day}  "
              f"{sp.priority:8s}  {coords}  {sp.reason[:60]}")

# ── Legacy flat lists ────────────────────────────
if response.food_stops:
    print(f"\n  Food Stops:")
    for fs in response.food_stops:
        print(f"    [FOOD] {fs.name} ({fs.notes})")

if response.stay_stops:
    print(f"\n  Where to Sleep:")
    for ss in response.stay_stops:
        print(f"    [SLEEP] {ss.name} ({ss.notes})")

if response.fuel_stops:
    print(f"\n  Fuel Stops:")
    for gs in response.fuel_stops:
        print(f"    [FUEL] {gs.name} -- {gs.notes}")

if response.weather_warnings:
    print(f"\n  Weather Warnings:")
    for w in response.weather_warnings:
        print(f"    [WARN] {w}")

if response.safety_notes:
    print(f"\n  Safety Notes:")
    for sn in response.safety_notes:
        print(f"    [SAFE] {sn}")

if response.retrieved_context_summary:
    print(f"\n  Semantic Context ({len(response.retrieved_context_summary)} chunks, route-filtered):")
    for ctx in response.retrieved_context_summary[:3]:
        print(f"    [{ctx.related_entity_id or '?'}] {ctx.snippet[:80]}...")

print(f"\n  LLM Explanation:")
print("  " + "-" * 56)
explanation = response.llm_explanation or "No LLM output"
for line in explanation.split("\n"):
    print(f"  {line}")

print(f"\n  Metadata: {response.metadata}")

print(f"\n{'='*60}")
print("  TEST COMPLETE")
print(f"{'='*60}")

# Save full response as JSON
output_path = PROJECT_ROOT / "test_output_yahya.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(response.model_dump(), f, indent=2, ensure_ascii=False, default=str)
print(f"\n  Full JSON response saved to: {output_path}")
