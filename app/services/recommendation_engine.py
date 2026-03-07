"""
Core recommendation engine.

Combines structured filtering, scoring, semantic retrieval,
context assembly, and LLM generation into one pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from app.models.domain import Route, Place, Stay, Restaurant, GasStation, WeatherRisk, RouteSegment
from app.schemas.request import RecommendationRequest, UserProfile
from app.schemas.response import (
    DayPlan,
    RecommendationResponse,
    RetrievedContextSummary,
    RouteCandidateScore,
    StopSummary,
)
from app.repositories.place_repo import PlaceRepository
from app.repositories.route_repo import RouteRepository
from app.repositories.entity_repos import (
    GasStationRepository,
    RestaurantRepository,
    RouteSegmentRepository,
    StayRepository,
    WeatherRiskRepository,
)
from app.vector_store.faiss_store import VectorStoreService
from app.services.ollama_client import OllamaClient
from app.services.prompt_builder import SYSTEM_PROMPT, build_recommendation_prompt

logger = logging.getLogger(__name__)

# ── difficulty mapping ────────────────────────────────

_DIFFICULTY_ORDER = {
    "easy": 1, "easy-intermediate": 2, "intermediate": 3,
    "intermediate-advanced": 4, "advanced": 5, "expert": 6,
}
_EXPERIENCE_TO_MAX = {
    "beginner": 2, "intermediate": 3, "advanced": 5, "expert": 6,
}

# ── budget tiers ──────────────────────────────────────

_BUDGET_TIERS = {
    "very low": 0, "low": 1, "low-medium": 2, "medium": 3,
    "medium-high": 4, "high": 5,
}


# ═══════════════════════════════════════════════════════
#  Scoring helpers
# ═══════════════════════════════════════════════════════

def _score_difficulty(route: Route, profile: UserProfile) -> float:
    """0-1 score — 1.0 if difficulty perfectly matches experience."""
    route_diff = _DIFFICULTY_ORDER.get(route.difficulty_level or "", 3)
    max_diff = _EXPERIENCE_TO_MAX.get(profile.rider_experience, 3)
    if route_diff > max_diff:
        return 0.0  # too hard
    gap = max_diff - route_diff
    return max(0.0, 1.0 - gap * 0.25)


def _score_bike_suitability(route: Route, profile: UserProfile) -> float:
    suits = [s.lower() for s in route.bike_type_suitability]
    bike = profile.bike_type.lower()
    if "all types" in suits or "all" in suits:
        return 1.0
    for s in suits:
        if bike in s or s in bike:
            return 1.0
    return 0.2


def _score_vibe(route: Route, places: List[Place], profile: UserProfile) -> float:
    """How well do the route's places match desired vibes."""
    if not profile.preferred_vibes or "mixed" in profile.preferred_vibes:
        return 0.7  # neutral
    wanted = set(v.lower() for v in profile.preferred_vibes)

    # ── Synonym / related-vibe expansion ─────────
    _VIBE_SYNONYMS: Dict[str, List[str]] = {
        "coastal": ["ocean", "beach", "sea", "atlantic", "mediterranean", "fishing"],
        "scenic": ["dramatic", "views", "epic views", "canyon", "gorge", "winding road", "high altitude"],
        "beach": ["coastal", "ocean", "sea", "atlantic", "fishing"],
        "nature": ["mountain", "gorge", "desert", "oasis", "palmery", "canyon"],
        "culture": ["authentic", "souks", "kasbah", "medina", "culture", "film"],
        "adventure": ["adventure", "remote", "wild", "dunes", "erg", "bivouac"],
        "relaxed": ["calm", "authentic", "transition", "oasis"],
    }

    all_place_ids = set(route.via_place_ids or [])
    if route.start_place_id:
        all_place_ids.add(route.start_place_id)
    if route.end_place_id:
        all_place_ids.add(route.end_place_id)

    place_map = {p.place_id: p for p in places}
    route_vibes = set()
    scenic_scores: List[int] = []
    for pid in all_place_ids:
        p = place_map.get(pid)
        if p:
            if p.primary_vibe:
                route_vibes.add(p.primary_vibe.lower())
            route_vibes.update(v.lower() for v in p.secondary_vibes)
            if hasattr(p, "scenic_score") and p.scenic_score is not None:
                scenic_scores.append(p.scenic_score)

    if not route_vibes:
        return 0.5

    # Direct overlap
    overlap = wanted & route_vibes

    # Expanded overlap via synonyms
    for w in wanted:
        if w in overlap:
            continue
        synonyms = _VIBE_SYNONYMS.get(w, [])
        for syn in synonyms:
            if syn in route_vibes:
                overlap.add(w)
                break

    base = len(overlap) / len(wanted) if wanted else 0.5

    # Bonus for high scenic scores when user wants scenic/coastal vibes
    scenic_vibes = {"scenic", "coastal", "beach", "nature"}
    if wanted & scenic_vibes and scenic_scores:
        avg_scenic = sum(scenic_scores) / len(scenic_scores)
        scenic_bonus = min(0.2, (avg_scenic / 10.0) * 0.2)
        base = min(1.0, base + scenic_bonus)

    return base


def _score_duration(route: Route, profile: UserProfile) -> float:
    """Does the route fit in the rider's trip duration?"""
    rec = route.recommended_days or "1"
    # Parse "7-10" → take midpoint
    parts = rec.replace(" ", "").split("-")
    try:
        if len(parts) == 2:
            mid = (int(parts[0]) + int(parts[1])) / 2
        else:
            mid = float(parts[0])
    except ValueError:
        mid = 5
    avail = profile.trip_duration_days
    if mid > avail + 2:
        return 0.1  # way too long
    if mid > avail:
        return 0.5
    return 1.0


def _score_fuel_safety(route: Route) -> float:
    gap = route.fuel_gap_km or 0
    if gap <= 60:
        return 1.0
    if gap <= 100:
        return 0.8
    if gap <= 150:
        return 0.5
    return 0.3


def _score_season(route: Route, travel_month: Optional[str]) -> float:
    if not travel_month:
        return 0.7
    month_lower = travel_month.lower()[:3]
    avoid = (route.avoid_season or "").lower()
    best = (route.best_season or "").lower()
    if month_lower in avoid:
        return 0.1
    if month_lower in best:
        return 1.0
    return 0.6


def score_route(
    route: Route,
    places: List[Place],
    profile: UserProfile,
) -> Tuple[float, Dict[str, float]]:
    """Return (total_score, breakdown_dict) for a route candidate."""
    weights = {
        "difficulty":  0.20,
        "bike_suit":   0.15,
        "vibe":        0.15,
        "duration":    0.20,
        "fuel_safety": 0.10,
        "season":      0.20,
    }

    scores = {
        "difficulty":  _score_difficulty(route, profile),
        "bike_suit":   _score_bike_suitability(route, profile),
        "vibe":        _score_vibe(route, places, profile),
        "duration":    _score_duration(route, profile),
        "fuel_safety": _score_fuel_safety(route),
        "season":      _score_season(route, profile.travel_month),
    }

    total = sum(scores[k] * weights[k] for k in weights)
    return round(total, 4), {k: round(v, 3) for k, v in scores.items()}


# ═══════════════════════════════════════════════════════
#  Main engine
# ═══════════════════════════════════════════════════════

class RecommendationEngine:
    """Orchestrates filtering → scoring → retrieval → generation."""

    def __init__(
        self,
        place_repo: PlaceRepository,
        route_repo: RouteRepository,
        segment_repo: RouteSegmentRepository,
        stay_repo: StayRepository,
        restaurant_repo: RestaurantRepository,
        gas_station_repo: GasStationRepository,
        weather_risk_repo: WeatherRiskRepository,
        vector_store: VectorStoreService,
        ollama: OllamaClient,
    ) -> None:
        self._place_repo = place_repo
        self._route_repo = route_repo
        self._segment_repo = segment_repo
        self._stay_repo = stay_repo
        self._restaurant_repo = restaurant_repo
        self._gas_station_repo = gas_station_repo
        self._weather_risk_repo = weather_risk_repo
        self._vector_store = vector_store
        self._ollama = ollama

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        """Full recommendation pipeline."""
        profile = request.user_profile
        places = self._place_repo.get_all()
        routes = self._route_repo.get_all()

        # ── Step 1: Score & rank routes ──────────────
        scored: List[Tuple[Route, float, Dict]] = []
        for route in routes:
            total, breakdown = score_route(route, places, profile)
            scored.append((route, total, breakdown))
        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            return RecommendationResponse(
                llm_explanation="No routes found in the knowledge base.",
                metadata={"error": "empty_kb"},
            )

        top_route, top_score, top_breakdown = scored[0]
        alternatives = scored[1:4]

        # ── Step 2: Retrieve related entities ────────
        segments = self._segment_repo.get_by_route(top_route.route_id)
        stays = self._stay_repo.get_by_ids(top_route.recommended_stay_ids)
        restaurants = self._restaurant_repo.get_by_ids(top_route.recommended_restaurant_ids)
        fuel_stations = self._gas_station_repo.get_by_ids(top_route.recommended_fuel_station_ids)
        weather_risks = self._weather_risk_repo.get_for_route(top_route.route_id)

        # Filter weather by season if travel month given
        weather_warnings = self._format_weather_warnings(weather_risks, profile.travel_month)

        # ── Step 3: Semantic retrieval ───────────────
        search_query = (
            f"motorcycle route {top_route.route_name} Morocco "
            f"{profile.rider_experience} rider {profile.bike_type} "
            f"{' '.join(profile.preferred_vibes)}"
        )
        semantic_hits = self._vector_store.search(search_query, top_k=5)
        semantic_chunks = [h.get("text_preview", "") for h in semantic_hits]
        context_summaries = [
            RetrievedContextSummary(
                chunk_id=h.get("chunk_id", ""),
                title=h.get("title"),
                related_entity_id=h.get("related_entity_id"),
                snippet=h.get("text_preview", "")[:200],
            )
            for h in semantic_hits
        ]

        # ── Step 4: Build stop summaries ─────────────
        food_stops = self._make_food_stops(restaurants)
        stay_stops = self._make_stay_stops(stays)
        fuel_stop_summaries = self._make_fuel_stops(fuel_stations)

        # ── Step 5: Build day plan skeleton ──────────
        day_plan = self._build_day_plan(top_route, segments, stays, profile)

        # ── Step 6: Build route candidate scores ─────
        def _make_candidate(r: Route, s: float, bd: Dict) -> RouteCandidateScore:
            return RouteCandidateScore(
                route_id=r.route_id,
                route_name=r.route_name,
                total_score=s,
                score_breakdown=bd,
                distance_km=r.distance_km,
                difficulty_level=r.difficulty_level,
                recommended_days=r.recommended_days,
                why_choose=r.why_choose_this_route,
            )

        recommended = _make_candidate(top_route, top_score, top_breakdown)
        alt_candidates = [_make_candidate(r, s, bd) for r, s, bd in alternatives]

        # ── Step 7: LLM generation ───────────────────
        profile_summary = self._format_profile(profile)
        prompt = build_recommendation_prompt(
            profile_summary=profile_summary,
            top_route=top_route.model_dump(),
            alternative_routes=[r.model_dump() for r, _, _ in alternatives],
            segments_text=self._format_segments(segments),
            stays_text=self._format_stays(stays),
            restaurants_text=self._format_restaurants(restaurants),
            fuel_stations_text=self._format_fuel_stations(fuel_stations),
            weather_warnings=weather_warnings,
            semantic_chunks=semantic_chunks,
            additional_preferences=request.additional_preferences,
        )

        llm_text = self._ollama.generate(prompt, system=SYSTEM_PROMPT, max_tokens=512)

        # ── Assemble response ────────────────────────
        return RecommendationResponse(
            recommended_route=recommended,
            alternative_routes=alt_candidates,
            day_plan=day_plan,
            food_stops=food_stops,
            stay_stops=stay_stops,
            fuel_stops=fuel_stop_summaries,
            weather_warnings=weather_warnings,
            safety_notes=[top_route.safety_advice] if top_route.safety_advice else [],
            llm_explanation=llm_text,
            retrieved_context_summary=context_summaries,
            metadata={
                "model": self._ollama._model,
                "total_routes_scored": len(scored),
                "semantic_chunks_used": len(semantic_chunks),
            },
        )

    # ── formatting helpers ───────────────────────────

    @staticmethod
    def _format_profile(p: UserProfile) -> str:
        return (
            f"Experience: {p.rider_experience}\n"
            f"Bike type: {p.bike_type}\n"
            f"Group size: {p.group_size}\n"
            f"Budget: €{p.budget_per_day_eur}/day\n"
            f"Trip duration: {p.trip_duration_days} days\n"
            f"Preferred vibes: {', '.join(p.preferred_vibes)}\n"
            f"Sleep preference: {p.sleep_preference}\n"
            f"Daily km tolerance: {p.daily_ride_km_tolerance} km\n"
            f"Travel month: {p.travel_month or 'not specified'}\n"
            f"Country: {p.country_of_origin or 'not specified'}"
        )

    @staticmethod
    def _format_segments(segments: List[RouteSegment]) -> str:
        lines = []
        for s in segments:
            lines.append(
                f"Seg {s.segment_order}: {s.segment_name} — "
                f"{s.distance_km}km, {s.estimated_duration}, "
                f"surface: {s.surface_type}, difficulty: {s.difficulty_level}. "
                f"Hazards: {s.hazard_notes or 'none'}. "
                f"Scenic: {s.scenic_notes or '-'}. "
                f"Stop: {s.stop_recommendation or '-'}"
            )
        return "\n".join(lines) if lines else "No segment data available."

    @staticmethod
    def _format_stays(stays: List[Stay]) -> str:
        lines = []
        for s in stays:
            lines.append(
                f"- {s.name} ({s.stay_type}, {s.price_range}) at {s.place_id}. "
                f"Parking: {'secure' if s.secure_parking else 'basic'}. "
                f"Shower: {'hot' if s.hot_shower else 'cold/basic'}. "
                f"Booking: {s.booking_recommended or 'unknown'}."
            )
        return "\n".join(lines) if lines else "No accommodation data."

    @staticmethod
    def _format_restaurants(restaurants: List[Restaurant]) -> str:
        lines = []
        for r in restaurants:
            lines.append(
                f"- {r.name} ({r.cuisine_type}, {r.price_level}) at {r.place_id}. "
                f"Meals: {', '.join(r.meal_type)}. "
                f"Quick stop: {'yes' if r.quick_stop_friendly else 'no'}."
            )
        return "\n".join(lines) if lines else "No restaurant data."

    @staticmethod
    def _format_fuel_stations(stations: List[GasStation]) -> str:
        lines = []
        for g in stations:
            lines.append(
                f"- {g.name} ({', '.join(g.fuel_types)}) at {g.place_id}. "
                f"24h: {'yes' if g.open_24h else 'no'}. "
                f"Gap note: {g.critical_gap_note or '-'}."
            )
        return "\n".join(lines) if lines else "No fuel station data."

    @staticmethod
    def _format_weather_warnings(
        risks: List[WeatherRisk], travel_month: Optional[str]
    ) -> List[str]:
        warnings: List[str] = []
        for wr in risks:
            month_match = True
            if travel_month and wr.season:
                month_lower = travel_month.lower()[:3]
                season_lower = wr.season.lower()
                month_match = month_lower in season_lower or not season_lower
            if month_match:
                warnings.append(
                    f"[{wr.severity or '?'}] {wr.risk_type}: {wr.description} "
                    f"— Advice: {wr.rider_advice}"
                )
        return warnings

    @staticmethod
    def _make_food_stops(restaurants: List[Restaurant]) -> List[StopSummary]:
        return [
            StopSummary(
                name=r.name,
                entity_id=r.restaurant_id,
                stop_type="meal",
                latitude=r.latitude,
                longitude=r.longitude,
                notes=f"{r.cuisine_type}, {r.price_level}",
            )
            for r in restaurants
        ]

    @staticmethod
    def _make_stay_stops(stays: List[Stay]) -> List[StopSummary]:
        return [
            StopSummary(
                name=s.name,
                entity_id=s.stay_id,
                stop_type="sleep",
                latitude=s.latitude,
                longitude=s.longitude,
                notes=f"{s.stay_type}, {s.price_range}",
            )
            for s in stays
        ]

    @staticmethod
    def _make_fuel_stops(stations: List[GasStation]) -> List[StopSummary]:
        return [
            StopSummary(
                name=g.name,
                entity_id=g.station_id,
                stop_type="fuel",
                latitude=g.latitude,
                longitude=g.longitude,
                notes=g.critical_gap_note,
            )
            for g in stations
        ]

    def _build_day_plan(
        self,
        route: Route,
        segments: List[RouteSegment],
        stays: List[Stay],
        profile: UserProfile,
    ) -> List[DayPlan]:
        """Build a skeleton day plan from segments, respecting daily km tolerance."""
        if not segments:
            return []

        daily_max = profile.daily_ride_km_tolerance or 150
        days: List[DayPlan] = []
        day_num = 1
        day_km = 0.0
        day_segments: List[RouteSegment] = []

        place_lookup = {s.place_id: s.name for s in stays}

        for seg in segments:
            seg_km = seg.distance_km or 0
            if day_km + seg_km > daily_max and day_segments:
                # Close current day
                start_name = day_segments[0].segment_name.split(" to ")[0] if day_segments else "?"
                end_name = day_segments[-1].segment_name.split(" to ")[-1] if day_segments else "?"
                days.append(DayPlan(
                    day_number=day_num,
                    start_place=start_name,
                    end_place=end_name,
                    distance_km=round(day_km, 1),
                    difficulty=max(
                        (s.difficulty_level or "easy" for s in day_segments),
                        key=lambda d: _DIFFICULTY_ORDER.get(d, 0),
                    ),
                    stops=self._segment_stops(day_segments),
                ))
                day_num += 1
                day_km = 0.0
                day_segments = []

            day_segments.append(seg)
            day_km += seg_km

        # Close last day
        if day_segments:
            start_name = day_segments[0].segment_name.split(" to ")[0]
            end_name = day_segments[-1].segment_name.split(" to ")[-1]
            days.append(DayPlan(
                day_number=day_num,
                start_place=start_name,
                end_place=end_name,
                distance_km=round(day_km, 1),
                difficulty=max(
                    (s.difficulty_level or "easy" for s in day_segments),
                    key=lambda d: _DIFFICULTY_ORDER.get(d, 0),
                ),
                stops=self._segment_stops(day_segments),
            ))

        return days

    @staticmethod
    def _segment_stops(segments: List[RouteSegment]) -> List[StopSummary]:
        stops: List[StopSummary] = []
        for seg in segments:
            if seg.stop_recommendation:
                stops.append(StopSummary(
                    name=seg.segment_name or "waypoint",
                    entity_id=seg.nearest_place_id,
                    stop_type="waypoint",
                    latitude=seg.end_latitude,
                    longitude=seg.end_longitude,
                    notes=seg.stop_recommendation,
                ))
        return stops
