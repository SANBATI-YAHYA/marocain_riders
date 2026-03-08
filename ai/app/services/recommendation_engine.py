"""
Core recommendation engine.

Combines structured filtering, scoring, semantic retrieval,
context assembly, and LLM generation into one pipeline.

KEY DESIGN RULES:
- Route selection is purely deterministic (scoring engine).
- Day planning is deterministic (stop planner + segment splitting).
- The LLM only *explains* — it never selects routes or invents stops.
- Semantic retrieval is filtered to the selected route's entities.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from app.models.domain import (
    GasStation, Place, Restaurant, Route, RouteSegment, Stay, WeatherRisk,
)
from app.schemas.request import RecommendationRequest, UserProfile
from app.schemas.response import (
    DayPlan,
    RecommendationResponse,
    RetrievedContextSummary,
    RouteCandidateScore,
    RouteStopPoint,
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
from app.services.stop_planner import StopPlanner, PlannedStop

logger = logging.getLogger(__name__)

# ── difficulty mapping ────────────────────────────────

_DIFFICULTY_ORDER = {
    "easy": 1, "easy-intermediate": 2, "intermediate": 3,
    "intermediate-advanced": 4, "advanced": 5, "expert": 6,
}
_EXPERIENCE_TO_MAX = {
    "beginner": 2, "intermediate": 3, "advanced": 5, "expert": 6,
}

# ── vibe synonym map (used by scorer AND mismatch detector) ──

_VIBE_SYNONYMS: Dict[str, List[str]] = {
    "coastal": ["ocean", "beach", "sea", "atlantic", "mediterranean", "fishing"],
    "scenic": ["dramatic", "views", "epic views", "canyon", "gorge", "winding road", "high altitude", "blue city", "rif"],
    "beach": ["coastal", "ocean", "sea", "atlantic", "fishing"],
    "nature": ["mountain", "gorge", "desert", "oasis", "palmery", "canyon", "forest", "rif", "hiking"],
    "culture": ["authentic", "souks", "kasbah", "medina", "culture", "film", "imperial", "crafts"],
    "adventure": ["adventure", "remote", "wild", "dunes", "erg", "bivouac", "hiking"],
    "relaxed": ["calm", "authentic", "transition", "oasis"],
    "mountain": ["rif", "atlas", "hiking", "forest", "high altitude", "winding road", "mountain_town"],
    "hiking": ["rif", "forest", "mountain", "nature", "trekking", "trails"],
}


# ═══════════════════════════════════════════════════════
#  Scoring helpers
# ═══════════════════════════════════════════════════════

def _score_difficulty(route: Route, profile: UserProfile) -> float:
    route_diff = _DIFFICULTY_ORDER.get(route.difficulty_level or "", 3)
    max_diff = _EXPERIENCE_TO_MAX.get(profile.rider_experience, 3)
    if route_diff > max_diff:
        return 0.0
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


def _collect_route_vibes(route: Route, places: Dict[str, Place]) -> set:
    """Collect all vibes from a route's associated places."""
    all_pids = set(route.via_place_ids or [])
    if route.start_place_id:
        all_pids.add(route.start_place_id)
    if route.end_place_id:
        all_pids.add(route.end_place_id)
    vibes = set()
    for pid in all_pids:
        p = places.get(pid)
        if p:
            if p.primary_vibe:
                vibes.add(p.primary_vibe.lower())
            vibes.update(v.lower() for v in p.secondary_vibes)
    return vibes


def _score_vibe(route: Route, places: List[Place], profile: UserProfile) -> float:
    if not profile.preferred_vibes or "mixed" in profile.preferred_vibes:
        return 0.7
    wanted = set(v.lower() for v in profile.preferred_vibes)
    place_map = {p.place_id: p for p in places}
    route_vibes = _collect_route_vibes(route, place_map)

    if not route_vibes:
        return 0.5

    # Direct overlap
    overlap = wanted & route_vibes
    # Expanded overlap via synonyms
    for w in wanted:
        if w in overlap:
            continue
        for syn in _VIBE_SYNONYMS.get(w, []):
            if syn in route_vibes:
                overlap.add(w)
                break

    base = len(overlap) / len(wanted) if wanted else 0.5

    # Scenic-score bonus
    scenic_vibes = {"scenic", "coastal", "beach", "nature"}
    if wanted & scenic_vibes:
        scores = []
        for pid in set(route.via_place_ids or []):
            p = place_map.get(pid)
            if p and p.scenic_score is not None:
                scores.append(p.scenic_score)
        if scores:
            base = min(1.0, base + min(0.2, (sum(scores) / len(scores) / 10) * 0.2))

    return base


def _score_duration(route: Route, profile: UserProfile) -> float:
    rec = route.recommended_days or "1"
    parts = str(rec).replace(" ", "").split("-")
    try:
        mid = (int(parts[0]) + int(parts[1])) / 2 if len(parts) == 2 else float(parts[0])
    except ValueError:
        mid = 5
    avail = profile.trip_duration_days
    if mid > avail + 2:
        return 0.1
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


_MONTH_NUM: Dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _months_in_range(season_str: str) -> set:
    """Parse a season string like 'Apr-Jun, Sep-Oct' into a set of month numbers."""
    months: set = set()
    if not season_str:
        return months
    for part in season_str.lower().split(","):
        part = part.strip()
        if "-" in part:
            tokens = part.split("-")
            start = _MONTH_NUM.get(tokens[0].strip()[:3], 0)
            end = _MONTH_NUM.get(tokens[1].strip()[:3], 0)
            if start and end:
                if start <= end:
                    months.update(range(start, end + 1))
                else:  # wraps around year, e.g. Oct-Mar
                    months.update(range(start, 13))
                    months.update(range(1, end + 1))
        else:
            m = _MONTH_NUM.get(part[:3], 0)
            if m:
                months.add(m)
    return months


def _score_season(route: Route, travel_month: Optional[str]) -> float:
    if not travel_month:
        return 0.7
    month_num = _MONTH_NUM.get(travel_month.lower()[:3], 0)
    if not month_num:
        return 0.7
    avoid_months = _months_in_range(route.avoid_season or "")
    best_months = _months_in_range(route.best_season or "")
    if month_num in avoid_months:
        return 0.1
    if month_num in best_months:
        return 1.0
    return 0.6


def score_route(
    route: Route,
    places: List[Place],
    profile: UserProfile,
) -> Tuple[float, Dict[str, float]]:
    """Return (total_score, breakdown_dict) for a route candidate."""
    weights = {
        "difficulty": 0.20, "bike_suit": 0.15, "vibe": 0.15,
        "duration": 0.20, "fuel_safety": 0.10, "season": 0.20,
    }
    scores = {
        "difficulty": _score_difficulty(route, profile),
        "bike_suit": _score_bike_suitability(route, profile),
        "vibe": _score_vibe(route, places, profile),
        "duration": _score_duration(route, profile),
        "fuel_safety": _score_fuel_safety(route),
        "season": _score_season(route, profile.travel_month),
    }
    total = sum(scores[k] * weights[k] for k in weights)
    return round(total, 4), {k: round(v, 3) for k, v in scores.items()}


# ═══════════════════════════════════════════════════════
#  Vibe-mismatch detector
# ═══════════════════════════════════════════════════════

def _detect_vibe_mismatch(
    route: Route,
    places: Dict[str, Place],
    profile: UserProfile,
) -> List[str]:
    """Return human-readable notes about vibes the route can't satisfy."""
    if not profile.preferred_vibes:
        return []
    wanted = set(v.lower() for v in profile.preferred_vibes)
    route_vibes = _collect_route_vibes(route, places)
    notes: List[str] = []
    for w in wanted:
        matched = w in route_vibes
        if not matched:
            for syn in _VIBE_SYNONYMS.get(w, []):
                if syn in route_vibes:
                    matched = True
                    break
        if not matched:
            notes.append(
                f"Your preference for '{w}' could not be fully matched. "
                f"The selected route offers: {', '.join(sorted(route_vibes)[:5])}. "
                f"This was the best overall fit for your skill level, bike, and trip duration."
            )
    return notes


# ═══════════════════════════════════════════════════════
#  Main engine
# ═══════════════════════════════════════════════════════

class RecommendationEngine:
    """Orchestrates: scoring -> stop planning -> day planning -> retrieval -> LLM."""

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

    # ── main pipeline ────────────────────────────────

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        profile = request.user_profile
        places = self._place_repo.get_all()
        routes = self._route_repo.get_all()
        place_map = {p.place_id: p for p in places}

        # ── 1. Score & rank routes ───────────────────
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

        top_route, top_score, top_bd = scored[0]
        alternatives = scored[1:4]

        # ── 2. Load related entities ─────────────────
        segments = self._segment_repo.get_by_route(top_route.route_id)
        stays = self._stay_repo.get_by_ids(top_route.recommended_stay_ids)
        restaurants = self._restaurant_repo.get_by_ids(top_route.recommended_restaurant_ids)
        fuel_stations = self._gas_station_repo.get_by_ids(top_route.recommended_fuel_station_ids)
        weather_risks = self._weather_risk_repo.get_for_route(top_route.route_id)
        weather_warnings = self._format_weather_warnings(weather_risks, profile.travel_month)

        # ── 3. Deterministic stop planning ───────────
        planner = StopPlanner(
            route=top_route,
            places=places,
            segments=segments,
            gas_stations=fuel_stations,
            restaurants=restaurants,
            stays=stays,
            weather_risks=weather_risks,
            profile=profile,
        )
        planned_stops = planner.plan_all_stops()

        # Convert to response models
        stop_points = [
            RouteStopPoint(
                stop_id=ps.stop_id, stop_type=ps.stop_type, name=ps.name,
                route_id=ps.route_id, segment_id=ps.segment_id,
                related_entity_id=ps.related_entity_id,
                latitude=ps.latitude, longitude=ps.longitude,
                reason=ps.reason, priority=ps.priority,
                estimated_day=ps.estimated_day, notes=ps.notes,
                km_from_start=ps.km_from_start,
            )
            for ps in planned_stops
        ]

        # Legacy flat lists
        food_stops = [
            StopSummary(
                name=ps.name, entity_id=ps.related_entity_id,
                stop_type="meal", latitude=ps.latitude,
                longitude=ps.longitude, notes=ps.notes,
            )
            for ps in planned_stops if ps.stop_type == "meal"
        ]
        stay_stops = [
            StopSummary(
                name=ps.name, entity_id=ps.related_entity_id,
                stop_type="sleep", latitude=ps.latitude,
                longitude=ps.longitude, notes=ps.notes,
            )
            for ps in planned_stops if ps.stop_type == "sleep"
        ]
        fuel_stop_summaries = [
            StopSummary(
                name=ps.name, entity_id=ps.related_entity_id,
                stop_type="fuel", latitude=ps.latitude,
                longitude=ps.longitude, notes=ps.notes,
            )
            for ps in planned_stops if ps.stop_type == "fuel"
        ]

        # ── 4. Build structured day plan ─────────────
        day_plan = self._build_day_plan(top_route, segments, planned_stops, profile)

        # ── 5. Vibe mismatch detection ───────────────
        vibe_notes = _detect_vibe_mismatch(top_route, place_map, profile)

        # ── 6. Route-filtered semantic retrieval ─────
        # Only retrieve chunks relevant to the selected route and its entities
        route_entity_ids = (
            [top_route.route_id]
            + list(top_route.via_place_ids or [])
            + list(top_route.recommended_stay_ids)
            + list(top_route.recommended_restaurant_ids)
        )
        search_query = f"motorcycle {top_route.route_name}"
        raw_hits = self._vector_store.search(search_query, top_k=8)

        # Filter: keep only chunks whose related_entity_id matches the route
        filtered_hits = [
            h for h in raw_hits
            if h.get("related_entity_id", "") in route_entity_ids
        ]
        # If filtering killed all results, take top 2 from raw
        if not filtered_hits:
            filtered_hits = raw_hits[:2]
        else:
            filtered_hits = filtered_hits[:3]

        context_summaries = [
            RetrievedContextSummary(
                chunk_id=h.get("chunk_id", ""),
                title=h.get("title"),
                related_entity_id=h.get("related_entity_id"),
                snippet=h.get("text_preview", "")[:200],
            )
            for h in filtered_hits
        ]

        # ── 7. Build route candidate scores ──────────
        def _cand(r: Route, s: float, bd: Dict) -> RouteCandidateScore:
            return RouteCandidateScore(
                route_id=r.route_id, route_name=r.route_name,
                total_score=s, score_breakdown=bd,
                distance_km=r.distance_km, difficulty_level=r.difficulty_level,
                recommended_days=r.recommended_days, why_choose=r.why_choose_this_route,
            )

        recommended = _cand(top_route, top_score, top_bd)
        alt_candidates = [_cand(r, s, bd) for r, s, bd in alternatives]

        # ── 8. LLM generation (explanation only) ─────
        profile_summary = self._format_profile(profile)
        top_route_summary = self._format_route_summary(top_route)
        day_plan_text = self._format_day_plan(day_plan)
        stops_text = self._format_planned_stops(planned_stops)
        weather_text = "\n".join(weather_warnings) if weather_warnings else ""
        vibe_mismatch_note = " ".join(vibe_notes) if vibe_notes else ""

        prompt = build_recommendation_prompt(
            profile_summary=profile_summary,
            day_plan_text=day_plan_text,
            top_route_summary=top_route_summary,
            stops_text=stops_text,
            weather_text=weather_text,
            vibe_mismatch_note=vibe_mismatch_note,
            additional_preferences=request.additional_preferences,
        )

        llm_text = self._ollama.generate(prompt, system=SYSTEM_PROMPT)

        # ── 9. Safety notes ──────────────────────────
        safety_notes: List[str] = []
        if top_route.safety_advice:
            safety_notes.append(top_route.safety_advice)
        for ps in planned_stops:
            if ps.stop_type == "warning" and ps.reason:
                safety_notes.append(ps.reason)
        # Deduplicate
        safety_notes = list(dict.fromkeys(safety_notes))

        # ── 10. Load GPX route polyline ───────────────
        route_polyline: List[List[float]] = []
        if top_route.linked_gpx_name:
            try:
                from app.services.gpx_parser import GpxParserService
                gpx_svc = GpxParserService(places=places)
                geometry = gpx_svc.get_route_geometry(
                    top_route.linked_gpx_name, max_points=800
                )
                if geometry:
                    route_polyline = geometry
                    logger.info(
                        "Loaded GPX polyline for '%s': %d points (simplified)",
                        top_route.linked_gpx_name, len(route_polyline),
                    )
            except Exception as e:
                logger.warning("Could not load GPX geometry: %s", e)

        # ── Assemble response ────────────────────────
        return RecommendationResponse(
            recommended_route=recommended,
            alternative_routes=alt_candidates,
            day_plan=day_plan,
            food_stops=food_stops,
            stay_stops=stay_stops,
            fuel_stops=fuel_stop_summaries,
            stop_points=stop_points,
            route_polyline=route_polyline,
            weather_warnings=weather_warnings,
            safety_notes=safety_notes,
            vibe_match_notes=vibe_notes,
            llm_explanation=llm_text,
            retrieved_context_summary=context_summaries,
            metadata={
                "model": self._ollama._model,
                "total_routes_scored": len(scored),
                "semantic_chunks_used": len(filtered_hits),
                "planned_stops_count": len(planned_stops),
                "route_polyline_points": len(route_polyline),
            },
        )

    # ═══════════════════════════════════════════════════
    #  Day plan builder
    # ═══════════════════════════════════════════════════

    def _build_day_plan(
        self,
        route: Route,
        segments: List[RouteSegment],
        planned_stops: List[PlannedStop],
        profile: UserProfile,
    ) -> List[DayPlan]:
        """Build day plan from segments + planned stops.

        Falls back to distance-based splitting when segments are missing.
        """
        # Determine the number of days
        rec = route.recommended_days or "1"
        parts = str(rec).replace(" ", "").split("-")
        try:
            num_days = int(parts[0])
        except ValueError:
            num_days = 1
        num_days = max(1, min(num_days, profile.trip_duration_days or 7))

        daily_km = profile.daily_ride_km_tolerance or 150
        total_km = route.distance_km or 0

        # Resolve place names
        place_map = {p.place_id: p for p in self._place_repo.get_all()}

        def _place_name(pid: Optional[str]) -> str:
            if pid and pid in place_map:
                return place_map[pid].name
            return pid or "?"

        # ── Path A: segment-based planning ───────────
        if segments:
            return self._day_plan_from_segments(
                route, segments, planned_stops, num_days, daily_km, _place_name,
            )

        # ── Path B: distance-split planning (no segments) ──
        return self._day_plan_from_distance(
            route, planned_stops, num_days, total_km, _place_name,
        )

    def _day_plan_from_segments(
        self,
        route: Route,
        segments: List[RouteSegment],
        planned_stops: List[PlannedStop],
        num_days: int,
        daily_km: int,
        place_name_fn,
    ) -> List[DayPlan]:
        """Split segments across days respecting daily-km tolerance."""
        segments = sorted(segments, key=lambda s: s.segment_order)
        days: List[DayPlan] = []
        day_num = 1
        day_km = 0.0
        day_segs: List[RouteSegment] = []

        for seg in segments:
            seg_km = seg.distance_km or 0
            if day_km + seg_km > daily_km and day_segs:
                days.append(self._close_day(
                    day_num, day_segs, day_km, planned_stops, place_name_fn,
                ))
                day_num += 1
                day_km = 0.0
                day_segs = []
            day_segs.append(seg)
            day_km += seg_km

        if day_segs:
            days.append(self._close_day(
                day_num, day_segs, day_km, planned_stops, place_name_fn,
            ))
        return days

    def _close_day(
        self,
        day_num: int,
        segs: List[RouteSegment],
        km: float,
        planned_stops: List[PlannedStop],
        place_name_fn,
    ) -> DayPlan:
        """Finalize one day from accumulated segments."""
        start = segs[0].segment_name.split(" to ")[0] if segs[0].segment_name else "?"
        end = segs[-1].segment_name.split(" to ")[-1] if segs[-1].segment_name else "?"
        diff = max(
            (s.difficulty_level or "easy" for s in segs),
            key=lambda d: _DIFFICULTY_ORDER.get(d, 0),
        )
        # Collect stops assigned to this day
        day_stops = [
            StopSummary(
                name=ps.name,
                entity_id=ps.related_entity_id,
                stop_type=ps.stop_type,
                latitude=ps.latitude,
                longitude=ps.longitude,
                notes=ps.reason,
            )
            for ps in planned_stops if ps.estimated_day == day_num
        ]
        return DayPlan(
            day_number=day_num,
            start_place=start,
            end_place=end,
            distance_km=round(km, 1),
            difficulty=diff,
            stops=day_stops,
        )

    def _day_plan_from_distance(
        self,
        route: Route,
        planned_stops: List[PlannedStop],
        num_days: int,
        total_km: float,
        place_name_fn,
    ) -> List[DayPlan]:
        """When no segments exist, split by route distance and via-places."""
        via = list(route.via_place_ids or [])
        start_pid = route.start_place_id
        end_pid = route.end_place_id

        # Build a list of place waypoints along the route
        waypoints = []
        if start_pid:
            waypoints.append(start_pid)
        waypoints.extend(via)
        if end_pid and end_pid not in waypoints:
            waypoints.append(end_pid)

        km_per_day = total_km / num_days if num_days > 0 else total_km
        days: List[DayPlan] = []

        for d in range(num_days):
            # Distribute waypoints roughly evenly
            wp_start_idx = int(len(waypoints) * d / num_days)
            wp_end_idx = int(len(waypoints) * (d + 1) / num_days)
            wp_end_idx = min(wp_end_idx, len(waypoints) - 1)

            start_name = place_name_fn(waypoints[wp_start_idx]) if wp_start_idx < len(waypoints) else "?"
            end_name = place_name_fn(waypoints[wp_end_idx]) if wp_end_idx < len(waypoints) else "?"

            day_num = d + 1
            day_stops = [
                StopSummary(
                    name=ps.name,
                    entity_id=ps.related_entity_id,
                    stop_type=ps.stop_type,
                    latitude=ps.latitude,
                    longitude=ps.longitude,
                    notes=ps.reason,
                )
                for ps in planned_stops if ps.estimated_day == day_num
            ]

            days.append(DayPlan(
                day_number=day_num,
                start_place=start_name,
                end_place=end_name,
                distance_km=round(km_per_day, 1),
                difficulty=route.difficulty_level,
                stops=day_stops,
            ))
        return days

    # ═══════════════════════════════════════════════════
    #  Formatting helpers
    # ═══════════════════════════════════════════════════

    @staticmethod
    def _format_profile(p: UserProfile) -> str:
        return (
            f"{p.rider_experience} rider, {p.bike_type}, "
            f"budget EUR{p.budget_per_day_eur}/day, "
            f"{p.trip_duration_days} days, "
            f"vibes: {', '.join(p.preferred_vibes)}, "
            f"max {p.daily_ride_km_tolerance}km/day, "
            f"month: {p.travel_month or '?'}"
        )

    @staticmethod
    def _format_route_summary(r: Route) -> str:
        return (
            f"{r.route_name} ({r.route_id})\n"
            f"Distance: {r.distance_km}km, Difficulty: {r.difficulty_level}, "
            f"Days: {r.recommended_days}\n"
            f"{r.route_summary or ''}"
        )

    @staticmethod
    def _format_day_plan(days: List[DayPlan]) -> str:
        if not days:
            return "No day plan available."
        lines = []
        for d in days:
            stops_str = "; ".join(
                f"{s.stop_type}: {s.name}" for s in d.stops
            ) if d.stops else "no assigned stops"
            lines.append(
                f"Day {d.day_number}: {d.start_place} -> {d.end_place} "
                f"({d.distance_km}km). Stops: {stops_str}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_planned_stops(stops: List[PlannedStop]) -> str:
        if not stops:
            return "No stops data."
        lines = []
        for s in stops:
            if s.stop_type in ("fuel", "meal", "sleep"):
                lines.append(f"[{s.stop_type.upper()}] {s.name} - {s.reason}")
        return "\n".join(lines) if lines else "No stops data."

    @staticmethod
    def _format_weather_warnings(
        risks: List[WeatherRisk], travel_month: Optional[str],
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
                    f"-- {wr.rider_advice}"
                )
        return warnings
