"""
GPX-driven deterministic stop planner.

Proposes fuel, meal, scenic, overnight, and warning stops by combining:
- route distance & segment metadata,
- GPX track progression (when available),
- nearby known entities (gas stations, restaurants, stays, places),
- rider profile constraints (daily km, budget, experience).

The planner is purely deterministic — no LLM involved.
Its output feeds the day-plan builder and the prompt builder.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.models.domain import (
    GasStation,
    Place,
    Restaurant,
    Route,
    RouteSegment,
    Stay,
    WeatherRisk,
)
from app.models.gpx_models import GpxTrackPoint, ParsedGpxFile
from app.schemas.request import UserProfile

logger = logging.getLogger(__name__)


# ── Stop-point data class ────────────────────────────

@dataclass
class PlannedStop:
    """A single proposed stop along a route."""
    stop_id: str
    stop_type: str              # fuel | meal | sleep | scenic | warning | rest
    name: str
    route_id: str
    segment_id: Optional[str] = None
    related_entity_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    reason: str = ""
    priority: str = "normal"    # critical | high | normal | optional
    estimated_day: int = 1
    notes: str = ""
    km_from_start: float = 0.0


# ── Haversine helper ─────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Parse recommended_days safely ────────────────────

def _parse_days(raw: Optional[str]) -> int:
    """'2-3' -> 2,  '7-10' -> 7,  '2' -> 2."""
    if not raw:
        return 1
    parts = str(raw).replace(" ", "").split("-")
    try:
        return int(parts[0])
    except (ValueError, IndexError):
        return 1


# ═══════════════════════════════════════════════════════
#  Main planner
# ═══════════════════════════════════════════════════════

class StopPlanner:
    """Deterministic stop planner using route + GPX + entity data."""

    def __init__(
        self,
        route: Route,
        places: List[Place],
        segments: List[RouteSegment],
        gas_stations: List[GasStation],
        restaurants: List[Restaurant],
        stays: List[Stay],
        weather_risks: List[WeatherRisk],
        profile: UserProfile,
        gpx: Optional[ParsedGpxFile] = None,
    ) -> None:
        self._route = route
        self._places = {p.place_id: p for p in places}
        self._segments = sorted(segments, key=lambda s: s.segment_order)
        self._gas_stations = {g.station_id: g for g in gas_stations}
        self._restaurants = {r.restaurant_id: r for r in restaurants}
        self._stays = {s.stay_id: s for s in stays}
        self._weather_risks = weather_risks
        self._profile = profile
        self._gpx = gpx

        self._total_km = route.distance_km or 0.0
        self._num_days = max(
            _parse_days(route.recommended_days),
            1,
        )
        # Cap num_days to rider's trip duration
        if profile.trip_duration_days and self._num_days > profile.trip_duration_days:
            self._num_days = profile.trip_duration_days

        self._daily_km = profile.daily_ride_km_tolerance or 150

    # ── public API ────────────────────────────────────

    def plan_all_stops(self) -> List[PlannedStop]:
        """Return a combined, sorted list of all planned stops."""
        stops: List[PlannedStop] = []
        stops.extend(self._plan_fuel_stops())
        stops.extend(self._plan_meal_stops())
        stops.extend(self._plan_overnight_stops())
        stops.extend(self._plan_scenic_stops())
        stops.extend(self._plan_warning_stops())
        # Sort by estimated_day then km_from_start
        stops.sort(key=lambda s: (s.estimated_day, s.km_from_start))
        return stops

    # ── 1. Fuel stops ────────────────────────────────

    def _plan_fuel_stops(self) -> List[PlannedStop]:
        """Propose fuel stops based on fuel gap and station proximity."""
        stops: List[PlannedStop] = []
        fuel_gap = self._route.fuel_gap_km or 80
        # Conservative: suggest fuel every min(fuel_gap * 0.7, 100) km
        refuel_interval = min(fuel_gap * 0.7, 100)

        # Use segment-level nearest fuel if available
        cum_km = 0.0
        last_fuel_km = 0.0
        _id = 0

        for seg in self._segments:
            seg_km = seg.distance_km or 0
            cum_km += seg_km

            if (cum_km - last_fuel_km) >= refuel_interval and seg.nearest_fuel_station_id:
                gs = self._gas_stations.get(seg.nearest_fuel_station_id)
                if gs:
                    _id += 1
                    stops.append(PlannedStop(
                        stop_id=f"fuel-{_id}",
                        stop_type="fuel",
                        name=gs.name,
                        route_id=self._route.route_id,
                        segment_id=seg.segment_id,
                        related_entity_id=gs.station_id,
                        latitude=gs.latitude,
                        longitude=gs.longitude,
                        reason=f"Refuel after ~{cum_km:.0f}km (fuel gap: {fuel_gap}km)",
                        priority="critical" if fuel_gap > 100 else "high",
                        estimated_day=self._km_to_day(cum_km),
                        notes=gs.critical_gap_note or "",
                        km_from_start=cum_km,
                    ))
                    last_fuel_km = cum_km

        # Fallback: if no segment-level fuel stations, use route-recommended ones
        if not stops:
            for i, sid in enumerate(self._route.recommended_fuel_station_ids):
                gs = self._gas_stations.get(sid)
                if gs:
                    stops.append(PlannedStop(
                        stop_id=f"fuel-{i+1}",
                        stop_type="fuel",
                        name=gs.name,
                        route_id=self._route.route_id,
                        related_entity_id=gs.station_id,
                        latitude=gs.latitude,
                        longitude=gs.longitude,
                        reason="Route-recommended fuel station",
                        priority="high",
                        estimated_day=1,
                        notes=gs.critical_gap_note or "",
                    ))
        return stops

    # ── 2. Meal stops ────────────────────────────────

    def _plan_meal_stops(self) -> List[PlannedStop]:
        """Propose meal stops at day midpoints or segment transitions."""
        stops: List[PlannedStop] = []
        _id = 0

        # Try segment-level nearest restaurants first
        cum_km = 0.0
        for seg in self._segments:
            seg_km = seg.distance_km or 0
            cum_km += seg_km
            day = self._km_to_day(cum_km)

            # Place a meal near the midpoint of each day
            day_start_km = (day - 1) * self._daily_km
            day_mid_km = day_start_km + self._daily_km / 2

            if abs(cum_km - day_mid_km) < (self._daily_km * 0.3) and seg.nearest_restaurant_id:
                rest = self._restaurants.get(seg.nearest_restaurant_id)
                if rest and not any(s.related_entity_id == rest.restaurant_id for s in stops):
                    _id += 1
                    stops.append(PlannedStop(
                        stop_id=f"meal-{_id}",
                        stop_type="meal",
                        name=rest.name,
                        route_id=self._route.route_id,
                        segment_id=seg.segment_id,
                        related_entity_id=rest.restaurant_id,
                        latitude=rest.latitude,
                        longitude=rest.longitude,
                        reason=f"Lunch stop (midday, ~{cum_km:.0f}km from start)",
                        priority="normal",
                        estimated_day=day,
                        notes=f"{rest.cuisine_type or 'local'}, {rest.price_level or '?'}",
                        km_from_start=cum_km,
                    ))

        # Fallback: route-recommended restaurants
        if not stops:
            for i, rid in enumerate(self._route.recommended_restaurant_ids):
                rest = self._restaurants.get(rid)
                if rest:
                    _id += 1
                    stops.append(PlannedStop(
                        stop_id=f"meal-{_id}",
                        stop_type="meal",
                        name=rest.name,
                        route_id=self._route.route_id,
                        related_entity_id=rest.restaurant_id,
                        latitude=rest.latitude,
                        longitude=rest.longitude,
                        reason="Route-recommended restaurant",
                        priority="normal",
                        estimated_day=min(i + 1, self._num_days),
                        notes=f"{rest.cuisine_type or 'local'}, {rest.price_level or '?'}",
                    ))
        return stops

    # ── 3. Overnight stops ───────────────────────────

    def _plan_overnight_stops(self) -> List[PlannedStop]:
        """Propose overnight stays at end-of-day positions."""
        stops: List[PlannedStop] = []

        # For each day except the last, we need an overnight stop
        for day in range(1, self._num_days + 1):
            target_km = day * self._daily_km
            best_stay: Optional[Stay] = None
            best_seg_id: Optional[str] = None

            # Find the segment closest to this day's end km
            cum_km = 0.0
            for seg in self._segments:
                cum_km += (seg.distance_km or 0)
                if cum_km >= target_km * 0.8 and seg.nearest_stay_id:
                    best_stay = self._stays.get(seg.nearest_stay_id)
                    best_seg_id = seg.segment_id
                    break

            # Fallback: use route-recommended stays
            if not best_stay and day <= len(self._route.recommended_stay_ids):
                sid = self._route.recommended_stay_ids[day - 1]
                best_stay = self._stays.get(sid)

            # Second fallback: any recommended stay
            if not best_stay:
                for sid in self._route.recommended_stay_ids:
                    s = self._stays.get(sid)
                    if s and not any(x.related_entity_id == sid for x in stops):
                        best_stay = s
                        break

            if best_stay:
                stops.append(PlannedStop(
                    stop_id=f"sleep-{day}",
                    stop_type="sleep",
                    name=best_stay.name,
                    route_id=self._route.route_id,
                    segment_id=best_seg_id,
                    related_entity_id=best_stay.stay_id,
                    latitude=best_stay.latitude,
                    longitude=best_stay.longitude,
                    reason=f"Overnight stop end of day {day}",
                    priority="high",
                    estimated_day=day,
                    notes=f"{best_stay.stay_type or '?'}, {best_stay.price_range or '?'}",
                    km_from_start=target_km,
                ))
        return stops

    # ── 4. Scenic / photo / rest stops ───────────────

    def _plan_scenic_stops(self) -> List[PlannedStop]:
        """Mark scenic places along the route as optional photo stops."""
        stops: List[PlannedStop] = []

        # Use via_place_ids to find high-scenic-score places
        all_via = list(self._route.via_place_ids or [])
        if self._route.start_place_id:
            all_via.insert(0, self._route.start_place_id)
        if self._route.end_place_id:
            all_via.append(self._route.end_place_id)

        _id = 0
        for pid in all_via:
            p = self._places.get(pid)
            if not p:
                continue
            scenic = p.scenic_score or 0
            # Mark places with scenic_score >= 8 as scenic stops
            if scenic >= 8 or p.place_type in ("natural_site", "mountain_pass"):
                _id += 1
                # Rough km estimate from position in via list
                frac = all_via.index(pid) / max(len(all_via) - 1, 1)
                est_km = frac * self._total_km
                stops.append(PlannedStop(
                    stop_id=f"scenic-{_id}",
                    stop_type="scenic",
                    name=p.name,
                    route_id=self._route.route_id,
                    related_entity_id=p.place_id,
                    latitude=p.latitude,
                    longitude=p.longitude,
                    reason=f"Scenic stop (score {scenic}/10): {p.primary_vibe or 'views'}",
                    priority="optional",
                    estimated_day=self._km_to_day(est_km),
                    notes=", ".join(p.available_activities[:3]) if p.available_activities else "",
                    km_from_start=round(est_km, 1),
                ))
        return stops

    # ── 5. Warning / caution stops ───────────────────

    def _plan_warning_stops(self) -> List[PlannedStop]:
        """Flag route sections that need extra caution."""
        stops: List[PlannedStop] = []
        _id = 0

        # From segments: hazard notes
        cum_km = 0.0
        for seg in self._segments:
            cum_km += (seg.distance_km or 0)
            if seg.hazard_notes:
                _id += 1
                stops.append(PlannedStop(
                    stop_id=f"warn-{_id}",
                    stop_type="warning",
                    name=seg.segment_name or "Caution zone",
                    route_id=self._route.route_id,
                    segment_id=seg.segment_id,
                    latitude=seg.start_latitude,
                    longitude=seg.start_longitude,
                    reason=seg.hazard_notes,
                    priority="high",
                    estimated_day=self._km_to_day(cum_km),
                    notes=f"Surface: {seg.surface_type or '?'}, difficulty: {seg.difficulty_level or '?'}",
                    km_from_start=cum_km,
                ))

        # From weather risks
        for wr in self._weather_risks:
            if wr.severity and wr.severity.lower() in ("high", "extreme"):
                _id += 1
                stops.append(PlannedStop(
                    stop_id=f"warn-{_id}",
                    stop_type="warning",
                    name=f"Weather: {wr.risk_type or 'risk'}",
                    route_id=self._route.route_id,
                    reason=wr.description or "",
                    priority="critical" if wr.severity.lower() == "extreme" else "high",
                    estimated_day=1,
                    notes=wr.rider_advice or "",
                ))

        # Route-level safety note
        if self._route.safety_advice:
            _id += 1
            stops.append(PlannedStop(
                stop_id=f"warn-{_id}",
                stop_type="warning",
                name="Route safety note",
                route_id=self._route.route_id,
                reason=self._route.safety_advice,
                priority="normal",
                estimated_day=1,
            ))

        return stops

    # ── Helpers ───────────────────────────────────────

    def _km_to_day(self, km: float) -> int:
        """Estimate which day a km-marker falls on."""
        if self._daily_km <= 0:
            return 1
        day = int(km / self._daily_km) + 1
        return min(day, self._num_days)
