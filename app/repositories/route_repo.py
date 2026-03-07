"""
Repository for querying the Routes table.
"""

from __future__ import annotations

import json
import sqlite3
from typing import List, Optional

from app.models.domain import Route


class RouteRepository:
    """CRUD-lite access to the *routes* table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ── writes ───────────────────────────────────────

    def upsert(self, route: Route) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO routes
                (route_id, route_name, linked_gpx_name, start_place_id,
                 end_place_id, via_place_ids, region, route_type,
                 distance_km, estimated_ride_hours, recommended_days,
                 difficulty_level, bike_type_suitability, beginner_friendliness,
                 road_surface_summary, elevation_profile, scenic_features,
                 hazard_level, fuel_gap_km, cell_coverage_level,
                 best_season, avoid_season, weather_dangers,
                 recommended_stay_ids, recommended_restaurant_ids,
                 recommended_fuel_station_ids, route_summary,
                 rider_advice, safety_advice, why_choose_this_route)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                route.route_id, route.route_name, route.linked_gpx_name,
                route.start_place_id, route.end_place_id,
                json.dumps(route.via_place_ids), route.region,
                route.route_type, route.distance_km,
                route.estimated_ride_hours, route.recommended_days,
                route.difficulty_level,
                json.dumps(route.bike_type_suitability),
                route.beginner_friendliness, route.road_surface_summary,
                route.elevation_profile, json.dumps(route.scenic_features),
                route.hazard_level, route.fuel_gap_km,
                route.cell_coverage_level, route.best_season,
                route.avoid_season, json.dumps(route.weather_dangers),
                json.dumps(route.recommended_stay_ids),
                json.dumps(route.recommended_restaurant_ids),
                json.dumps(route.recommended_fuel_station_ids),
                route.route_summary, route.rider_advice,
                route.safety_advice, route.why_choose_this_route,
            ),
        )
        self._conn.commit()

    def upsert_many(self, routes: List[Route]) -> int:
        for r in routes:
            self.upsert(r)
        return len(routes)

    # ── reads ────────────────────────────────────────

    def _row_to_model(self, row: sqlite3.Row) -> Route:
        d = dict(row)
        for key in (
            "via_place_ids", "bike_type_suitability", "scenic_features",
            "weather_dangers", "recommended_stay_ids",
            "recommended_restaurant_ids", "recommended_fuel_station_ids",
        ):
            val = d.get(key)
            d[key] = json.loads(val) if val else []
        return Route(**d)

    def get_by_id(self, route_id: str) -> Optional[Route]:
        cur = self._conn.execute("SELECT * FROM routes WHERE route_id = ?", (route_id,))
        row = cur.fetchone()
        return self._row_to_model(row) if row else None

    def get_all(self) -> List[Route]:
        cur = self._conn.execute("SELECT * FROM routes ORDER BY route_id")
        return [self._row_to_model(r) for r in cur.fetchall()]

    def get_by_difficulty(self, level: str) -> List[Route]:
        cur = self._conn.execute(
            "SELECT * FROM routes WHERE difficulty_level LIKE ?",
            (f"%{level}%",),
        )
        return [self._row_to_model(r) for r in cur.fetchall()]
