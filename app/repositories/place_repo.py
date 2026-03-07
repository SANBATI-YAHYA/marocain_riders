"""
Repository for querying the Places table.
"""

from __future__ import annotations

import json
import sqlite3
from typing import List, Optional

from app.models.domain import Place


class PlaceRepository:
    """CRUD-lite access to the *places* table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ── writes ───────────────────────────────────────

    def upsert(self, place: Place) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO places
                (place_id, name, region, province, latitude, longitude,
                 place_type, primary_vibe, secondary_vibes, budget_level,
                 recommended_stay_duration, best_season, altitude_category,
                 road_accessibility, bike_type_suitability, beginner_friendliness,
                 scenic_score, comfort_score, fuel_access_score,
                 available_activities, safety_notes, weather_notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                place.place_id, place.name, place.region, place.province,
                place.latitude, place.longitude, place.place_type,
                place.primary_vibe, json.dumps(place.secondary_vibes),
                place.budget_level, place.recommended_stay_duration,
                place.best_season, place.altitude_category,
                place.road_accessibility,
                json.dumps(place.bike_type_suitability),
                place.beginner_friendliness, place.scenic_score,
                place.comfort_score, place.fuel_access_score,
                json.dumps(place.available_activities),
                place.safety_notes, place.weather_notes,
            ),
        )
        self._conn.commit()

    def upsert_many(self, places: List[Place]) -> int:
        for p in places:
            self.upsert(p)
        return len(places)

    # ── reads ────────────────────────────────────────

    def _row_to_model(self, row: sqlite3.Row) -> Place:
        d = dict(row)
        for key in ("secondary_vibes", "bike_type_suitability", "available_activities"):
            val = d.get(key)
            d[key] = json.loads(val) if val else []
        return Place(**d)

    def get_by_id(self, place_id: str) -> Optional[Place]:
        cur = self._conn.execute("SELECT * FROM places WHERE place_id = ?", (place_id,))
        row = cur.fetchone()
        return self._row_to_model(row) if row else None

    def get_all(self) -> List[Place]:
        cur = self._conn.execute("SELECT * FROM places ORDER BY place_id")
        return [self._row_to_model(r) for r in cur.fetchall()]

    def search_by_vibe(self, vibe: str) -> List[Place]:
        cur = self._conn.execute(
            "SELECT * FROM places WHERE primary_vibe LIKE ? OR secondary_vibes LIKE ?",
            (f"%{vibe}%", f"%{vibe}%"),
        )
        return [self._row_to_model(r) for r in cur.fetchall()]
