"""
Repository for route_segments, stays, restaurants, gas_stations, weather_risks.
"""

from __future__ import annotations

import json
import sqlite3
from typing import List, Optional

from app.models.domain import (
    GasStation,
    Restaurant,
    RouteSegment,
    Stay,
    WeatherRisk,
)


# ═══════════════════════════════════════════════════════
#  RouteSegmentRepository
# ═══════════════════════════════════════════════════════

class RouteSegmentRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, seg: RouteSegment) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO route_segments
               (segment_id, route_id, segment_order, segment_name,
                start_latitude, start_longitude, end_latitude, end_longitude,
                distance_km, estimated_duration, surface_type, difficulty_level,
                hazard_notes, scenic_notes, nearest_place_id,
                nearest_restaurant_id, nearest_stay_id,
                nearest_fuel_station_id, stop_recommendation)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                seg.segment_id, seg.route_id, seg.segment_order,
                seg.segment_name, seg.start_latitude, seg.start_longitude,
                seg.end_latitude, seg.end_longitude, seg.distance_km,
                seg.estimated_duration, seg.surface_type, seg.difficulty_level,
                seg.hazard_notes, seg.scenic_notes, seg.nearest_place_id,
                seg.nearest_restaurant_id, seg.nearest_stay_id,
                seg.nearest_fuel_station_id, seg.stop_recommendation,
            ),
        )
        self._conn.commit()

    def upsert_many(self, segments: List[RouteSegment]) -> int:
        for s in segments:
            self.upsert(s)
        return len(segments)

    def _row_to_model(self, row: sqlite3.Row) -> RouteSegment:
        return RouteSegment(**dict(row))

    def get_by_route(self, route_id: str) -> List[RouteSegment]:
        cur = self._conn.execute(
            "SELECT * FROM route_segments WHERE route_id = ? ORDER BY segment_order",
            (route_id,),
        )
        return [self._row_to_model(r) for r in cur.fetchall()]

    def get_all(self) -> List[RouteSegment]:
        cur = self._conn.execute("SELECT * FROM route_segments ORDER BY route_id, segment_order")
        return [self._row_to_model(r) for r in cur.fetchall()]


# ═══════════════════════════════════════════════════════
#  StayRepository
# ═══════════════════════════════════════════════════════

def _bool_to_int(v) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        return 1 if v.lower() in ("yes", "true", "1") else 0
    return int(bool(v))


class StayRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, stay: Stay) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO stays
               (stay_id, name, place_id, region, latitude, longitude,
                stay_type, price_range, facilities, motorbike_friendly,
                secure_parking, hot_shower, breakfast_available,
                private_room_available, group_friendly,
                booking_recommended, ideal_for)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                stay.stay_id, stay.name, stay.place_id, stay.region,
                stay.latitude, stay.longitude, stay.stay_type,
                stay.price_range, json.dumps(stay.facilities),
                _bool_to_int(stay.motorbike_friendly),
                _bool_to_int(stay.secure_parking),
                _bool_to_int(stay.hot_shower),
                _bool_to_int(stay.breakfast_available),
                _bool_to_int(stay.private_room_available),
                _bool_to_int(stay.group_friendly),
                stay.booking_recommended,
                json.dumps(stay.ideal_for),
            ),
        )
        self._conn.commit()

    def upsert_many(self, stays: List[Stay]) -> int:
        for s in stays:
            self.upsert(s)
        return len(stays)

    def _row_to_model(self, row: sqlite3.Row) -> Stay:
        d = dict(row)
        d["facilities"] = json.loads(d.get("facilities") or "[]")
        d["ideal_for"] = json.loads(d.get("ideal_for") or "[]")
        for k in ("motorbike_friendly", "secure_parking", "hot_shower",
                   "breakfast_available", "private_room_available", "group_friendly"):
            d[k] = bool(d.get(k))
        return Stay(**d)

    def get_by_place(self, place_id: str) -> List[Stay]:
        cur = self._conn.execute("SELECT * FROM stays WHERE place_id = ?", (place_id,))
        return [self._row_to_model(r) for r in cur.fetchall()]

    def get_all(self) -> List[Stay]:
        cur = self._conn.execute("SELECT * FROM stays ORDER BY stay_id")
        return [self._row_to_model(r) for r in cur.fetchall()]

    def get_by_ids(self, ids: List[str]) -> List[Stay]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        cur = self._conn.execute(
            f"SELECT * FROM stays WHERE stay_id IN ({placeholders})", ids
        )
        return [self._row_to_model(r) for r in cur.fetchall()]


# ═══════════════════════════════════════════════════════
#  RestaurantRepository
# ═══════════════════════════════════════════════════════

class RestaurantRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, rest: Restaurant) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO restaurants
               (restaurant_id, name, place_id, region, latitude, longitude,
                cuisine_type, price_level, vibe, biker_friendly,
                quick_stop_friendly, group_friendly, meal_type, ideal_for)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rest.restaurant_id, rest.name, rest.place_id, rest.region,
                rest.latitude, rest.longitude, rest.cuisine_type,
                rest.price_level, rest.vibe,
                _bool_to_int(rest.biker_friendly),
                _bool_to_int(rest.quick_stop_friendly),
                _bool_to_int(rest.group_friendly),
                json.dumps(rest.meal_type),
                json.dumps(rest.ideal_for),
            ),
        )
        self._conn.commit()

    def upsert_many(self, restaurants: List[Restaurant]) -> int:
        for r in restaurants:
            self.upsert(r)
        return len(restaurants)

    def _row_to_model(self, row: sqlite3.Row) -> Restaurant:
        d = dict(row)
        d["meal_type"] = json.loads(d.get("meal_type") or "[]")
        d["ideal_for"] = json.loads(d.get("ideal_for") or "[]")
        for k in ("biker_friendly", "quick_stop_friendly", "group_friendly"):
            d[k] = bool(d.get(k))
        return Restaurant(**d)

    def get_by_place(self, place_id: str) -> List[Restaurant]:
        cur = self._conn.execute("SELECT * FROM restaurants WHERE place_id = ?", (place_id,))
        return [self._row_to_model(r) for r in cur.fetchall()]

    def get_all(self) -> List[Restaurant]:
        cur = self._conn.execute("SELECT * FROM restaurants ORDER BY restaurant_id")
        return [self._row_to_model(r) for r in cur.fetchall()]

    def get_by_ids(self, ids: List[str]) -> List[Restaurant]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        cur = self._conn.execute(
            f"SELECT * FROM restaurants WHERE restaurant_id IN ({placeholders})", ids
        )
        return [self._row_to_model(r) for r in cur.fetchall()]


# ═══════════════════════════════════════════════════════
#  GasStationRepository
# ═══════════════════════════════════════════════════════

class GasStationRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, gs: GasStation) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO gas_stations
               (station_id, name, place_id, region, latitude, longitude,
                fuel_types, open_24h, hours, payment_notes,
                reliability_notes, air_pump_available,
                basic_repair_nearby, critical_gap_note)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                gs.station_id, gs.name, gs.place_id, gs.region,
                gs.latitude, gs.longitude,
                json.dumps(gs.fuel_types), _bool_to_int(gs.open_24h),
                gs.hours, gs.payment_notes, gs.reliability_notes,
                _bool_to_int(gs.air_pump_available),
                _bool_to_int(gs.basic_repair_nearby),
                gs.critical_gap_note,
            ),
        )
        self._conn.commit()

    def upsert_many(self, stations: List[GasStation]) -> int:
        for gs in stations:
            self.upsert(gs)
        return len(stations)

    def _row_to_model(self, row: sqlite3.Row) -> GasStation:
        d = dict(row)
        d["fuel_types"] = json.loads(d.get("fuel_types") or "[]")
        for k in ("open_24h", "air_pump_available", "basic_repair_nearby"):
            d[k] = bool(d.get(k))
        return GasStation(**d)

    def get_by_place(self, place_id: str) -> List[GasStation]:
        cur = self._conn.execute("SELECT * FROM gas_stations WHERE place_id = ?", (place_id,))
        return [self._row_to_model(r) for r in cur.fetchall()]

    def get_all(self) -> List[GasStation]:
        cur = self._conn.execute("SELECT * FROM gas_stations ORDER BY station_id")
        return [self._row_to_model(r) for r in cur.fetchall()]

    def get_by_ids(self, ids: List[str]) -> List[GasStation]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        cur = self._conn.execute(
            f"SELECT * FROM gas_stations WHERE station_id IN ({placeholders})", ids
        )
        return [self._row_to_model(r) for r in cur.fetchall()]


# ═══════════════════════════════════════════════════════
#  WeatherRiskRepository
# ═══════════════════════════════════════════════════════

class WeatherRiskRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, wr: WeatherRisk) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO weather_risks
               (risk_id, related_route_id, related_route_ids,
                related_place_id, related_place_ids, region, season,
                risk_type, severity, description, rider_advice)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                wr.risk_id, wr.related_route_id,
                json.dumps(wr.related_route_ids),
                wr.related_place_id,
                json.dumps(wr.related_place_ids),
                wr.region, wr.season, wr.risk_type,
                wr.severity, wr.description, wr.rider_advice,
            ),
        )
        self._conn.commit()

    def upsert_many(self, risks: List[WeatherRisk]) -> int:
        for wr in risks:
            self.upsert(wr)
        return len(risks)

    def _row_to_model(self, row: sqlite3.Row) -> WeatherRisk:
        d = dict(row)
        d["related_route_ids"] = json.loads(d.get("related_route_ids") or "[]")
        d["related_place_ids"] = json.loads(d.get("related_place_ids") or "[]")
        return WeatherRisk(**d)

    def get_for_route(self, route_id: str) -> List[WeatherRisk]:
        cur = self._conn.execute(
            "SELECT * FROM weather_risks WHERE related_route_id = ? "
            "OR related_route_ids LIKE ?",
            (route_id, f'%"{route_id}"%'),
        )
        return [self._row_to_model(r) for r in cur.fetchall()]

    def get_for_place(self, place_id: str) -> List[WeatherRisk]:
        cur = self._conn.execute(
            "SELECT * FROM weather_risks WHERE related_place_id = ? "
            "OR related_place_ids LIKE ?",
            (place_id, f'%"{place_id}"%'),
        )
        return [self._row_to_model(r) for r in cur.fetchall()]

    def get_all(self) -> List[WeatherRisk]:
        cur = self._conn.execute("SELECT * FROM weather_risks ORDER BY risk_id")
        return [self._row_to_model(r) for r in cur.fetchall()]
