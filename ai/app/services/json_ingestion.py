"""
JSON Knowledge Base ingestion service.

Loads the structured knowledge base JSON, validates each entity,
and stores everything in SQLite via the repository layer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from app.models.domain import (
    GasStation,
    Place,
    Restaurant,
    Route,
    RouteSegment,
    Stay,
    WeatherRisk,
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

logger = logging.getLogger(__name__)


# ── helper: coerce truthy/falsy strings coming from JSON ──────

def _to_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "yes", "1")
    return bool(val)


# ── public API ────────────────────────────────────────────────

class JsonIngestionService:
    """Loads the morocco_moto_kb.json file and persists entities to SQLite."""

    def __init__(
        self,
        place_repo: PlaceRepository,
        route_repo: RouteRepository,
        segment_repo: RouteSegmentRepository,
        stay_repo: StayRepository,
        restaurant_repo: RestaurantRepository,
        gas_station_repo: GasStationRepository,
        weather_risk_repo: WeatherRiskRepository,
    ) -> None:
        self._place_repo = place_repo
        self._route_repo = route_repo
        self._segment_repo = segment_repo
        self._stay_repo = stay_repo
        self._restaurant_repo = restaurant_repo
        self._gas_station_repo = gas_station_repo
        self._weather_risk_repo = weather_risk_repo

    def ingest(self, json_path: Path) -> Dict[str, int]:
        """Read, validate and store the knowledge base.  Returns counts."""
        logger.info("Loading JSON knowledge base from %s", json_path)
        raw: Dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))

        counts: Dict[str, int] = {}

        # Places
        places = self._parse_places(raw.get("places", []))
        counts["places"] = self._place_repo.upsert_many(places)

        # Routes
        routes = self._parse_routes(raw.get("routes", []))
        counts["routes"] = self._route_repo.upsert_many(routes)

        # Route segments
        segments = self._parse_segments(raw.get("route_segments", []))
        counts["route_segments"] = self._segment_repo.upsert_many(segments)

        # Stays
        stays = self._parse_stays(raw.get("stays", []))
        counts["stays"] = self._stay_repo.upsert_many(stays)

        # Restaurants
        restaurants = self._parse_restaurants(raw.get("restaurants", []))
        counts["restaurants"] = self._restaurant_repo.upsert_many(restaurants)

        # Gas stations
        gas_stations = self._parse_gas_stations(raw.get("gas_stations", []))
        counts["gas_stations"] = self._gas_station_repo.upsert_many(gas_stations)

        # Weather risks
        risks = self._parse_weather_risks(raw.get("weather_risks", []))
        counts["weather_risks"] = self._weather_risk_repo.upsert_many(risks)

        logger.info("JSON ingestion complete: %s", counts)
        return counts

    # ── entity parsers ────────────────────────────────

    @staticmethod
    def _parse_places(raw_list: List[Dict]) -> List[Place]:
        out: List[Place] = []
        for d in raw_list:
            d.setdefault("secondary_vibes", [])
            d.setdefault("bike_type_suitability", [])
            d.setdefault("available_activities", [])
            out.append(Place(**d))
        return out

    @staticmethod
    def _parse_routes(raw_list: List[Dict]) -> List[Route]:
        out: List[Route] = []
        for d in raw_list:
            for key in (
                "via_place_ids", "bike_type_suitability", "scenic_features",
                "weather_dangers", "recommended_stay_ids",
                "recommended_restaurant_ids", "recommended_fuel_station_ids",
            ):
                d.setdefault(key, [])
            # Coerce numeric fields that should be strings
            for str_key in ("recommended_days", "estimated_ride_hours"):
                if str_key in d and not isinstance(d[str_key], str):
                    d[str_key] = str(d[str_key])
            out.append(Route(**d))
        return out

    @staticmethod
    def _parse_segments(raw_list: List[Dict]) -> List[RouteSegment]:
        return [RouteSegment(**d) for d in raw_list]

    @staticmethod
    def _parse_stays(raw_list: List[Dict]) -> List[Stay]:
        out: List[Stay] = []
        for d in raw_list:
            for k in ("motorbike_friendly", "secure_parking", "hot_shower",
                       "breakfast_available", "private_room_available",
                       "group_friendly"):
                d[k] = _to_bool(d.get(k, False))
            # booking_recommended may be bool in JSON — coerce to string
            br = d.get("booking_recommended")
            if isinstance(br, bool):
                d["booking_recommended"] = "yes" if br else "no"
            d.setdefault("facilities", [])
            d.setdefault("ideal_for", [])
            out.append(Stay(**d))
        return out

    @staticmethod
    def _parse_restaurants(raw_list: List[Dict]) -> List[Restaurant]:
        out: List[Restaurant] = []
        for d in raw_list:
            for k in ("biker_friendly", "quick_stop_friendly", "group_friendly"):
                d[k] = _to_bool(d.get(k, False))
            d.setdefault("meal_type", [])
            d.setdefault("ideal_for", [])
            out.append(Restaurant(**d))
        return out

    @staticmethod
    def _parse_gas_stations(raw_list: List[Dict]) -> List[GasStation]:
        out: List[GasStation] = []
        for d in raw_list:
            for k in ("open_24h", "air_pump_available", "basic_repair_nearby"):
                d[k] = _to_bool(d.get(k, False))
            d.setdefault("fuel_types", [])
            out.append(GasStation(**d))
        return out

    @staticmethod
    def _parse_weather_risks(raw_list: List[Dict]) -> List[WeatherRisk]:
        out: List[WeatherRisk] = []
        for d in raw_list:
            # Normalise: some entries use related_route_ids (plural) instead of singular
            if "related_route_ids" in d and isinstance(d["related_route_ids"], list):
                d.setdefault("related_route_id", d["related_route_ids"][0] if d["related_route_ids"] else None)
            else:
                d.setdefault("related_route_ids", [])
            if "related_place_ids" in d and isinstance(d["related_place_ids"], list):
                d.setdefault("related_place_id", d["related_place_ids"][0] if d["related_place_ids"] else None)
            else:
                d.setdefault("related_place_ids", [])
            out.append(WeatherRisk(**d))
        return out
