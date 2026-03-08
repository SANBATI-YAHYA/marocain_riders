"""
Domain models for the Morocco Motorcycle knowledge base entities.

These are plain Pydantic models used for internal logic, DB I/O, and ranking.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Places ────────────────────────────────────────

class Place(BaseModel):
    place_id: str
    name: str
    region: Optional[str] = None
    province: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    place_type: Optional[str] = None
    primary_vibe: Optional[str] = None
    secondary_vibes: List[str] = Field(default_factory=list)
    budget_level: Optional[str] = None
    recommended_stay_duration: Optional[str] = None
    best_season: Optional[str] = None
    altitude_category: Optional[str] = None
    road_accessibility: Optional[str] = None
    bike_type_suitability: List[str] = Field(default_factory=list)
    beginner_friendliness: Optional[int] = None
    scenic_score: Optional[int] = None
    comfort_score: Optional[int] = None
    fuel_access_score: Optional[int] = None
    available_activities: List[str] = Field(default_factory=list)
    safety_notes: Optional[str] = None
    weather_notes: Optional[str] = None


# ── Routes ────────────────────────────────────────

class Route(BaseModel):
    route_id: str
    route_name: str
    linked_gpx_name: Optional[str] = None
    start_place_id: Optional[str] = None
    end_place_id: Optional[str] = None
    via_place_ids: List[str] = Field(default_factory=list)
    region: Optional[str] = None
    route_type: Optional[str] = None
    distance_km: Optional[float] = None
    estimated_ride_hours: Optional[str] = None
    recommended_days: Optional[str] = None
    difficulty_level: Optional[str] = None
    bike_type_suitability: List[str] = Field(default_factory=list)
    beginner_friendliness: Optional[int] = None
    road_surface_summary: Optional[str] = None
    elevation_profile: Optional[str] = None
    scenic_features: List[str] = Field(default_factory=list)
    hazard_level: Optional[str] = None
    fuel_gap_km: Optional[float] = None
    cell_coverage_level: Optional[str] = None
    best_season: Optional[str] = None
    avoid_season: Optional[str] = None
    weather_dangers: List[str] = Field(default_factory=list)
    recommended_stay_ids: List[str] = Field(default_factory=list)
    recommended_restaurant_ids: List[str] = Field(default_factory=list)
    recommended_fuel_station_ids: List[str] = Field(default_factory=list)
    route_summary: Optional[str] = None
    rider_advice: Optional[str] = None
    safety_advice: Optional[str] = None
    why_choose_this_route: Optional[str] = None


# ── Route Segments ────────────────────────────────

class RouteSegment(BaseModel):
    segment_id: str
    route_id: str
    segment_order: int
    segment_name: Optional[str] = None
    start_latitude: Optional[float] = None
    start_longitude: Optional[float] = None
    end_latitude: Optional[float] = None
    end_longitude: Optional[float] = None
    distance_km: Optional[float] = None
    estimated_duration: Optional[str] = None
    surface_type: Optional[str] = None
    difficulty_level: Optional[str] = None
    hazard_notes: Optional[str] = None
    scenic_notes: Optional[str] = None
    nearest_place_id: Optional[str] = None
    nearest_restaurant_id: Optional[str] = None
    nearest_stay_id: Optional[str] = None
    nearest_fuel_station_id: Optional[str] = None
    stop_recommendation: Optional[str] = None


# ── Stays ─────────────────────────────────────────

class Stay(BaseModel):
    stay_id: str
    name: str
    place_id: Optional[str] = None
    region: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    stay_type: Optional[str] = None
    price_range: Optional[str] = None
    facilities: List[str] = Field(default_factory=list)
    motorbike_friendly: bool = False
    secure_parking: bool = False
    hot_shower: bool = False
    breakfast_available: bool = False
    private_room_available: bool = False
    group_friendly: bool = False
    booking_recommended: Optional[str] = None
    ideal_for: List[str] = Field(default_factory=list)


# ── Restaurants ───────────────────────────────────

class Restaurant(BaseModel):
    restaurant_id: str
    name: str
    place_id: Optional[str] = None
    region: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    cuisine_type: Optional[str] = None
    price_level: Optional[str] = None
    vibe: Optional[str] = None
    biker_friendly: bool = False
    quick_stop_friendly: bool = False
    group_friendly: bool = False
    meal_type: List[str] = Field(default_factory=list)
    ideal_for: List[str] = Field(default_factory=list)


# ── Gas Stations ──────────────────────────────────

class GasStation(BaseModel):
    station_id: str
    name: str
    place_id: Optional[str] = None
    region: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    fuel_types: List[str] = Field(default_factory=list)
    open_24h: bool = False
    hours: Optional[str] = None
    payment_notes: Optional[str] = None
    reliability_notes: Optional[str] = None
    air_pump_available: bool = False
    basic_repair_nearby: bool = False
    critical_gap_note: Optional[str] = None


# ── Weather Risks ─────────────────────────────────

class WeatherRisk(BaseModel):
    risk_id: str
    related_route_id: Optional[str] = None
    related_route_ids: List[str] = Field(default_factory=list)
    related_place_id: Optional[str] = None
    related_place_ids: List[str] = Field(default_factory=list)
    region: Optional[str] = None
    season: Optional[str] = None
    risk_type: Optional[str] = None
    severity: Optional[str] = None
    description: Optional[str] = None
    rider_advice: Optional[str] = None


# ── Text Chunk (for vector store) ─────────────────

class TextChunk(BaseModel):
    chunk_id: str
    source_type: str   # "json_kb", "markdown", "pdf_text"
    source_file: Optional[str] = None
    title: Optional[str] = None
    related_entity_id: Optional[str] = None
    chunk_text: str
    chunk_index: int = 0
