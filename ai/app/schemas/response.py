"""
Pydantic schemas for API response bodies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StopSummary(BaseModel):
    """Short description of a recommended stop."""
    name: str
    entity_id: Optional[str] = None
    stop_type: str  # "meal", "sleep", "fuel", "scenic", "warning", "rest", "waypoint"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    notes: Optional[str] = None


class RouteStopPoint(BaseModel):
    """A planned stop derived from GPX / segment / entity analysis.

    Richer than StopSummary — designed for frontend map rendering.
    """
    stop_id: str
    stop_type: str                        # fuel | meal | sleep | scenic | warning | rest
    name: str
    route_id: str
    segment_id: Optional[str] = None
    related_entity_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    reason: str = ""
    priority: str = "normal"              # critical | high | normal | optional
    estimated_day: int = 1
    notes: str = ""
    km_from_start: float = 0.0


class DayPlan(BaseModel):
    """One day of the recommended itinerary."""
    day_number: int
    start_place: str
    end_place: str
    distance_km: Optional[float] = None
    ride_hours: Optional[str] = None
    difficulty: Optional[str] = None
    stops: List[StopSummary] = Field(default_factory=list)
    narrative: Optional[str] = None
    weather_warnings: List[str] = Field(default_factory=list)


class RouteCandidateScore(BaseModel):
    """A scored route candidate shown for transparency."""
    route_id: str
    route_name: str
    total_score: float
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    distance_km: Optional[float] = None
    difficulty_level: Optional[str] = None
    recommended_days: Optional[str] = None
    why_choose: Optional[str] = None


class RetrievedContextSummary(BaseModel):
    """Summary of a semantic chunk retrieved for context."""
    chunk_id: str
    title: Optional[str] = None
    related_entity_id: Optional[str] = None
    snippet: str  # first N chars of the chunk text


class RecommendationResponse(BaseModel):
    """Full response for POST /recommend."""

    recommended_route: Optional[RouteCandidateScore] = None
    alternative_routes: List[RouteCandidateScore] = Field(default_factory=list)

    day_plan: List[DayPlan] = Field(default_factory=list)

    # Legacy flat stop lists (kept for backward compat)
    food_stops: List[StopSummary] = Field(default_factory=list)
    stay_stops: List[StopSummary] = Field(default_factory=list)
    fuel_stops: List[StopSummary] = Field(default_factory=list)

    # New: full planned stop points (for map/frontend)
    stop_points: List[RouteStopPoint] = Field(default_factory=list)

    # Route polyline from GPX: [[lat, lng], [lat, lng], ...]
    route_polyline: List[List[float]] = Field(default_factory=list)

    weather_warnings: List[str] = Field(default_factory=list)
    safety_notes: List[str] = Field(default_factory=list)

    # Vibe match transparency
    vibe_match_notes: List[str] = Field(default_factory=list)

    llm_explanation: Optional[str] = None
    retrieved_context_summary: List[RetrievedContextSummary] = Field(
        default_factory=list
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    llm_reachable: bool = False
    db_ready: bool = False
    vector_store_ready: bool = False


class IngestResponse(BaseModel):
    """Generic response for ingestion endpoints."""
    status: str = "ok"
    message: str = ""
    records_processed: int = 0
