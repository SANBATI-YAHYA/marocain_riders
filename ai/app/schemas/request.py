"""
Pydantic schemas for API request bodies.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Rider profile supplied in the recommendation request."""

    rider_experience: str = Field(
        ...,
        description="beginner | intermediate | advanced | expert",
    )
    bike_type: str = Field(
        ...,
        description="road/sport | touring | adventure | dual-sport | scooter",
    )
    group_size: int = Field(default=1, ge=1, le=15)
    budget_per_day_eur: float = Field(default=60.0, ge=0)
    trip_duration_days: int = Field(default=7, ge=1, le=21)
    preferred_vibes: List[str] = Field(
        default_factory=lambda: ["mixed"],
        description="culture | adventure | desert | coastal | scenic | mixed",
    )
    sleep_preference: str = Field(
        default="mid-range",
        description="budget/auberge | mid-range | comfort | camping",
    )
    daily_ride_km_tolerance: int = Field(
        default=150,
        description="Maximum comfortable daily riding distance in km",
    )
    travel_month: Optional[str] = Field(
        default=None,
        description="Planned travel month (e.g. 'October')",
    )
    country_of_origin: Optional[str] = None


class RecommendationRequest(BaseModel):
    """Body for POST /recommend."""

    user_profile: UserProfile
    additional_preferences: Optional[str] = Field(
        default=None,
        description="Free-text rider preferences or questions",
    )
