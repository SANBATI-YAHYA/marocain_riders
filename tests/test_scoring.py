"""
Basic test suite for the recommendation engine scoring.
"""

from __future__ import annotations

import pytest
from app.models.domain import Route, Place
from app.schemas.request import UserProfile
from app.services.recommendation_engine import score_route


@pytest.fixture
def sample_route() -> Route:
    return Route(
        route_id="RT001",
        route_name="Grand Atlas Circuit",
        distance_km=1450,
        difficulty_level="intermediate",
        bike_type_suitability=["all-road touring", "adventure"],
        beginner_friendliness=3,
        recommended_days="7-10",
        fuel_gap_km=90,
        best_season="Oct-Apr",
        avoid_season="Jul-Aug",
        via_place_ids=["PL007", "PL002"],
        start_place_id="PL001",
        end_place_id="PL001",
        route_summary="Classic loop",
        why_choose_this_route="Complete experience",
    )


@pytest.fixture
def sample_places() -> list:
    return [
        Place(place_id="PL001", name="Marrakech", primary_vibe="urban_gateway", secondary_vibes=["culture"]),
        Place(place_id="PL007", name="Tizi n'Tichka", primary_vibe="mountain", secondary_vibes=["dramatic"]),
        Place(place_id="PL002", name="Ouarzazate", primary_vibe="desert_gateway", secondary_vibes=["culture"]),
    ]


def test_score_intermediate_adventure_rider(sample_route: Route, sample_places: list):
    """An intermediate adventure rider in October should score high on RT001."""
    profile = UserProfile(
        rider_experience="intermediate",
        bike_type="adventure",
        trip_duration_days=10,
        preferred_vibes=["desert", "culture"],
        travel_month="October",
    )
    total, breakdown = score_route(sample_route, sample_places, profile)
    assert total > 0.5, f"Expected >0.5, got {total}"
    assert breakdown["difficulty"] > 0, "Difficulty score should be positive"
    assert breakdown["season"] == 1.0, "October is peak season for RT001"


def test_beginner_penalised_on_intermediate_route(sample_route: Route, sample_places: list):
    """A beginner should score lower on an intermediate route."""
    beginner = UserProfile(
        rider_experience="beginner",
        bike_type="touring",
        trip_duration_days=10,
    )
    advanced = UserProfile(
        rider_experience="advanced",
        bike_type="adventure",
        trip_duration_days=10,
    )
    beg_score, _ = score_route(sample_route, sample_places, beginner)
    adv_score, _ = score_route(sample_route, sample_places, advanced)
    assert adv_score > beg_score, "Advanced rider should score higher than beginner"


def test_short_trip_penalised(sample_route: Route, sample_places: list):
    """A 3-day trip is too short for a 7-10-day route."""
    short = UserProfile(
        rider_experience="intermediate",
        bike_type="adventure",
        trip_duration_days=3,
    )
    long = UserProfile(
        rider_experience="intermediate",
        bike_type="adventure",
        trip_duration_days=10,
    )
    short_score, _ = score_route(sample_route, sample_places, short)
    long_score, _ = score_route(sample_route, sample_places, long)
    assert long_score > short_score, "Longer trip should score higher for a 7-10 day route"
