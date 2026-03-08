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


# ═══════════════════════════════════════════════════════
#  RT006 – Rif Mountains Loop tests
# ═══════════════════════════════════════════════════════


@pytest.fixture
def rif_route() -> Route:
    return Route(
        route_id="RT006",
        route_name="Rif Mountains Loop",
        distance_km=480,
        difficulty_level="intermediate",
        bike_type_suitability=["adventure", "touring", "dual-sport"],
        beginner_friendliness=3,
        recommended_days="3-4",
        fuel_gap_km=90,
        best_season="Apr-Jun, Sep-Oct",
        avoid_season="Dec-Feb",
        via_place_ids=["PL013"],
        start_place_id="PL012",
        end_place_id="PL012",
        route_summary="Rif Mountains Loop via Chefchaouen",
        why_choose_this_route="Blue City, Rif forests, Mediterranean viewpoints",
    )


@pytest.fixture
def rif_places() -> list:
    return [
        Place(
            place_id="PL012",
            name="Fès",
            latitude=34.037,
            longitude=-5.000,
            primary_vibe="culture",
            secondary_vibes=["medina", "imperial", "crafts"],
        ),
        Place(
            place_id="PL013",
            name="Chefchaouen",
            latitude=35.168,
            longitude=-5.268,
            primary_vibe="scenic",
            secondary_vibes=["blue city", "Rif", "hiking"],
        ),
    ]


def test_rif_scores_high_for_scenic_mountain_rider(rif_route: Route, rif_places: list):
    """An intermediate rider wanting scenic + mountain vibes for 4 days in May
    should score RT006 above 0.5."""
    profile = UserProfile(
        rider_experience="intermediate",
        bike_type="adventure",
        trip_duration_days=4,
        preferred_vibes=["scenic", "mountain", "culture"],
        travel_month="May",
    )
    total, breakdown = score_route(rif_route, rif_places, profile)
    assert total > 0.5, f"Expected >0.5, got {total}"
    assert breakdown["vibe"] > 0.5, f"Vibe match should be strong, got {breakdown['vibe']}"
    assert breakdown["season"] == 1.0, "May is peak season for RT006"


def test_rif_beats_atlas_for_short_scenic_trip(
    rif_route: Route, rif_places: list, sample_route: Route, sample_places: list
):
    """For a 4-day scenic trip, RT006 (480km, 3-4 days) should outperform
    RT001 (1450km, 7-10 days) on duration fit alone."""
    profile = UserProfile(
        rider_experience="intermediate",
        bike_type="adventure",
        trip_duration_days=4,
        preferred_vibes=["scenic", "culture"],
        travel_month="May",
    )
    rif_score, _ = score_route(rif_route, rif_places, profile)
    atlas_score, _ = score_route(sample_route, sample_places, profile)
    assert rif_score > atlas_score, (
        f"RT006 ({rif_score:.3f}) should beat RT001 ({atlas_score:.3f}) for a 4-day trip"
    )


def test_fes_longitude_is_negative(rif_places: list):
    """PL012 Fès must have negative longitude (west of Greenwich)."""
    fes = next(p for p in rif_places if p.place_id == "PL012")
    assert fes.longitude is not None and fes.longitude < 0, (
        f"Fès longitude should be negative, got {fes.longitude}"
    )
