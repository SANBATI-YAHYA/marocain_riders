from __future__ import annotations

from app.services.recommendation_engine import align_planned_stops_to_route_geometry
from app.services.stop_planner import PlannedStop


def test_align_planned_stops_to_route_geometry_snaps_and_orders_stops():
    route_polyline = [
        [30.000000, -8.000000],
        [30.000000, -7.000000],
        [31.000000, -7.000000],
    ]
    stops = [
        PlannedStop(
            stop_id="sleep-1",
            stop_type="sleep",
            name="End stop",
            route_id="RTX",
            latitude=30.910000,
            longitude=-6.850000,
            estimated_day=1,
            km_from_start=0.0,
        ),
        PlannedStop(
            stop_id="fuel-1",
            stop_type="fuel",
            name="Start stop",
            route_id="RTX",
            latitude=30.120000,
            longitude=-7.920000,
            estimated_day=1,
            km_from_start=0.0,
        ),
        PlannedStop(
            stop_id="meal-1",
            stop_type="meal",
            name="Middle stop",
            route_id="RTX",
            latitude=30.150000,
            longitude=-7.020000,
            estimated_day=1,
            km_from_start=0.0,
        ),
    ]

    aligned = align_planned_stops_to_route_geometry(
        planned_stops=stops,
        route_polyline=route_polyline,
        daily_km=70,
        max_days=3,
    )

    assert [stop.name for stop in aligned] == ["Start stop", "Middle stop", "End stop"]
    assert [stop.km_from_start for stop in aligned] == sorted(
        stop.km_from_start for stop in aligned
    )
    assert aligned[0].latitude == 30.0
    assert aligned[1].longitude == -7.0
    assert aligned[2].longitude == -7.0
    assert [stop.estimated_day for stop in aligned] == [1, 2, 3]
