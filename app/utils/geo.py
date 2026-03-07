"""
Geo-math and general utility helpers.
"""

from __future__ import annotations

import math
from typing import Tuple


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (km) between two WGS-84 points."""
    R = 6371.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def midpoint(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
    """Geographic midpoint between two WGS-84 points."""
    rlat1 = math.radians(lat1)
    rlon1 = math.radians(lon1)
    rlat2 = math.radians(lat2)
    rlon2 = math.radians(lon2)
    dlon = rlon2 - rlon1
    bx = math.cos(rlat2) * math.cos(dlon)
    by = math.cos(rlat2) * math.sin(dlon)
    mid_lat = math.atan2(
        math.sin(rlat1) + math.sin(rlat2),
        math.sqrt((math.cos(rlat1) + bx) ** 2 + by ** 2),
    )
    mid_lon = rlon1 + math.atan2(by, math.cos(rlat1) + bx)
    return math.degrees(mid_lat), math.degrees(mid_lon)
