"""
GPX-specific internal data structures.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class GpxWaypoint(BaseModel):
    """A single waypoint from a GPX file."""
    name: Optional[str] = None
    latitude: float
    longitude: float
    elevation: Optional[float] = None
    symbol: Optional[str] = None


class GpxTrackPoint(BaseModel):
    """A single track point."""
    latitude: float
    longitude: float
    elevation: Optional[float] = None


class GpxSegment(BaseModel):
    """A contiguous segment of track points."""
    points: List[GpxTrackPoint]


class GpxTrack(BaseModel):
    """A named track composed of segments."""
    name: Optional[str] = None
    segments: List[GpxSegment] = []


class ParsedGpxFile(BaseModel):
    """Complete parsed GPX file representation."""
    filename: str
    route_name: Optional[str] = None
    waypoints: List[GpxWaypoint] = []
    tracks: List[GpxTrack] = []

    # computed metadata
    total_distance_km: Optional[float] = None
    start_lat: Optional[float] = None
    start_lon: Optional[float] = None
    end_lat: Optional[float] = None
    end_lon: Optional[float] = None
    total_points: int = 0
    nearest_place_ids: List[str] = []
