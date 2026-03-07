"""
GPX file parser service.

Reads .gpx files, extracts waypoints, track points, and route metadata.
Links parsed GPX data to structured route entities.
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.models.gpx_models import (
    GpxSegment,
    GpxTrack,
    GpxTrackPoint,
    GpxWaypoint,
    ParsedGpxFile,
)
from app.models.domain import Place

logger = logging.getLogger(__name__)

# GPX XML namespace
_NS = {"gpx": "http://www.topografix.com/GPX/1/1"}


# ── math helpers ──────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _total_distance(points: List[GpxTrackPoint]) -> float:
    """Cumulative distance along a sequence of track points."""
    total = 0.0
    for i in range(1, len(points)):
        total += _haversine_km(
            points[i - 1].latitude, points[i - 1].longitude,
            points[i].latitude, points[i].longitude,
        )
    return round(total, 2)


def _nearest_places(
    lat: float, lon: float, places: List[Place], max_km: float = 30.0
) -> List[str]:
    """Return place_ids within *max_km* of a coordinate, closest first."""
    hits: List[Tuple[float, str]] = []
    for p in places:
        if p.latitude is not None and p.longitude is not None:
            d = _haversine_km(lat, lon, p.latitude, p.longitude)
            if d <= max_km:
                hits.append((d, p.place_id))
    hits.sort()
    return [pid for _, pid in hits]


# ── public service ────────────────────────────────────

class GpxParserService:
    """Parses GPX files and enriches them with route metadata."""

    def __init__(self, places: Optional[List[Place]] = None) -> None:
        self._places = places or []

    def parse_file(self, gpx_path: Path) -> ParsedGpxFile:
        """Parse a single GPX file and return enriched model."""
        logger.info("Parsing GPX file: %s", gpx_path.name)
        tree = ET.parse(str(gpx_path))
        root = tree.getroot()

        waypoints = self._parse_waypoints(root)
        tracks = self._parse_tracks(root)

        # Try to extract a route name from metadata or first track
        route_name = self._extract_route_name(root, tracks)

        # Collect all track points for distance/endpoint computation
        all_points: List[GpxTrackPoint] = []
        for trk in tracks:
            for seg in trk.segments:
                all_points.extend(seg.points)

        total_distance = _total_distance(all_points) if all_points else None
        start_lat = all_points[0].latitude if all_points else None
        start_lon = all_points[0].longitude if all_points else None
        end_lat = all_points[-1].latitude if all_points else None
        end_lon = all_points[-1].longitude if all_points else None

        # Find nearest known places to start/end
        nearest_ids: List[str] = []
        if start_lat and start_lon:
            nearest_ids.extend(_nearest_places(start_lat, start_lon, self._places))
        if end_lat and end_lon:
            nearest_ids.extend(_nearest_places(end_lat, end_lon, self._places))
        nearest_ids = list(dict.fromkeys(nearest_ids))  # deduplicate preserving order

        return ParsedGpxFile(
            filename=gpx_path.name,
            route_name=route_name,
            waypoints=waypoints,
            tracks=tracks,
            total_distance_km=total_distance,
            start_lat=start_lat,
            start_lon=start_lon,
            end_lat=end_lat,
            end_lon=end_lon,
            total_points=len(all_points),
            nearest_place_ids=nearest_ids,
        )

    def parse_directory(self, gpx_dir: Path) -> List[ParsedGpxFile]:
        """Parse every .gpx file in a directory."""
        results: List[ParsedGpxFile] = []
        for fpath in sorted(gpx_dir.iterdir()):
            if fpath.suffix.lower() == ".gpx":
                results.append(self.parse_file(fpath))
        return results

    # ── internal parsing ─────────────────────────────

    @staticmethod
    def _parse_waypoints(root: ET.Element) -> List[GpxWaypoint]:
        waypoints: List[GpxWaypoint] = []
        for wpt in root.findall("gpx:wpt", _NS):
            lat = float(wpt.get("lat", 0))
            lon = float(wpt.get("lon", 0))
            name_el = wpt.find("gpx:name", _NS)
            sym_el = wpt.find("gpx:sym", _NS)
            ele_el = wpt.find("gpx:ele", _NS)
            waypoints.append(GpxWaypoint(
                name=name_el.text if name_el is not None else None,
                latitude=lat,
                longitude=lon,
                elevation=float(ele_el.text) if ele_el is not None else None,
                symbol=sym_el.text if sym_el is not None else None,
            ))
        return waypoints

    @staticmethod
    def _parse_tracks(root: ET.Element) -> List[GpxTrack]:
        tracks: List[GpxTrack] = []
        for trk in root.findall("gpx:trk", _NS):
            name_el = trk.find("gpx:name", _NS)
            segments: List[GpxSegment] = []
            for trkseg in trk.findall("gpx:trkseg", _NS):
                points: List[GpxTrackPoint] = []
                for trkpt in trkseg.findall("gpx:trkpt", _NS):
                    lat = float(trkpt.get("lat", 0))
                    lon = float(trkpt.get("lon", 0))
                    ele_el = trkpt.find("gpx:ele", _NS)
                    points.append(GpxTrackPoint(
                        latitude=lat,
                        longitude=lon,
                        elevation=float(ele_el.text) if ele_el is not None else None,
                    ))
                if points:
                    segments.append(GpxSegment(points=points))
            tracks.append(GpxTrack(
                name=name_el.text if name_el is not None else None,
                segments=segments,
            ))
        return tracks

    @staticmethod
    def _extract_route_name(root: ET.Element, tracks: List[GpxTrack]) -> Optional[str]:
        # Try metadata link text
        meta = root.find("gpx:metadata", _NS)
        if meta is not None:
            link = meta.find("gpx:link", _NS)
            if link is not None:
                text_el = link.find("gpx:text", _NS)
                if text_el is not None and text_el.text:
                    return text_el.text
        # Fallback: first track name
        for trk in tracks:
            if trk.name:
                return trk.name
        return None
