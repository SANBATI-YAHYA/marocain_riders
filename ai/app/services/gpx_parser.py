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


# ── Douglas-Peucker line simplification ───────────────

def _perpendicular_distance(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:
    """Perpendicular distance from *point* to line(line_start→line_end) in degrees.

    Good enough for small-area simplification (Morocco).
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(x0 - x1, y0 - y1)
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(x0 - proj_x, y0 - proj_y)


def _project_point_onto_segment(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> Tuple[Tuple[float, float], float, float]:
    """Project *point* onto a segment.

    Returns ``((lat, lon), t, distance)`` where ``t`` is the clamped progress
    ratio along the segment and ``distance`` is the cartesian offset in degrees.
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return (x1, y1), 0.0, math.hypot(x0 - x1, y0 - y1)
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return (proj_x, proj_y), t, math.hypot(x0 - proj_x, y0 - proj_y)


def _douglas_peucker(
    coords: List[Tuple[float, float]], epsilon: float
) -> List[Tuple[float, float]]:
    """Simplify a polyline using the Douglas-Peucker algorithm."""
    if len(coords) <= 2:
        return list(coords)

    # Find the point with max distance from the line start→end
    max_dist = 0.0
    max_idx = 0
    for i in range(1, len(coords) - 1):
        d = _perpendicular_distance(coords[i], coords[0], coords[-1])
        if d > max_dist:
            max_dist = d
            max_idx = i

    if max_dist > epsilon:
        left = _douglas_peucker(coords[: max_idx + 1], epsilon)
        right = _douglas_peucker(coords[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [coords[0], coords[-1]]


def simplify_track(
    points: List[Tuple[float, float]],
    max_points: int = 800,
) -> List[Tuple[float, float]]:
    """Simplify a track to at most *max_points*, preserving shape.

    Uses iterative Douglas-Peucker with increasing epsilon.
    """
    if len(points) <= max_points:
        return points

    # Start with a small epsilon and grow until under budget
    epsilon = 0.0001  # ~11 m at equator
    simplified = points
    for _ in range(30):
        simplified = _douglas_peucker(points, epsilon)
        if len(simplified) <= max_points:
            break
        epsilon *= 1.8
    return simplified


def snap_point_to_polyline(
    point: Tuple[float, float],
    polyline: List[Tuple[float, float]],
) -> Optional[Tuple[float, float, float]]:
    """Snap a point to the nearest position on a route polyline.

    Returns ``(latitude, longitude, km_from_start)`` or ``None`` when the
    polyline cannot be projected.
    """
    if len(polyline) < 2:
        return None

    best_match: Optional[Tuple[float, float, float, float]] = None
    cum_km = 0.0

    for start, end in zip(polyline, polyline[1:]):
        seg_km = _haversine_km(start[0], start[1], end[0], end[1])
        snapped, ratio, distance = _project_point_onto_segment(point, start, end)
        km_from_start = cum_km + seg_km * ratio

        if best_match is None or distance < best_match[3]:
            best_match = (snapped[0], snapped[1], km_from_start, distance)

        cum_km += seg_km

    if best_match is None:
        return None

    return (
        round(best_match[0], 6),
        round(best_match[1], 6),
        round(best_match[2], 1),
    )


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

    # ── route geometry extraction ─────────────────

    def get_route_geometry(
        self,
        gpx_name: str,
        max_points: int = 800,
    ) -> Optional[List[List[float]]]:
        """Return simplified [[lat, lng], ...] for a GPX file.

        Looks up the file in the configured GPX directory.
        Returns *None* if the file is not found.
        """
        from app.core.config import get_settings
        gpx_dir = get_settings().gpx_data_dir

        # Try exact match first
        target = gpx_dir / gpx_name
        if not target.exists():
            # Fallback: find any GPX file (useful when only one exists)
            candidates = list(gpx_dir.glob("*.gpx"))
            if len(candidates) == 1:
                target = candidates[0]
                logger.info("GPX exact match for '%s' not found; using '%s'", gpx_name, target.name)
            else:
                logger.warning("GPX file '%s' not found and %d candidates in %s", gpx_name, len(candidates), gpx_dir)
                return None

        parsed = self.parse_file(target)
        all_points: List[Tuple[float, float]] = []
        for trk in parsed.tracks:
            for seg in trk.segments:
                for pt in seg.points:
                    all_points.append((pt.latitude, pt.longitude))

        if not all_points:
            return None

        simplified = simplify_track(all_points, max_points=max_points)
        return [[lat, lng] for lat, lng in simplified]

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
