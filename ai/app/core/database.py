"""
Database initialisation helpers.

Creates all required SQLite tables for the structured knowledge base.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

_connection: Optional[sqlite3.Connection] = None

# ── Schema DDL ────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS places (
    place_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    region          TEXT,
    province        TEXT,
    latitude        REAL,
    longitude       REAL,
    place_type      TEXT,
    primary_vibe    TEXT,
    secondary_vibes TEXT,          -- JSON array stored as text
    budget_level    TEXT,
    recommended_stay_duration TEXT,
    best_season     TEXT,
    altitude_category TEXT,
    road_accessibility TEXT,
    bike_type_suitability TEXT,    -- JSON array
    beginner_friendliness INTEGER,
    scenic_score    INTEGER,
    comfort_score   INTEGER,
    fuel_access_score INTEGER,
    available_activities TEXT,     -- JSON array
    safety_notes    TEXT,
    weather_notes   TEXT
);

CREATE TABLE IF NOT EXISTS routes (
    route_id            TEXT PRIMARY KEY,
    route_name          TEXT NOT NULL,
    linked_gpx_name     TEXT,
    start_place_id      TEXT,
    end_place_id        TEXT,
    via_place_ids       TEXT,      -- JSON array
    region              TEXT,
    route_type          TEXT,
    distance_km         REAL,
    estimated_ride_hours TEXT,
    recommended_days    TEXT,
    difficulty_level    TEXT,
    bike_type_suitability TEXT,   -- JSON array
    beginner_friendliness INTEGER,
    road_surface_summary TEXT,
    elevation_profile   TEXT,
    scenic_features     TEXT,     -- JSON array
    hazard_level        TEXT,
    fuel_gap_km         REAL,
    cell_coverage_level TEXT,
    best_season         TEXT,
    avoid_season        TEXT,
    weather_dangers     TEXT,     -- JSON array
    recommended_stay_ids TEXT,    -- JSON array
    recommended_restaurant_ids TEXT,
    recommended_fuel_station_ids TEXT,
    route_summary       TEXT,
    rider_advice        TEXT,
    safety_advice       TEXT,
    why_choose_this_route TEXT
);

CREATE TABLE IF NOT EXISTS route_segments (
    segment_id          TEXT PRIMARY KEY,
    route_id            TEXT,
    segment_order       INTEGER,
    segment_name        TEXT,
    start_latitude      REAL,
    start_longitude     REAL,
    end_latitude        REAL,
    end_longitude       REAL,
    distance_km         REAL,
    estimated_duration  TEXT,
    surface_type        TEXT,
    difficulty_level    TEXT,
    hazard_notes        TEXT,
    scenic_notes        TEXT,
    nearest_place_id    TEXT,
    nearest_restaurant_id TEXT,
    nearest_stay_id     TEXT,
    nearest_fuel_station_id TEXT,
    stop_recommendation TEXT,
    FOREIGN KEY (route_id) REFERENCES routes(route_id)
);

CREATE TABLE IF NOT EXISTS stays (
    stay_id             TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    place_id            TEXT,
    region              TEXT,
    latitude            REAL,
    longitude           REAL,
    stay_type           TEXT,
    price_range         TEXT,
    facilities          TEXT,    -- JSON array
    motorbike_friendly  INTEGER, -- boolean 0/1
    secure_parking      INTEGER,
    hot_shower          INTEGER,
    breakfast_available  INTEGER,
    private_room_available INTEGER,
    group_friendly      INTEGER,
    booking_recommended TEXT,
    ideal_for           TEXT,    -- JSON array
    FOREIGN KEY (place_id) REFERENCES places(place_id)
);

CREATE TABLE IF NOT EXISTS restaurants (
    restaurant_id       TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    place_id            TEXT,
    region              TEXT,
    latitude            REAL,
    longitude           REAL,
    cuisine_type        TEXT,
    price_level         TEXT,
    vibe                TEXT,
    biker_friendly      INTEGER,
    quick_stop_friendly INTEGER,
    group_friendly      INTEGER,
    meal_type           TEXT,    -- JSON array
    ideal_for           TEXT,    -- JSON array
    FOREIGN KEY (place_id) REFERENCES places(place_id)
);

CREATE TABLE IF NOT EXISTS gas_stations (
    station_id          TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    place_id            TEXT,
    region              TEXT,
    latitude            REAL,
    longitude           REAL,
    fuel_types          TEXT,    -- JSON array
    open_24h            INTEGER,
    hours               TEXT,
    payment_notes       TEXT,
    reliability_notes   TEXT,
    air_pump_available  INTEGER,
    basic_repair_nearby INTEGER,
    critical_gap_note   TEXT,
    FOREIGN KEY (place_id) REFERENCES places(place_id)
);

CREATE TABLE IF NOT EXISTS weather_risks (
    risk_id             TEXT PRIMARY KEY,
    related_route_id    TEXT,
    related_route_ids   TEXT,    -- JSON array (some rows have multiple)
    related_place_id    TEXT,
    related_place_ids   TEXT,    -- JSON array
    region              TEXT,
    season              TEXT,
    risk_type           TEXT,
    severity            TEXT,
    description         TEXT,
    rider_advice        TEXT
);

CREATE TABLE IF NOT EXISTS text_chunks (
    chunk_id            TEXT PRIMARY KEY,
    source_type         TEXT,
    source_file         TEXT,
    title               TEXT,
    related_entity_id   TEXT,
    chunk_text          TEXT,
    chunk_index         INTEGER
);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Return a shared SQLite connection (creates file + tables if needed)."""
    global _connection
    if _connection is not None:
        return _connection

    db_path.parent.mkdir(parents=True, exist_ok=True)
    _connection = sqlite3.connect(str(db_path), check_same_thread=False)
    _connection.row_factory = sqlite3.Row
    _connection.executescript(_SCHEMA_SQL)
    return _connection


def close_connection() -> None:
    """Close the shared connection if open."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None
