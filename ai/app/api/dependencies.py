"""
FastAPI dependency injection.

Instantiates shared singletons for DB connection, repositories,
services, and the recommendation engine.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings, Settings
from app.core.database import get_connection
from app.repositories.place_repo import PlaceRepository
from app.repositories.route_repo import RouteRepository
from app.repositories.entity_repos import (
    GasStationRepository,
    RestaurantRepository,
    RouteSegmentRepository,
    StayRepository,
    WeatherRiskRepository,
)
from app.services.json_ingestion import JsonIngestionService
from app.services.text_ingestion import TextIngestionService
from app.services.gpx_parser import GpxParserService
from app.services.ollama_client import OllamaClient
from app.services.recommendation_engine import RecommendationEngine
from app.vector_store.faiss_store import VectorStoreService


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return get_settings()


# ── Database ──────────────────────────────────────────

def get_db():
    return get_connection(_settings().sqlite_db_path)


# ── Repositories ──────────────────────────────────────

def get_place_repo() -> PlaceRepository:
    return PlaceRepository(get_db())

def get_route_repo() -> RouteRepository:
    return RouteRepository(get_db())

def get_segment_repo() -> RouteSegmentRepository:
    return RouteSegmentRepository(get_db())

def get_stay_repo() -> StayRepository:
    return StayRepository(get_db())

def get_restaurant_repo() -> RestaurantRepository:
    return RestaurantRepository(get_db())

def get_gas_station_repo() -> GasStationRepository:
    return GasStationRepository(get_db())

def get_weather_risk_repo() -> WeatherRiskRepository:
    return WeatherRiskRepository(get_db())


# ── Services ──────────────────────────────────────────

def get_json_ingestion_service() -> JsonIngestionService:
    return JsonIngestionService(
        place_repo=get_place_repo(),
        route_repo=get_route_repo(),
        segment_repo=get_segment_repo(),
        stay_repo=get_stay_repo(),
        restaurant_repo=get_restaurant_repo(),
        gas_station_repo=get_gas_station_repo(),
        weather_risk_repo=get_weather_risk_repo(),
    )

def get_text_ingestion_service() -> TextIngestionService:
    return TextIngestionService(conn=get_db())

def get_gpx_parser_service() -> GpxParserService:
    places = get_place_repo().get_all()
    return GpxParserService(places=places)

def get_vector_store() -> VectorStoreService:
    s = _settings()
    return VectorStoreService(
        index_path=s.vector_store_path,
        metadata_path=s.chunk_metadata_path,
        embedding_model=s.embedding_model,
    )

def get_ollama_client() -> OllamaClient:
    s = _settings()
    return OllamaClient(base_url=s.ollama_base_url, model=s.ollama_model)

def get_recommendation_engine() -> RecommendationEngine:
    return RecommendationEngine(
        place_repo=get_place_repo(),
        route_repo=get_route_repo(),
        segment_repo=get_segment_repo(),
        stay_repo=get_stay_repo(),
        restaurant_repo=get_restaurant_repo(),
        gas_station_repo=get_gas_station_repo(),
        weather_risk_repo=get_weather_risk_repo(),
        vector_store=get_vector_store(),
        ollama=get_ollama_client(),
    )
