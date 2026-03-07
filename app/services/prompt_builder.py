"""
Prompt builder for LLM-based itinerary generation.

Assembles a structured prompt from user profile, route candidates,
stops, weather data, and retrieved semantic context.
"""

from __future__ import annotations

from typing import Dict, List, Optional


SYSTEM_PROMPT = """\
You are an expert motorcycle travel advisor specialising in Morocco.
You recommend routes, stops, restaurants, accommodations, fuel stations,
and safety advice based on structured data about the rider and the route.
Never invent place names, distances, or coordinates — use only the data provided.
Always explain *why* your recommendation fits this specific rider.
Format your answer as a clear day-by-day itinerary with practical details.
"""


def build_recommendation_prompt(
    *,
    profile_summary: str,
    top_route: Dict,
    alternative_routes: List[Dict],
    segments_text: str,
    stays_text: str,
    restaurants_text: str,
    fuel_stations_text: str,
    weather_warnings: List[str],
    semantic_chunks: List[str],
    additional_preferences: Optional[str] = None,
) -> str:
    """Return the full prompt string to send to Ollama."""

    parts: List[str] = []

    parts.append("=== RIDER PROFILE ===")
    parts.append(profile_summary)

    if additional_preferences:
        parts.append(f"\nAdditional rider request: {additional_preferences}")

    parts.append("\n=== RECOMMENDED ROUTE ===")
    parts.append(
        f"Route: {top_route.get('route_name', 'N/A')} "
        f"({top_route.get('route_id', '')})\n"
        f"Distance: {top_route.get('distance_km', '?')} km\n"
        f"Difficulty: {top_route.get('difficulty_level', '?')}\n"
        f"Recommended days: {top_route.get('recommended_days', '?')}\n"
        f"Summary: {top_route.get('route_summary', '')}\n"
        f"Why choose: {top_route.get('why_choose_this_route', '')}\n"
        f"Rider advice: {top_route.get('rider_advice', '')}\n"
        f"Safety advice: {top_route.get('safety_advice', '')}"
    )

    if alternative_routes:
        parts.append("\n=== ALTERNATIVE ROUTES ===")
        for alt in alternative_routes[:2]:
            parts.append(
                f"- {alt.get('route_name', '')} ({alt.get('distance_km', '?')} km, "
                f"difficulty: {alt.get('difficulty_level', '?')})"
            )

    parts.append("\n=== ROUTE SEGMENTS ===")
    parts.append(segments_text)

    parts.append("\n=== ACCOMMODATION OPTIONS ===")
    parts.append(stays_text)

    parts.append("\n=== RESTAURANT OPTIONS ===")
    parts.append(restaurants_text)

    parts.append("\n=== FUEL STATIONS ===")
    parts.append(fuel_stations_text)

    if weather_warnings:
        parts.append("\n=== WEATHER & SAFETY WARNINGS ===")
        for w in weather_warnings:
            parts.append(f"- {w}")

    if semantic_chunks:
        parts.append("\n=== ADDITIONAL TRAVEL KNOWLEDGE ===")
        for chunk in semantic_chunks[:2]:
            parts.append(chunk[:200])

    parts.append(
        "\n=== TASK ===\n"
        "Generate a concise day-by-day motorcycle itinerary for this rider.\n"
        "Include: route choice reason, daily plan, eat/sleep/fuel stops, weather notes.\n"
        "Be brief and practical. Use only the data provided."
    )

    return "\n".join(parts)
