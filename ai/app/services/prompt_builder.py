"""
Prompt builder for LLM-based itinerary explanation.

Builds a compact, grounded prompt that tells the LLM *exactly* what was
selected by the structured engine and asks it only to explain / narrate —
never to invent new stops, cities, or route branches.

Designed for small local models (gemma2:2b) where short context = faster
and less hallucination.
"""

from __future__ import annotations

from typing import List, Optional

# ── System prompt — hard grounding constraints ───────

SYSTEM_PROMPT = """\
You are a motorcycle travel assistant for Morocco.
STRICT RULES:
1. Use ONLY the cities, stops, and data provided below. Never invent places.
2. The route, stops, and day plan are already decided. Your job is to EXPLAIN them.
3. If the rider wanted a vibe (e.g. coastal) that the route does not provide, \
say so honestly and explain why this route was chosen instead.
4. Keep your answer under 300 words. Be practical. No poetry.
5. Refer to stops by their exact names from the data.
6. Do not add extra days, extra cities, or alternative routes."""


def build_recommendation_prompt(
    *,
    profile_summary: str,
    day_plan_text: str,
    top_route_summary: str,
    stops_text: str,
    weather_text: str,
    vibe_mismatch_note: str = "",
    additional_preferences: Optional[str] = None,
) -> str:
    """Return a compact, grounded prompt for LLM explanation.

    All heavy data (route selection, day splitting, stop assignment)
    is resolved *before* this function is called.  The LLM's only job
    is to narrate the pre-built plan.
    """
    parts: List[str] = []

    # ── Rider snapshot (concise) ─────────────────────
    parts.append("RIDER:")
    parts.append(profile_summary)

    if additional_preferences:
        parts.append(f"Rider's note: {additional_preferences[:150]}")

    # ── Vibe mismatch disclosure ─────────────────────
    if vibe_mismatch_note:
        parts.append(f"\nIMPORTANT: {vibe_mismatch_note}")

    # ── Pre-selected route ───────────────────────────
    parts.append(f"\nSELECTED ROUTE:\n{top_route_summary}")

    # ── Pre-built day plan ───────────────────────────
    parts.append(f"\nDAY PLAN:\n{day_plan_text}")

    # ── Selected stops only ──────────────────────────
    parts.append(f"\nSTOPS:\n{stops_text}")

    # ── Weather / safety ─────────────────────────────
    if weather_text:
        parts.append(f"\nWEATHER/SAFETY:\n{weather_text}")

    # ── Task instruction ─────────────────────────────
    parts.append(
        "\nTASK: Write a short, friendly explanation of this trip plan "
        "for the rider. Mention why this route fits them, summarise each "
        "day briefly, note key stops, and include any safety advice. "
        "Do NOT add places, stops, or days not listed above."
    )

    return "\n".join(parts)
