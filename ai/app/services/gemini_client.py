"""
Google Gemini LLM client service.

Uses the Google GenAI SDK (google-genai) for text generation.
Replaces the previous local Ollama integration with cloud-based Gemini API.
"""

from __future__ import annotations

import logging
from typing import Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_FALLBACK_MSG = (
    "[LLM unavailable] The structured recommendation above is complete. "
    "An AI-generated narrative explanation could not be produced at this time."
)


class GeminiClient:
    """Wrapper around the Google Gemini API for text generation."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_name = model

    # ── generation ───────────────────────────────────

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 1024,
    ) -> str:
        """Generate text via Gemini. Returns text or a user-friendly fallback."""
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            if system:
                config.system_instruction = system

            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=config,
            )
            return response.text

        except Exception as exc:
            logger.error("Gemini generation failed: %s", exc)
            return _FALLBACK_MSG

    # ── health check ─────────────────────────────────

    def is_reachable(self) -> bool:
        """Quick check that the Gemini API is accessible."""
        try:
            models = self._client.models.list()
            return any(True for _ in models)
        except Exception:
            return False
