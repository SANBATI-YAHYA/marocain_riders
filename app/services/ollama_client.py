"""
Ollama LLM client service.

Sends prompt requests to a local Ollama instance for text generation.
Supports non-streaming inference (MVP).
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120.0  # generous for large generations


class OllamaClient:
    """Thin wrapper around the Ollama HTTP API."""

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    # ── generation ───────────────────────────────────

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Send a non-streaming generate request and return the full response text."""
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        url = f"{self._base_url}/api/generate"
        try:
            resp = httpx.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s", self._base_url)
            return "[ERROR] Ollama is not reachable. Please ensure it is running."
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama HTTP error: %s", exc)
            return f"[ERROR] Ollama returned status {exc.response.status_code}"
        except Exception as exc:
            logger.error("Ollama generation failed: %s", exc)
            return f"[ERROR] Generation failed: {exc}"

    # ── health ───────────────────────────────────────

    def is_reachable(self) -> bool:
        """Quick check whether Ollama is responding."""
        try:
            resp = httpx.get(f"{self._base_url}/", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
