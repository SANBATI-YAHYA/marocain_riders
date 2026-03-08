"""
Ollama LLM client service.

Sends prompt requests to a local Ollama instance for text generation.
Includes health/model checks and graceful fallback for MVP.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

# ── defaults ──────────────────────────────────────────
DEFAULT_TIMEOUT = 180.0       # total read timeout (seconds)
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_MAX_TOKENS = 512      # keep small for 2B models
DEFAULT_TEMPERATURE = 0.4     # lower = less creative = fewer hallucinations

_FALLBACK_MSG = (
    "[LLM unavailable] The structured recommendation above is complete. "
    "An AI-generated narrative explanation could not be produced at this time."
)


class OllamaClient:
    """Thin wrapper around the Ollama HTTP API with health checks."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = httpx.Timeout(timeout, connect=DEFAULT_CONNECT_TIMEOUT)

    # ── generation ───────────────────────────────────

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Non-streaming generate. Returns text or a user-friendly error."""
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
            resp = httpx.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")

        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s", self._base_url)
            return _FALLBACK_MSG

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 404:
                # Most likely: model not pulled yet
                logger.error(
                    "Ollama 404 — model '%s' is probably not installed. "
                    "Run:  ollama pull %s", self._model, self._model,
                )
                return (
                    f"[ERROR] Model '{self._model}' not found on Ollama. "
                    f"Run: ollama pull {self._model}"
                )
            logger.error("Ollama HTTP %d: %s", status, exc)
            return f"[ERROR] Ollama returned HTTP {status}."

        except httpx.ReadTimeout:
            logger.error("Ollama generation timed out after %.0fs", self._timeout.read)
            return (
                "[ERROR] LLM generation timed out. "
                "The structured recommendation above is still valid."
            )

        except Exception as exc:
            logger.error("Ollama generation failed: %s", exc)
            return _FALLBACK_MSG

    # ── health / diagnostics ─────────────────────────

    def is_reachable(self) -> bool:
        """Quick check whether Ollama daemon is responding."""
        try:
            resp = httpx.get(f"{self._base_url}/", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def is_model_available(self) -> bool:
        """Check whether the configured model is actually pulled."""
        try:
            resp = httpx.get(
                f"{self._base_url}/api/tags", timeout=10.0,
            )
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            # Model names may include tags like "gemma2:2b"
            available = {m.get("name", "") for m in models}
            # Also match without tag  (e.g. "gemma2:2b" ⊂ "gemma2:2b")
            return any(self._model in name or name in self._model for name in available)
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return names of locally-available models."""
        try:
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=10.0)
            if resp.status_code == 200:
                return [m.get("name", "?") for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []
