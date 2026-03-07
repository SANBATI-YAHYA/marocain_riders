"""
FAISS-backed vector store for semantic search over text chunks.

Uses sentence-transformers `all-MiniLM-L6-v2` for embeddings.
Chunk metadata is persisted alongside the FAISS index.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports — heavy libraries loaded only when needed
_model = None
_index = None
_chunk_meta: List[Dict] = []


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load the SentenceTransformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
    return _model


def _get_faiss():
    """Lazy-import faiss."""
    import faiss
    return faiss


# ── public API ────────────────────────────────────────

class VectorStoreService:
    """Manages a FAISS flat (L2) index with chunk metadata."""

    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._index_path = index_path
        self._metadata_path = metadata_path
        self._embedding_model = embedding_model
        self._dimension: Optional[int] = None

        # Try loading existing index
        self._load()

    # ── build / add ──────────────────────────────────

    def add_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
    ) -> int:
        """Embed *texts* and add them (with metadata) to the index."""
        faiss = _get_faiss()
        model = _get_model(self._embedding_model)

        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        global _index, _chunk_meta
        if _index is None:
            self._dimension = embeddings.shape[1]
            _index = faiss.IndexFlatIP(self._dimension)  # inner product on normalised vecs

        _index.add(embeddings)

        meta = metadata or [{} for _ in texts]
        for cid, txt, m in zip(chunk_ids, texts, meta):
            _chunk_meta.append({"chunk_id": cid, "text_preview": txt[:200], **m})

        self._save()
        logger.info("Added %d vectors to FAISS index (total: %d)", len(texts), _index.ntotal)
        return len(texts)

    # ── search ───────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return the top-k most similar chunks for a natural-language query."""
        global _index, _chunk_meta
        if _index is None or _index.ntotal == 0:
            logger.warning("Vector store is empty — returning no results")
            return []

        model = _get_model(self._embedding_model)
        q_emb = model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")

        distances, indices = _index.search(q_emb, min(top_k, _index.ntotal))

        results: List[Dict] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(_chunk_meta[idx])
            entry["score"] = float(dist)
            results.append(entry)
        return results

    # ── persistence ──────────────────────────────────

    def _save(self) -> None:
        global _index, _chunk_meta
        if _index is None:
            return
        faiss = _get_faiss()
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(_index, str(self._index_path))
        self._metadata_path.write_text(
            json.dumps(_chunk_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        global _index, _chunk_meta
        faiss = _get_faiss()
        if self._index_path.exists() and self._metadata_path.exists():
            try:
                _index = faiss.read_index(str(self._index_path))
                _chunk_meta = json.loads(self._metadata_path.read_text(encoding="utf-8"))
                self._dimension = _index.d
                logger.info("Loaded FAISS index with %d vectors", _index.ntotal)
            except Exception as exc:
                logger.warning("Could not load FAISS index: %s", exc)
                _index = None
                _chunk_meta = []

    @property
    def total_vectors(self) -> int:
        global _index
        return _index.ntotal if _index is not None else 0

    def clear(self) -> None:
        """Remove all vectors and metadata — fresh start."""
        global _index, _chunk_meta
        _index = None
        _chunk_meta = []
        # Delete persisted files
        if self._index_path.exists():
            self._index_path.unlink()
        if self._metadata_path.exists():
            self._metadata_path.unlink()
        logger.info("Cleared FAISS index and metadata")

    def is_ready(self) -> bool:
        return self.total_vectors > 0
