"""
Text / Markdown / PDF-derived content ingestion service.

Reads text files, chunks them intelligently, and prepares them
for embedding and storage.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from app.models.domain import TextChunk

logger = logging.getLogger(__name__)

# ── chunking parameters ──────────────────────────────

DEFAULT_CHUNK_SIZE = 600      # tokens ≈ words
DEFAULT_CHUNK_OVERLAP = 80


# ── helpers ──────────────────────────────────────────

def _make_chunk_id(source: str, index: int) -> str:
    digest = hashlib.md5(f"{source}:{index}".encode()).hexdigest()[:12]
    return f"CHK-{digest}"


def _extract_entity_id(title: str) -> Optional[str]:
    """Try to pull a PL/RT/ST/RS/GS id from the section title."""
    m = re.search(r"(PL\d{3}|RT\d{3}|ST\d{3}[a-z]?|RS\d{3}|GS\d{3}|SEG\d{3})", title)
    return m.group(1) if m else None


# ── main service ─────────────────────────────────────

class TextIngestionService:
    """Loads text/markdown files, chunks, and stores metadata in SQLite."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._conn = conn
        self._chunk_size = chunk_size
        self._overlap = chunk_overlap

    def ingest_directory(self, docs_dir: Path) -> Dict[str, int]:
        """Ingest all .txt and .md files from a directory."""
        total_chunks = 0
        files_processed = 0

        for fpath in sorted(docs_dir.iterdir()):
            if fpath.suffix.lower() not in (".txt", ".md"):
                continue
            chunks = self._ingest_file(fpath)
            total_chunks += len(chunks)
            files_processed += 1

        return {"files": files_processed, "chunks": total_chunks}

    def ingest_file(self, file_path: Path) -> List[TextChunk]:
        """Public single-file entry point."""
        return self._ingest_file(file_path)

    # ── internal ────────────────────────────────────

    def _ingest_file(self, file_path: Path) -> List[TextChunk]:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        source_type = "markdown" if file_path.suffix == ".md" else "text"
        sections = self._split_into_sections(text)

        all_chunks: List[TextChunk] = []
        for title, body in sections:
            entity_id = _extract_entity_id(title)
            chunks = self._chunk_text(body)
            for idx, chunk_text in enumerate(chunks):
                chunk = TextChunk(
                    chunk_id=_make_chunk_id(f"{file_path.name}:{title}", idx),
                    source_type=source_type,
                    source_file=file_path.name,
                    title=title or file_path.stem,
                    related_entity_id=entity_id,
                    chunk_text=chunk_text,
                    chunk_index=idx,
                )
                all_chunks.append(chunk)
                self._store_chunk(chunk)

        logger.info("Ingested %d chunks from %s", len(all_chunks), file_path.name)
        return all_chunks

    def _split_into_sections(self, text: str) -> List[tuple]:
        """Split markdown by ## or ### headings.  Returns (title, body) pairs."""
        pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
        parts = pattern.split(text)

        sections: List[tuple] = []
        # parts[0] is text before first heading (if any)
        if parts[0].strip():
            sections.append(("introduction", parts[0].strip()))

        i = 1
        while i < len(parts) - 1:
            # parts[i] = "##" or "###", parts[i+1] = heading text, parts[i+2] = body
            title = parts[i + 1].strip()
            body = parts[i + 2].strip() if (i + 2) < len(parts) else ""
            if body:
                sections.append((title, body))
            i += 3

        # Fallback: if no headings found, treat entire text as one section
        if not sections:
            sections.append((Path("doc").stem, text.strip()))

        return sections

    def _chunk_text(self, text: str) -> List[str]:
        """Split long text into overlapping word-chunks."""
        words = text.split()
        if len(words) <= self._chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start + self._chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            start = end - self._overlap

        return chunks

    def _store_chunk(self, chunk: TextChunk) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO text_chunks
               (chunk_id, source_type, source_file, title,
                related_entity_id, chunk_text, chunk_index)
               VALUES (?,?,?,?,?,?,?)""",
            (
                chunk.chunk_id, chunk.source_type, chunk.source_file,
                chunk.title, chunk.related_entity_id,
                chunk.chunk_text, chunk.chunk_index,
            ),
        )
        self._conn.commit()

    def get_all_chunks(self) -> List[TextChunk]:
        """Retrieve all stored text chunks."""
        cur = self._conn.execute("SELECT * FROM text_chunks ORDER BY source_file, chunk_index")
        return [TextChunk(**dict(r)) for r in cur.fetchall()]
