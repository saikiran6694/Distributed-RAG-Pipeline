"""
Context expander: enriches retrieved chunks with their text and parent context.

Two problems this solves:

1. Chunk text is not stored in Qdrant (only metadata/payload is).
   Text lives in Postgres (chunks.raw_text — added in schema migration below).
   We batch-fetch text for all retrieved chunk IDs in one query.

2. Hierarchical chunks retrieved at L2 (paragraph level) often lack context.
   "The retention period is 90 days" means nothing without knowing
   it's the refund policy section of a terms-of-service document.
   We fetch the L1 parent chunk and prepend it to the context window.

The LLM prompt builder receives ExpandedChunk objects with:
  - chunk.text         (the specific retrieved paragraph)
  - chunk.parent_text  (the section it belongs to, if hierarchical)
  - chunk.source_url   (for citation)
  - chunk.section_title (for display)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import asyncpg

from query.retriever import RetrievedChunk
from shared.telemetry import traced_span

logger = logging.getLogger(__name__)


@dataclass
class ExpandedChunk:
    """
    A retrieved chunk enriched with text and optional parent context.
    This is what the prompt builder receives.
    """
    chunk_id:        str
    doc_id:          str
    text:            str            # the retrieved chunk's own text
    parent_text:     str | None     # L1 section text (if hierarchy_level == 2)
    score:           float
    source_url:      str
    section_title:   str | None
    page_number:     int | None
    hierarchy_level: int
    chunk_index:     int
    dense_rank:      int | None
    sparse_rank:     int | None

    def to_context_string(self, include_parent: bool = True) -> str:
        """
        Format chunk for inclusion in an LLM prompt.
        Includes section heading and optional parent context.
        """
        parts = []

        if self.section_title:
            parts.append(f"[Section: {self.section_title}]")

        if include_parent and self.parent_text and self.hierarchy_level == 2:
            parts.append(f"Context: {self.parent_text.strip()}")
            parts.append("---")

        parts.append(self.text.strip())

        return "\n".join(parts)

    def to_citation(self) -> dict:
        """Structured citation for the UI."""
        return {
            "source_url":    self.source_url,
            "section_title": self.section_title,
            "page_number":   self.page_number,
            "chunk_index":   self.chunk_index,
            "score":         self.score,
        }


class ContextExpander:
    """
    Enriches RetrievedChunk objects with text from Postgres.

    Two operations per expansion:
      1. Batch fetch chunk texts by chunk_id
      2. For L2 chunks: fetch parent L1 text by parent_chunk_id

    Both are done in single queries (IN clause) for efficiency.
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self._db = db_pool

    async def expand(self, chunks: list[RetrievedChunk]) -> list[ExpandedChunk]:
        """
        Enrich all retrieved chunks with text and parent context.
        Returns ExpandedChunk list in the same order as input.
        """
        if not chunks:
            return []

        with traced_span("context_expander.expand", {"chunk_count": len(chunks)}):
            chunk_ids = [c.chunk_id for c in chunks]

            # Batch fetch chunk texts
            text_map = await self._fetch_chunk_texts(chunk_ids)

            # Collect parent IDs for L2 chunks
            parent_ids = [
                c.parent_chunk_id
                for c in chunks
                if c.hierarchy_level == 2 and c.parent_chunk_id
            ]
            parent_text_map = await self._fetch_chunk_texts(parent_ids) if parent_ids else {}

            expanded = []
            for chunk in chunks:
                text = text_map.get(chunk.chunk_id, "")
                if not text:
                    logger.warning("No text found in Postgres for chunk_id=%s", chunk.chunk_id)

                parent_text = None
                if chunk.hierarchy_level == 2 and chunk.parent_chunk_id:
                    parent_text = parent_text_map.get(chunk.parent_chunk_id)

                expanded.append(ExpandedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=text,
                    parent_text=parent_text,
                    score=chunk.score,
                    source_url=chunk.source_url,
                    section_title=chunk.section_title,
                    page_number=chunk.page_number,
                    hierarchy_level=chunk.hierarchy_level,
                    chunk_index=chunk.chunk_index,
                    dense_rank=chunk.dense_rank,
                    sparse_rank=chunk.sparse_rank,
                ))

        return expanded

    async def _fetch_chunk_texts(self, chunk_ids: list[str]) -> dict[str, str]:
        """
        Batch-fetch chunk texts from Postgres by ID.
        Returns {chunk_id: text} mapping.
        """
        if not chunk_ids:
            return {}

        rows = await self._db.fetch(
            """
            SELECT id::text, raw_text
            FROM chunks
            WHERE id = ANY($1::uuid[])
            """,
            chunk_ids,
        )
        return {str(r["id"]): r["raw_text"] for r in rows}
