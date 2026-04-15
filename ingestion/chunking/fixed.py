"""
Fixed-size chunker with token-aware splitting and configurable overlap.

Strategy: split every N tokens, carry the last `overlap` tokens of
chunk N into the start of chunk N+1. Fast and deterministic.

Best for: code files, structured data exports, uniform content.
Avoid for: narrative text where topic shifts mid-chunk are common.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import tiktoken

from shared.config import get_settings
from shared.models import Chunk, ChunkingStrategy, ParsedDocument

logger = logging.getLogger(__name__)
settings = get_settings()

# Use cl100k_base tokenizer (same as OpenAI text-embedding-3-*)
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


class FixedChunker:
    """
    Token-aware fixed-size chunker.

    Args:
        chunk_size:    Target chunk size in tokens (default from config)
        overlap:       Overlap in tokens between consecutive chunks (default from config)
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        overlap:    int | None = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_FIXED_SIZE_TOKENS
        self.overlap    = overlap    or settings.CHUNK_FIXED_OVERLAP_TOKENS

        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"Overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})"
            )

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """
        Split document into fixed-size token chunks with overlap.
        Preserves section title context from the nearest heading above each chunk.
        """
        if not document.raw_text.strip():
            logger.warning("Empty document text for doc_id=%s", document.doc_id)
            return []

        tokens = _TOKENIZER.encode(document.raw_text)
        if not tokens:
            return []

        # Build a section title lookup: token_position → nearest section title
        section_map = self._build_section_map(document)

        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            text = _TOKENIZER.decode(chunk_tokens)

            if text.strip():
                section_title = self._nearest_section(section_map, start, document.raw_text)
                chunks.append(
                    Chunk(
                        id=uuid4(),         # will be overridden by content hash in validator
                        doc_id=document.doc_id,
                        chunk_index=chunk_index,
                        text=text,
                        section_title=section_title,
                        source_url=document.source_url,
                        chunking_strategy=ChunkingStrategy.FIXED,
                        chunk_overlap_tokens=self.overlap if chunk_index > 0 else 0,
                    )
                )
                chunk_index += 1

            # Advance by (chunk_size - overlap) to create the sliding window
            start += self.chunk_size - self.overlap

        logger.debug(
            "Fixed chunker produced %d chunks for doc_id=%s (tokens=%d)",
            len(chunks), document.doc_id, len(tokens),
        )
        return chunks

    def _build_section_map(self, document: ParsedDocument) -> dict[int, str]:
        """
        Map character positions of section titles to their token positions.
        Used to annotate each chunk with the nearest preceding heading.
        """
        mapping: dict[int, str] = {}
        full_text = document.raw_text

        for section in document.sections:
            if not section.title:
                continue
            pos = full_text.find(section.content[:50])  # find section start
            if pos >= 0:
                token_pos = len(_TOKENIZER.encode(full_text[:pos]))
                mapping[token_pos] = section.title

        return mapping

    def _nearest_section(
        self,
        section_map: dict[int, str],
        token_start: int,
        full_text: str,
    ) -> str | None:
        """Return the title of the section that precedes token_start."""
        best_pos = -1
        best_title = None
        for pos, title in section_map.items():
            if pos <= token_start and pos > best_pos:
                best_pos = pos
                best_title = title
        return best_title