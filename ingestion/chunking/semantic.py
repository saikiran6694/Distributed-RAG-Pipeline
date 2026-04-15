"""
ingestion/chunking/semantic.py

Semantic chunker: splits at points where topic shifts.

Algorithm:
  1. Split document into sentences
  2. Embed each sentence with a lightweight sentence-transformer
  3. Compute cosine similarity between adjacent sentence embeddings
  4. Split where similarity drops below a threshold (topic boundary)
  5. Merge small segments up to max_tokens

This produces chunks that are semantically coherent — each chunk
is about one thing — unlike fixed-size which splits blindly.

Best for: articles, documentation, PDFs with narrative text.
Cost: ~2x slower than fixed due to sentence embedding step.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from shared.config import get_settings
from shared.models import Chunk, ChunkingStrategy, ParsedDocument

logger = logging.getLogger(__name__)
settings = get_settings()

_TOKENIZER  = tiktoken.get_encoding("cl100k_base")

# Lightweight model — fast CPU inference, good sentence similarity
_SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_sentence_model: SentenceTransformer | None = None


def _get_sentence_model() -> SentenceTransformer:
    """Lazy-load the sentence model once (expensive to initialize)."""
    global _sentence_model
    if _sentence_model is None:
        logger.info("Loading sentence-transformer model: %s", _SENTENCE_MODEL_NAME)
        _sentence_model = SentenceTransformer(_SENTENCE_MODEL_NAME, device=settings.HF_DEVICE)
    return _sentence_model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticChunker:
    """
    Sentence-similarity based semantic chunker.

    Args:
        threshold:  Cosine similarity below which a split is inserted (0.0-1.0)
        min_tokens: Minimum tokens per chunk (prevents micro-chunks)
        max_tokens: Maximum tokens per chunk (safety ceiling)
    """

    def __init__(
        self,
        threshold:  float | None = None,
        min_tokens: int   | None = None,
        max_tokens: int   | None = None,
    ):
        self.threshold  = threshold  or settings.CHUNK_SEMANTIC_THRESHOLD
        self.min_tokens = min_tokens or settings.CHUNK_SEMANTIC_MIN_TOKENS
        self.max_tokens = max_tokens or settings.CHUNK_SEMANTIC_MAX_TOKENS

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        if not document.raw_text.strip():
            return []

        sentences = self._split_sentences(document.raw_text)
        if len(sentences) < 2:
            # Single sentence or very short doc — return as one chunk
            return [self._make_chunk(document, document.raw_text, 0, None)]

        # Embed all sentences in one batched call
        model = _get_sentence_model()
        embeddings = model.encode(sentences, batch_size=64, show_progress_bar=False)

        # Find split points (where similarity drops below threshold)
        split_indices = self._find_split_points(sentences, embeddings)

        # Build raw segments from split points
        segments = self._build_segments(sentences, split_indices)

        # Merge segments that are below min_tokens, enforce max_tokens ceiling
        merged = self._merge_segments(segments)

        # Annotate with section context
        section_map = self._build_section_map(document)

        chunks = []
        for idx, text in enumerate(merged):
            section_title = self._nearest_section(section_map, text, document.raw_text)
            chunks.append(self._make_chunk(document, text, idx, section_title))

        logger.debug(
            "Semantic chunker produced %d chunks for doc_id=%s",
            len(chunks), document.doc_id,
        )
        return chunks

    # ─────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> list[str]:
        """
        Sentence tokenizer using pure regex — zero external dependencies.
        No NLTK data downloads, works in any environment including CI.

        Strategy: naive split on .!? + uppercase, then re-join false positives
        caused by abbreviations (Dr., Mr., St., etc.) using a post-pass merge.
        """
        import re
        _abbrevs = re.compile(
            r"\b(Mr|Ms|Mrs|Dr|St|Prof|vs|etc|al|fig|no|vol|pp|Jr|Sr)\.$",
            re.IGNORECASE,
        )
        sentences: list[str] = []

        for para in re.split(r"\n{2,}", text):
            para = para.strip()
            if not para:
                continue

            # Naive split: .!? followed by whitespace + uppercase or quote
            raw = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', para)

            # Re-join fragments where the split was on an abbreviation
            merged: list[str] = []
            buf = ""
            for part in raw:
                if buf:
                    if _abbrevs.search(buf.rstrip()):
                        buf = buf + " " + part   # abbreviation — keep joining
                    else:
                        merged.append(buf)
                        buf = part
                else:
                    buf = part
            if buf:
                merged.append(buf)

            sentences.extend(s.strip() for s in merged if s.strip())

        return sentences if sentences else [text]

    def _find_split_points(
        self,
        sentences:  list[str],
        embeddings: np.ndarray,
    ) -> list[int]:
        """
        Return indices AFTER which a chunk boundary should be inserted.
        A boundary is inserted where cosine similarity < threshold.
        """
        split_after: list[int] = []
        for i in range(len(sentences) - 1):
            sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self.threshold:
                split_after.append(i)
        return split_after

    def _build_segments(
        self,
        sentences:    list[str],
        split_after:  list[int],
    ) -> list[str]:
        """Group sentences into segments using the split indices."""
        segments: list[str] = []
        current: list[str] = []

        for i, sentence in enumerate(sentences):
            current.append(sentence)
            if i in split_after:
                segments.append(" ".join(current))
                current = []

        if current:
            segments.append(" ".join(current))

        return segments

    def _merge_segments(self, segments: list[str]) -> list[str]:
        """
        Merge consecutive small segments and enforce max_tokens ceiling.
        Guarantees: min_tokens <= len(chunk) <= max_tokens (approximately).
        """
        merged: list[str] = []
        buffer = ""
        buffer_tokens = 0

        for segment in segments:
            seg_tokens = len(_TOKENIZER.encode(segment))

            if buffer_tokens + seg_tokens > self.max_tokens and buffer:
                merged.append(buffer.strip())
                buffer = segment
                buffer_tokens = seg_tokens
            else:
                if not buffer and seg_tokens > self.max_tokens:
                    # Oversized single segment — hard split it with the tokenizer
                    tokens = _TOKENIZER.encode(segment)
                    for i in range(0, len(tokens), self.max_tokens):
                        chunk_text = _TOKENIZER.decode(tokens[i:i + self.max_tokens]).strip()
                        if chunk_text:
                            merged.append(chunk_text)
                    continue

                buffer = (buffer + " " + segment).strip() if buffer else segment
                buffer_tokens += seg_tokens

                if buffer_tokens >= self.min_tokens:
                    # Segment is big enough — can flush now if next one would exceed max
                    pass  # flush lazily on next iteration

        if buffer.strip():
            merged.append(buffer.strip())

        return [m for m in merged if m.strip()]

    def _make_chunk(
        self,
        document: ParsedDocument,
        text: str,
        index: int,
        section_title: str | None,
    ) -> Chunk:
        return Chunk(
            id=uuid4(),
            doc_id=document.doc_id,
            chunk_index=index,
            text=text,
            section_title=section_title,
            source_url=document.source_url,
            chunking_strategy=ChunkingStrategy.SEMANTIC,
        )

    def _build_section_map(self, document: ParsedDocument) -> dict[str, str]:
        """Map first 50 chars of section content → section title."""
        return {
            section.content[:50]: section.title
            for section in document.sections
            if section.title and section.content
        }

    def _nearest_section(
        self,
        section_map: dict[str, str],
        chunk_text: str,
        full_text: str,
    ) -> str | None:
        """Find which section this chunk falls within by text proximity."""
        full_lower = full_text.lower()
        chunk_start = full_lower.find(chunk_text[:40].lower())
        if chunk_start < 0:
            return None

        best_pos = -1
        best_title = None
        for prefix, title in section_map.items():
            pos = full_lower.find(prefix[:30].lower())
            if 0 <= pos <= chunk_start and pos > best_pos:
                best_pos = pos
                best_title = title
        return best_title
