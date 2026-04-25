"""
CrossEncoder reranker: re-scores retrieved chunks by passing
(query, chunk_text) pairs directly through a cross-encoder model.

Why reranking matters:
  Bi-encoders (used for retrieval) embed query and chunk independently
  then compare vectors. Fast, but loses the interaction signal between
  query and document — the model never sees them together.

  A cross-encoder sees the full (query, chunk) pair concatenated and
  produces a single relevance score. Much more accurate, but O(n) slower
  since it must run a forward pass per chunk.

  The standard pattern: retrieve top-30 with fast bi-encoder ANN,
  rerank with slow cross-encoder, keep top-10 for the prompt.
  You get recall of 30 with precision of 10.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M parameters, runs fast on CPU (~50ms for 30 chunks)
  - Trained on MS MARCO passage ranking (web search relevance)
  - Scores are raw logits — higher = more relevant
  - Does not require calibration; relative ordering is what matters
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from shared.config import get_settings
from shared.telemetry import traced_span, RERANKER_DURATION

logger = logging.getLogger(__name__)
settings = get_settings()

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder = None   # lazy-loaded singleton


def _get_cross_encoder():
    """Lazy-load the cross-encoder model once per process."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading CrossEncoder model: %s", _MODEL_NAME)
        _cross_encoder = CrossEncoder(
            _MODEL_NAME,
            max_length=512,
            device=settings.HF_DEVICE,
        )
        logger.info("CrossEncoder loaded")
    return _cross_encoder


@dataclass
class RankedChunk:
    """A chunk with its reranker score attached."""
    chunk_id:        str
    doc_id:          str
    text:            str
    rerank_score:    float      # raw cross-encoder logit (higher = more relevant)
    retrieval_score: float      # original RRF score from retriever
    source_url:      str
    section_title:   str | None
    page_number:     int | None
    hierarchy_level: int
    chunk_index:     int
    parent_text:     str | None
    dense_rank:      int | None
    sparse_rank:     int | None


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.

    Usage:
        reranker = CrossEncoderReranker(top_k=10)
        ranked = reranker.rerank(query, expanded_chunks)
        # ranked is sorted by rerank_score descending, top_k items
    """

    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    def update_top_k(self, top_k: int = 10):
        self.top_k = top_k

    def rerank(
        self,
        query:  str,
        chunks: list,           # list[ExpandedChunk]
    ) -> list[RankedChunk]:
        """
        Score all chunks against the query and return top_k by rerank score.

        Args:
            query:  The user's query string
            chunks: ExpandedChunk list from the context expander

        Returns:
            List of RankedChunk sorted by rerank_score descending,
            length = min(top_k, len(chunks))
        """
        if not chunks:
            return []

        # Single chunk — no need to rerank
        if len(chunks) == 1:
            return [self._to_ranked(chunks[0], score=1.0)]

        with traced_span("reranker.rerank", {
            "query_len":   len(query),
            "num_chunks":  len(chunks),
            "top_k":       self.top_k,
        }):
            t0 = time.monotonic()

            model  = _get_cross_encoder()
            # Build (query, passage) pairs — cross-encoder input format
            pairs  = [[query, self._build_passage(chunk)] for chunk in chunks]
            scores = model.predict(pairs, show_progress_bar=False)

            elapsed_ms = (time.monotonic() - t0) * 1000
            RERANKER_DURATION.observe(elapsed_ms / 1000)
            logger.debug(
                "Reranked %d chunks in %.1fms, top score=%.3f",
                len(chunks), elapsed_ms, float(max(scores)),
            )

            # Zip scores with chunks, sort descending, take top_k
            scored = sorted(
                zip(scores, chunks),
                key=lambda x: float(x[0]),
                reverse=True,
            )[:self.top_k]

            return [
                self._to_ranked(chunk, score=float(score))
                for score, chunk in scored
            ]

    def _build_passage(self, chunk) -> str:
        """
        Build the passage string for the cross-encoder.
        Prepend section title so the model has structural context.
        Keep under 400 tokens to stay within cross-encoder max_length=512
        after the query is prepended.
        """
        parts = []
        if chunk.section_title:
            parts.append(f"[{chunk.section_title}]")
        parts.append(chunk.text[:1200])   # ~300 tokens, leaves room for query
        return " ".join(parts)

    def _to_ranked(self, chunk, score: float) -> RankedChunk:
        return RankedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            rerank_score=score,
            retrieval_score=chunk.score,
            source_url=chunk.source_url,
            section_title=chunk.section_title,
            page_number=chunk.page_number,
            hierarchy_level=chunk.hierarchy_level,
            chunk_index=chunk.chunk_index,
            parent_text=chunk.parent_text,
            dense_rank=chunk.dense_rank,
            sparse_rank=chunk.sparse_rank,
        )