"""
Hybrid retriever: runs dense (semantic) and sparse (BM25) searches
simultaneously, then fuses results with Reciprocal Rank Fusion (RRF).

Why hybrid over pure vector search:
  Dense search excels at semantic similarity ("what is machine learning")
  but misses exact keyword matches ("GPT-4o" "RFC 2616" "errno ECONNREFUSED").
  Sparse (BM25) catches those exact matches.
  RRF merges both ranked lists without needing to normalize incompatible scores.

Reciprocal Rank Fusion formula:
  RRF(d) = Σ 1 / (k + rank_i(d))
  where k=60 (constant that dampens the impact of very high ranks),
  rank_i is the position of document d in result list i (1-indexed).

  A document ranked #1 in both lists scores: 1/(60+1) + 1/(60+1) ≈ 0.033
  A document ranked #1 in one and #10 in other: 1/61 + 1/70 ≈ 0.030
  The fusion naturally rewards documents that appear in both result sets.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    ScoredPoint,
    SparseVector,
)

from ingestion.embedding.services import EmbeddingService
from ingestion.embedding.sparse import BM25Encoder, get_encoder
from shared.config import get_settings
from shared.telemetry import traced_span

logger = logging.getLogger(__name__)
settings = get_settings()

# RRF constant — standard value, rarely needs tuning
_RRF_K = 60


@dataclass
class RetrievedChunk:
    """A single chunk returned from hybrid retrieval with its fusion score."""
    chunk_id:        str
    doc_id:          str
    text:            str  # not stored in Qdrant — fetched from Postgres separately
    score:           float
    source_url:      str
    section_title:   str | None
    page_number:     int | None
    hierarchy_level: int
    parent_chunk_id: str | None
    chunk_index:     int
    dense_rank:      int | None   = None   # rank in dense results (debug)
    sparse_rank:     int | None   = None   # rank in sparse results (debug)
    payload:         dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Full retrieval result for a single query."""
    query:            str
    chunks:           list[RetrievedChunk]
    dense_only_count: int     # how many chunks only appeared in dense results
    sparse_only_count: int    # how many chunks only appeared in sparse results
    overlap_count:    int     # how many appeared in both (high overlap = good query)
    latency_ms:       float   = 0.0


class HybridRetriever:
    """
    Runs hybrid search (dense + sparse) and fuses results with RRF.

    Usage:
        retriever = HybridRetriever(qdrant_client, embedding_service)
        result = await retriever.retrieve("what is gradient descent", top_k=10)
    """

    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        embedding_service: EmbeddingService,
        bm25_encoder: BM25Encoder | None = None,
    ):
        self._qdrant    = qdrant
        self._embedder  = embedding_service
        self._bm25      = bm25_encoder or get_encoder()
        self._collection = settings.QDRANT_COLLECTION_NAME

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        ef: int = 128,
        doc_id_filter: str | None = None,
        source_url_filter: str | None = None,
        hierarchy_level_filter: int | None = None,
        prefetch_k: int | None = None,
    ) -> RetrievalResult:
        """
        Run hybrid retrieval for a query string.

        Args:
            query:                  Natural language query
            top_k:                  Number of chunks to return after fusion
            ef:                     HNSW ef parameter — higher = better recall, slower
                                    64=fast, 128=balanced (default), 256=best recall
            doc_id_filter:          Restrict to a single document
            source_url_filter:      Restrict to a specific source URL
            hierarchy_level_filter: Restrict to a specific hierarchy level (0/1/2)
            prefetch_k:             How many results to fetch from each sub-search
                                    before fusion. Defaults to top_k * 3.
        """
        import time
        t0 = time.monotonic()

        prefetch = prefetch_k or top_k * 3

        with traced_span("retriever.retrieve", {"query_len": len(query), "top_k": top_k}):

            # Build filter (shared by both dense and sparse searches)
            qdrant_filter = self._build_filter(
                doc_id=doc_id_filter,
                source_url=source_url_filter,
                hierarchy_level=hierarchy_level_filter,
            )

            # Run dense and sparse searches concurrently
            dense_task  = asyncio.create_task(
                self._dense_search(query, prefetch, ef, qdrant_filter)
            )
            sparse_task = asyncio.create_task(
                self._sparse_search(query, prefetch, qdrant_filter)
            )
            dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)

            # Fuse results with RRF
            fused = self._reciprocal_rank_fusion(dense_hits, sparse_hits, top_k)

            latency_ms = (time.monotonic() - t0) * 1000

            # Compute overlap stats
            dense_ids  = {h.id for h in dense_hits}
            sparse_ids = {h.id for h in sparse_hits}
            overlap    = dense_ids & sparse_ids

            logger.debug(
                "Hybrid retrieve: query=%r top_k=%d dense=%d sparse=%d overlap=%d latency=%.1fms",
                query[:50], top_k, len(dense_hits), len(sparse_hits), len(overlap), latency_ms,
            )

        return RetrievalResult(
            query=query,
            chunks=fused,
            dense_only_count=len(dense_ids - sparse_ids),
            sparse_only_count=len(sparse_ids - dense_ids),
            overlap_count=len(overlap),
            latency_ms=latency_ms,
        )

    # ─────────────────────────────────────────────────────────
    #  Dense search
    # ─────────────────────────────────────────────────────────

    async def _dense_search(
        self,
        query: str,
        limit: int,
        ef: int,
        qdrant_filter: Filter | None,
    ) -> list[ScoredPoint]:
        """Embed query and run ANN search on the dense vector index."""
        # Embed the query — single text, not batched
        chunks_with_vectors = await self._embedder.embed_chunks([
            _make_query_chunk(query)
        ])
        query_vector = chunks_with_vectors[0].vector

        results = await self._qdrant.search(
            collection_name=self._collection,
            query_vector=NamedVector(name="dense", vector=query_vector),
            limit=limit,
            query_filter=qdrant_filter,
            search_params={"hnsw_ef": ef, "exact": False},
            with_payload=True,
            with_vectors=False,
        )
        return results

    # ─────────────────────────────────────────────────────────
    #  Sparse search
    # ─────────────────────────────────────────────────────────

    async def _sparse_search(
        self,
        query: str,
        limit: int,
        qdrant_filter: Filter | None,
    ) -> list[ScoredPoint]:
        """Encode query as BM25 sparse vector and run keyword search."""
        sparse_vec = self._bm25.encode_query(query)

        if not sparse_vec.indices:
            logger.debug("Sparse query produced empty vector for: %r", query[:50])
            return []

        results = await self._qdrant.search(
            collection_name=self._collection,
            query_vector=NamedSparseVector(
                name="sparse",
                vector=SparseVector(
                    indices=sparse_vec.indices,
                    values=sparse_vec.values,
                ),
            ),
            limit=limit,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,
        )
        return results

    # ─────────────────────────────────────────────────────────
    #  RRF fusion
    # ─────────────────────────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        dense_hits:  list[ScoredPoint],
        sparse_hits: list[ScoredPoint],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Merge dense and sparse ranked lists using Reciprocal Rank Fusion.

        RRF score for a document d across result lists L1, L2:
            score(d) = Σ_i  1 / (k + rank_i(d))

        Documents not appearing in a list get no contribution from that list.
        Documents appearing in both lists get double contribution — naturally
        rewarding cross-modal agreement without manual score normalization.
        """
        rrf_scores:   dict[str, float]         = {}
        dense_ranks:  dict[str, int]           = {}
        sparse_ranks: dict[str, int]           = {}
        payloads:     dict[str, dict]          = {}

        # Score from dense results (1-indexed ranks)
        for rank, hit in enumerate(dense_hits, start=1):
            pid = str(hit.id)
            rrf_scores[pid]  = rrf_scores.get(pid, 0.0) + 1.0 / (_RRF_K + rank)
            dense_ranks[pid] = rank
            if hit.payload:
                payloads[pid] = hit.payload

        # Score from sparse results
        for rank, hit in enumerate(sparse_hits, start=1):
            pid = str(hit.id)
            rrf_scores[pid]   = rrf_scores.get(pid, 0.0) + 1.0 / (_RRF_K + rank)
            sparse_ranks[pid] = rank
            if hit.payload and pid not in payloads:
                payloads[pid] = hit.payload

        # Sort by RRF score descending, take top_k
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for chunk_id, score in ranked:
            p = payloads.get(chunk_id, {})
            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=p.get("doc_id", ""),
                text="",   # populated by ContextExpander
                score=round(score, 6),
                source_url=p.get("source_url", ""),
                section_title=p.get("section_title"),
                page_number=p.get("page_number"),
                hierarchy_level=p.get("hierarchy_level", 0),
                parent_chunk_id=p.get("parent_chunk_id"),
                chunk_index=p.get("chunk_index", 0),
                dense_rank=dense_ranks.get(chunk_id),
                sparse_rank=sparse_ranks.get(chunk_id),
                payload=p,
            ))

        return results

    # ─────────────────────────────────────────────────────────
    #  Filter builder
    # ─────────────────────────────────────────────────────────

    def _build_filter(
        self,
        doc_id:          str | None,
        source_url:      str | None,
        hierarchy_level: int | None,
    ) -> Filter | None:
        conditions = []
        if doc_id:
            conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        if source_url:
            conditions.append(FieldCondition(key="source_url", match=MatchValue(value=source_url)))
        if hierarchy_level is not None:
            conditions.append(FieldCondition(key="hierarchy_level", match=MatchValue(value=hierarchy_level)))

        if not conditions:
            return None
        return Filter(must=conditions)


# ─────────────────────────────────────────────────────────────────
#  Helper: make a minimal Chunk object for query embedding
# ─────────────────────────────────────────────────────────────────

def _make_query_chunk(text: str):
    """Create a minimal Chunk-like object for the embedding service."""
    from shared.models import Chunk, ChunkingStrategy, DocType
    import uuid
    return Chunk(
        id=uuid.uuid4(),
        doc_id=uuid.uuid4(),
        chunk_index=0,
        text=text,
        source_url="query://",
        chunking_strategy=ChunkingStrategy.FIXED,
    )
