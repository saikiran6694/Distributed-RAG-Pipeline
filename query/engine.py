"""
query/engine.py

Query engine: full RAG pipeline orchestrator (Phase 4).

Sequence:
  1. Semantic cache check    — return cached response on hit
  2. Query decomposition     — LLM splits compound queries
  3. Hybrid retrieval        — dense + sparse search, top-30 (prefetch)
  4. Context expansion       — fetch text + parent context from Postgres
  5. Deduplication           — remove near-duplicate chunks
  6. CrossEncoder rerank     — top-30 → top-10 with cross-encoder model  ← NEW
  7. Prompt building         — token-budget-aware context assembly
  8. LLM generation          — streaming response with inline citations
  9. Cache store             — persist response for future similar queries ← NEW
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import AsyncIterator

import asyncpg
import redis.asyncio as aioredis
from qdrant_client import AsyncQdrantClient

from ingestion.embedding.services import EmbeddingService
from ingestion.embedding.sparse import BM25Encoder
from query.cache import SemanticCache, build_cache
from query.context_expander import ContextExpander, ExpandedChunk
from query.decomposer import QueryDecomposer
from query.generator import GeneratorChunk, LLMGenerator
from query.prompt_builder import BuiltPrompt, ConversationTurn, PromptBuilder
from query.reranker import CrossEncoderReranker, RankedChunk
from query.retriever import HybridRetriever, RetrievalResult
from shared.config import get_settings
from shared.telemetry import traced_span

logger = logging.getLogger(__name__)
settings = get_settings()

# Retrieve this many candidates before reranking, then keep top_k
_RERANK_PREFETCH = 30


@dataclass
class QueryRequest:
    query:                  str
    top_k:                  int  = 10
    ef:                     int  = 128
    doc_id_filter:          str | None = None
    source_url_filter:      str | None = None
    hierarchy_level_filter: int | None = None
    include_parent_context: bool = True
    max_context_tokens:     int  = 8000
    history:                list[ConversationTurn] = field(default_factory=list)
    stream:                 bool = True
    use_cache:              bool = True     # set False to bypass cache (debug)
    use_reranker:           bool = True     # set False to skip reranking


@dataclass
class QueryResponse:
    """Returned when stream=False."""
    query:             str
    answer:            str
    citations:         list[dict]
    chunks_used:       int
    latency_ms:        float
    sub_queries:       list[str]       = field(default_factory=list)
    dense_only_count:  int             = 0
    sparse_only_count: int             = 0
    overlap_count:     int             = 0
    cache_hit:         bool            = False
    cache_similarity:  float | None    = None
    reranked:          bool            = False


class QueryEngine:
    """
    Full RAG pipeline: cache → decompose → retrieve → expand →
                       deduplicate → rerank → build → generate → cache.

    Usage (streaming):
        engine = QueryEngine(qdrant, embedding_service, db_pool, redis)
        async for chunk in engine.stream(QueryRequest(query="what is RAG?")):
            print(chunk.delta, end="", flush=True)

    Usage (non-streaming):
        response = await engine.query(QueryRequest(query="what is RAG?", stream=False))
    """

    def __init__(
        self,
        qdrant:            AsyncQdrantClient,
        embedding_service: EmbeddingService,
        db_pool:           asyncpg.Pool,
        redis:             aioredis.Redis | None = None,
        bm25_encoder:      BM25Encoder    | None = None,
        reranker_top_k:    int                   = 10,
    ):
        self._retriever  = HybridRetriever(qdrant, embedding_service, bm25_encoder)
        self._expander   = ContextExpander(db_pool)
        self._decomposer = QueryDecomposer()
        self._generator  = LLMGenerator()
        self._reranker   = CrossEncoderReranker(top_k=reranker_top_k)
        self._embedder   = embedding_service
        self._cache      = build_cache(redis) if redis else None

    # ─────────────────────────────────────────────────────────
    #  Public interface
    # ─────────────────────────────────────────────────────────

    async def stream(self, request: QueryRequest) -> AsyncIterator[GeneratorChunk]:
        """Stream the full RAG response chunk by chunk."""

        # Cache check
        if request.use_cache and self._cache:
            cached = await self._check_cache(request)
            if cached:
                yield GeneratorChunk(
                    delta=cached.get("answer", ""),
                    done=True,
                    citations=cached.get("citations", []),
                    usage={"cache_hit": True},
                )
                return

        _, prompt, _ = await self._retrieve_and_build(request)

        full_text = ""
        async for gen_chunk in self._generator.stream(prompt, request.history):
            full_text += gen_chunk.delta
            yield gen_chunk

        # Store in cache after generation completes
        if request.use_cache and self._cache and full_text:
            await self._store_cache(request, prompt, full_text)

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Run the full pipeline and return the complete response (non-streaming)."""
        import time
        t0 = time.monotonic()

        # Cache check
        if request.use_cache and self._cache:
            cached = await self._check_cache(request)
            if cached:
                return QueryResponse(
                    query=request.query,
                    answer=cached.get("answer", ""),
                    citations=cached.get("citations", []),
                    chunks_used=cached.get("chunks_used", 0),
                    latency_ms=round((time.monotonic() - t0) * 1000, 1),
                    cache_hit=True,
                    cache_similarity=cached.get("cache_similarity"),
                )

        _, prompt, retrieval = await self._retrieve_and_build(request)
        sub_queries = getattr(self, "_last_sub_queries", [])
        reranked    = getattr(self, "_last_reranked", False)

        full_text = ""
        async for gen_chunk in self._generator.stream(prompt, request.history):
            full_text += gen_chunk.delta

        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        response = QueryResponse(
            query=request.query,
            answer=full_text,
            citations=prompt.citations,
            chunks_used=prompt.chunks_used,
            latency_ms=latency_ms,
            sub_queries=sub_queries,
            dense_only_count=retrieval.dense_only_count,
            sparse_only_count=retrieval.sparse_only_count,
            overlap_count=retrieval.overlap_count,
            reranked=reranked,
        )

        # Store in cache
        if request.use_cache and self._cache:
            import dataclasses
            await self._store_cache(request, prompt, full_text)

        return response

    # ─────────────────────────────────────────────────────────
    #  Internal pipeline
    # ─────────────────────────────────────────────────────────

    async def _retrieve_and_build(
        self,
        request: QueryRequest,
    ) -> tuple[list, BuiltPrompt, RetrievalResult]:
        """Steps 2-7: decompose → retrieve → expand → dedup → rerank → build."""

        with traced_span("query_engine.pipeline", {"query": request.query[:80]}):

            # Step 2: Decompose
            sub_queries = await self._decomposer.decompose(request.query)
            self._last_sub_queries = sub_queries

            # Step 3: Retrieve — fetch more candidates when reranking
            prefetch = _RERANK_PREFETCH if request.use_reranker else request.top_k
            if len(sub_queries) == 1:
                retrieval = await self._retriever.retrieve(
                    query=sub_queries[0],
                    top_k=prefetch,
                    ef=request.ef,
                    doc_id_filter=request.doc_id_filter,
                    source_url_filter=request.source_url_filter,
                    hierarchy_level_filter=request.hierarchy_level_filter,
                )
            else:
                retrieval = await self._retrieve_multi(request, sub_queries, prefetch)

            # Step 4: Expand context
            expanded = await self._expander.expand(retrieval.chunks)

            # Step 5: Deduplicate
            deduped = self._deduplicate(expanded)

            # Step 6: Rerank (top-30 → top-10)
            if request.use_reranker and len(deduped) > 1:
                ranked = self._reranker.rerank(request.query, deduped)
                self._last_reranked = True
                logger.info(
                    "Reranked %d → %d chunks for query: %r",
                    len(deduped), len(ranked), request.query[:50],
                )
                # Convert RankedChunk back to ExpandedChunk-compatible for PromptBuilder
                final_chunks = self._ranked_to_expanded(ranked)
            else:
                final_chunks = deduped[:request.top_k]
                self._last_reranked = False

            # Step 7: Build prompt
            builder = PromptBuilder(max_context_tokens=request.max_context_tokens)
            prompt  = builder.build(request.query, final_chunks, request.history)

        return final_chunks, prompt, retrieval

    def _ranked_to_expanded(self, ranked: list[RankedChunk]) -> list:
        """
        Convert RankedChunk list back to ExpandedChunk-compatible objects
        so the PromptBuilder can work with them unchanged.
        """
        from query.context_expander import ExpandedChunk
        return [
            ExpandedChunk(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                text=r.text,
                parent_text=r.parent_text,
                score=r.rerank_score,       # use rerank score as the display score
                source_url=r.source_url,
                section_title=r.section_title,
                page_number=r.page_number,
                hierarchy_level=r.hierarchy_level,
                chunk_index=r.chunk_index,
                dense_rank=r.dense_rank,
                sparse_rank=r.sparse_rank,
            )
            for r in ranked
        ]

    async def _retrieve_multi(
        self,
        request:     QueryRequest,
        sub_queries: list[str],
        prefetch:    int,
    ) -> RetrievalResult:
        import asyncio
        tasks = [
            self._retriever.retrieve(
                query=q,
                top_k=prefetch,
                ef=request.ef,
                doc_id_filter=request.doc_id_filter,
                source_url_filter=request.source_url_filter,
                hierarchy_level_filter=request.hierarchy_level_filter,
            )
            for q in sub_queries
        ]
        results = await asyncio.gather(*tasks)

        seen: dict = {}
        for result in results:
            for chunk in result.chunks:
                if chunk.chunk_id not in seen or chunk.score > seen[chunk.chunk_id].score:
                    seen[chunk.chunk_id] = chunk

        merged = sorted(seen.values(), key=lambda c: c.score, reverse=True)[:prefetch]
        return RetrievalResult(
            query=request.query,
            chunks=merged,
            dense_only_count=sum(r.dense_only_count for r in results),
            sparse_only_count=sum(r.sparse_only_count for r in results),
            overlap_count=sum(r.overlap_count for r in results),
            latency_ms=max(r.latency_ms for r in results),
        )

    def _deduplicate(self, chunks: list[ExpandedChunk]) -> list[ExpandedChunk]:
        seen_ids:  set[str]   = set()
        seen_keys: set[tuple] = set()
        result = []
        for chunk in chunks:
            if chunk.chunk_id in seen_ids:
                continue
            adjacent_key = (chunk.doc_id, chunk.chunk_index // 2)
            if adjacent_key in seen_keys:
                continue
            seen_ids.add(chunk.chunk_id)
            seen_keys.add(adjacent_key)
            result.append(chunk)
        return result

    # ─────────────────────────────────────────────────────────
    #  Cache helpers
    # ─────────────────────────────────────────────────────────

    async def _embed_query(self, query: str) -> list[float] | None:
        """Embed the query for cache key lookup."""
        try:
            from query.retriever import _make_query_chunk
            chunks = await self._embedder.embed_chunks([_make_query_chunk(query)])
            return chunks[0].vector
        except Exception as e:
            logger.warning("Failed to embed query for cache: %s", e)
            return None

    async def _check_cache(self, request: QueryRequest) -> dict | None:
        """Embed query and check semantic cache. Returns cached response or None."""
        if not self._cache:
            return None
        vector = await self._embed_query(request.query)
        if not vector:
            return None
        return await self._cache.get(request.query, vector)

    async def _store_cache(
        self,
        request:   QueryRequest,
        prompt:    BuiltPrompt,
        full_text: str,
    ) -> None:
        """Embed query and store response in semantic cache."""
        if not self._cache:
            return
        vector = await self._embed_query(request.query)
        if not vector:
            return
        response_dict = {
            "query":       request.query,
            "answer":      full_text,
            "citations":   prompt.citations,
            "chunks_used": prompt.chunks_used,
        }
        try:
            await self._cache.set(request.query, vector, response_dict)
        except Exception as e:
            logger.warning("Failed to store in cache: %s", e)