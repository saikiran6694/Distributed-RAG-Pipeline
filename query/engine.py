"""
Query engine: full RAG pipeline orchestrator.

Sequence:
  1. Query decomposition  — LLM splits compound queries (fallback: original)
  2. Hybrid retrieval     — dense + sparse search with RRF fusion
  3. Context expansion    — fetch text + parent context from Postgres
  4. Deduplication        — remove near-duplicate chunks
  5. Prompt building      — token-budget-aware context assembly
  6. LLM generation       — streaming response with inline citations

Phase 4 will add: CrossEncoder reranker between steps 4 and 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import AsyncIterator

import asyncpg
from qdrant_client import AsyncQdrantClient

from ingestion.embedding.services import EmbeddingService
from ingestion.embedding.sparse import BM25Encoder
from query.context_expander import ContextExpander, ExpandedChunk
from query.decomposer import QueryDecomposer
from query.generator import GeneratorChunk, LLMGenerator
from query.prompt_builder import BuiltPrompt, ConversationTurn, PromptBuilder
from query.retriever import HybridRetriever, RetrievalResult
from shared.config import get_settings
from shared.telemetry import traced_span

logger = logging.getLogger(__name__)
settings = get_settings()


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


class QueryEngine:
    """
    Full RAG pipeline: decompose → retrieve → expand → build → generate.

    Usage (streaming):
        engine = QueryEngine(qdrant, embedding_service, db_pool)
        async for chunk in engine.stream(QueryRequest(query="what is RAG?")):
            print(chunk.delta, end="", flush=True)

    Usage (non-streaming):
        response = await engine.query(QueryRequest(query="what is RAG?", stream=False))
        print(response.answer)
    """

    def __init__(
        self,
        qdrant:            AsyncQdrantClient,
        embedding_service: EmbeddingService,
        db_pool:           asyncpg.Pool,
        bm25_encoder:      BM25Encoder | None = None,
    ):
        self._retriever  = HybridRetriever(qdrant, embedding_service, bm25_encoder)
        self._expander   = ContextExpander(db_pool)
        self._decomposer = QueryDecomposer()
        self._generator  = LLMGenerator()

    async def stream(self, request: QueryRequest) -> AsyncIterator[GeneratorChunk]:
        """Stream the full RAG response chunk by chunk."""
        chunks, prompt, retrieval = await self._retrieve_and_build(request)
        sub_queries = getattr(self, "_last_sub_queries", [])

        async for gen_chunk in self._generator.stream(prompt, request.history):
            yield gen_chunk

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Run the full pipeline and return the complete response (non-streaming)."""
        import time
        t0 = time.monotonic()

        chunks, prompt, retrieval = await self._retrieve_and_build(request)
        sub_queries = getattr(self, "_last_sub_queries", [])

        # Collect full response
        full_text = ""
        final_chunk = None
        async for gen_chunk in self._generator.stream(prompt, request.history):
            full_text += gen_chunk.delta
            if gen_chunk.done:
                final_chunk = gen_chunk

        return QueryResponse(
            query=request.query,
            answer=full_text,
            citations=prompt.citations,
            chunks_used=prompt.chunks_used,
            latency_ms=(time.monotonic() - t0) * 1000,
            sub_queries=sub_queries,
            dense_only_count=retrieval.dense_only_count,
            sparse_only_count=retrieval.sparse_only_count,
            overlap_count=retrieval.overlap_count,
        )

    # ─────────────────────────────────────────────────────────
    #  Internal pipeline
    # ─────────────────────────────────────────────────────────

    async def _retrieve_and_build(
        self,
        request: QueryRequest,
    ) -> tuple[list[ExpandedChunk], BuiltPrompt, RetrievalResult]:
        """Steps 1-5: decompose → retrieve → expand → dedup → build prompt."""

        with traced_span("query_engine.pipeline", {"query": request.query[:80]}):

            # Step 1: LLM-based query decomposition
            sub_queries = await self._decomposer.decompose(request.query)
            self._last_sub_queries = sub_queries

            # Step 2: Hybrid retrieval
            if len(sub_queries) == 1:
                retrieval = await self._retriever.retrieve(
                    query=sub_queries[0],
                    top_k=request.top_k,
                    ef=request.ef,
                    doc_id_filter=request.doc_id_filter,
                    source_url_filter=request.source_url_filter,
                    hierarchy_level_filter=request.hierarchy_level_filter,
                )
            else:
                retrieval = await self._retrieve_multi(request, sub_queries)

            # Step 3: Context expansion
            expanded = await self._expander.expand(retrieval.chunks)

            # Step 4: Deduplication
            deduped = self._deduplicate(expanded)

            logger.info(
                "Pipeline: query=%r sub_queries=%d retrieved=%d deduped=%d",
                request.query[:50], len(sub_queries),
                len(retrieval.chunks), len(deduped),
            )

            # Step 5: Prompt building
            builder = PromptBuilder(max_context_tokens=request.max_context_tokens)
            prompt  = builder.build(request.query, deduped, request.history)

        return deduped, prompt, retrieval

    async def _retrieve_multi(
        self,
        request:     QueryRequest,
        sub_queries: list[str],
    ) -> RetrievalResult:
        import asyncio
        tasks = [
            self._retriever.retrieve(
                query=q,
                top_k=request.top_k,
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

        merged = sorted(seen.values(), key=lambda c: c.score, reverse=True)[:request.top_k]
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
