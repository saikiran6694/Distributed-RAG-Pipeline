"""
api/main.py

FastAPI application — Phase 3 query endpoint.

Endpoints:
  POST /query         — non-streaming, returns full answer + citations
  POST /query/stream  — Server-Sent Events streaming response
  GET  /health        — liveness check
  GET  /ready         — readiness check (verifies Qdrant + Postgres)

Startup:
  - Creates asyncpg pool
  - Connects to Qdrant
  - Runs collection_setup (idempotent)
  - Initialises embedding service + BM25 encoder
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient

from infrastructure.qdrant.collection_setup import setup_collection
from ingestion.embedding.services import EmbeddingService, build_backend
from ingestion.embedding.sparse import get_encoder
from query.engine import QueryEngine, QueryRequest
from query.prompt_builder import ConversationTurn
from shared.config import get_settings
from shared.telemetry import configure_logging, configure_telemetry

logger = logging.getLogger(__name__)
settings = get_settings()

# ── App-level singletons ──────────────────────────────────────────

_db_pool:         asyncpg.Pool       | None = None
_qdrant:          AsyncQdrantClient  | None = None
_query_engine:    QueryEngine        | None = None


# ── Lifespan ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_pool, _qdrant, _query_engine

    configure_logging()
    configure_telemetry("rag-api")
    logger.info("Starting RAG API...")

    # Postgres
    _db_pool = await asyncpg.create_pool(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        database=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        min_size=settings.POSTGRES_POOL_MIN,
        max_size=settings.POSTGRES_POOL_MAX,
    )

    # Qdrant
    _qdrant = AsyncQdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        prefer_grpc=settings.QDRANT_USE_GRPC,
    )
    await setup_collection(_qdrant)

    # Embedding + query engine
    backend   = build_backend()
    embed_svc = EmbeddingService(backend=backend, db_pool=_db_pool)
    bm25      = get_encoder()

    _query_engine = QueryEngine(
        qdrant=_qdrant,
        embedding_service=embed_svc,
        db_pool=_db_pool,
        bm25_encoder=bm25,
    )

    logger.info("RAG API ready")
    yield

    # Shutdown
    await _db_pool.close()
    await _qdrant.close()
    logger.info("RAG API shut down")


# ── FastAPI app ───────────────────────────────────────────────────

app = FastAPI(
    title="Distributed RAG Pipeline",
    version="0.3.0",
    description="Phase 3 — Hybrid retrieval + LLM generation",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────

class TurnSchema(BaseModel):
    role:    str
    content: str


class QueryRequestSchema(BaseModel):
    query:                  str   = Field(..., min_length=1, max_length=2000)
    top_k:                  int   = Field(default=10, ge=1, le=50)
    ef:                     int   = Field(default=128, ge=32, le=512)
    doc_id_filter:          str | None = None
    source_url_filter:      str | None = None
    hierarchy_level_filter: int | None = Field(default=None, ge=0, le=2)
    max_context_tokens:     int   = Field(default=8000, ge=1000, le=64000)
    history:                list[TurnSchema] = Field(default_factory=list)


class CitationSchema(BaseModel):
    index:         int
    source_url:    str
    section_title: str | None
    score:         float


class QueryResponseSchema(BaseModel):
    query:             str
    answer:            str
    citations:         list[CitationSchema]
    chunks_used:       int
    latency_ms:        float
    sub_queries:       list[str]


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness check — verifies all downstream services are reachable."""
    errors = []
    try:
        await _db_pool.fetchval("SELECT 1")
    except Exception as e:
        errors.append(f"postgres: {e}")
    try:
        await _qdrant.get_collections()
    except Exception as e:
        errors.append(f"qdrant: {e}")

    if errors:
        raise HTTPException(status_code=503, detail={"errors": errors})
    return {"status": "ready"}


@app.post("/query", response_model=QueryResponseSchema)
async def query(request: QueryRequestSchema):
    """Non-streaming query — returns complete answer with citations."""
    if _query_engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    history = [ConversationTurn(role=t.role, content=t.content) for t in request.history]

    result = await _query_engine.query(QueryRequest(
        query=request.query,
        top_k=request.top_k,
        ef=request.ef,
        doc_id_filter=request.doc_id_filter,
        source_url_filter=request.source_url_filter,
        hierarchy_level_filter=request.hierarchy_level_filter,
        max_context_tokens=request.max_context_tokens,
        history=history,
        stream=False,
    ))

    return QueryResponseSchema(
        query=result.query,
        answer=result.answer,
        citations=[CitationSchema(**c) for c in result.citations],
        chunks_used=result.chunks_used,
        latency_ms=round(result.latency_ms, 1),
        sub_queries=result.sub_queries,
    )


@app.post("/query/stream")
async def query_stream(request: QueryRequestSchema):
    """
    Streaming query — returns Server-Sent Events.

    Each event is JSON with shape:
      {"delta": "...", "done": false}

    Final event:
      {"delta": "", "done": true, "citations": [...], "usage": {...}}

    Client usage (JavaScript):
      const es = new EventSource("/query/stream");
      es.onmessage = (e) => {
        const chunk = JSON.parse(e.data);
        if (chunk.done) { showCitations(chunk.citations); es.close(); }
        else appendText(chunk.delta);
      };
    """
    if _query_engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    history = [ConversationTurn(role=t.role, content=t.content) for t in request.history]
    req = QueryRequest(
        query=request.query,
        top_k=request.top_k,
        ef=request.ef,
        doc_id_filter=request.doc_id_filter,
        source_url_filter=request.source_url_filter,
        hierarchy_level_filter=request.hierarchy_level_filter,
        max_context_tokens=request.max_context_tokens,
        history=history,
        stream=True,
    )

    async def event_generator():
        async for chunk in _query_engine.stream(req):
            yield chunk.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )
