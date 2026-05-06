"""
FastAPI application — Distributed RAG Pipeline.

Endpoints:
  POST /ingest/file            — upload a file
  POST /ingest/url             — ingest from URL or S3
  POST /ingest/text            — ingest raw text
  GET  /ingest/list            — list all documents
  GET  /ingest/{doc_id}/status — document processing status
  POST /query                  — non-streaming query
  POST /query/stream           — SSE streaming query
  GET  /cache/stats            — semantic cache statistics
  GET  /health                 — liveness check
  GET  /ready                  — readiness check
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import asyncpg
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import AsyncQdrantClient

from api.ingest import router as ingest_router
from api.ingest import set_ingest_dependencies
from infrastructure.qdrant.collection_setup import setup_collection
from ingestion.embedding.services import EmbeddingService, build_backend
from ingestion.embedding.sparse import get_encoder
from ingestion.intake.producer import ensure_topics_exist
from query.engine import QueryEngine, QueryRequest
from query.prompt_builder import ConversationTurn
from shared.config import get_settings
from shared.telemetry import configure_logging, configure_telemetry

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Module-level singletons ───────────────────────────────────────

_db_pool: asyncpg.Pool | None = None
_qdrant: AsyncQdrantClient | None = None
_redis: aioredis.Redis | None = None
_query_engine: QueryEngine | None = None


# ── Lifespan ─────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Bug fix: all module-level singletons must be listed here
    global _db_pool, _qdrant, _redis, _query_engine

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
    logger.info("PostgreSQL pool created")

    # Redis
    _redis = aioredis.from_url(settings.redis_url, decode_responses=False)
    await _redis.ping()
    logger.info("Redis connected")

    # Kafka topics (idempotent — safe to call every startup)
    try:
        ensure_topics_exist()
        logger.info("Kafka topics ready")
    except Exception as e:
        logger.warning("Kafka topic setup failed (workers may not function): %s", e)

    # Qdrant
    _qdrant = AsyncQdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        prefer_grpc=settings.QDRANT_USE_GRPC,
    )
    await setup_collection(_qdrant)
    logger.info("Qdrant collection ready")

    # Embedding service (shared by ingest + query)
    backend = build_backend()
    embed_svc = EmbeddingService(backend=backend, db_pool=_db_pool)
    bm25 = get_encoder()
    logger.info("Embedding service ready (backend=%s)", settings.EMBED_BACKEND)

    # Wire ingest router dependencies
    set_ingest_dependencies(_db_pool, _qdrant, embed_svc)

    # Query engine (Phase 3 + 4)
    _query_engine = QueryEngine(
        qdrant=_qdrant,
        embedding_service=embed_svc,
        db_pool=_db_pool,
        redis=_redis,
        bm25_encoder=bm25,
    )

    logger.info("RAG API ready — all services initialised")
    yield

    # Graceful shutdown
    logger.info("Shutting down RAG API...")
    await _db_pool.close()
    await _qdrant.close()
    await _redis.aclose()
    logger.info("RAG API shut down cleanly")


# ── App ───────────────────────────────────────────────────────────

app = FastAPI(
    title="Distributed RAG Pipeline",
    version="1.0.0",
    description="Phases 1–4: Ingestion · Hybrid Search · Generation · Reranking · Cache",
    lifespan=lifespan,
)

# Ingest router must be included before middleware
app.include_router(ingest_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────


class TurnSchema(BaseModel):
    role: str
    content: str


class QueryRequestSchema(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    ef: int = Field(default=128, ge=32, le=512)
    doc_id_filter: str | None = None
    source_url_filter: str | None = None
    hierarchy_level_filter: int | None = Field(default=None, ge=0, le=2)
    max_context_tokens: int = Field(default=8000, ge=1000, le=64000)
    history: list[TurnSchema] = Field(default_factory=list)
    use_cache: bool = True
    use_reranker: bool = True
    score_threshold: float = Field(default=0.01, ge=0.0, le=1.0)


class CitationSchema(BaseModel):
    # Bug fix: added chunk_id — prompt_builder includes it in citation dicts.
    # Without this, CitationSchema(**citation_dict) raises ValidationError
    # because Pydantic v2 rejects unexpected fields by default.
    index: int
    source_url: str
    section_title: str | None
    score: float
    chunk_id: str | None = None  # optional so old responses still deserialise

    model_config = ConfigDict(extra="ignore")  # silently drop any other extra fields


class QueryResponseSchema(BaseModel):
    query: str
    answer: str
    citations: list[CitationSchema]
    chunks_used: int
    latency_ms: float
    sub_queries: list[str]
    cache_hit: bool = False
    reranked: bool = False
    groundedness: dict | None = None


# ── Health endpoints ──────────────────────────────────────────────


@app.get("/health", tags=["Health"])
async def health():
    """Liveness check — always returns 200 if the process is running."""
    return {"status": "ok", "version": app.version}


@app.get("/ready", tags=["Health"])
async def ready():
    """
    Readiness check — verifies Postgres and Qdrant are reachable.
    Returns 503 with error details if any service is down.
    """
    errors = []
    try:
        await _db_pool.fetchval("SELECT 1")
    except Exception as e:
        errors.append(f"postgres: {e}")
    try:
        await _qdrant.get_collections()
    except Exception as e:
        errors.append(f"qdrant: {e}")
    try:
        await _redis.ping()
    except Exception as e:
        errors.append(f"redis: {e}")

    if errors:
        raise HTTPException(status_code=503, detail={"errors": errors})
    return {"status": "ready"}


@app.get("/cache/stats", tags=["Cache"])
async def cache_stats():
    """Semantic cache statistics — entry count, total hits, threshold."""
    if _redis is None:
        raise HTTPException(status_code=503, detail="Redis not connected")
    from query.cache import build_cache

    return await build_cache(_redis).stats()


# ── Query endpoints ───────────────────────────────────────────────


@app.post("/query", response_model=QueryResponseSchema, tags=["Query"])
async def query(request: QueryRequestSchema):
    """
    Non-streaming RAG query.
    Runs the full pipeline: decompose → retrieve → rerank → generate.
    Returns the complete answer with inline citations.
    """
    if _query_engine is None:
        raise HTTPException(status_code=503, detail="Query engine not ready")

    history = [ConversationTurn(role=t.role, content=t.content) for t in request.history]

    result = await _query_engine.query(
        QueryRequest(
            query=request.query,
            top_k=request.top_k,
            ef=request.ef,
            doc_id_filter=request.doc_id_filter,
            source_url_filter=request.source_url_filter,
            hierarchy_level_filter=request.hierarchy_level_filter,
            max_context_tokens=request.max_context_tokens,
            history=history,
            stream=False,
            use_cache=request.use_cache,
            use_reranker=request.use_reranker,
            score_threshold=request.score_threshold,
        )
    )

    return QueryResponseSchema(
        query=result.query,
        answer=result.answer,
        citations=[CitationSchema(**c) for c in result.citations],
        chunks_used=result.chunks_used,
        latency_ms=round(result.latency_ms, 1),
        sub_queries=result.sub_queries,
        cache_hit=result.cache_hit,
        reranked=result.reranked,
        groundedness=result.groundedness,
    )


@app.post("/query/stream", tags=["Query"])
async def query_stream(request: QueryRequestSchema):
    """
    Streaming RAG query via Server-Sent Events.

    Stream format — each event:
      data: {"delta": "text fragment", "done": false}

    Final event:
      data: {"delta": "", "done": true, "citations": [...], "usage": {...}}

    JavaScript client:
      const res = await fetch("/query/stream", {method:"POST", body: JSON.stringify({query:"..."})});
      const reader = res.body.getReader();
      // read lines, parse JSON after "data: " prefix
    """
    if _query_engine is None:
        raise HTTPException(status_code=503, detail="Query engine not ready")

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
        use_cache=request.use_cache,
        use_reranker=request.use_reranker,
        score_threshold=request.score_threshold,
    )

    async def event_generator():
        async for chunk in _query_engine.stream(req):
            yield chunk.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
        },
    )
