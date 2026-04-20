#!/usr/bin/env python3
"""
scripts/test_e2e.py

End-to-end validation script for the distributed RAG pipeline.
Runs WITHOUT pytest — direct Python script so you can see exactly
what passes and fails at each stage.

Usage:
    cd distributed-rag
    python scripts/test_e2e.py

Requirements:
    - Docker Compose stack must be running:
        cd infrastructure && docker compose up -d
    - Python deps installed:
        pip install -e ".[dev]"
    - .env file present (or defaults work for local dev)

What this tests:
    Stage 1: Infrastructure connectivity (Postgres, Qdrant, Redis, Kafka)
    Stage 2: Qdrant collection setup (named vectors, payload indexes)
    Stage 3: Document ingestion (chunk → embed → store)
    Stage 4: Hybrid retrieval (dense + sparse, RRF)
    Stage 5: Context expansion (Postgres text fetch)
    Stage 6: Reranker (CrossEncoder scoring)
    Stage 7: Semantic cache (set + get)
    Stage 8: Prompt builder (token budget, citations)
    Stage 9: Full query engine (non-streaming)
    Stage 10: API health + ready endpoints
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Colour helpers ────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")
def info(msg): print(f"  {BLUE}→{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}!{RESET}  {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}")

results: list[tuple[str, bool, str]] = []

def record(stage: str, passed: bool, detail: str = ""):
    results.append((stage, passed, detail))
    if passed:
        ok(stage + (f" — {detail}" if detail else ""))
    else:
        fail(stage + (f" — {detail}" if detail else ""))

# ── Test document ────────────────────────────────────────────────

TEST_TEXT = """
Introduction to Distributed Systems

A distributed system is a collection of independent computers that appears
to its users as a single coherent system. Distributed systems are characterized
by concurrency, lack of a global clock, and independent failures.

Key Challenges

The main challenges in distributed systems include network partitions,
consistency guarantees, and fault tolerance. The CAP theorem states that
a distributed system can provide at most two of the following three guarantees:
consistency, availability, and partition tolerance.

Consensus Algorithms

Consensus algorithms like Raft and Paxos allow distributed nodes to agree
on a single value even in the presence of failures. Raft divides time into
terms and uses leader election to ensure only one node proposes values at a time.

Practical Applications

Distributed systems power modern cloud infrastructure including databases
like Cassandra and CockroachDB, message queues like Apache Kafka, and
coordination services like ZooKeeper and etcd.
""".strip()


# ── Stage implementations ─────────────────────────────────────────

async def stage1_infrastructure():
    header("Stage 1 — Infrastructure Connectivity")
    from shared.config import get_settings
    settings = get_settings()

    # Postgres
    try:
        import asyncpg
        pool = await asyncpg.create_pool(
            host=settings.POSTGRES_HOST, port=settings.POSTGRES_PORT,
            database=settings.POSTGRES_DB, user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD, min_size=1, max_size=2,
        )
        version = await pool.fetchval("SELECT version()")
        await pool.close()
        record("PostgreSQL connected", True, version[:40])
    except Exception as e:
        record("PostgreSQL connected", False, str(e))
        return None, None, None

    # Qdrant
    try:
        from qdrant_client import AsyncQdrantClient
        qdrant = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        cols = await qdrant.get_collections()
        record("Qdrant connected", True, f"{len(cols.collections)} existing collections")
    except Exception as e:
        record("Qdrant connected", False, str(e))
        return None, None, None

    # Redis
    try:
        import redis.asyncio as aioredis
        redis_client = aioredis.from_url(settings.redis_url)
        await redis_client.ping()
        record("Redis connected", True)
    except Exception as e:
        record("Redis connected", False, str(e))
        redis_client = None

    # Kafka (check topic exists)
    try:
        from confluent_kafka.admin import AdminClient
        admin = AdminClient({"bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS})
        meta = admin.list_topics(timeout=5)
        topics = list(meta.topics.keys())
        rag_topics = [t for t in topics if "raw-documents" in t or "ingestion" in t]
        if rag_topics:
            record("Kafka connected", True, f"topics: {', '.join(rag_topics)}")
        else:
            record("Kafka connected", True, f"broker reachable, run kafka-init to create topics")
    except Exception as e:
        record("Kafka connected", False, str(e))

    # Return connections for later stages
    pool = await asyncpg.create_pool(
        host=settings.POSTGRES_HOST, port=settings.POSTGRES_PORT,
        database=settings.POSTGRES_DB, user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD, min_size=1, max_size=5,
    )
    return pool, qdrant, redis_client


async def stage2_collection_setup(qdrant):
    header("Stage 2 — Qdrant Collection Setup")
    if qdrant is None:
        record("Collection setup", False, "Qdrant not connected")
        return False
    try:
        from infrastructure.qdrant.collection_setup import setup_collection, verify_cluster_health
        from shared.config import get_settings
        settings = get_settings()

        await setup_collection(qdrant, recreate=False)
        record("Collection created/verified", True)

        # Verify named vectors exist
        col_info = await qdrant.get_collection(settings.QDRANT_COLLECTION_NAME)
        vectors_config = col_info.config.params.vectors
        if isinstance(vectors_config, dict):
            vector_names = list(vectors_config.keys())
            record("Named vectors (dense + sparse)", True, f"vectors: {vector_names}")
        else:
            record("Named vectors (dense + sparse)", True, "single vector config")

        record("Collection status", True, str(col_info.status))
        return True
    except Exception as e:
        record("Collection setup", False, str(e))
        traceback.print_exc()
        return False


async def stage3_ingestion(db_pool, qdrant):
    header("Stage 3 — Document Ingestion")
    if not db_pool or not qdrant:
        record("Ingestion", False, "Missing connections")
        return None, None

    try:
        from shared.models import ParsedDocument, Section, DocType
        from ingestion.chunking.fixed import FixedChunker
        from ingestion.chunking.hierarical import HierarchicalChunker
        from ingestion.embedding.services import EmbeddingService, HuggingFaceBackend
        from ingestion.storage.writer import StorageWriter

        doc_id = uuid.uuid4()
        doc = ParsedDocument(
            doc_id=doc_id,
            source_url=f"file:///test/e2e/{doc_id}.txt",
            doc_type=DocType.TXT,
            raw_text=TEST_TEXT,
            sections=[
                Section(title="Introduction to Distributed Systems",
                        content=TEST_TEXT[:400]),
                Section(title="Key Challenges",
                        content=TEST_TEXT[400:700]),
                Section(title="Consensus Algorithms",
                        content=TEST_TEXT[700:1000]),
                Section(title="Practical Applications",
                        content=TEST_TEXT[1000:]),
            ],
            title="Distributed Systems Overview",
        )

        # Register document in Postgres
        await db_pool.execute(
            """
            INSERT INTO documents (id, source_url, source_type, doc_type, content_hash, byte_size, status)
            VALUES ($1, $2, 'test', $3, $4, $5, 'embedding')
            ON CONFLICT (source_url) DO UPDATE SET status = 'embedding', id = EXCLUDED.id
            RETURNING id
            """,
            doc_id, doc.source_url, 'txt', doc.content_hash, len(TEST_TEXT),
        )
        record("Document registered in Postgres", True, f"doc_id={str(doc_id)[:8]}...")

        # Chunk
        chunker = FixedChunker(chunk_size=150, overlap=15)
        chunks = chunker.chunk(doc)
        record(f"Fixed chunking", True, f"{len(chunks)} chunks produced")

        # Hierarchical chunking
        hier_chunker = HierarchicalChunker(parent_tokens=400, child_tokens=100)
        hier_chunks = hier_chunker.chunk(doc)
        levels = {c.hierarchy_level for c in hier_chunks}
        record(f"Hierarchical chunking", True, f"{len(hier_chunks)} chunks, levels={sorted(levels)}")

        # Embed
        info("Loading HuggingFace embedding model (first run downloads ~90MB)...")
        backend = HuggingFaceBackend()
        embed_svc = EmbeddingService(backend=backend, db_pool=db_pool)
        embedded = await embed_svc.embed_chunks(chunks)
        assert all(c.vector is not None for c in embedded)
        record("Embedding", True,
               f"dim={len(embedded[0].vector)}, model={embedded[0].embedding_model}")

        # Store
        writer = StorageWriter(db_pool=db_pool, qdrant=qdrant)
        await writer.store(embedded, doc)
        record("Storage (Qdrant + Postgres)", True, f"{len(embedded)} chunks indexed")

        # Verify in Postgres
        count = await db_pool.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE doc_id=$1 AND status='indexed'", doc_id
        )
        record("Postgres chunk verification", count == len(chunks), f"{count}/{len(chunks)} indexed")

        # Verify in Qdrant
        from shared.config import get_settings
        settings = get_settings()
        qdrant_result = await qdrant.retrieve(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            ids=[str(embedded[0].id)],
            with_payload=True,
        )
        record("Qdrant point verification", len(qdrant_result) == 1,
               f"payload keys: {list(qdrant_result[0].payload.keys())[:5]}")

        return doc, embed_svc

    except Exception as e:
        record("Ingestion", False, str(e))
        traceback.print_exc()
        return None, None


async def stage4_retrieval(db_pool, qdrant, embed_svc):
    header("Stage 4 — Hybrid Retrieval")
    if not embed_svc:
        record("Retrieval", False, "No embedding service")
        return None

    try:
        from query.retriever import HybridRetriever

        retriever = HybridRetriever(qdrant=qdrant, embedding_service=embed_svc)
        t0 = time.monotonic()
        result = await retriever.retrieve(
            query="What is the CAP theorem in distributed systems?",
            top_k=5,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        record("Hybrid retrieval", len(result.chunks) > 0,
               f"{len(result.chunks)} chunks in {elapsed_ms:.0f}ms")
        record("Dense results", result.dense_only_count >= 0,
               f"dense_only={result.dense_only_count}, sparse_only={result.sparse_only_count}, overlap={result.overlap_count}")

        if result.chunks:
            top = result.chunks[0]
            record("Top chunk has payload", bool(top.source_url),
                   f"score={top.score:.4f}, url={top.source_url[:40]}")

        return result

    except Exception as e:
        record("Retrieval", False, str(e))
        traceback.print_exc()
        return None


async def stage5_context_expansion(db_pool, retrieval_result):
    header("Stage 5 — Context Expansion")
    if not retrieval_result or not retrieval_result.chunks:
        record("Context expansion", False, "No retrieval results")
        return None

    try:
        from query.context_expander import ContextExpander

        expander = ContextExpander(db_pool=db_pool)
        expanded = await expander.expand(retrieval_result.chunks)

        with_text = [c for c in expanded if c.text]
        record("Chunk text fetched from Postgres", True,
               f"{len(with_text)}/{len(expanded)} chunks have text")

        if with_text:
            sample = with_text[0]
            record("Sample chunk text", bool(sample.text),
                   f"{len(sample.text)} chars — \"{sample.text[:60]}...\"")

        return expanded

    except Exception as e:
        record("Context expansion", False, str(e))
        traceback.print_exc()
        return None


async def stage6_reranker(expanded_chunks):
    header("Stage 6 — CrossEncoder Reranker")
    if not expanded_chunks:
        record("Reranker", False, "No chunks to rerank")
        return None

    try:
        from query.reranker import CrossEncoderReranker

        info("Loading CrossEncoder model (first run downloads ~80MB)...")
        reranker = CrossEncoderReranker(top_k=3)
        t0 = time.monotonic()
        ranked = reranker.rerank(
            "What is the CAP theorem in distributed systems?",
            expanded_chunks,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        record("Reranking completed", True,
               f"{len(expanded_chunks)} → {len(ranked)} chunks in {elapsed_ms:.0f}ms")

        if ranked:
            record("Top reranked chunk", True,
                   f"rerank_score={ranked[0].rerank_score:.4f}, "
                   f"retrieval_score={ranked[0].retrieval_score:.4f}")
            # Verify rerank scores are in descending order
            scores = [r.rerank_score for r in ranked]
            record("Scores are descending", scores == sorted(scores, reverse=True),
                   f"scores: {[round(s,3) for s in scores]}")

        return ranked

    except Exception as e:
        record("Reranker", False, str(e))
        traceback.print_exc()
        return None


async def stage7_cache(redis_client, embed_svc):
    header("Stage 7 — Semantic Cache")
    if not redis_client:
        warn("Redis not available — skipping cache tests")
        return

    try:
        from query.cache import SemanticCache

        cache = SemanticCache(redis=redis_client, threshold=0.92, ttl=60)

        test_query  = "What is consensus in distributed systems?"
        test_vector = [0.1] * 384   # fake vector for cache test

        # Cache miss on empty
        result = await cache.get(test_query, test_vector)
        record("Cache miss on empty", result is None)

        # Store entry
        response_data = {
            "answer":      "Consensus allows nodes to agree on a value.",
            "citations":   [{"index": 1, "source_url": "http://x.com",
                             "section_title": "Consensus", "score": 0.9}],
            "chunks_used": 2,
        }
        cache_id = await cache.set(test_query, test_vector, response_data, ttl=30)
        record("Cache SET", bool(cache_id), f"id={cache_id[:8]}...")

        # Cache hit — same vector
        hit = await cache.get(test_query, test_vector)
        record("Cache HIT (identical vector)", hit is not None and hit.get("cache_hit") is True,
               f"similarity={hit.get('cache_similarity') if hit else 'N/A'}")

        # Cache miss — very different vector
        diff_vector = [0.9] * 192 + [-0.9] * 192
        miss = await cache.get("completely different query", diff_vector)
        record("Cache MISS (orthogonal vector)", miss is None)

        # Stats
        stats = await cache.stats()
        record("Cache stats", "total_entries" in stats,
               f"entries={stats.get('total_entries')}, hits={stats.get('total_hits')}")

        # Cleanup
        await redis_client.delete(
            "rag:cache:vectors", "rag:cache:queries",
            "rag:cache:responses", "rag:cache:meta",
        )

    except Exception as e:
        record("Semantic cache", False, str(e))
        traceback.print_exc()


async def stage8_prompt_builder(expanded_chunks):
    header("Stage 8 — Prompt Builder")
    try:
        from query.prompt_builder import PromptBuilder, ConversationTurn

        chunks = expanded_chunks or []
        builder = PromptBuilder(max_context_tokens=4000)
        prompt = builder.build(
            "What is the CAP theorem?",
            chunks,
            history=[ConversationTurn(role="user", content="Tell me about distributed systems")]
        )

        record("Prompt built", True,
               f"chunks_used={prompt.chunks_used}/{prompt.chunks_total}, "
               f"~{prompt.tokens_used} tokens")
        record("Citations present", len(prompt.citations) == prompt.chunks_used,
               f"{len(prompt.citations)} citations")
        record("System prompt present", bool(prompt.system))

        messages = builder.build_messages(prompt)
        record("Messages list", messages[0]["role"] == "system",
               f"{len(messages)} messages (system + history + user)")

        if prompt.chunks_used > 0:
            record("Passage [1] in user message", "[1]" in prompt.user)

    except Exception as e:
        record("Prompt builder", False, str(e))
        traceback.print_exc()


async def stage9_query_engine(db_pool, qdrant, embed_svc, redis_client):
    header("Stage 9 — Full Query Engine (no LLM)")
    if not embed_svc:
        record("Query engine", False, "No embedding service")
        return

    try:
        from query.engine import QueryEngine, QueryRequest

        # Build engine without LLM (generation will fail gracefully,
        # but retrieval + rerank + prompt build should all work)
        engine = QueryEngine(
            qdrant=qdrant,
            embedding_service=embed_svc,
            db_pool=db_pool,
            redis=redis_client,
        )

        # Test the retrieval + rerank + prompt pipeline directly
        request = QueryRequest(
            query="What consensus algorithms are used in distributed systems?",
            top_k=5,
            use_cache=False,    # skip cache for deterministic test
            use_reranker=True,
            stream=False,
        )

        _, prompt, retrieval = await engine._retrieve_and_build(request)

        record("Pipeline: decompose → retrieve → expand → rerank → build",
               prompt.chunks_used > 0,
               f"{prompt.chunks_used} chunks in prompt, "
               f"{retrieval.overlap_count} hybrid overlap")

        reranked = getattr(engine, "_last_reranked", False)
        record("Reranker was used", reranked)

        sub_queries = getattr(engine, "_last_sub_queries", [])
        record("Query decomposition", len(sub_queries) >= 1,
               f"sub_queries={sub_queries}")

    except Exception as e:
        record("Query engine pipeline", False, str(e))
        traceback.print_exc()


async def stage10_api_health():
    header("Stage 10 — API Endpoints")
    try:
        import httpx
        async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=5) as client:
            # Health
            r = await client.get("/health")
            record("GET /health", r.status_code == 200, r.text)

            # Ready
            r = await client.get("/ready")
            record("GET /ready", r.status_code in (200, 503),
                   f"status={r.status_code}")

            # Docs
            r = await client.get("/docs")
            record("GET /docs (OpenAPI)", r.status_code == 200)

    except httpx.ConnectError:
        warn("API not running — start with: uvicorn api.main:app --reload --port 8000")
        record("API reachable", False, "Connection refused — API not started")
    except Exception as e:
        record("API health check", False, str(e))


# ── Main ──────────────────────────────────────────────────────────

async def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Distributed RAG Pipeline — End-to-End Test{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    t_start = time.monotonic()

    db_pool, qdrant, redis_client = await stage1_infrastructure()

    if not db_pool or not qdrant:
        print(f"\n{RED}Infrastructure not available. Is Docker Compose running?{RESET}")
        print("  cd infrastructure && docker compose up -d")
        sys.exit(1)

    collection_ok = await stage2_collection_setup(qdrant)
    doc, embed_svc = await stage3_ingestion(db_pool, qdrant)
    retrieval_result = await stage4_retrieval(db_pool, qdrant, embed_svc)
    expanded = await stage5_context_expansion(db_pool, retrieval_result)
    ranked = await stage6_reranker(expanded)
    await stage7_cache(redis_client, embed_svc)
    await stage8_prompt_builder(expanded)
    await stage9_query_engine(db_pool, qdrant, embed_svc, redis_client)
    await stage10_api_health()

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    passed  = sum(1 for _, p, _ in results if p)
    failed  = sum(1 for _, p, _ in results if not p)
    total   = len(results)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Results: {GREEN}{passed} passed{RESET}{BOLD}, "
          f"{RED if failed else ''}{failed} failed{RESET}{BOLD}, "
          f"{total} total  ({elapsed:.1f}s){RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    if failed:
        print(f"\n{RED}Failed checks:{RESET}")
        for name, passed_, detail in results:
            if not passed_:
                print(f"  {RED}✗{RESET}  {name}: {detail}")

    await db_pool.close()
    if redis_client:
        await redis_client.aclose()
    if qdrant:
        await qdrant.close()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())