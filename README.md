# Distributed RAG Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![Qdrant](https://img.shields.io/badge/Qdrant-1.9-red)
![Kafka](https://img.shields.io/badge/Kafka-KRaft-black?logo=apachekafka)
![Redis](https://img.shields.io/badge/Redis-7-red?logo=redis)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql)
![License](https://img.shields.io/badge/License-MIT-yellow)

A production-grade distributed Retrieval-Augmented Generation (RAG) pipeline built across 4 phases — from async document ingestion through Kafka, to hybrid BM25+vector search, CrossEncoder reranking, and semantic caching. Designed as a portfolio project targeting ML/AI Engineer roles.

---

## Architecture

```
Document Sources
      │
      ▼
┌─────────────────┐
│ Intake Validator │  ← dedup, MIME detection, size checks
└────────┬────────┘
         │ direct (sync)          │ async (Kafka)
         ▼                        ▼
┌──────────────────┐    ┌──────────────────┐
│  Inline Pipeline │    │  Kafka Workers   │
└────────┬─────────┘    │  PDF · HTML      │
         │              └────────┬─────────┘
         └──────────┬────────────┘
                    ▼
         ┌─────────────────────┐
         │  Chunking Selector  │  ← Fixed · Semantic · Hierarchical
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Embedding Service  │  ← HuggingFace · OpenAI · Ollama
         └──────────┬──────────┘
                    ▼
         ┌──────────────────────────────┐
         │  Storage Writer              │
         │  Qdrant (vectors)            │
         │  PostgreSQL (metadata)       │
         └──────────────────────────────┘

Query:
  ┌──────────────────────────────────────────────────┐
  │  Semantic Cache (Redis)  ← cosine similarity hit │
  └──────────────────┬───────────────────────────────┘
                     │ miss
                     ▼
         ┌─────────────────────┐
         │  Query Decomposer   │  ← LLM-based (OpenAI / Ollama / Groq)
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────────────────┐
         │  Hybrid Retriever               │
         │  Dense (ANN) + Sparse (BM25)    │
         │  Reciprocal Rank Fusion (RRF)   │
         └──────────┬──────────────────────┘
                    ▼
         ┌─────────────────────┐
         │  Context Expander   │  ← fetch text + L1 parent from Postgres
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  CrossEncoder       │  ← ms-marco-MiniLM-L-6-v2
         │  Reranker           │  ← top-30 → top-10
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Prompt Builder     │  ← token budget, citations [1][2]
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  LLM Generator      │  ← OpenAI · Groq · Ollama streaming
         └──────────┬──────────┘
                    ▼
              SSE Stream → Client
```

---

## Stack

| Layer            | Technology                                      |
|------------------|-------------------------------------------------|
| Message queue    | Apache Kafka (KRaft — no Zookeeper)             |
| Vector DB        | Qdrant (HNSW + INT8 quantization)               |
| Metadata store   | PostgreSQL 16                                   |
| Cache            | Redis 7 (semantic cosine similarity)            |
| Embeddings       | sentence-transformers / OpenAI / Ollama         |
| Reranker         | cross-encoder/ms-marco-MiniLM-L-6-v2            |
| LLM backends     | OpenAI · Groq · Ollama                          |
| API framework    | FastAPI (async, SSE streaming)                  |
| Tracing          | OpenTelemetry → Jaeger                          |
| Metrics          | Prometheus                                      |
| Infrastructure   | Docker Compose                                  |

---

## Phases

### Phase 1 — Ingestion Pipeline

End-to-end async document ingestion over Kafka with at-least-once delivery guarantees.

**Key components:**
- `IntakeValidator` — SHA-256 content deduplication, MIME type detection via libmagic, size limits
- `IngestionProducer` — Kafka producer with `acks=all`, idempotent delivery, manual offset commit
- `PDFWorker` — Unstructured.io layout-aware extraction, OCR confidence gating, per-page error isolation, markdown table serialization
- `HTMLWorker` — trafilatura main-content extraction, encoding detection, section reconstruction
- `BaseWorker` — consumer loop with `RetryableError` / `PoisonPillError` classification, exponential backoff retries, DLQ routing
- `WorkerManager` — supervisor that starts workers as OS processes and auto-restarts on crash

**Three chunking strategies (auto-selected by doc type):**

| Strategy      | Best for                        | Mechanism                                      |
|---------------|---------------------------------|------------------------------------------------|
| Fixed         | Code, TXT, structured exports   | Token-aware sliding window with configurable overlap |
| Semantic      | Articles, documentation, HTML   | Sentence-transformer cosine similarity boundary detection |
| Hierarchical  | PDFs, DOCX, long-form docs      | L0/L1/L2 parent-child tree — enables context expansion at query time |

**Storage:**
- Postgres-first two-store atomicity saga — chunk written as `pending_vector`, Qdrant upserted, then marked `indexed`
- Background reconciliation job re-queues any stuck `pending_vector` chunks
- Embedding model version stamped on every chunk — enables zero-downtime model migration

---

### Phase 2 — Qdrant + Hybrid Search

Qdrant collection with HNSW tuning, INT8 quantization, BM25 sparse vectors, and hybrid RRF retrieval.

**Qdrant collection configuration:**
- Named vectors: `dense` (float32, cosine) + `sparse` (BM25 token weights)
- HNSW: `m=16`, `ef_construct=200` — Recall@10 ≈ 0.97
- INT8 scalar quantization — 4× memory reduction, ~1-2% recall cost
- Payload indexes on all filter fields — O(log n) filtered search

**BM25 sparse encoder (`BM25Encoder`):**
- Corpus-aware IDF fitting with TF saturation (`k1=1.5`, `b=0.75`)
- Consistent token IDs via cl100k tokenizer — same vocabulary as dense embeddings
- `save()` / `load()` — persists fitted IDF table across restarts

**Hybrid retrieval (`HybridRetriever`):**
- Dense + sparse searches run concurrently via `asyncio.gather`
- Reciprocal Rank Fusion merges results: `score(d) = Σ 1/(k + rank_i(d))` with `k=60`
- Documents appearing in both lists get a natural boost without manual score normalization

**Context expansion:**
- Chunk text fetched from Postgres in a single batch query
- L2 (paragraph) chunks enriched with their L1 parent section — solves the "orphaned chunk" problem

---

### Phase 3 — Query Engine + FastAPI

Full RAG pipeline with LLM-based query decomposition, streaming generation, and a FastAPI gateway.

**Query decomposer (`QueryDecomposer`):**
- Calls `gpt-4o-mini` (OpenAI) or `llama3.2` (Ollama) with a strict JSON schema
- Splits compound queries — `"What is X and how does Y work?"` → two focused sub-queries
- Falls back to the original query on any failure
- Short queries (≤6 words) skip the LLM call entirely

**Prompt builder (`PromptBuilder`):**
- Token-budget-aware context assembly — never exceeds configured context window
- Citation numbering `[1]`, `[2]` for inline source attribution
- L1 parent context prepended for L2 hierarchy chunks
- Multi-turn conversation history support

**LLM generator (`LLMGenerator`):**
- Streaming via Server-Sent Events — yields `GeneratorChunk` with `.to_sse()`
- OpenAI: native async SSE streaming
- Groq: OpenAI-compatible API, 10-20× faster than local models
- Ollama: chunked HTTP streaming for local models
- Final chunk carries citations and token usage stats

**FastAPI endpoints:**

| Method | Path                    | Description                              |
|--------|-------------------------|------------------------------------------|
| POST   | `/ingest/file`          | Multipart file upload                    |
| POST   | `/ingest/url`           | HTTP URL or S3 path                      |
| POST   | `/ingest/text`          | Raw text with metadata                   |
| GET    | `/ingest/list`          | List all documents with status           |
| GET    | `/ingest/{doc_id}/status` | Poll processing status                 |
| POST   | `/query`                | Non-streaming RAG query                  |
| POST   | `/query/stream`         | SSE streaming query                      |
| GET    | `/cache/stats`          | Semantic cache statistics                |
| GET    | `/health`               | Liveness check                           |
| GET    | `/ready`                | Readiness check (Postgres + Qdrant + Redis) |

---

### Phase 4 — CrossEncoder Reranker + Semantic Cache

Two-stage retrieval quality improvement: reranking with a cross-encoder model, and semantic deduplication of repeated queries via Redis.

**CrossEncoder reranker (`CrossEncoderReranker`):**
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, ~50ms on CPU for 30 chunks)
- Bi-encoders embed query and chunk independently — cross-encoder sees both together, producing a significantly more accurate relevance score
- Pattern: retrieve top-30 with fast ANN, rerank to top-10 with cross-encoder
- Lazy-loaded singleton — model loaded once per process on first request

**Semantic cache (`SemanticCache`):**
- Cache key is the query embedding vector, not the query string
- On lookup: computes cosine similarity against all cached vectors
- Hit threshold: 92% similarity — catches paraphrased and semantically equivalent queries
- TTL tracked via Redis expiry keys
- On hit: skips retrieval, reranking, and LLM call entirely — returns cached response in milliseconds

**Prometheus metrics added:**
- `reranker_duration_seconds` — histogram of reranking latency
- `semantic_cache_hits_total` — cache hit counter
- `semantic_cache_misses_total` — cache miss counter

**Pipeline flags** (per-request, configurable from the API):
- `use_reranker: bool` — toggle reranking on/off
- `use_cache: bool` — bypass cache for debugging

---

## Quick Start

### 1. Prerequisites

- Docker Desktop
- Python 3.11+
- `uv` or `pip`

### 2. Clone and configure

```bash
git clone https://github.com/your-username/distributed-rag.git
cd distributed-rag
cp .env.example .env
# Edit .env — defaults work for local dev
```

### 3. Start infrastructure

```bash
make up
# Wait ~20 seconds for all services to be healthy
make setup-collection
```

### 4. Install dependencies

```bash
pip install -e ".[dev]"
```

### 5. Start the API

```bash
make api
# API running at http://localhost:8000
# OpenAPI docs at http://localhost:8000/docs
```

### 6. Ingest documents

```bash
# Single file
make ingest-file FILE=docs/report.pdf

# Entire directory
make ingest-dir DIR=./knowledge-base/

# URL
make ingest-url URL=https://docs.example.com/page

# Raw text via API
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your content here", "title": "My Document"}'
```

### 7. Query

```bash
# Non-streaming
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "use_reranker": true}'

# Streaming (SSE)
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the architecture"}'
```

### 8. Run tests

```bash
# Unit tests (no Docker needed)
make unit

# Full end-to-end test (requires Docker stack + API running)
make e2e
```

---

## Configuration

All configuration via `.env` (see `.env.example` for full reference):

```bash
# LLM backend — set ONE of these
OPENAI_API_KEY=sk-...          # uses gpt-4o-mini
GROQ_API_KEY=gsk_...           # uses llama-3.3-70b-versatile (fastest)
# Neither = Ollama on localhost:11434

# Embedding backend
EMBED_BACKEND=huggingface      # huggingface | openai | ollama

# Groq model (optional)
GROQ_MODEL=llama-3.3-70b-versatile
```

---

## Make Commands

```bash
make up                        # start Docker Compose stack
make down                      # stop stack
make reset                     # wipe volumes and restart
make setup-collection          # create Qdrant collection (run once)
make api                       # start FastAPI on :8000
make workers                   # start Kafka workers (async ingest mode)
make unit                      # run unit tests
make integration               # run integration tests
make e2e                       # run end-to-end validation script
make ingest-file FILE=path     # ingest a single file
make ingest-dir DIR=path       # ingest a directory
make ingest-url URL=https://…  # ingest a web page
make lint                      # ruff linting
make typecheck                 # mypy type checking
```

---

## Project Structure

```
distributed-rag/
├── api/
│   ├── main.py              — FastAPI app, lifespan, query endpoints
│   └── ingest.py            — Ingest router (file/url/text/status/list)
├── ingestion/
│   ├── chunking/
│   │   ├── fixed.py         — Token-aware sliding window
│   │   ├── semantic.py      — Cosine similarity boundary detection
│   │   ├── hierarchical.py  — L0/L1/L2 parent-child tree
│   │   └── selector.py      — DocType → chunker mapping
│   ├── embedding/
│   │   ├── service.py       — OpenAI/HuggingFace/Ollama backends
│   │   └── sparse.py        — BM25 encoder (fit/encode/save/load)
│   ├── intake/
│   │   ├── validator.py     — Dedup, MIME, size, lifecycle
│   │   └── producer.py      — Kafka producer
│   ├── storage/
│   │   ├── writer.py        — Two-store atomic writer
│   │   └── reconciliation.py — Background pending_vector cleanup
│   └── workers/
│       ├── base_worker.py   — Consumer loop, retry, DLQ
│       ├── pdf_worker.py    — Unstructured.io PDF parser
│       ├── html_worker.py   — trafilatura HTML parser
│       └── worker_manager.py — Supervisor process
├── query/
│   ├── retriever.py         — HybridRetriever: concurrent dense+sparse, RRF
│   ├── context_expander.py  — Text fetch, L1 parent expansion
│   ├── decomposer.py        — LLM query decomposer
│   ├── prompt_builder.py    — Token-budget prompt assembly
│   ├── generator.py         — Streaming LLM (OpenAI/Groq/Ollama)
│   ├── reranker.py          — CrossEncoder reranker
│   ├── cache.py             — Semantic Redis cache
│   └── engine.py            — Full pipeline orchestrator
├── shared/
│   ├── models.py            — Pydantic v2 models
│   ├── config.py            — Environment-based settings
│   └── telemetry.py         — OTel + Prometheus metrics
├── infrastructure/
│   ├── docker-compose.yml   — Kafka, Qdrant, Postgres, Redis, Jaeger
│   └── postgres/
│       ├── schema.sql       — Full schema with indexes and views
│       └── migrations/      — Incremental schema changes
├── scripts/
│   ├── ingest.py            — Bulk ingestion CLI
│   └── test_e2e.py          — 10-stage end-to-end validation
├── tests/
│   ├── unit/                — 50+ unit tests (no Docker needed)
│   └── integration/         — Full pipeline integration tests
├── Makefile
└── pyproject.toml
```

---

## Key Engineering Decisions

| Decision | Rationale |
|----------|-----------|
| Manual Kafka offset commit | Offsets advance only after confirmed write — no data loss on crash |
| Content-hash chunk IDs | Qdrant upserts are idempotent — safe to retry without duplicates |
| Postgres-first two-store write | Postgres row at `pending_vector` acts as crash recovery checkpoint |
| Embedding model version on every chunk | Enables zero-downtime model migration via parallel collection + alias swap |
| Three chunking strategies auto-selected | Chunking quality has the largest single impact on retrieval precision |
| RRF for hybrid fusion | Merges ranked lists without normalizing incompatible score scales |
| CrossEncoder after ANN retrieval | Recall of 30 with precision of 10 — best of both worlds |
| Semantic cache with cosine similarity | Catches paraphrased queries that string-match caches miss entirely |

---

## Infrastructure Services

| Service      | URL                          | Purpose                    |
|--------------|------------------------------|----------------------------|
| FastAPI      | http://localhost:8000        | RAG API + OpenAPI docs      |
| Kafka UI     | http://localhost:8090        | Topic lag, message rates    |
| Qdrant       | http://localhost:6333/dashboard | Vector collection inspector |
| Jaeger       | http://localhost:16686       | Distributed traces          |
| Prometheus   | http://localhost:9090        | Metrics scraping            |

---

## License

MIT
