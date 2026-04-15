# Distributed RAG Pipeline

A production-grade, distributed Retrieval-Augmented Generation system.

## Phase 1 — Ingestion Pipeline (this branch)

### Architecture

```
Document Sources
      │
      ▼
Intake Validator ──→ DLQ (failed/rejected)
      │
      ▼
Kafka: raw-documents (4 partitions)
      │
   ┌──┴────┐
   ▼       ▼
PDF       HTML    (parallel workers, same consumer group)
Worker   Worker
   └──┬────┘
      ▼
  Chunker
  (fixed / semantic / hierarchical — selected by doc type)
      │
      ▼
Embedding Service
  (OpenAI / HuggingFace / Ollama — swappable)
      │
      ▼
Storage Writer
  ├── Qdrant (vectors)
  └── PostgreSQL (metadata registry)
```

### Stack

| Layer | Technology |
|---|---|
| Message queue | Apache Kafka |
| Vector DB | Qdrant |
| Metadata store | PostgreSQL 16 |
| Cache / broker | Redis 7 |
| PDF parsing | Unstructured.io |
| HTML parsing | trafilatura |
| Tokenization | tiktoken (cl100k_base) |
| Embeddings | sentence-transformers / OpenAI / Ollama |
| Tracing | OpenTelemetry → Jaeger |
| Metrics | Prometheus |
| API framework | FastAPI (Phase 5) |

---

## Quick start

### 1. Prerequisites

- Docker Desktop (required — all services run in containers)
- Python 3.11+
- `pip` or `uv`

### 2. Clone and configure

```bash
git clone https://github.com/your-username/distributed-rag.git
cd distributed-rag
cp .env.example .env
# Edit .env if needed — defaults work for local development
```

### 3. Start infrastructure

```bash
cd infrastructure
docker compose up -d

# Check all services are healthy
docker compose ps
```

Services and their UIs:
| Service | URL |
|---|---|
| Kafka UI | http://localhost:8090 |
| Qdrant dashboard | http://localhost:6333/dashboard |
| Jaeger tracing | http://localhost:16686 |

### 4. Install Python dependencies

```bash
# Using uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Or using pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 5. Run unit tests

```bash
pytest tests/unit/ -v
```

### 6. Run integration tests (requires Docker stack)

```bash
pytest tests/integration/ -v
```

---

## Project structure

```
distributed-rag/
├── ingestion/
│   ├── intake/
│   │   ├── validator.py          # Dedup, size, MIME checks
│   │   └── producer.py           # Kafka producer
│   ├── workers/
│   │   ├── base_worker.py        # Consumer loop + retry + DLQ
│   │   ├── pdf_worker.py         # Unstructured.io PDF parser
│   │   └── html_worker.py        # trafilatura HTML parser
│   ├── chunking/
│   │   ├── fixed.py              # Fixed-size + overlap
│   │   ├── semantic.py           # Cosine-similarity boundary detection
│   │   └── hierarchical.py       # L0/L1/L2 parent-child tree
│   ├── embedding/
│   │   └── service.py            # Batched, multi-backend embedding
│   └── storage/
│       ├── writer.py             # Idempotent Qdrant + Postgres writer
│       └── reconciliation.py     # Background pending_vector cleanup
├── shared/
│   ├── models.py                 # Pydantic schemas (all inter-service contracts)
│   ├── config.py                 # Environment-based settings
│   └── telemetry.py              # OTel tracing + Prometheus metrics
├── infrastructure/
│   ├── docker-compose.yml
│   └── postgres/
│       └── schema.sql
├── tests/
│   ├── unit/
│   │   └── test_chunking.py
│   └── integration/
│       └── test_ingestion_pipeline.py
├── .env.example
├── pyproject.toml
└── .github/workflows/ci.yml
```

---

## Key engineering decisions

| Decision | Rationale |
|---|---|
| Manual Kafka offset commit | Offsets advance only after confirmed write — no data loss on crash |
| Content-hash chunk IDs | Qdrant upserts are idempotent — safe to retry without duplicates |
| Postgres-first two-store write | Postgres row at `pending_vector` acts as crash recovery checkpoint |
| Embedding model version on every chunk | Enables zero-downtime model migration via parallel collection + alias swap |
| Three chunking strategies | Fixed for code, semantic for narrative, hierarchical for long-form |
| Strategy per doc_type | Chunking quality has the largest impact on retrieval precision |

---

## Observability

After starting the stack:

- **Kafka topics**: http://localhost:8090 — monitor consumer lag, message rates
- **Traces**: http://localhost:16686 — search by `doc_id` to trace a document end-to-end
- **Qdrant**: http://localhost:6333/dashboard — inspect collection, vector count, payload

---

## Coming next

- **Phase 2**: Qdrant cluster configuration, HNSW index tuning, hybrid BM25+vector collection
- **Phase 3**: Query engine — semantic search, query decomposition, hybrid retrieval fusion
- **Phase 4**: Reranker (CrossEncoder) + semantic Redis cache
- **Phase 5**: FastAPI gateway + streaming Next.js UI with citations
- **Phase 6**: Full observability dashboards + Kubernetes manifests