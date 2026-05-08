-- =============================================================
--  Distributed RAG Pipeline — PostgreSQL Schema
--  Phase 1: Ingestion metadata registry
-- =============================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================
--  ENUM TYPES
-- =============================================================

CREATE TYPE doc_status AS ENUM (
    'pending',          -- Accepted, not yet queued
    'queued',           -- On Kafka, awaiting worker
    'parsing',          -- Worker is parsing
    'chunking',         -- Chunker is running
    'embedding',        -- Embedding service is running
    'partially_indexed',-- Some pages/chunks failed
    'indexed',          -- Fully indexed in Qdrant
    'failed',           -- Exceeded max retries → DLQ
    'deleted'           -- Source deleted, chunks removed
);

CREATE TYPE chunk_status AS ENUM (
    'pending_vector',   -- Written to Postgres, Qdrant write pending
    'indexed',          -- Written to both stores
    'stale',            -- Document updated, chunk superseded
    'failed'            -- Embedding or storage write failed
);

CREATE TYPE doc_type AS ENUM (
    'pdf', 'html', 'docx', 'markdown', 'txt', 'code', 'unknown'
);

-- =============================================================
--  DOCUMENTS  — one row per source document
-- =============================================================

CREATE TABLE documents (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    source_url          TEXT        NOT NULL,
    content_hash        TEXT        NOT NULL,           -- SHA-256 of raw file bytes
    doc_type            doc_type    NOT NULL DEFAULT 'unknown',
    status              doc_status  NOT NULL DEFAULT 'pending',

    -- Source metadata
    title               TEXT,
    author              TEXT,
    source_type         TEXT        NOT NULL DEFAULT 'unknown', -- 's3', 'http', 'notion', etc.

    -- Size tracking
    byte_size           BIGINT,
    page_count          INT,

    -- Version tracking (for updates)
    version             INT         NOT NULL DEFAULT 1,
    previous_hash       TEXT,                           -- hash before last update

    -- Processing stats
    total_chunks        INT         DEFAULT 0,
    failed_chunks       INT         DEFAULT 0,
    ingested_at         TIMESTAMPTZ,
    processing_started  TIMESTAMPTZ,
    processing_duration_ms BIGINT,

    -- Error tracking
    last_error          TEXT,
    retry_count         INT         NOT NULL DEFAULT 0,

    tags                JSONB,                             -- arbitrary key-value pairs for enrichment and filtering

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Unique on source_url (one active record per document location)
CREATE UNIQUE INDEX idx_documents_source_url ON documents(source_url);

-- Fast lookup by hash for deduplication
CREATE INDEX idx_documents_hash ON documents(content_hash);

-- Status polling (reconciliation job, monitoring)
CREATE INDEX idx_documents_status ON documents(status);

-- Source-type filtering
CREATE INDEX idx_documents_source_type ON documents(source_type);

-- =============================================================
--  CHUNKS  — one row per chunk, mirrors Qdrant point payload
-- =============================================================

CREATE TABLE chunks (
    id                  UUID        PRIMARY KEY,        -- content hash of chunk text (idempotent)
    doc_id              UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index         INT         NOT NULL,           -- 0-based position within document
    status              chunk_status NOT NULL DEFAULT 'pending_vector',

    -- Content identifiers
    content_hash        TEXT        NOT NULL,           -- SHA-256 of chunk text
    token_count         INT,
    char_count          INT,

    -- Structural metadata (carried into Qdrant payload)
    section_title       TEXT,
    page_number         INT,
    source_url          TEXT        NOT NULL,
    raw_text            TEXT,

    -- Hierarchical chunking (null for flat strategies)
    parent_chunk_id     UUID        REFERENCES chunks(id),
    hierarchy_level     INT         NOT NULL DEFAULT 0, -- 0=doc, 1=section, 2=paragraph

    -- Chunking provenance
    chunking_strategy   TEXT        NOT NULL,           -- 'fixed', 'semantic', 'hierarchical'
    chunk_overlap_tokens INT,

    -- Embedding provenance (critical for migration)
    embedding_model     TEXT        NOT NULL,           -- e.g. 'text-embedding-3-small'
    embedding_model_version TEXT    NOT NULL,           -- e.g. '1.0.0'
    embedding_dim       INT         NOT NULL,           -- e.g. 1536

    -- Schema evolution
    schema_version      INT         NOT NULL DEFAULT 1,

    -- Timestamps
    ingested_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Most common query: all chunks for a document (for delete-on-update)
CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);

-- Reconciliation job: find stuck pending_vector chunks
CREATE INDEX idx_chunks_status ON chunks(status);

-- Embedding migration: find all chunks using a specific model
CREATE INDEX idx_chunks_embedding_model ON chunks(embedding_model);

-- Hierarchy traversal: find children of a parent chunk
CREATE INDEX idx_chunks_parent_id ON chunks(parent_chunk_id);

-- =============================================================
--  INGESTION COST LOG  — per-batch embedding API spend
-- =============================================================

CREATE TABLE ingestion_cost_log (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          UUID        REFERENCES documents(id) ON DELETE SET NULL,
    model           TEXT        NOT NULL,
    batch_size      INT         NOT NULL,
    token_count     INT         NOT NULL,
    cost_usd        NUMERIC(10, 8),                    -- NULL if model is local/free
    latency_ms      INT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cost_log_doc_id ON ingestion_cost_log(doc_id);
CREATE INDEX idx_cost_log_created ON ingestion_cost_log(created_at);

-- =============================================================
--  DLQ EVENTS  — failed ingestion attempts for manual replay
-- =============================================================

CREATE TABLE dlq_events (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          UUID        REFERENCES documents(id) ON DELETE SET NULL,
    kafka_topic     TEXT        NOT NULL,
    kafka_partition INT,
    kafka_offset    BIGINT,
    error_type      TEXT        NOT NULL,
    error_message   TEXT,
    payload         JSONB,                             -- original Kafka message for replay
    retry_count     INT         NOT NULL DEFAULT 0,
    resolved        BOOLEAN     NOT NULL DEFAULT FALSE,
    resolved_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_dlq_resolved ON dlq_events(resolved, created_at);

-- =============================================================
--  HELPER: auto-update updated_at on documents
-- =============================================================

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================
--  VIEWS  — useful for monitoring queries
-- =============================================================

-- Ingestion funnel: how many documents are at each stage
CREATE VIEW v_ingestion_funnel AS
SELECT
    status,
    COUNT(*)            AS doc_count,
    SUM(byte_size)      AS total_bytes,
    AVG(retry_count)    AS avg_retries
FROM documents
GROUP BY status
ORDER BY doc_count DESC;

-- Chunks pending vector write (reconciliation target)
CREATE VIEW v_pending_reconciliation AS
SELECT
    c.id,
    c.doc_id,
    c.created_at,
    EXTRACT(EPOCH FROM (NOW() - c.created_at)) AS age_seconds,
    d.source_url
FROM chunks c
JOIN documents d ON c.doc_id = d.id
WHERE c.status = 'pending_vector'
  AND c.created_at < NOW() - INTERVAL '10 minutes'
ORDER BY c.created_at;

-- Embedding model distribution (migration planning)
CREATE VIEW v_embedding_model_stats AS
SELECT
    embedding_model,
    embedding_model_version,
    COUNT(*)    AS chunk_count,
    MIN(ingested_at) AS first_ingested,
    MAX(ingested_at) AS last_ingested
FROM chunks
WHERE status = 'indexed'
GROUP BY embedding_model, embedding_model_version
ORDER BY chunk_count DESC;
