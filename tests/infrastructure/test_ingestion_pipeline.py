"""
End-to-end integration test for Phase 1 ingestion pipeline.
Requires the Docker Compose stack to be running:
  - Kafka on localhost:9092
  - PostgreSQL on localhost:5432
  - Qdrant on localhost:6333

Run with:
  docker compose -f infrastructure/docker-compose.yml up -d
  pytest tests/integration/ -v
"""

from __future__ import annotations

import uuid

import asyncpg
import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from ingestion.chunking.fixed import FixedChunker
from ingestion.chunking.hierarical import HierarchicalChunker
from ingestion.embedding.services import EmbeddingService, HuggingFaceBackend
from ingestion.storage.writer import StorageWriter
from shared.config import get_settings
from shared.models import DocType, ParsedDocument, Section

settings = get_settings()

# ── Fixtures ──────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="module")
async def db_pool():
    pool = await asyncpg.create_pool(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        database=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        min_size=1,
        max_size=3,
    )
    yield pool
    await pool.close()


@pytest_asyncio.fixture(scope="module")
async def qdrant():
    client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    yield client
    await client.close()


@pytest_asyncio.fixture(scope="module")
async def storage_writer(db_pool, qdrant):
    writer = StorageWriter(db_pool=db_pool, qdrant=qdrant)
    await writer.ensure_collection()
    return writer


@pytest_asyncio.fixture(scope="module")
async def embedding_service(db_pool):
    backend = HuggingFaceBackend()
    return EmbeddingService(backend=backend, db_pool=db_pool)


def make_test_document(text: str | None = None) -> ParsedDocument:
    doc_id = uuid.uuid4()
    body = text or (
        "Distributed systems are systems in which components located on "
        "networked computers communicate and coordinate their actions by "
        "passing messages. " * 30
    )
    return ParsedDocument(
        doc_id=doc_id,
        source_url=f"file:///test/integration/{doc_id}.pdf",
        doc_type=DocType.PDF,
        raw_text=body,
        sections=[
            Section(title="Introduction", content=body[:len(body)//2]),
            Section(title="Conclusion", content=body[len(body)//2:]),
        ],
        title="Integration Test Document",
    )


async def register_test_document(db_pool: asyncpg.Pool, doc: ParsedDocument) -> None:
    """Insert a document row so foreign key constraints are satisfied."""
    await db_pool.execute(
        """
        INSERT INTO documents (id, source_url, source_type, doc_type, content_hash, byte_size, status)
        VALUES ($1, $2, 'test', $3, $4, 1000, 'embedding')
        ON CONFLICT (source_url) DO UPDATE SET status = 'embedding'
        """,
        doc.doc_id,
        doc.source_url,
        doc.doc_type.value if hasattr(doc.doc_type, 'value') else doc.doc_type,
        doc.content_hash,
    )


# ── Tests ─────────────────────────────────────────────────────────

class TestFixedChunkPipeline:
    """Full pipeline: fixed chunk → embed → store → verify in Qdrant and Postgres."""

    @pytest.mark.asyncio
    async def test_end_to_end_fixed_chunking(self, db_pool, storage_writer, embedding_service):
        doc = make_test_document()
        await register_test_document(db_pool, doc)

        # Chunk
        chunker = FixedChunker(chunk_size=128, overlap=15)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0, "Chunker produced no chunks"

        # Embed
        embedded = await embedding_service.embed_chunks(chunks)
        assert all(c.vector is not None for c in embedded), "Some chunks missing vectors"
        assert all(len(c.vector) > 0 for c in embedded), "Empty vectors found"

        # Store
        await storage_writer.store(embedded, doc)

        # Verify in Postgres
        row = await db_pool.fetchrow(
            "SELECT total_chunks, status FROM documents WHERE id = $1",
            doc.doc_id,
        )
        assert row is not None, "Document not found in Postgres"
        assert row["status"] == "indexed", f"Unexpected status: {row['status']}"
        assert row["total_chunks"] == len(chunks)

        # Verify chunk rows in Postgres
        chunk_rows = await db_pool.fetch(
            "SELECT status FROM chunks WHERE doc_id = $1",
            doc.doc_id,
        )
        assert len(chunk_rows) == len(chunks)
        assert all(r["status"] == "indexed" for r in chunk_rows)

        # Verify vectors in Qdrant
        result = await storage_writer._qdrant.retrieve(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            ids=[str(chunks[0].id)],
            with_vectors=True,
            with_payload=True,
        )
        assert len(result) == 1, "First chunk not found in Qdrant"
        point = result[0]
        assert point.payload["doc_id"] == str(doc.doc_id)
        assert point.payload["chunking_strategy"] == "fixed"
        assert len(point.vector) > 0


class TestHierarchicalChunkPipeline:
    """Pipeline test using hierarchical chunking strategy."""

    @pytest.mark.asyncio
    async def test_hierarchical_levels_stored_correctly(self, db_pool, storage_writer, embedding_service):
        doc = make_test_document()
        await register_test_document(db_pool, doc)

        chunker = HierarchicalChunker(parent_tokens=300, child_tokens=80)
        chunks = chunker.chunk(doc)

        levels = {c.hierarchy_level for c in chunks}
        assert 0 in levels
        assert 1 in levels
        assert 2 in levels

        embedded = await embedding_service.embed_chunks(chunks)
        await storage_writer.store(embedded, doc)

        # Verify parent-child relationships preserved in Postgres
        l2_chunks = [c for c in chunks if c.hierarchy_level == 2]
        if l2_chunks:
            row = await db_pool.fetchrow(
                "SELECT parent_chunk_id FROM chunks WHERE id = $1",
                l2_chunks[0].id,
            )
            assert row["parent_chunk_id"] is not None, "L2 chunk missing parent_chunk_id"


class TestIdempotency:
    """Verify that re-ingesting the same document is safe (upsert semantics)."""

    @pytest.mark.asyncio
    async def test_double_store_does_not_duplicate(self, db_pool, storage_writer, embedding_service):
        doc = make_test_document()
        await register_test_document(db_pool, doc)

        chunker = FixedChunker(chunk_size=128, overlap=10)
        chunks = chunker.chunk(doc)
        embedded = await embedding_service.embed_chunks(chunks)

        # Store twice
        await storage_writer.store(embedded, doc)
        await storage_writer.store(embedded, doc)

        # Should still have exactly len(chunks) indexed rows (not doubled)
        count = await db_pool.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = $1 AND status = 'indexed'",
            doc.doc_id,
        )
        assert count == len(chunks), f"Expected {len(chunks)} chunks, got {count}"


class TestDocumentUpdateFlow:
    """Verify that updating a document replaces old chunks cleanly."""

    @pytest.mark.asyncio
    async def test_update_replaces_old_chunks(self, db_pool, storage_writer, embedding_service):
        # First version
        doc_v1 = make_test_document("Version one content. " * 40)
        await register_test_document(db_pool, doc_v1)

        chunker = FixedChunker(chunk_size=100, overlap=10)
        chunks_v1 = chunker.chunk(doc_v1)
        embedded_v1 = await embedding_service.embed_chunks(chunks_v1)
        await storage_writer.store(embedded_v1, doc_v1)

        v1_count = await db_pool.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = $1 AND status = 'indexed'",
            doc_v1.doc_id,
        )

        # Second version — same doc_id, different text (simulates update)
        doc_v2 = ParsedDocument(
            doc_id=doc_v1.doc_id,
            source_url=doc_v1.source_url,
            doc_type=doc_v1.doc_type,
            raw_text="Version two completely different content. " * 40,
            sections=[],
        )

        chunks_v2 = chunker.chunk(doc_v2)
        embedded_v2 = await embedding_service.embed_chunks(chunks_v2)
        await storage_writer.store(embedded_v2, doc_v2)

        # Old chunks should be stale, new chunks indexed
        stale = await db_pool.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = $1 AND status = 'stale'",
            doc_v1.doc_id,
        )
        indexed = await db_pool.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = $1 AND status = 'indexed'",
            doc_v1.doc_id,
        )
        assert stale == v1_count, "Old chunks should be marked stale"
        assert indexed == len(chunks_v2), "New chunks should be indexed"
