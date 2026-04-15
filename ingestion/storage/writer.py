"""
Stage 7: Storage writer.
Atomically writes embedded chunks to Qdrant (vectors) and Postgres (metadata).

Design:
  1. Write chunk metadata to Postgres with status='pending_vector'
  2. Upsert vector to Qdrant (idempotent by content-hash ID)
  3. Update Postgres status to 'indexed'

If step 2 or 3 fails, the row stays at 'pending_vector'.
The reconciliation job (reconciliation.py) picks these up and re-queues.

On document update (is_update=True): delete all old chunks for the
document from Qdrant BEFORE inserting new ones to prevent stale results.
"""

from __future__ import annotations

import logging
from uuid import UUID

import asyncpg
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    PointStruct,
    UpdateResult,
)

from ingestion.embedding.sparse import get_encoder
from shared.config import get_settings
from shared.models import Chunk, ParsedDocument
from shared.telemetry import traced_span

logger = logging.getLogger(__name__)
settings = get_settings()


class StorageWriter:
    """
    Writes embedded chunks to Qdrant + Postgres.
    One instance per worker process — reuse the client connections.
    """

    def __init__(self, db_pool: asyncpg.Pool, qdrant: AsyncQdrantClient):
        self._db     = db_pool
        self._qdrant = qdrant

    async def ensure_collection(self) -> None:
        """
        Delegates to collection_setup.py which handles named vectors
        (dense + sparse) and all HNSW/quantization config.
        Safe to call on every startup — idempotent.
        """
        from infrastructure.qdrant.collection_setup import setup_collection
        await setup_collection(self._qdrant)

    async def store(self, chunks: list[Chunk], document: ParsedDocument) -> None:
        """
        Full atomic store operation for one document's chunks.
        If document is an update, old chunks are deleted first.
        """
        if not chunks:
            logger.warning("store() called with empty chunk list for doc_id=%s", document.doc_id)
            return

        with traced_span("storage.store", {"doc_id": str(document.doc_id), "chunk_count": len(chunks)}):
            # On update: delete old chunks from Qdrant before inserting new ones
            await self._delete_old_chunks_if_update(document)

            # Step 1: Write all chunk metadata to Postgres (pending_vector status)
            await self._write_postgres_batch(chunks)

            # Step 2: Upsert vectors to Qdrant
            await self._upsert_qdrant_batch(chunks)

            # Step 3: Mark all chunks as indexed in Postgres
            await self._mark_indexed(chunks, document.doc_id)

            logger.info(
                "Stored %d chunks for doc_id=%s",
                len(chunks), document.doc_id,
            )

    # ─────────────────────────────────────────────────────────
    #  Delete old chunks on document update
    # ─────────────────────────────────────────────────────────

    async def _delete_old_chunks_if_update(self, document: ParsedDocument) -> None:
        """
        If this document was previously indexed, delete all its old chunks
        from both Qdrant and Postgres before writing the new version.

        Uses a Postgres advisory lock to prevent race conditions if two
        workers try to update the same document simultaneously.
        """
        # Check if this doc has existing indexed chunks
        row = await self._db.fetchrow(
            "SELECT COUNT(*) AS n FROM chunks WHERE doc_id = $1 AND status = 'indexed'",
            document.doc_id,
        )
        if row["n"] == 0:
            return

        logger.info("Deleting %d old chunks for updated doc_id=%s", row["n"], document.doc_id)

        # Fetch old chunk IDs (needed for Qdrant delete)
        old_ids = await self._db.fetch(
            "SELECT id FROM chunks WHERE doc_id = $1",
            document.doc_id,
        )
        qdrant_ids = [str(r["id"]) for r in old_ids]

        # Delete from Qdrant first
        if qdrant_ids:
            await self._qdrant.delete(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points_selector=qdrant_ids,
            )

        # Mark old Postgres chunks as stale (don't DELETE — keep for audit trail)
        await self._db.execute(
            "UPDATE chunks SET status = 'stale' WHERE doc_id = $1",
            document.doc_id,
        )

    # ─────────────────────────────────────────────────────────
    #  Postgres writes
    # ─────────────────────────────────────────────────────────

    async def _write_postgres_batch(self, chunks: list[Chunk]) -> None:
        """
        Insert all chunk metadata rows in a single transaction.
        Status is 'pending_vector' — reconciliation will re-queue if stuck here.
        """
        records = [
            (
                chunk.id,
                chunk.doc_id,
                chunk.chunk_index,
                chunk.content_hash,
                chunk.char_count,
                chunk.section_title,
                chunk.page_number,
                chunk.source_url,
                str(chunk.parent_chunk_id) if chunk.parent_chunk_id else None,
                chunk.hierarchy_level,
                chunk.chunking_strategy,
                chunk.chunk_overlap_tokens,
                chunk.embedding_model,
                chunk.embedding_model_version or "unknown",
                chunk.embedding_dim or 0,
                chunk.schema_version,
                chunk.text,
            )
            for chunk in chunks
        ]

        await self._db.executemany(
            """
            INSERT INTO chunks (
                id, doc_id, chunk_index, content_hash, char_count,
                section_title, page_number, source_url, parent_chunk_id,
                hierarchy_level, chunking_strategy, chunk_overlap_tokens,
                embedding_model, embedding_model_version, embedding_dim,
                schema_version, raw_text, status
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9,
                $10, $11, $12,
                $13, $14, $15,
                $16, $17, 'pending_vector'
            )
            ON CONFLICT (id) DO UPDATE
                SET status = 'pending_vector',
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_model_version = EXCLUDED.embedding_model_version
            """,
            records,
        )

    async def _mark_indexed(self, chunks: list[Chunk], doc_id: UUID) -> None:
        """Update chunk status to 'indexed' and set ingested_at timestamp."""
        chunk_ids = [chunk.id for chunk in chunks]
        await self._db.execute(
            """
            UPDATE chunks
            SET status = 'indexed', ingested_at = NOW()
            WHERE id = ANY($1::uuid[])
            """,
            chunk_ids,
        )

        # Update document-level stats
        await self._db.execute(
            """
            UPDATE documents
            SET
                status         = 'indexed',
                total_chunks   = $2,
                ingested_at    = NOW(),
                updated_at     = NOW()
            WHERE id = $1
            """,
            doc_id,
            len(chunks),
        )

    # ─────────────────────────────────────────────────────────
    #  Qdrant writes
    # ─────────────────────────────────────────────────────────

    async def _upsert_qdrant_batch(self, chunks: list[Chunk]) -> None:
        """
        Upsert all chunk vectors to Qdrant in one operation.
        Each point carries BOTH dense and sparse vectors for hybrid search.
        Qdrant upsert is idempotent — safe to retry.
        """
        bm25 = get_encoder()
        points = []
        for chunk in chunks:
            if chunk.vector is None:
                logger.warning("Chunk %s has no vector — skipping Qdrant upsert", chunk.id)
                continue

            # Compute BM25 sparse vector for this chunk
            sparse_vec = bm25.encode_document(chunk.text)

            points.append(
                PointStruct(
                    id=str(chunk.id),
                    vector={
                        "dense":  chunk.vector,
                        "sparse": sparse_vec.to_qdrant_dict(),
                    },
                    payload=chunk.to_qdrant_payload(),
                )
            )

        if not points:
            return

        # Batch in groups of 100 for Qdrant API efficiency
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            result: UpdateResult = await self._qdrant.upsert(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points=batch,
                wait=True,     # wait for indexing to complete before returning
            )
            if result.status.value != "completed":
                raise RuntimeError(f"Qdrant upsert returned status: {result.status}")
