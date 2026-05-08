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
from qdrant_client.http.models import PointStruct, UpdateResult

from ingestion.auto_tagger import AutoTagger
from ingestion.embedding.sparse import get_encoder
from shared.config import get_settings
from shared.models import Chunk, ParsedDocument
from shared.telemetry import traced_span

logger = logging.getLogger(__name__)
settings = get_settings()


class StorageWriter:
    def __init__(self, db_pool: asyncpg.Pool, qdrant: AsyncQdrantClient):
        self._db = db_pool
        self._qdrant = qdrant
        self._tagger = AutoTagger()

    async def ensure_collection(self) -> None:
        from infrastructure.qdrant.collection_setup import setup_collection

        await setup_collection(self._qdrant)

    async def store(self, chunks: list[Chunk], document: ParsedDocument) -> None:
        if not chunks:
            logger.warning("store() called with empty chunk list for doc_id=%s", document.doc_id)
            return

        with traced_span(
            "storage.store", {"doc_id": str(document.doc_id), "chunk_count": len(chunks)}
        ):
            await self._delete_old_chunks_if_update(document)
            await self._write_postgres_batch(chunks)
            await self._upsert_qdrant_batch(chunks)
            await self._mark_indexed(chunks, document.doc_id)
            logger.info("Stored %d chunks for doc_id=%s", len(chunks), document.doc_id)

        # ── Auto-tagging (after store succeeds, never blocks/fails the store) ──
        try:
            tags = await self._tagger.tag(
                raw_text=document.raw_text,
                existing_title=getattr(document, "title", None),
            )

            logger.debug("AutoTagger returned tags for doc_id=%s: %s", document.doc_id, tags)

            if tags:
                import json

                await self._db.execute(
                    "UPDATE documents SET tags=$1 WHERE id=$2",
                    json.dumps(tags.to_dict()),
                    document.doc_id,
                )
                logger.debug(
                    "Auto-tagged doc_id=%s category=%s topics=%s",
                    document.doc_id,
                    tags.doc_category,
                    tags.topics[:3],
                )
        except Exception as e:
            logger.warning("Auto-tagging failed for %s (non-critical): %s", document.doc_id, e)

    async def _delete_old_chunks_if_update(self, document: ParsedDocument) -> None:
        row = await self._db.fetchrow(
            "SELECT COUNT(*) AS n FROM chunks WHERE doc_id=$1 AND status='indexed'",
            document.doc_id,
        )
        if row["n"] == 0:
            return
        logger.info("Deleting %d old chunks for updated doc_id=%s", row["n"], document.doc_id)
        old_ids = await self._db.fetch("SELECT id FROM chunks WHERE doc_id=$1", document.doc_id)
        qdrant_ids = [str(r["id"]) for r in old_ids]
        if qdrant_ids:
            await self._qdrant.delete(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points_selector=qdrant_ids,
            )
        await self._db.execute("UPDATE chunks SET status='stale' WHERE doc_id=$1", document.doc_id)

    async def _write_postgres_batch(self, chunks: list[Chunk]) -> None:
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
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,
                'pending_vector'
            )
            ON CONFLICT (id) DO UPDATE
                SET status='pending_vector',
                    embedding_model=EXCLUDED.embedding_model,
                    embedding_model_version=EXCLUDED.embedding_model_version
            """,
            records,
        )

    async def _mark_indexed(self, chunks: list[Chunk], doc_id: UUID) -> None:
        await self._db.execute(
            "UPDATE chunks SET status='indexed', ingested_at=NOW() WHERE id=ANY($1::uuid[])",
            [chunk.id for chunk in chunks],
        )
        await self._db.execute(
            """
            UPDATE documents
            SET status='indexed', total_chunks=$2, ingested_at=NOW(), updated_at=NOW()
            WHERE id=$1
            """,
            doc_id,
            len(chunks),
        )

    async def _upsert_qdrant_batch(self, chunks: list[Chunk]) -> None:
        bm25 = get_encoder()
        points = []
        for chunk in chunks:
            if chunk.vector is None:
                logger.warning("Chunk %s has no vector — skipping", chunk.id)
                continue
            sparse_vec = bm25.encode_document(chunk.text)
            points.append(
                PointStruct(
                    id=str(chunk.id),
                    vector={
                        "dense": chunk.vector,
                        "sparse": sparse_vec.to_qdrant_dict(),
                    },
                    payload=chunk.to_qdrant_payload(),
                )
            )
        if not points:
            return
        for i in range(0, len(points), 100):
            result: UpdateResult = await self._qdrant.upsert(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points=points[i : i + 100],
                wait=True,
            )
            if result.status.value != "completed":
                raise RuntimeError(f"Qdrant upsert returned status: {result.status}")
