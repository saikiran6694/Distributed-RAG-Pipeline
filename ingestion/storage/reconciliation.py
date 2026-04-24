"""
Background reconciliation job.
Runs every N minutes and re-queues any chunks stuck in 'pending_vector'
status — meaning Postgres write succeeded but Qdrant write failed.

This is the safety net for two-store atomicity.
"""

from __future__ import annotations

import asyncio
import logging

import asyncpg
from confluent_kafka import Producer

from shared.config import get_settings
from shared.models import DocType, DocumentIngestionMessage

logger = logging.getLogger(__name__)
settings = get_settings()


class ReconciliationJob:
    """
    Periodically checks for chunks stuck at 'pending_vector' and re-queues
    their parent documents for re-ingestion.

    Runs as a background asyncio task alongside the main worker process.
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self._db = db_pool
        self._producer = Producer({"bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS})

    async def run_forever(self) -> None:
        """Main loop — runs until cancelled."""
        logger.info("Reconciliation job started (interval=%ds)", settings.RECONCILIATION_INTERVAL_SECONDS)
        while True:
            try:
                await self._run_once()
            except Exception as e:
                logger.error("Reconciliation job error: %s", e)
            await asyncio.sleep(settings.RECONCILIATION_INTERVAL_SECONDS)

    async def _run_once(self) -> None:
        """Find and re-queue all stale pending_vector chunks."""
        stale_docs = await self._db.fetch(
            """
            SELECT DISTINCT d.id, d.source_url, d.source_type, d.doc_type, d.byte_size, d.content_hash
            FROM chunks c
            JOIN documents d ON c.doc_id = d.id
            WHERE c.status = 'pending_vector'
              AND c.created_at < NOW() - ($1 || ' minutes')::interval
            LIMIT 50
            """,
            str(settings.RECONCILIATION_STALE_MINUTES),
        )

        if not stale_docs:
            logger.debug("Reconciliation: no stale chunks found")
            return

        logger.warning("Reconciliation: found %d documents with stale chunks — re-queuing", len(stale_docs))

        for row in stale_docs:
            msg = DocumentIngestionMessage(
                doc_id=row["id"],
                source_url=row["source_url"],
                source_type=row["source_type"],
                doc_type=DocType(row["doc_type"]),
                byte_size=row["byte_size"] or 0,
                content_hash=row["content_hash"],
                is_update=True,
            )
            self._producer.produce(
                topic=settings.KAFKA_TOPIC_RAW_DOCUMENTS,
                key=msg.source_type.encode(),
                value=msg.model_dump_json().encode(),
            )

        self._producer.flush(timeout=5)
        logger.info("Reconciliation: re-queued %d documents", len(stale_docs))