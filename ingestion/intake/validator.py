"""
Stage 2: Intake validator.
Checks every incoming document before it touches the queue:
  - Content-hash deduplication against Postgres
  - File size limits
  - MIME type detection and mapping to DocType
  - Poison pill / basic sanity checks

Returns a validated DocumentIngestionMessage or raises ValidationError.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from uuid import uuid4
from typing import Literal

import magic                   # python-magic: libmagic bindings
import asyncpg

from shared.config import get_settings
from shared.models import DocType, DocumentIngestionMessage

logger = logging.getLogger(__name__)
settings = get_settings()

# MIME type → DocType mapping
_MIME_TO_DOCTYPE: dict[str, DocType] = {
    "application/pdf":                                          DocType.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocType.DOCX,
    "application/msword":                                       DocType.DOCX,
    "text/html":                                                DocType.HTML,
    "text/markdown":                                            DocType.MARKDOWN,
    "text/x-markdown":                                          DocType.MARKDOWN,
    "text/plain":                                               DocType.TXT,
    "text/x-python":                                            DocType.CODE,
    "text/javascript":                                          DocType.CODE,
    "application/json":                                         DocType.CODE,
}


class ValidationError(Exception):
    """
    Raised when a document fails validation. Contains a reason string.
    """
    def __init__(self, reason: str, doc_url: str):
        self.reason = reason
        self.doc_url = doc_url
        super().__init__(f"Validation failed for {doc_url}: {reason}")


class DuplicateDocumentError(Exception):
    """Raised when an identical document (same hash) is already indexed."""
    def __init__(self, source_url: str, existing_doc_id: str):
        self.source_url = source_url
        self.existing_doc_id = existing_doc_id
        super().__init__(f"Duplicate: {source_url} already indexed as {existing_doc_id}")


class IntakeValidator:
    """
    Validates incoming documents and produces DocumentIngestionMessage objects.

    Usage:
        validator = IntakeValidator(db_pool)
        message = await validator.validate(
            source_url="s3://my-bucket/report.pdf",
            source_type="s3",
            file_bytes=raw_bytes,
        )
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self._db = db_pool
        self._max_bytes = settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024

    async def validate(
        self,
        source_url: str,
        source_type: str,
        file_bytes: bytes,
    ) -> DocumentIngestionMessage:
        """
        Full validation pipeline. Returns a ready-to-enqueue message.
        Raises ValidationError or DuplicateDocumentError on failure.
        """
        # 1. Size check (fast — before any hashing)
        byte_size = len(file_bytes)
        if byte_size == 0:
            raise ValidationError("Empty file", source_url)
        if byte_size > self._max_bytes:
            raise ValidationError(
                f"File too large: {byte_size / 1e6:.1f}MB > {settings.MAX_DOCUMENT_SIZE_MB}MB",
                source_url,
            )

        # 2. MIME detection (uses libmagic — more reliable than file extension)
        mime_type = magic.from_buffer(file_bytes[:2048], mime=True)
        doc_type = self._resolve_doc_type(source_url, mime_type)

        # 3. Content hash
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        # 4. Deduplication check
        existing = await self._check_duplicate(source_url, content_hash)

        if existing == "identical":
            # Same URL, same content — nothing to do
            raise DuplicateDocumentError(source_url, "unknown")

        is_update = existing == "updated"  # Same URL, new content

        # 5. Register document in Postgres (or mark as updating)
        doc_id = await self._register_document(
            source_url=source_url,
            source_type=source_type,
            doc_type=doc_type,
            content_hash=content_hash,
            byte_size=byte_size,
            is_update=is_update,
        )

        logger.info(
            "Document validated",
            extra={
                "doc_id": str(doc_id),
                "source_url": source_url,
                "doc_type": doc_type,
                "byte_size": byte_size,
                "is_update": is_update,
            },
        )

        return DocumentIngestionMessage(
            doc_id=doc_id,
            source_url=source_url,
            source_type=source_type,
            doc_type=doc_type,
            byte_size=byte_size,
            content_hash=content_hash,
            is_update=is_update,
        )

    def _resolve_doc_type(self, source_url: str, mime_type: str) -> DocType:
        """
        Map MIME type to DocType, falling back to extension-based guess.
        """
        if mime_type in _MIME_TO_DOCTYPE:
            return _MIME_TO_DOCTYPE[mime_type]

        # Extension fallback
        suffix = Path(source_url).suffix.lower()
        extension_map = {
            ".pdf":  DocType.PDF,
            ".docx": DocType.DOCX,
            ".doc":  DocType.DOCX,
            ".html": DocType.HTML,
            ".htm":  DocType.HTML,
            ".md":   DocType.MARKDOWN,
            ".txt":  DocType.TXT,
            ".py":   DocType.CODE,
            ".js":   DocType.CODE,
            ".ts":   DocType.CODE,
        }
        doc_type = extension_map.get(suffix, DocType.UNKNOWN)
        if doc_type == DocType.UNKNOWN:
            logger.warning("Unknown doc type for %s (MIME: %s)", source_url, mime_type)
        return doc_type

    async def _check_duplicate(
        self,
        source_url: str,
        content_hash: str,
    ) -> Literal["new", "identical", "updated"]:
        """
        Check Postgres for existing document.
        Returns:
            'new'       — never seen this URL or hash
            'identical' — same URL, same hash (skip)
            'updated'   — same URL, different hash (re-ingest)
        """
        row = await self._db.fetchrow(
            "SELECT id, content_hash, status FROM documents WHERE source_url = $1",
            source_url,
        )
        if row is None:
            return "new"
        if row["content_hash"] == content_hash:
            logger.info("Duplicate document skipped: %s", source_url)
            return "identical"
        logger.info("Document updated, will re-ingest: %s", source_url)
        return "updated"

    async def _register_document(
        self,
        source_url: str,
        source_type: str,
        doc_type: DocType,
        content_hash: str,
        byte_size: int,
        is_update: bool,
    ):
        """
        Insert or update document row in Postgres.
        Returns the doc_id UUID.
        """
        if is_update:
            # Fetch existing id, store previous hash, bump version
            row = await self._db.fetchrow(
                """
                UPDATE documents
                SET
                    content_hash   = $1,
                    status         = 'pending',
                    previous_hash  = content_hash,
                    version        = version + 1,
                    retry_count    = 0,
                    last_error     = NULL,
                    updated_at     = NOW()
                WHERE source_url = $2
                RETURNING id
                """,
                content_hash,
                source_url,
            )
            return row["id"]
        else:
            row = await self._db.fetchrow(
                """
                INSERT INTO documents (source_url, source_type, doc_type, content_hash, byte_size)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
                """,
                source_url,
                source_type,
                doc_type.value,
                content_hash,
                byte_size,
            )
            return row["id"]