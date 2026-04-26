"""
api/ingest.py

Ingest router — accepts documents via:
  POST /ingest/file              — multipart file upload
  POST /ingest/url               — HTTP URL or S3 path
  POST /ingest/text              — raw text with metadata
  GET  /ingest/list              — list all ingested documents
  GET  /ingest/{doc_id}/status   — poll processing status
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import asyncpg
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient

from ingestion.chunking.selector import ChunkingStrategySelector
from ingestion.intake.validator import (
    DuplicateDocumentError,
    IntakeValidator,
    ValidationError,
)
from shared.config import get_settings
from shared.models import DocType, DocumentIngestionMessage

logger   = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# ── Singletons (set by api/main.py lifespan) ──────────────────────

_db_pool:   asyncpg.Pool      | None = None
_qdrant:    AsyncQdrantClient | None = None
_embed_svc                           = None   # EmbeddingService
_writer                              = None   # StorageWriter


def set_ingest_dependencies(db_pool, qdrant, embed_svc):
    global _db_pool, _qdrant, _embed_svc, _writer
    from ingestion.storage.writer import StorageWriter
    _db_pool   = db_pool
    _qdrant    = qdrant
    _embed_svc = embed_svc
    _writer    = StorageWriter(db_pool=db_pool, qdrant=qdrant)


# ── Request / Response schemas ────────────────────────────────────

class IngestResponse(BaseModel):
    doc_id:     str
    status:     str
    source_url: str
    chunks:     int = 0
    message:    str = ""


class StatusResponse(BaseModel):
    doc_id:        str
    status:        str
    source_url:    str
    total_chunks:  int
    failed_chunks: int
    ingested_at:   str | None
    last_error:    str | None


class URLIngestRequest(BaseModel):
    url:    str  = Field(..., description="HTTP URL or s3:// path")
    direct: bool = Field(True, description="Process inline (True) or queue to Kafka (False)")


class TextIngestRequest(BaseModel):
    text:       str  = Field(..., min_length=10)
    title:      str  = Field(default="Untitled")
    source_url: str  = Field(default="")
    doc_type:   str  = Field(default="txt")
    direct:     bool = True


# ── Endpoints — NOTE: /list must come before /{doc_id}/status ─────
# FastAPI matches routes top-to-bottom. If /{doc_id}/status came first,
# GET /ingest/list would be caught by it with doc_id="list", fail UUID
# parsing, and return 422 instead of the document list.

@router.get("/list", summary="List all ingested documents")
async def list_documents(limit: int = 20, offset: int = 0):
    """List all documents with their current processing status."""
    _require_deps()
    rows = await _db_pool.fetch(
        """
        SELECT id, source_url, doc_type, status, total_chunks, ingested_at
        FROM documents
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
        """,
        limit, offset,
    )
    total = await _db_pool.fetchval("SELECT COUNT(*) FROM documents")
    return {
        "total": total,
        "documents": [
            {
                "doc_id":       str(r["id"]),
                "source_url":   r["source_url"],
                "doc_type":     r["doc_type"],
                "status":       r["status"],
                "total_chunks": r["total_chunks"] or 0,
                "ingested_at":  r["ingested_at"].isoformat() if r["ingested_at"] else None,
            }
            for r in rows
        ],
    }


@router.get("/{doc_id}/status", response_model=StatusResponse)
async def ingest_status(doc_id: str):
    """Poll the processing status of a document by its doc_id."""
    _require_deps()
    try:
        uid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid doc_id — must be a UUID")

    row = await _db_pool.fetchrow(
        """
        SELECT id, source_url, status, total_chunks, failed_chunks,
               ingested_at, last_error
        FROM documents WHERE id = $1
        """,
        uid,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    return StatusResponse(
        doc_id=str(row["id"]),
        status=row["status"],
        source_url=row["source_url"],
        total_chunks=row["total_chunks"] or 0,
        failed_chunks=row["failed_chunks"] or 0,
        ingested_at=row["ingested_at"].isoformat() if row["ingested_at"] else None,
        last_error=row["last_error"],
    )


@router.post("/file", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_file(
    file:   UploadFile = File(...),
    direct: bool       = Form(True),
):
    """
    Upload a file for ingestion.
    Supported formats: .pdf  .html  .htm  .md  .txt  .docx

    direct=true  (default) — process synchronously, return when indexed.
    direct=false           — save file to disk, publish to Kafka, workers fetch by path.
    """
    _require_deps()
    file_bytes = await file.read()

    if not direct:
        # Workers fetch the file from disk — save to a persistent upload dir
        import tempfile, os
        upload_dir = Path(settings.UPLOAD_DIR) if hasattr(settings, "UPLOAD_DIR")                      else Path(tempfile.gettempdir()) / "rag_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name  = file.filename.replace(" ", "_")
        saved_path = upload_dir / safe_name
        saved_path.write_bytes(file_bytes)
        source_url = f"file://{saved_path.resolve()}"
    else:
        source_url = f"file://uploads/{file.filename}"

    return await _ingest_bytes(
        file_bytes=file_bytes,
        source_url=source_url,
        source_type="upload",
        direct=direct,
    )


@router.post("/url", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_url(body: URLIngestRequest):
    """
    Fetch and ingest a document from a URL or S3 path.

    Examples:
      {"url": "https://docs.example.com/page.html"}
      {"url": "s3://my-bucket/reports/q3.pdf"}
    """
    _require_deps()
    try:
        file_bytes = await _fetch_url(body.url)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to fetch URL: {e}")

    return await _ingest_bytes(
        file_bytes=file_bytes,
        source_url=body.url,
        source_type="http" if body.url.startswith("http") else "s3",
        direct=body.direct,
    )


@router.post("/text", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_text(body: TextIngestRequest):
    """
    Ingest raw text directly — useful for testing and programmatic use.

    The doc_type field controls chunking strategy:
      txt / code    → fixed-size chunker
      markdown      → semantic chunker
      html          → semantic chunker (trafilatura extraction applied)
    """
    _require_deps()
    file_bytes = body.text.encode("utf-8")
    source_url = body.source_url or f"text://inline/{uuid.uuid4()}"
    return await _ingest_bytes(
        file_bytes=file_bytes,
        source_url=source_url,
        source_type="text",
        direct=body.direct,
        title=body.title,
        forced_doc_type=body.doc_type,
    )


# ── Core ingestion logic ──────────────────────────────────────────

async def _ingest_bytes(
    file_bytes:      bytes,
    source_url:      str,
    source_type:     str,
    direct:          bool,
    title:           str | None = None,
    forced_doc_type: str | None = None,
) -> IngestResponse:
    """
    Validate → register → (process inline | queue to Kafka).
    """
    validator = IntakeValidator(db_pool=_db_pool)
    try:
        message = await validator.validate(
            source_url=source_url,
            source_type=source_type,
            file_bytes=file_bytes,
        )
    except DuplicateDocumentError as e:
        # Return the real existing doc_id so caller can look up its status
        existing_row = await _db_pool.fetchrow(
            "SELECT id FROM documents WHERE source_url = $1", source_url
        )
        existing_id = str(existing_row["id"]) if existing_row else str(uuid.uuid4())
        return IngestResponse(
            doc_id=existing_id,
            status="duplicate",
            source_url=source_url,
            message="Document already indexed with identical content",
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.reason)

    # Override doc_type if caller specified one (e.g. /ingest/text with doc_type=markdown)
    if forced_doc_type:
        try:
            message.doc_type = DocType(forced_doc_type)
        except ValueError:
            pass   # unknown type — keep what MIME detection set

    if not direct:
        from ingestion.intake.producer import IngestionProducer
        producer = IngestionProducer()
        producer.publish(message)
        producer.close()
        return IngestResponse(
            doc_id=str(message.doc_id),
            status="queued",
            source_url=source_url,
            message="Queued for processing. Poll GET /ingest/{doc_id}/status",
        )

    # Direct mode — process inline
    try:
        embedded = await _process_inline(file_bytes, message, title)
        return IngestResponse(
            doc_id=str(message.doc_id),
            status="indexed",
            source_url=source_url,
            chunks=len(embedded),
            message=f"Indexed {len(embedded)} chunks",
        )
    except Exception as e:
        logger.error("Inline ingestion failed for %s: %s", source_url, e)
        await _db_pool.execute(
            "UPDATE documents SET status='failed', last_error=$1 WHERE id=$2",
            str(e)[:500], message.doc_id,
        )
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


async def _process_inline(
    file_bytes: bytes,
    message:    DocumentIngestionMessage,
    title:      str | None,
) -> list:
    """Parse → chunk → embed → store. Returns the embedded chunks."""
    parsed  = _parse_bytes(file_bytes, message, title)
    chunker = ChunkingStrategySelector().get(message.doc_type)
    chunks  = chunker.chunk(parsed)

    if not chunks:
        raise ValueError("No chunks produced — document may be empty or unparseable")

    embedded = await _embed_svc.embed_chunks(chunks)
    await _writer.store(embedded, parsed)
    return embedded


def _parse_bytes(
    file_bytes: bytes,
    message:    DocumentIngestionMessage,
    title:      str | None,
):
    """
    Parse raw bytes → ParsedDocument based on doc_type.

    Each branch returns a ParsedDocument. The function never falls through
    without returning or raising — explicit handling for every DocType.
    """
    from shared.models import ParsedDocument, DocType as DT
    import chardet

    doc_type = message.doc_type
    # Normalise enum value
    dt = doc_type.value if hasattr(doc_type, "value") else str(doc_type)

    # ── Plain text formats ─────────────────────────────────────
    if dt in ("txt", "code", "unknown", "markdown"):
        enc  = chardet.detect(file_bytes[:8192]).get("encoding") or "utf-8"
        text = file_bytes.decode(enc, errors="replace")
        if len(text.strip()) < 10:
            raise ValueError("Document is empty or too short")
        return ParsedDocument(
            doc_id=message.doc_id, source_url=message.source_url,
            doc_type=message.doc_type, raw_text=text, title=title,
        )

    # ── HTML ──────────────────────────────────────────────────
    if dt == "html":
        import trafilatura
        enc      = chardet.detect(file_bytes[:8192]).get("encoding") or "utf-8"
        html_str = file_bytes.decode(enc, errors="replace")
        text     = trafilatura.extract(html_str, favor_precision=True)
        if not text or len(text.strip()) < 10:
            # Fall back to raw HTML truncated
            text = html_str[:5000]
        return ParsedDocument(
            doc_id=message.doc_id, source_url=message.source_url,
            doc_type=message.doc_type, raw_text=text, title=title,
        )

    # ── PDF ───────────────────────────────────────────────────
    if dt == "pdf":
        # Use a standalone parse function — avoids __new__ footgun
        parsed = _parse_pdf(file_bytes, message)
        if title:
            parsed.title = title
        return parsed

    # ── DOCX ──────────────────────────────────────────────────
    if dt == "docx":
        import docx2txt, io
        text = docx2txt.process(io.BytesIO(file_bytes))
        if not text or len(text.strip()) < 10:
            raise ValueError("DOCX appears to be empty")
        return ParsedDocument(
            doc_id=message.doc_id, source_url=message.source_url,
            doc_type=message.doc_type, raw_text=text, title=title,
        )

    # ── Fallback (should never reach here) ────────────────────
    raise ValueError(f"Unsupported doc_type for inline parsing: {dt}")


def _parse_pdf(file_bytes: bytes, message: DocumentIngestionMessage):
    """
    Standalone PDF parser — extracts text using Unstructured.io.
    Does not require a PDFWorker instance.
    """
    import tempfile
    import chardet
    from shared.models import ParsedDocument, Section

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        try:
            from unstructured.partition.pdf import partition_pdf
            from unstructured.documents.elements import (
                Title, Table, Header, Footer, PageBreak,
            )
            elements = partition_pdf(
                filename=tmp.name,
                strategy="hi_res",
                infer_table_structure=True,
                include_page_breaks=True,
                languages=["eng"],
            )
        except Exception as e:
            raise ValueError(f"PDF parsing failed: {e}") from e

    text_parts, sections, current_title, current_content = [], [], None, []
    current_page = 1

    for el in elements:
        if isinstance(el, PageBreak):
            current_page += 1
            continue
        if isinstance(el, (Header, Footer)):
            continue

        text = (el.text or "").strip()
        if not text:
            continue

        if isinstance(el, Table):
            text_parts.append(text)
            continue

        if isinstance(el, Title):
            if current_title and current_content:
                sections.append(Section(
                    title=current_title,
                    content="\n".join(current_content),
                    start_page=current_page,
                ))
            current_title   = text
            current_content = []

        text_parts.append(text)
        current_content.append(text)

    if current_title and current_content:
        sections.append(Section(title=current_title, content="\n".join(current_content)))

    raw_text = "\n\n".join(text_parts)
    if len(raw_text.strip()) < 10:
        raise ValueError("PDF extraction produced no usable text")

    return ParsedDocument(
        doc_id=message.doc_id, source_url=message.source_url,
        doc_type=message.doc_type, raw_text=raw_text,
        sections=sections, page_count=current_page,
    )


async def _fetch_url(url: str) -> bytes:
    if url.startswith("s3://"):
        import boto3
        parts = url.replace("s3://", "").split("/", 1)
        obj   = boto3.client("s3").get_object(Bucket=parts[0], Key=parts[1])
        return obj["Body"].read()
    import httpx
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.get(url, headers={"User-Agent": "RAG-Ingest/1.0"})
        r.raise_for_status()
        return r.content


def _require_deps():
    if not _db_pool or not _qdrant or not _embed_svc or not _writer:
        raise HTTPException(status_code=503, detail="Service not initialised")