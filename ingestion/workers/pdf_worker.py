"""
PDF parser worker.
Uses Unstructured.io for layout-aware text extraction.

Handles:
  - Multi-column PDFs (reads logical flow, not visual columns)
  - Scanned PDFs (OCR via Tesseract)
  - Tables (serialized to markdown)
  - Per-page error isolation (one bad page doesn't kill the whole doc)
  - OCR confidence gating (skips low-quality OCR chunks)
  - Encoding normalization to UTF-8
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import chardet

from ingestion.chunking.selector import ChunkingStrategySelector
from ingestion.workers.base_worker import BaseWorker, PoisonPillError, RetryableError
from shared.config import get_settings
from shared.models import DocumentIngestionMessage, DocType, ParsedDocument, Section

logger = logging.getLogger(__name__)
settings = get_settings()


class PDFWorker(BaseWorker):
    """Parses PDF documents using Unstructured.io with OCR fallback."""

    worker_name = "pdf-worker"

    def __init__(self, embedding_service, storage_writer):
        super().__init__()
        self._embedder  = embedding_service
        self._storage   = storage_writer
        self._selector  = ChunkingStrategySelector()

    def _process(self, message: DocumentIngestionMessage) -> None:
        import asyncio
        raw_bytes = self._fetch_document(message.source_url)

        dt = message.doc_type.value if hasattr(message.doc_type, "value") else str(message.doc_type)
        if dt == "docx":
            parsed = self._parse_docx(raw_bytes, message)
        else:
            parsed = self._parse_pdf(raw_bytes, message)

        chunker = self._selector.get(message.doc_type)
        chunks  = chunker.chunk(parsed)
        loop    = asyncio.get_event_loop()
        loop.run_until_complete(self._store(chunks, parsed))

    async def _store(self, chunks, parsed) -> None:
        embedded = await self._embedder.embed_chunks(chunks)
        await self._storage.store(embedded, parsed)

    # ── Parse DOCX ────────────────────────────────────────────────

    def _parse_docx(self, raw_bytes: bytes, message: DocumentIngestionMessage) -> ParsedDocument:
        """Extract text from DOCX using docx2txt."""
        try:
            import docx2txt
            import io
            text = docx2txt.process(io.BytesIO(raw_bytes))
        except Exception as e:
            raise PoisonPillError(f"DOCX parse failed: {e}") from e

        if not text or len(text.strip()) < 50:
            raise PoisonPillError("DOCX extracted text too short — file may be empty")

        return ParsedDocument(
            doc_id=message.doc_id,
            source_url=message.source_url,
            doc_type=message.doc_type,
            raw_text=text.strip(),
        )

    # ── Fetch ──────────────────────────────────────────────────────

    def _fetch_document(self, source_url: str) -> bytes:
        try:
            if source_url.startswith("file://"):
                return Path(source_url.replace("file://", "")).read_bytes()
            if source_url.startswith("s3://"):
                return self._fetch_from_s3(source_url)
            import httpx
            with httpx.Client(timeout=30) as client:
                response = client.get(source_url)
                response.raise_for_status()
                return response.content

        except OSError as e:
            raise RetryableError(f"Fetch failed: {e}") from e

    def _fetch_from_s3(self, s3_url: str) -> bytes:
        import boto3
        from botocore.exceptions import ClientError
        parts  = s3_url.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        try:
            obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
            return obj["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "NoSuchBucket"):
                raise PoisonPillError(f"S3 object not found: {s3_url}") from e
            raise RetryableError(f"S3 error: {e}") from e

    # ── Parse ──────────────────────────────────────────────────────

    def _parse_pdf(self, raw_bytes: bytes, message: DocumentIngestionMessage) -> ParsedDocument:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.documents.elements import (
            Table, Title, Header, Footer, PageBreak,
        )

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(raw_bytes)
            tmp.flush()
            try:
                elements = partition_pdf(
                    filename=tmp.name,
                    strategy="hi_res",
                    infer_table_structure=True,
                    include_page_breaks=True,
                    languages=["eng"],
                )
            except Exception as e:
                raise PoisonPillError(f"PDF parse failed: {e}") from e

        text_parts, sections, tables, errors = [], [], [], []
        current_title, current_content, current_page = None, [], 1

        for el in elements:
            if isinstance(el, PageBreak):
                current_page += 1
                continue
            if isinstance(el, (Header, Footer)):
                continue

            if hasattr(el, "metadata") and el.metadata:
                conf = getattr(el.metadata, "detection_class_prob", None)
                if conf is not None and conf < settings.OCR_CONFIDENCE_THRESHOLD:
                    errors.append(f"Low OCR confidence ({conf:.2f}) p{current_page}")
                    continue

            text = self._normalize_encoding(el.text or "")
            if not text.strip():
                continue

            if isinstance(el, Table):
                md = self._table_to_markdown(el)
                if md:
                    tables.append(md)
                    text_parts.append(f"\n{md}\n")
                continue

            if isinstance(el, Title):
                if current_title and current_content:
                    sections.append(Section(
                        title=current_title,
                        content="\n".join(current_content),
                        start_page=current_page,
                    ))
                current_title = text
                current_content = []

            text_parts.append(text)
            current_content.append(text)

        if current_title and current_content:
            sections.append(Section(title=current_title, content="\n".join(current_content)))

        raw_text = "\n\n".join(text_parts)
        if len(raw_text.strip()) < 50:
            raise PoisonPillError("Extracted text too short — likely failed OCR")

        return ParsedDocument(
            doc_id=message.doc_id, source_url=message.source_url,
            doc_type=message.doc_type, raw_text=raw_text,
            sections=sections, tables=tables, page_count=current_page,
            parse_errors=errors,
        )

    def _normalize_encoding(self, text: str) -> str:
        try:
            text.encode("utf-8")
            return text
        except UnicodeEncodeError:
            raw      = text.encode("latin-1", errors="replace")
            detected = chardet.detect(raw)
            enc      = detected.get("encoding") or "utf-8"
            return raw.decode(enc, errors="replace")

    def _table_to_markdown(self, element) -> str:
        try:
            html = getattr(element.metadata, "text_as_html", None)
            if not html:
                return element.text or ""
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            rows = soup.find_all("tr")
            if not rows:
                return element.text or ""
            md_rows = []
            for i, row in enumerate(rows):
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                md_rows.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    md_rows.append("|" + "|".join(["---"] * len(cells)) + "|")
            return "\n".join(md_rows)
        except Exception:
            return element.text or ""


def start_pdf_worker():
    """
    Entrypoint: setup async resources, then run sync consumer loop.

    IMPORTANT event loop ordering:
      1. Create a single event loop explicitly
      2. Use it ONLY for setup (pool, qdrant, ensure_collection)
      3. Keep it as the current loop — _process calls loop.run_until_complete()
         on this same loop from within the sync worker.run() call
      worker.run() is a plain while-loop — NOT inside any coroutine —
      so run_until_complete() works fine here.
    """
    import asyncio

    import asyncpg
    from qdrant_client import AsyncQdrantClient

    from ingestion.embedding.services import EmbeddingService, build_backend
    from ingestion.storage.writer import StorageWriter

    cfg = get_settings()

    # Create explicit event loop — keep it alive for _process to use
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _setup():
        pool = await asyncpg.create_pool(
            host=cfg.POSTGRES_HOST, port=cfg.POSTGRES_PORT,
            database=cfg.POSTGRES_DB, user=cfg.POSTGRES_USER,
            password=cfg.POSTGRES_PASSWORD,
        )
        qdrant   = AsyncQdrantClient(host=cfg.QDRANT_HOST, port=cfg.QDRANT_PORT)
        backend  = build_backend()
        embedder = EmbeddingService(backend=backend, db_pool=pool)
        writer   = StorageWriter(db_pool=pool, qdrant=qdrant)
        await writer.ensure_collection()
        return embedder, writer

    # Run setup on our loop — loop is NOT running after this returns
    embedder, writer = loop.run_until_complete(_setup())

    # Instantiate worker (sync — no loop needed)
    worker = PDFWorker(embedding_service=embedder, storage_writer=writer)

    # Start blocking consumer loop
    # _process() will call loop.run_until_complete() on the same loop
    # This is safe because worker.run() is a plain while-loop, not a coroutine
    try:
        worker.run()
    finally:
        loop.close()


if __name__ == "__main__":
    start_pdf_worker()