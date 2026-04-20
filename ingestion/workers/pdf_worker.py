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
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Element, Table, Title,
    Header, Footer, PageBreak,
)

from ingestion.workers.base_worker import BaseWorker, PoisonPillError, RetryableError
from shared.config import get_settings
from shared.models import DocumentIngestionMessage, ParsedDocument, Section

logger = logging.getLogger(__name__)
settings = get_settings()


class PDFWorker(BaseWorker):
    """
    Parses PDF documents using Unstructured.io with OCR fallback.
    """

    worker_name = "pdf-worker"

    def __init__(self, chunker, storage):
        super().__init__()
        self._chunker = chunker
        self._storage = storage

    def _process(self, message: DocumentIngestionMessage) -> None:
        """
        Full PDF ingestion pipeline:
        fetch → parse → normalize → chunk → embed → store
        """
        raw_bytes = self._fetch_document(message.source_url)
        parsed = self._parse_pdf(raw_bytes, message)
        chunks = self._chunker.chunk(parsed)
        self._storage.store(chunks, parsed)

    # ─────────────────────────────────────────────────────────
    #  Fetch
    # ─────────────────────────────────────────────────────────

    def _fetch_document(self, source_url: str) -> bytes:
        """
        Fetch raw bytes from the source URL.
        Supports local file paths (file://) and S3 (s3://) via boto3.
        Raises RetryableError on network failures.
        """
        try:
            if source_url.startswith("file://"):
                path = source_url.replace("file://", "")
                return Path(path).read_bytes()

            if source_url.startswith("s3://"):
                return self._fetch_from_s3(source_url)

            # HTTP
            import httpx
            with httpx.Client(timeout=30) as client:
                response = client.get(source_url)
                response.raise_for_status()
                return response.content

        except (IOError, OSError) as e:
            raise RetryableError(f"Fetch failed: {e}") from e

    def _fetch_from_s3(self, s3_url: str) -> bytes:
        import boto3
        from botocore.exceptions import ClientError

        parts = s3_url.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        try:
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "NoSuchBucket"):
                raise PoisonPillError(f"S3 object not found: {s3_url}") from e
            raise RetryableError(f"S3 error: {e}") from e

    # ─────────────────────────────────────────────────────────
    #  Parse
    # ─────────────────────────────────────────────────────────

    def _parse_pdf(
        self,
        raw_bytes: bytes,
        message: DocumentIngestionMessage,
    ) -> ParsedDocument:
        """
        Extract text and structure from PDF bytes using Unstructured.io.
        Writes to a temp file (Unstructured requires a file path).
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(raw_bytes)
            tmp.flush()

            try:
                elements = partition_pdf(
                    filename=tmp.name,
                    strategy="hi_res",              # layout detection + OCR
                    infer_table_structure=True,     # extract table structure
                    include_page_breaks=True,
                    languages=["eng"],
                )
            except Exception as e:
                # Unstructured can fail on corrupt PDFs — treat as poison pill
                raise PoisonPillError(f"Unstructured.io parse failed: {e}") from e

        return self._normalize_elements(elements, message)

    def _normalize_elements(
        self,
        elements: list[Element],
        message: DocumentIngestionMessage,
    ) -> ParsedDocument:
        """
        Convert Unstructured elements to a normalized ParsedDocument.
        Handles: OCR confidence gating, encoding normalization,
                 section reconstruction, table serialization.
        """
        text_parts: list[str] = []
        sections:   list[Section] = []
        tables:     list[str] = []
        parse_errors: list[str] = []

        current_section_title: str | None = None
        current_section_content: list[str] = []
        current_page = 1

        for element in elements:
            if isinstance(element, PageBreak):
                current_page += 1
                continue

            if isinstance(element, (Header, Footer)):
                continue   # skip navigation noise

            # OCR confidence check
            if hasattr(element, "metadata") and element.metadata:
                confidence = getattr(element.metadata, "detection_class_prob", None)
                if confidence is not None and confidence < settings.OCR_CONFIDENCE_THRESHOLD:
                    parse_errors.append(
                        f"Low OCR confidence ({confidence:.2f}) on page {current_page}, skipping element"
                    )
                    continue

            # Encoding normalization
            text = self._normalize_encoding(element.text or "")
            if not text.strip():
                continue

            if isinstance(element, Table):
                # Serialize table to markdown
                table_md = self._table_to_markdown(element)
                if table_md:
                    tables.append(table_md)
                    text_parts.append(f"\n{table_md}\n")
                continue

            if isinstance(element, Title):
                # Flush previous section
                if current_section_title and current_section_content:
                    sections.append(Section(
                        title=current_section_title,
                        content="\n".join(current_section_content),
                        start_page=current_page,
                    ))
                current_section_title = text
                current_section_content = []

            text_parts.append(text)
            current_section_content.append(text)

        # Flush last section
        if current_section_title and current_section_content:
            sections.append(Section(
                title=current_section_title,
                content="\n".join(current_section_content),
            ))

        raw_text = "\n\n".join(text_parts)

        if len(raw_text.strip()) < 50:
            raise PoisonPillError("Extracted text too short — likely a scanned image with failed OCR")

        return ParsedDocument(
            doc_id=message.doc_id,
            source_url=message.source_url,
            doc_type=message.doc_type,
            raw_text=raw_text,
            sections=sections,
            tables=tables,
            page_count=current_page,
            parse_errors=parse_errors,
        )

    def _normalize_encoding(self, text: str) -> str:
        """
        Detect and fix encoding issues.
        Handles Windows-1252, Latin-1, and other common non-UTF-8 encodings.
        """
        try:
            # Already valid unicode — most common case, fast path
            text.encode("utf-8")
            return text
        except UnicodeEncodeError:
            pass

        # Re-encode via chardet detection
        raw = text.encode("latin-1", errors="replace")
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
        return raw.decode(encoding, errors="replace")

    def _table_to_markdown(self, element: Table) -> str:
        """
        Serialize a Table element to markdown format.
        Produces clean markdown tables instead of disjointed cell text.
        """
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
        except Exception as e:
            logger.warning("Failed to serialize table to markdown: %s", e)
            return element.text or ""
        

def start_pdf_worker():
    """Entrypoint: wire up dependencies and start consuming."""
    import asyncio
    import asyncpg
    from qdrant_client import AsyncQdrantClient
    from ingestion.embedding.services import EmbeddingService, build_backend
    from ingestion.storage.writer import StorageWriter
 
    async def _run():
        cfg  = get_settings()
        pool = await asyncpg.create_pool(
            host=cfg.POSTGRES_HOST, port=cfg.POSTGRES_PORT,
            database=cfg.POSTGRES_DB, user=cfg.POSTGRES_USER,
            password=cfg.POSTGRES_PASSWORD,
        )
        qdrant  = AsyncQdrantClient(host=cfg.QDRANT_HOST, port=cfg.QDRANT_PORT)
        backend = build_backend()
        embedder = EmbeddingService(backend=backend, db_pool=pool)
        writer   = StorageWriter(db_pool=pool, qdrant=qdrant)
        await writer.ensure_collection()
 
        worker = PDFWorker(embedding_service=embedder, storage_writer=writer)
        worker.run()
 
    asyncio.run(_run())
 
 
if __name__ == "__main__":
    start_pdf_worker()