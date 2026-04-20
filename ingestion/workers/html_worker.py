"""
HTML parser worker.
Uses trafilatura for main-content extraction — strips navigation,
ads, footers, cookie banners, and other boilerplate noise.

Handles:
  - Encoding detection before parsing
  - Section reconstruction from heading structure
  - Relative URL resolution for source tracking
  - Minimum content length gating
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin, urlparse

import chardet
import httpx
import trafilatura
from trafilatura.settings import use_config
from bs4 import BeautifulSoup

from ingestion.workers.base_worker import BaseWorker, PoisonPillError, RetryableError
from shared.config import get_settings
from shared.models import DocumentIngestionMessage, ParsedDocument, Section

logger = logging.getLogger(__name__)
settings = get_settings()

# Trafilatura config: favour precision over recall
_TRAF_CONFIG = use_config()
_TRAF_CONFIG.set("DEFAULT", "MIN_EXTRACTED_SIZE", "200")
_TRAF_CONFIG.set("DEFAULT", "MIN_OUTPUT_SIZE", "100")


class HTMLWorker(BaseWorker):
    """
    Parses HTML documents using trafilatura main-content extraction.
    """

    worker_name = "html-worker"

    def __init__(self, chunker, storage):
        super().__init__()
        self._chunker = chunker
        self._storage = storage

    def _process(self, message: DocumentIngestionMessage) -> None:
        raw_bytes = self._fetch_html(message.source_url)
        parsed = self._parse_html(raw_bytes, message)
        chunks = self._chunker.chunk(parsed)
        self._storage.store(chunks, parsed)

    # ─────────────────────────────────────────────────────────
    #  Fetch
    # ─────────────────────────────────────────────────────────

    def _fetch_html(self, source_url: str) -> bytes:
        if source_url.startswith("file://"):
            from pathlib import Path
            return Path(source_url.replace("file://", "")).read_bytes()

        try:
            with httpx.Client(
                timeout=20,
                follow_redirects=True,
                headers={"User-Agent": "RAG-Ingestion-Bot/1.0"},
            ) as client:
                response = client.get(source_url)
                response.raise_for_status()
                return response.content
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (404, 410):
                raise PoisonPillError(f"Page not found: {source_url}") from e
            raise RetryableError(f"HTTP error: {e}") from e
        except httpx.RequestError as e:
            raise RetryableError(f"Network error: {e}") from e

    # ─────────────────────────────────────────────────────────
    #  Parse
    # ─────────────────────────────────────────────────────────

    def _parse_html(
        self,
        raw_bytes: bytes,
        message: DocumentIngestionMessage,
    ) -> ParsedDocument:
        # Detect and decode encoding before passing to trafilatura
        html_str = self._decode_html(raw_bytes)

        # Extract title and metadata from head (before trafilatura strips it)
        title, author = self._extract_meta(html_str, message.source_url)

        # Main content extraction
        main_text = trafilatura.extract(
            html_str,
            config=_TRAF_CONFIG,
            include_tables=True,
            include_links=False,
            include_images=False,
            no_fallback=False,
            favor_precision=True,
        )

        if not main_text or len(main_text.strip()) < 100:
            raise PoisonPillError(
                f"trafilatura extracted less than 100 chars from {message.source_url} "
                "— likely a JS-rendered page or mostly non-text content"
            )

        # Reconstruct sections from heading structure in original HTML
        sections = self._extract_sections(html_str)

        return ParsedDocument(
            doc_id=message.doc_id,
            source_url=message.source_url,
            doc_type=message.doc_type,
            raw_text=main_text,
            sections=sections,
            title=title,
            author=author,
        )

    def _decode_html(self, raw_bytes: bytes) -> str:
        """
        Detect charset from HTTP meta tag or chardet, then decode.
        Always returns valid UTF-8 string.
        """
        # Try to find charset in meta tag (fast path)
        head_sample = raw_bytes[:4096].decode("ascii", errors="replace")
        charset_match = re.search(r'charset=["\']?([\w-]+)', head_sample, re.IGNORECASE)
        if charset_match:
            declared_charset = charset_match.group(1)
            try:
                return raw_bytes.decode(declared_charset, errors="replace")
            except LookupError:
                pass

        # Fall back to chardet detection
        detected = chardet.detect(raw_bytes[:8192])
        encoding = detected.get("encoding") or "utf-8"
        return raw_bytes.decode(encoding, errors="replace")

    def _extract_meta(self, html: str, source_url: str) -> tuple[str | None, str | None]:
        """Extract title and author from HTML head metadata."""
        try:
            soup = BeautifulSoup(html[:8192], "html.parser")
            title = None
            author = None

            if soup.title:
                title = soup.title.string

            # Open Graph title takes priority over <title>
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                title = og_title["content"]

            for meta in soup.find_all("meta"):
                name = meta.get("name", "").lower()
                if name in ("author", "twitter:creator"):
                    author = meta.get("content")
                    break

            return title, author
        except Exception:
            return None, None

    def _extract_sections(self, html: str) -> list[Section]:
        """
        Reconstruct document sections from heading hierarchy.
        Preserves reading order and section context for chunker.
        """
        sections: list[Section] = []
        try:
            soup = BeautifulSoup(html, "html.parser")
            main = soup.find("main") or soup.find("article") or soup.body
            if not main:
                return sections

            current_title: str | None = None
            current_content: list[str] = []
            current_level = 1

            for tag in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
                if tag.name in ("h1", "h2", "h3", "h4"):
                    if current_title and current_content:
                        sections.append(Section(
                            title=current_title,
                            content=" ".join(current_content),
                            level=current_level,
                        ))
                    current_title = tag.get_text(strip=True)
                    current_content = []
                    current_level = int(tag.name[1])
                else:
                    text = tag.get_text(strip=True)
                    if text:
                        current_content.append(text)

            if current_title and current_content:
                sections.append(Section(
                    title=current_title,
                    content=" ".join(current_content),
                    level=current_level,
                ))
        except Exception as e:
            logger.warning("Section extraction failed: %s", e)

        return sections
    

def start_html_worker():
    """Entrypoint: wire up dependencies and start consuming."""
    import asyncio
    import asyncpg
    from qdrant_client import AsyncQdrantClient
    from ingestion.embedding.services import EmbeddingService, build_backend
    from ingestion.storage.writer import StorageWriter
 
    async def _run():
        cfg     = get_settings()
        pool    = await asyncpg.create_pool(
            host=cfg.POSTGRES_HOST, port=cfg.POSTGRES_PORT,
            database=cfg.POSTGRES_DB, user=cfg.POSTGRES_USER,
            password=cfg.POSTGRES_PASSWORD,
        )
        qdrant   = AsyncQdrantClient(host=cfg.QDRANT_HOST, port=cfg.QDRANT_PORT)
        backend  = build_backend()
        embedder = EmbeddingService(backend=backend, db_pool=pool)
        writer   = StorageWriter(db_pool=pool, qdrant=qdrant)
        await writer.ensure_collection()
 
        worker = HTMLWorker(embedding_service=embedder, storage_writer=writer)
        worker.run()
 
    asyncio.run(_run())
 
 
if __name__ == "__main__":
    start_html_worker()