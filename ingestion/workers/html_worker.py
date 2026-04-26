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

import chardet
import httpx
import trafilatura
from bs4 import BeautifulSoup
from trafilatura.settings import use_config

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

    def __init__(self, embedding_service, storage_writer):
        super().__init__()
        self._embedder = embedding_service
        self._storage  = storage_writer
        from ingestion.chunking.selector import ChunkingStrategySelector
        self._selector = ChunkingStrategySelector()

    def _process(self, message: DocumentIngestionMessage) -> None:
        import asyncio
        raw_bytes = self._fetch_html(message.source_url)
        parsed    = self._parse_html(raw_bytes, message)
        chunker   = self._selector.get(message.doc_type)
        chunks    = chunker.chunk(parsed)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._store(chunks, parsed))

    async def _store(self, chunks, parsed) -> None:
        embedded = await self._embedder.embed_chunks(chunks)
        await self._storage.store(embedded, parsed)

    def _fetch_html(self, source_url: str) -> bytes:
        if source_url.startswith("file://"):
            from pathlib import Path
            return Path(source_url.replace("file://", "")).read_bytes()
        try:
            with httpx.Client(timeout=20, follow_redirects=True,
                               headers={"User-Agent": "RAG-Ingestion-Bot/1.0"}) as client:
                response = client.get(source_url)
                response.raise_for_status()
                return response.content
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (404, 410):
                raise PoisonPillError(f"Page not found: {source_url}") from e
            raise RetryableError(f"HTTP error: {e}") from e
        except httpx.RequestError as e:
            raise RetryableError(f"Network error: {e}") from e

    def _parse_html(self, raw_bytes: bytes, message: DocumentIngestionMessage) -> ParsedDocument:
        html_str = self._decode_html(raw_bytes)
        title, author = self._extract_meta(html_str, message.source_url)
        main_text = trafilatura.extract(
            html_str, config=_TRAF_CONFIG, include_tables=True,
            include_links=False, no_fallback=False, favor_precision=True,
        )
        if not main_text or len(main_text.strip()) < 100:
            raise PoisonPillError(f"Too little content extracted from {message.source_url}")
        sections = self._extract_sections(html_str)
        return ParsedDocument(
            doc_id=message.doc_id, source_url=message.source_url,
            doc_type=message.doc_type, raw_text=main_text,
            sections=sections, title=title, author=author,
        )

    def _decode_html(self, raw_bytes: bytes) -> str:
        head_sample = raw_bytes[:4096].decode("ascii", errors="replace")
        charset_match = re.search(r'charset=["\']?([\w-]+)', head_sample, re.IGNORECASE)
        if charset_match:
            try:
                return raw_bytes.decode(charset_match.group(1), errors="replace")
            except LookupError:
                pass
        detected = chardet.detect(raw_bytes[:8192])
        encoding = detected.get("encoding") or "utf-8"
        return raw_bytes.decode(encoding, errors="replace")

    def _extract_meta(self, html: str, source_url: str) -> tuple[str | None, str | None]:
        try:
            soup = BeautifulSoup(html[:8192], "html.parser")
            title = soup.title.string if soup.title else None
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                title = og_title["content"]
            author = None
            for meta in soup.find_all("meta"):
                if meta.get("name", "").lower() in ("author", "twitter:creator"):
                    author = meta.get("content")
                    break
            return title, author
        except Exception:
            return None, None

    def _extract_sections(self, html: str) -> list[Section]:
        sections: list[Section] = []
        try:
            soup = BeautifulSoup(html, "html.parser")
            main = soup.find("main") or soup.find("article") or soup.body
            if not main:
                return sections
            current_title, current_content, current_level = None, [], 1
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
                sections.append(Section(title=current_title,
                                        content=" ".join(current_content),
                                        level=current_level))
        except Exception as e:
            logger.warning("Section extraction failed: %s", e)
        return sections


def start_html_worker():
    """Entrypoint: setup async resources, then run sync consumer loop."""
    import asyncio

    import asyncpg
    from qdrant_client import AsyncQdrantClient

    from ingestion.embedding.services import EmbeddingService, build_backend
    from ingestion.storage.writer import StorageWriter

    cfg = get_settings()

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

    embedder, writer = loop.run_until_complete(_setup())
    worker = HTMLWorker(embedding_service=embedder, storage_writer=writer)

    try:
        worker.run()
    finally:
        loop.close()


if __name__ == "__main__":
    start_html_worker()