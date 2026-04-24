"""
Embedding service: converts chunks to vectors in batches.

Key responsibilities:
  - Backend abstraction (OpenAI / HuggingFace / Ollama) via EmbeddingBackend protocol
  - Batching: collects chunks into groups before calling the model API
  - Exponential backoff on rate limits (429) and transient errors
  - Model version stamping on every chunk (critical for migration)
  - Cost tracking: logs token counts + estimated USD to Postgres
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Protocol

import asyncpg

from shared.config import get_settings
from shared.models import Chunk
from shared.telemetry import EMBEDDING_BATCH_DURATION, EMBEDDING_COST_USD

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────
#  Backend protocol — all backends implement this interface
# ─────────────────────────────────────────────────────────────────

class EmbeddingBackend(Protocol):
    model_name:    str
    model_version: str
    embedding_dim: int

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        ...


# ─────────────────────────────────────────────────────────────────
#  Backend implementations
# ─────────────────────────────────────────────────────────────────

class OpenAIBackend:
    """OpenAI text-embedding-3-* API backend."""

    def __init__(self):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_name    = settings.OPENAI_EMBED_MODEL
        self.model_version = settings.OPENAI_EMBED_MODEL_VERSION
        self.embedding_dim = settings.OPENAI_EMBED_DIM

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self.model_name,
            input=texts,
            encoding_format="float",
        )
        # Results are guaranteed ordered by index
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


class HuggingFaceBackend:
    """Local sentence-transformers backend — no API key, works offline."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading HuggingFace model: %s", settings.HF_EMBED_MODEL)
        self._model        = SentenceTransformer(settings.HF_EMBED_MODEL, device=settings.HF_DEVICE)
        self.model_name    = settings.HF_EMBED_MODEL
        self.model_version = settings.HF_EMBED_MODEL_VERSION
        self.embedding_dim = settings.HF_EMBED_DIM

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # SentenceTransformer is synchronous — run in thread pool
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, batch_size=64, show_progress_bar=False).tolist(),
        )
        return embeddings


class OllamaBackend:
    """Ollama local API backend (nomic-embed-text or similar)."""

    def __init__(self):
        import httpx
        self._client       = httpx.AsyncClient(base_url=settings.OLLAMA_BASE_URL, timeout=60)
        self.model_name    = settings.OLLAMA_EMBED_MODEL
        self.model_version = "local"
        self.embedding_dim = settings.OLLAMA_EMBED_DIM

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            response = await self._client.post(
                "/api/embeddings",
                json={"model": self.model_name, "prompt": text},
            )
            response.raise_for_status()
            results.append(response.json()["embedding"])
        return results


def build_backend() -> EmbeddingBackend:
    """Factory: select backend from EMBED_BACKEND env var."""
    backends = {
        "openai":      OpenAIBackend,
        "huggingface": HuggingFaceBackend,
        "ollama":      OllamaBackend,
    }
    cls = backends.get(settings.EMBED_BACKEND)
    if cls is None:
        raise ValueError(f"Unknown EMBED_BACKEND: {settings.EMBED_BACKEND}")
    return cls()


# ─────────────────────────────────────────────────────────────────
#  Embedding service
# ─────────────────────────────────────────────────────────────────

class EmbeddingService:
    """
    Wraps an embedding backend with batching, retries, and cost tracking.

    Usage:
        service = EmbeddingService(backend, db_pool)
        chunks_with_vectors = await service.embed_chunks(chunks)
    """

    def __init__(self, backend: EmbeddingBackend, db_pool: asyncpg.Pool):
        self._backend  = backend
        self._db       = db_pool
        self._batch_sz = settings.EMBED_BATCH_SIZE

    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Embed all chunks. Returns chunks with .vector and embedding metadata set.
        Processes in batches, handles retries, tracks cost.
        """
        if not chunks:
            return []

        # Process in batches
        result_chunks: list[Chunk] = []
        batches = [chunks[i:i + self._batch_sz] for i in range(0, len(chunks), self._batch_sz)]

        for batch in batches:
            embedded_batch = await self._embed_batch_with_retry(batch)
            result_chunks.extend(embedded_batch)

        return result_chunks

    async def _embed_batch_with_retry(self, batch: list[Chunk]) -> list[Chunk]:
        """
        Embed one batch with exponential backoff on failure.
        Max retries and delays come from config.
        """
        texts = [chunk.text for chunk in batch]
        delay = settings.EMBED_RETRY_BASE_DELAY

        for attempt in range(settings.EMBED_MAX_RETRIES):
            start = time.monotonic()
            try:
                vectors = await self._backend.embed_batch(texts)

                elapsed = time.monotonic() - start
                EMBEDDING_BATCH_DURATION.labels(backend=settings.EMBED_BACKEND).observe(elapsed)

                # Attach vectors and model metadata to chunks
                for chunk, vector in zip(batch, vectors, strict=True):
                    chunk.vector               = vector
                    chunk.embedding_model      = self._backend.model_name
                    chunk.embedding_model_version = self._backend.model_version
                    chunk.embedding_dim        = self._backend.embedding_dim

                # Log cost for this batch
                await self._log_cost(batch, texts)

                return batch

            except Exception as e:
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                is_last_attempt = attempt == settings.EMBED_MAX_RETRIES - 1

                if is_last_attempt:
                    logger.error("Embedding batch failed after %d attempts: %s", attempt + 1, e)
                    raise

                if is_rate_limit:
                    logger.warning("Rate limited by embedding API. Waiting %.1fs...", delay)
                else:
                    logger.warning("Embedding attempt %d failed: %s. Retrying in %.1fs", attempt + 1, e, delay)

                await asyncio.sleep(delay)
                delay = min(delay * 2, settings.EMBED_RETRY_MAX_DELAY)

        # Should never reach here
        raise RuntimeError("Embedding retry loop exited without result")

    async def _log_cost(self, batch: list[Chunk], texts: list[str]) -> None:
        """Record embedding API spend to Postgres for cost tracking."""
        total_chars = sum(len(t) for t in texts)
        # Rough token estimate: ~4 chars per token
        estimated_tokens = total_chars // 4

        # OpenAI pricing (as of 2025): $0.02 per 1M tokens for text-embedding-3-small
        cost_per_million = 0.02
        cost_usd = (estimated_tokens / 1_000_000) * cost_per_million if settings.EMBED_BACKEND == "openai" else None

        if cost_usd:
            EMBEDDING_COST_USD.labels(model=self._backend.model_name).inc(cost_usd)

        try:
            from uuid import UUID
            raw_doc_id = batch[0].doc_id if batch else None
            doc_id = None
            if raw_doc_id:
                exists = await self._db.fetchval(
                    "SELECT id FROM documents WHERE id = $1",
                    raw_doc_id if isinstance(raw_doc_id, UUID) else UUID(str(raw_doc_id)),
                )
                doc_id = exists  # UUID or None

            await self._db.execute(
                """
                INSERT INTO ingestion_cost_log (doc_id, model, batch_size, token_count, cost_usd)
                VALUES ($1, $2, $3, $4, $5)
                """,
                doc_id,
                self._backend.model_name,
                len(batch),
                estimated_tokens,
                cost_usd,
            )
        except Exception as e:
            # Non-critical — don't fail the pipeline for cost logging
            logger.warning("Failed to log embedding cost: %s", e)