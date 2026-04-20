"""
Semantic query cache backed by Redis.

Why semantic over string matching:
  "What is the refund policy?"
  "How do I get a refund?"
  "Tell me about refunds"
  → All three should hit the same cache entry.

  String-match caches miss all of these. A semantic cache stores the
  query embedding vector as the key and uses cosine similarity to find
  hits. Any query within `threshold` similarity of a cached query
  returns the cached response instantly.

Architecture:
  - On query: embed the query → scan cached vectors for similarity
  - On hit: return cached QueryResponse (skips retrieval + LLM entirely)
  - On miss: run full pipeline → store result in cache with TTL

Storage layout in Redis:
  rag:cache:vectors   → Hash { cache_id: json(vector) }
  rag:cache:queries   → Hash { cache_id: original_query_string }
  rag:cache:responses → Hash { cache_id: json(QueryResponse) }
  rag:cache:meta      → Hash { cache_id: json({created_at, hit_count}) }

Why not Redis Vector Search (RediSearch):
  RediSearch requires a paid module or Redis Stack. We implement
  brute-force cosine similarity over the cached vectors — this is
  perfectly fine for caches up to ~10k entries (sub-millisecond scan).
  Beyond that, switch to RediSearch or a dedicated vector store.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import redis.asyncio as aioredis

from shared.config import get_settings
from shared.telemetry import traced_span, CACHE_HITS, CACHE_MISSES

logger = logging.getLogger(__name__)
settings = get_settings()

# Redis key prefixes
_PFX          = "rag:cache"
_KEY_VECTORS  = f"{_PFX}:vectors"
_KEY_QUERIES  = f"{_PFX}:queries"
_KEY_RESPONSES = f"{_PFX}:responses"
_KEY_META     = f"{_PFX}:meta"

# Default TTL: 1 hour
_DEFAULT_TTL_SECONDS = 3600


@dataclass
class CacheEntry:
    cache_id:     str
    query:        str
    response:     dict        # serialized QueryResponse
    created_at:   str
    hit_count:    int = 0


class SemanticCache:
    """
    Semantic query cache using cosine similarity for cache key matching.

    Usage:
        cache = SemanticCache(redis_client, threshold=0.92)

        # Check cache
        hit = await cache.get(query, query_vector)
        if hit:
            return hit

        # Run pipeline...
        response = await engine.query(request)

        # Store result
        await cache.set(query, query_vector, response)
    """

    def __init__(
        self,
        redis:     aioredis.Redis,
        threshold: float = 0.92,    # cosine similarity threshold for cache hit
        ttl:       int   = _DEFAULT_TTL_SECONDS,
    ):
        self._redis     = redis
        self.threshold  = threshold
        self._ttl       = ttl

    async def get(
        self,
        query:        str,
        query_vector: list[float],
    ) -> dict | None:
        """
        Look up a semantically similar cached response.
        Returns the cached QueryResponse dict on hit, None on miss.
        """
        with traced_span("cache.get", {"query_len": len(query)}):
            t0 = time.monotonic()

            # Fetch all cached vectors
            raw_vectors = await self._redis.hgetall(_KEY_VECTORS)
            if not raw_vectors:
                return None

            # Find most similar cached query
            query_arr = np.array(query_vector, dtype=np.float32)
            best_id:    str | None = None
            best_score: float      = -1.0

            for cache_id, vec_json in raw_vectors.items():
                cached_vec = np.array(json.loads(vec_json), dtype=np.float32)
                score      = self._cosine_similarity(query_arr, cached_vec)
                if score > best_score:
                    best_score = score
                    best_id    = cache_id.decode() if isinstance(cache_id, bytes) else cache_id

            if best_score < self.threshold or best_id is None:
                logger.debug(
                    "Cache miss (best=%.3f < threshold=%.3f) for: %r",
                    best_score, self.threshold, query[:60],
                )
                return None

            # Fetch the cached response
            raw_response = await self._redis.hget(_KEY_RESPONSES, best_id)
            if not raw_response:
                return None

            # Increment hit count
            await self._increment_hit_count(best_id)

            elapsed_ms = (time.monotonic() - t0) * 1000
            CACHE_HITS.inc()
            logger.info(
                "Cache HIT (similarity=%.3f, %.1fms): %r",
                best_score, elapsed_ms, query[:60],
            )

            response = json.loads(raw_response)
            response["cache_hit"]       = True
            response["cache_similarity"] = round(best_score, 4)
            return response

    async def set(
        self,
        query:        str,
        query_vector: list[float],
        response:     dict,
        ttl:          int | None = None,
    ) -> str:
        """
        Store a query+response in the cache.
        Returns the cache_id for the new entry.
        """
        cache_id   = str(uuid.uuid4())
        ttl_       = ttl or self._ttl
        expiry_key = f"{_PFX}:expiry:{cache_id}"
        meta       = json.dumps({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hit_count":  0,
        })

        # pipeline() is a sync context manager in redis-py asyncio
        # Use it as a regular (non-async) context manager
        pipe = self._redis.pipeline(transaction=False)
        pipe.hset(_KEY_VECTORS,    cache_id, json.dumps(query_vector))
        pipe.hset(_KEY_QUERIES,    cache_id, query)
        pipe.hset(_KEY_RESPONSES,  cache_id, json.dumps(response))
        pipe.hset(_KEY_META,       cache_id, meta)
        pipe.set(expiry_key, "1", ex=ttl_)
        await pipe.execute()

        logger.debug("Cache SET: id=%s query=%r", cache_id, query[:60])
        return cache_id

    async def invalidate_expired(self) -> int:
        """
        Remove cache entries whose expiry key has elapsed.
        Run periodically (e.g. on each cache miss) to keep Redis lean.
        Returns number of entries removed.
        """
        all_ids = await self._redis.hkeys(_KEY_VECTORS)
        removed = 0

        for cache_id in all_ids:
            cid = cache_id.decode() if isinstance(cache_id, bytes) else cache_id
            expiry_key = f"{_PFX}:expiry:{cid}"
            still_alive = await self._redis.exists(expiry_key)
            if not still_alive:
                pipe = self._redis.pipeline()
                pipe.hdel(_KEY_VECTORS,   cid)
                pipe.hdel(_KEY_QUERIES,   cid)
                pipe.hdel(_KEY_RESPONSES, cid)
                pipe.hdel(_KEY_META,      cid)
                await pipe.execute()
                removed += 1

        if removed:
            logger.info("Cache: evicted %d expired entries", removed)
        return removed

    async def stats(self) -> dict:
        """Return cache statistics for monitoring."""
        total  = await self._redis.hlen(_KEY_VECTORS)
        metas  = await self._redis.hgetall(_KEY_META)

        total_hits = 0
        for raw in metas.values():
            try:
                total_hits += json.loads(raw).get("hit_count", 0)
            except Exception:
                pass

        return {
            "total_entries": total,
            "total_hits":    total_hits,
            "threshold":     self.threshold,
            "ttl_seconds":   self._ttl,
        }

    # ─────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def _increment_hit_count(self, cache_id: str) -> None:
        raw = await self._redis.hget(_KEY_META, cache_id)
        if raw:
            try:
                meta = json.loads(raw)
                meta["hit_count"] = meta.get("hit_count", 0) + 1
                await self._redis.hset(_KEY_META, cache_id, json.dumps(meta))
            except Exception:
                pass


def build_cache(redis_client: aioredis.Redis) -> SemanticCache:
    """Factory: build cache with settings from config."""
    return SemanticCache(
        redis=redis_client,
        threshold=0.92,
        ttl=_DEFAULT_TTL_SECONDS,
    )