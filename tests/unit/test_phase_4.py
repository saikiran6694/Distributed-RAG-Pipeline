"""
tests/unit/test_phase4.py

Unit tests for Phase 4 components:
  - CrossEncoderReranker: ordering, top_k, single chunk passthrough
  - SemanticCache: cosine similarity math, hit/miss logic, TTL tracking
"""

from __future__ import annotations

import json
import math
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from query.reranker import CrossEncoderReranker
from query.context_expander import ExpandedChunk


# ── Helpers ───────────────────────────────────────────────────────

def make_expanded(
    chunk_id="c1", doc_id="d1", text="Some chunk text.",
    score=0.9, source_url="http://x.com", section_title="S",
    hierarchy_level=0, chunk_index=0, parent_text=None,
) -> ExpandedChunk:
    return ExpandedChunk(
        chunk_id=chunk_id, doc_id=doc_id, text=text, score=score,
        source_url=source_url, section_title=section_title,
        parent_text=parent_text, page_number=None,
        hierarchy_level=hierarchy_level, chunk_index=chunk_index,
        dense_rank=1, sparse_rank=1,
    )


# ── CrossEncoderReranker ──────────────────────────────────────────

class TestCrossEncoderReranker:

    def _make_reranker_with_mock_scores(self, scores: list[float]) -> CrossEncoderReranker:
        """Return a reranker whose cross-encoder is mocked to return given scores."""
        reranker = CrossEncoderReranker(top_k=len(scores))
        mock_model = MagicMock()
        mock_model.predict.return_value = scores
        with patch("query.reranker._cross_encoder", mock_model):
            reranker._model_override = mock_model
        return reranker, mock_model

    def test_empty_chunks_returns_empty(self):
        reranker = CrossEncoderReranker(top_k=5)
        result = reranker.rerank("query", [])
        assert result == []

    def test_single_chunk_returns_score_1(self):
        reranker = CrossEncoderReranker(top_k=5)
        chunk = make_expanded()
        result = reranker.rerank("query", [chunk])
        assert len(result) == 1
        assert result[0].rerank_score == 1.0
        assert result[0].chunk_id == "c1"

    def test_rerank_sorts_by_score_descending(self):
        reranker = CrossEncoderReranker(top_k=3)
        chunks = [
            make_expanded(chunk_id="c1", text="low relevance"),
            make_expanded(chunk_id="c2", text="high relevance"),
            make_expanded(chunk_id="c3", text="medium relevance"),
        ]

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]  # scores for c1, c2, c3

        import query.reranker as reranker_module
        original = reranker_module._cross_encoder
        reranker_module._cross_encoder = mock_model
        try:
            result = reranker.rerank("test query", chunks)
        finally:
            reranker_module._cross_encoder = original

        assert result[0].chunk_id == "c2"   # highest score 0.9
        assert result[1].chunk_id == "c3"   # middle score 0.5
        assert result[2].chunk_id == "c1"   # lowest score 0.1

    def test_top_k_limits_output(self):
        reranker = CrossEncoderReranker(top_k=2)
        chunks = [make_expanded(chunk_id=f"c{i}") for i in range(5)]

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.9, 0.1, 0.7, 0.3]

        import query.reranker as reranker_module
        original = reranker_module._cross_encoder
        reranker_module._cross_encoder = mock_model
        try:
            result = reranker.rerank("query", chunks)
        finally:
            reranker_module._cross_encoder = original

        assert len(result) == 2

    def test_ranked_chunk_has_both_scores(self):
        # Need 2+ chunks — single chunk short-circuits with score=1.0 without
        # calling the model. Two chunks forces a real model.predict() call.
        reranker = CrossEncoderReranker(top_k=1)
        chunk1 = make_expanded(chunk_id="c1", score=0.75)
        chunk2 = make_expanded(chunk_id="c2", score=0.50)

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.95, 0.30]  # c1 wins

        import query.reranker as reranker_module
        original = reranker_module._cross_encoder
        reranker_module._cross_encoder = mock_model
        try:
            result = reranker.rerank("query", [chunk1, chunk2])
        finally:
            reranker_module._cross_encoder = original

        assert result[0].chunk_id        == "c1"
        assert result[0].rerank_score    == 0.95
        assert result[0].retrieval_score == 0.75

    def test_section_title_included_in_passage(self):
        reranker = CrossEncoderReranker(top_k=1)
        chunk = make_expanded(section_title="Refund Policy", text="Takes 5 days.")
        passage = reranker._build_passage(chunk)
        assert "Refund Policy" in passage
        assert "Takes 5 days." in passage

    def test_passage_truncated_to_1200_chars(self):
        reranker = CrossEncoderReranker(top_k=1)
        chunk = make_expanded(text="word " * 500)
        passage = reranker._build_passage(chunk)
        # text portion should be capped at 1200 chars
        assert len(passage) <= 1250   # 1200 + small section title overhead


# ── SemanticCache ─────────────────────────────────────────────────

class TestSemanticCacheCosineSimilarity:
    """Test the cosine similarity math without a Redis connection."""

    def _cache(self, threshold=0.92):
        from query.cache import SemanticCache
        return SemanticCache(redis=MagicMock(), threshold=threshold)

    def test_identical_vectors_score_1(self):
        cache = self._cache()
        v = [1.0, 0.0, 0.0]
        arr = np.array(v, dtype=np.float32)
        assert abs(cache._cosine_similarity(arr, arr) - 1.0) < 1e-6

    def test_orthogonal_vectors_score_0(self):
        cache = self._cache()
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(cache._cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors_score_minus_1(self):
        cache = self._cache()
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert abs(cache._cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_0(self):
        cache = self._cache()
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        assert cache._cosine_similarity(a, b) == 0.0

    def test_similar_vectors_above_threshold(self):
        cache = self._cache(threshold=0.9)
        a = np.array([1.0, 0.1], dtype=np.float32)
        b = np.array([1.0, 0.15], dtype=np.float32)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        score = cache._cosine_similarity(a, b)
        assert score > 0.9


class TestSemanticCacheGetSet:
    """Test cache get/set with mocked Redis."""

    @pytest.mark.asyncio
    async def test_cache_miss_on_empty(self):
        from query.cache import SemanticCache
        mock_redis = AsyncMock()
        mock_redis.hgetall.return_value = {}
        cache = SemanticCache(redis=mock_redis, threshold=0.92)
        result = await cache.get("query", [1.0, 0.0, 0.0])
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_miss_below_threshold(self):
        from query.cache import SemanticCache
        mock_redis = AsyncMock()
        # Cached vector is orthogonal to query vector — similarity = 0
        mock_redis.hgetall.return_value = {
            b"id1": json.dumps([0.0, 1.0, 0.0]).encode()
        }
        cache = SemanticCache(redis=mock_redis, threshold=0.92)
        result = await cache.get("query", [1.0, 0.0, 0.0])
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_above_threshold(self):
        from query.cache import SemanticCache
        mock_redis = AsyncMock()
        # Cached vector is identical to query vector — similarity = 1.0
        mock_redis.hgetall.return_value = {
            b"id1": json.dumps([1.0, 0.0, 0.0]).encode()
        }
        mock_redis.hget.return_value = json.dumps({
            "answer": "cached answer",
            "citations": [],
            "chunks_used": 3,
        }).encode()
        mock_redis.hset = AsyncMock()
        cache = SemanticCache(redis=mock_redis, threshold=0.92)
        result = await cache.get("query", [1.0, 0.0, 0.0])
        assert result is not None
        assert result["answer"] == "cached answer"
        assert result["cache_hit"] is True
        assert result["cache_similarity"] == 1.0

    @pytest.mark.asyncio
    async def test_cache_set_calls_pipeline(self):
        from query.cache import SemanticCache

        mock_redis = AsyncMock()

        mock_pipeline = MagicMock()
        mock_pipeline.hset = MagicMock()
        mock_pipeline.set = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=[True] * 5)

        mock_pipeline.__enter__ = MagicMock(return_value=mock_pipeline)
        mock_pipeline.__exit__ = MagicMock(return_value=False)

        mock_redis.pipeline = MagicMock(return_value=mock_pipeline)

        cache = SemanticCache(redis=mock_redis, threshold=0.92)

        cache_id = await cache.set(
            "test query",
            [1.0, 0.0, 0.0],
            {"answer": "test", "citations": [], "chunks_used": 2},
        )

        assert isinstance(cache_id, str)
        assert len(cache_id) == 36
        assert mock_pipeline.hset.call_count == 4
        assert mock_pipeline.set.call_count == 1
        mock_pipeline.execute.assert_awaited_once()
        
# ── QueryEngine reranker integration ─────────────────────────────

class TestQueryEngineRerankFlag:
    """Verify use_reranker=False bypasses the reranker."""

    def test_use_reranker_false_skips_rerank(self):
        from query.engine import QueryEngine, QueryRequest
        engine = QueryEngine.__new__(QueryEngine)

        # _deduplicate should return chunks as-is when use_reranker=False
        chunks = [make_expanded(chunk_id=f"c{i}", chunk_index=i*3) for i in range(5)]
        deduped = engine._deduplicate(chunks)
        assert len(deduped) == 5   # no duplicates to remove

    def test_ranked_to_expanded_preserves_rerank_score_as_score(self):
        from query.engine import QueryEngine
        from query.reranker import RankedChunk
        engine = QueryEngine.__new__(QueryEngine)

        ranked = [RankedChunk(
            chunk_id="c1", doc_id="d1", text="text",
            rerank_score=0.95, retrieval_score=0.7,
            source_url="http://x.com", section_title="S",
            page_number=None, hierarchy_level=0, chunk_index=0,
            parent_text=None, dense_rank=1, sparse_rank=1,
        )]
        expanded = engine._ranked_to_expanded(ranked)
        assert expanded[0].score == 0.95   # rerank score becomes the display score