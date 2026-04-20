"""
tests/unit/test_phase2.py

Unit tests for Phase 2 components:
  - BM25 sparse encoder
  - Reciprocal Rank Fusion logic
  - Query decomposition
  - Context expander dedup
"""

from __future__ import annotations

import math
import pytest
from unittest.mock import MagicMock, AsyncMock

from ingestion.embedding.sparse import BM25Encoder, SparseVector


# ─────────────────────────────────────────────────────────────────
#  BM25 Encoder
# ─────────────────────────────────────────────────────────────────

class TestBM25Encoder:

    def test_empty_text_returns_empty_vector(self):
        enc = BM25Encoder()
        result = enc.encode_document("")
        assert result.indices == []
        assert result.values == []

    def test_encode_document_without_fit(self):
        enc = BM25Encoder()
        result = enc.encode_document("hello world hello")
        assert len(result.indices) > 0
        assert len(result.values) == len(result.indices)

    def test_indices_sorted_ascending(self):
        enc = BM25Encoder()
        result = enc.encode_document("the quick brown fox jumps over the lazy dog")
        assert result.indices == sorted(result.indices)

    def test_repeated_term_does_not_dominate_score(self):
        """BM25 term frequency saturation: 10x repetition should NOT give 10x score."""
        enc = BM25Encoder()
        sparse_once  = enc.encode_document("machine learning")
        sparse_many  = enc.encode_document("machine " * 20 + "learning")

        # Find the score for "machine" token in both
        token_id = sparse_once.indices[0] if sparse_once.indices else None
        if token_id and token_id in sparse_many.indices:
            idx_once = sparse_once.indices.index(token_id)
            idx_many = sparse_many.indices.index(token_id)
            score_once = sparse_once.values[idx_once]
            score_many = sparse_many.values[idx_many]
            # BM25 saturation: 20x repetition should give less than 4x score
            assert score_many < score_once * 4, \
                f"No saturation: once={score_once:.3f} many={score_many:.3f}"

    def test_fit_and_encode_uses_idf(self):
        corpus = [
            "machine learning is great",
            "deep learning neural networks",
            "the cat sat on the mat",
            "the dog ran in the park",
        ]
        enc = BM25Encoder()
        enc.fit(corpus)
        assert enc._fitted
        assert enc._doc_count == 4
        assert enc._avgdl > 0

        # "the" appears in 2/4 docs — should have lower IDF than "machine" (1/4)
        the_tokens    = enc._tokenize("the")
        machine_tokens = enc._tokenize("machine")
        if the_tokens and machine_tokens:
            idf_the     = enc._idf.get(the_tokens[0], 0)
            idf_machine = enc._idf.get(machine_tokens[0], 0)
            assert idf_the < idf_machine, \
                f"Common word 'the' (idf={idf_the:.3f}) should have lower IDF than rare 'machine' (idf={idf_machine:.3f})"

    def test_encode_query_returns_vector(self):
        enc = BM25Encoder()
        result = enc.encode_query("what is machine learning")
        assert len(result.indices) > 0
        assert all(v > 0 for v in result.values)

    def test_save_and_load_roundtrip(self, tmp_path):
        corpus = ["hello world", "foo bar baz", "machine learning rocks"]
        enc = BM25Encoder()
        enc.fit(corpus)

        path = str(tmp_path / "idf.json")
        enc.save(path)

        loaded = BM25Encoder.load(path)
        assert loaded._fitted
        assert loaded._doc_count == enc._doc_count
        assert abs(loaded._avgdl - enc._avgdl) < 0.001

        # Encoding should produce identical results
        v1 = enc.encode_document("machine learning")
        v2 = loaded.encode_document("machine learning")
        assert v1.indices == v2.indices
        assert all(abs(a - b) < 1e-6 for a, b in zip(v1.values, v2.values))

    def test_to_qdrant_dict_format(self):
        enc = BM25Encoder()
        result = enc.encode_document("hello world")
        d = result.to_qdrant_dict()
        assert "indices" in d
        assert "values" in d
        assert isinstance(d["indices"], list)
        assert isinstance(d["values"], list)


# ─────────────────────────────────────────────────────────────────
#  RRF Fusion (test the math directly)
# ─────────────────────────────────────────────────────────────────

class TestRRFFusion:
    """Test the RRF scoring logic in isolation."""

    def _rrf_score(self, ranks: list[int], k: int = 60) -> float:
        return sum(1.0 / (k + r) for r in ranks)

    def test_document_in_both_lists_scores_higher(self):
        """A doc ranked #5 in both lists should beat one ranked #1 in only one list."""
        score_both = self._rrf_score([5, 5])
        score_one  = self._rrf_score([1])
        assert score_both > score_one, \
            f"Both-list doc ({score_both:.4f}) should beat single-list top doc ({score_one:.4f})"

    def test_higher_rank_beats_lower(self):
        """Rank 1 should score higher than rank 10 in the same list."""
        assert self._rrf_score([1]) > self._rrf_score([10])

    def test_rrf_formula_correctness(self):
        """Verify exact RRF formula: 1/(k+rank)."""
        k = 60
        assert abs(self._rrf_score([1], k) - 1/61) < 1e-10
        assert abs(self._rrf_score([60], k) - 1/120) < 1e-10

    def test_fusion_with_mock_retriever(self):
        """End-to-end RRF fusion via HybridRetriever._reciprocal_rank_fusion."""
        from query.retriever import HybridRetriever
        from unittest.mock import MagicMock
        from qdrant_client.http.models import ScoredPoint

        retriever = HybridRetriever.__new__(HybridRetriever)

        def make_hit(id_, score):
            h = MagicMock(spec=ScoredPoint)
            h.id = id_
            h.score = score
            h.payload = {
                "doc_id": "doc1", "source_url": "http://x.com",
                "section_title": None, "page_number": None,
                "hierarchy_level": 0, "parent_chunk_id": None, "chunk_index": 0,
            }
            return h

        dense_hits  = [make_hit("a", 0.95), make_hit("b", 0.80), make_hit("c", 0.70)]
        sparse_hits = [make_hit("b", 0.90), make_hit("a", 0.75), make_hit("d", 0.60)]

        result = retriever._reciprocal_rank_fusion(dense_hits, sparse_hits, top_k=4)

        ids = [r.chunk_id for r in result]
        scores = [r.score for r in result]

        # "a" is #1 dense, #2 sparse — should be top result
        assert ids[0] == "a", f"Expected 'a' first, got {ids}"
        # "b" is #2 dense, #1 sparse — should be second
        assert ids[1] == "b", f"Expected 'b' second, got {ids}"
        # Scores should be descending
        assert scores == sorted(scores, reverse=True)
        # "a" and "b" should have higher scores than "c" and "d" (both-list bonus)
        score_a = next(r.score for r in result if r.chunk_id == "a")
        score_c = next(r.score for r in result if r.chunk_id == "c")
        assert score_a > score_c


# ─────────────────────────────────────────────────────────────────
#  Query decomposition
# ─────────────────────────────────────────────────────────────────

class TestQueryDecomposition:
    """
    Tests the regex fallback path in QueryDecomposer._parse_response.
    The LLM-based decompose() is tested in test_phase3.py.
    Short queries (<=6 words) bypass the LLM and return as-is.
    """

    def _decompose_sync(self, query):
        """
        For unit tests we test _parse_response directly (no LLM call needed)
        and the short-query fast-path via decompose() which skips LLM entirely.
        """
        import asyncio
        from query.decomposer import QueryDecomposer
        decomposer = QueryDecomposer()
        # decompose() is async but short queries (<=6 words) return immediately
        return asyncio.get_event_loop().run_until_complete(decomposer.decompose(query))

    def test_simple_short_query_not_decomposed(self):
        # 4 words — hits the <=6 word fast-path, returns as-is without LLM
        result = self._decompose_sync("What is machine learning?")
        assert result == ["What is machine learning?"]

    def test_single_question_unchanged(self):
        # 5 words — also hits fast-path
        q = "How does gradient descent work?"
        result = self._decompose_sync(q)
        assert result[0] == q

    def test_very_short_query_unchanged(self):
        # 2 words — well under 6 word threshold
        result = self._decompose_sync("Why? What?")
        assert len(result) >= 1

    def test_parse_response_compound_splits(self):
        # Test the JSON parse logic directly — no LLM call
        from query.decomposer import QueryDecomposer
        d = QueryDecomposer()
        result = d._parse_response(
            '{"queries": ["What is the refund policy?", "How long does it take?"]}',
            "What is the refund policy and how long does it take?"
        )
        assert len(result) == 2
        assert result[0] == "What is the refund policy?"
        assert result[1] == "How long does it take?"


# ─────────────────────────────────────────────────────────────────
#  Deduplication
# ─────────────────────────────────────────────────────────────────

class TestDeduplication:

    def _make_chunk(self, chunk_id, doc_id, chunk_index, score):
        from query.context_expander import ExpandedChunk
        return ExpandedChunk(
            chunk_id=chunk_id, doc_id=doc_id, text="text", parent_text=None,
            score=score, source_url="http://x.com", section_title=None,
            page_number=None, hierarchy_level=2, chunk_index=chunk_index,
            dense_rank=1, sparse_rank=1,
        )

    def test_exact_duplicate_removed(self):
        from query.engine import QueryEngine
        engine = QueryEngine.__new__(QueryEngine)
        chunks = [
            self._make_chunk("id1", "doc1", 0, 0.9),
            self._make_chunk("id1", "doc1", 0, 0.8),  # duplicate
        ]
        result = engine._deduplicate(chunks)
        assert len(result) == 1
        assert result[0].chunk_id == "id1"

    def test_different_docs_both_kept(self):
        from query.engine import QueryEngine
        engine = QueryEngine.__new__(QueryEngine)
        chunks = [
            self._make_chunk("id1", "doc1", 0, 0.9),
            self._make_chunk("id2", "doc2", 0, 0.8),
        ]
        result = engine._deduplicate(chunks)
        assert len(result) == 2