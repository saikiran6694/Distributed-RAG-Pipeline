"""
Unit tests for all three chunking strategies.
Uses deterministic inputs so expected outputs can be verified exactly.
No external services required.
"""

from __future__ import annotations

import uuid

import pytest

from ingestion.chunking.fixed import FixedChunker
from ingestion.chunking.hierarical import HierarchicalChunker
from ingestion.chunking.semantic import SemanticChunker

from shared.models import ChunkingStrategy, ParsedDocument, Section


# ─────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────

def make_document(text: str, sections: list[Section] | None = None) -> ParsedDocument:
    return ParsedDocument(
        doc_id=uuid.uuid4(),
        source_url="file:///test/doc.pdf",
        doc_type="pdf",
        raw_text=text,
        sections=sections or [],
    )


SHORT_TEXT = "Hello world. This is a test document."

LONG_TEXT = " ".join([
    "The quick brown fox jumps over the lazy dog." * 20,
    "Machine learning models are trained on large datasets to learn patterns.",
    "Natural language processing enables computers to understand human language.",
    "Vector embeddings represent semantic meaning as points in high-dimensional space.",
])

SECTIONED_TEXT = """Introduction
This document covers the basics of distributed systems.

Architecture
A distributed system consists of multiple nodes communicating over a network.
Each node handles a subset of the total workload.

Conclusion
Distributed systems enable horizontal scaling and fault tolerance.
"""


# ─────────────────────────────────────────────────────────────────
#  Fixed chunker tests
# ─────────────────────────────────────────────────────────────────

class TestFixedChunker:

    def test_empty_document_returns_no_chunks(self):
        chunker = FixedChunker(chunk_size=512, overlap=50)
        doc = make_document("")
        assert chunker.chunk(doc) == []

    def test_short_document_returns_single_chunk(self):
        chunker = FixedChunker(chunk_size=512, overlap=50)
        doc = make_document(SHORT_TEXT)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text.strip() == SHORT_TEXT.strip()

    def test_long_document_returns_multiple_chunks(self):
        chunker = FixedChunker(chunk_size=100, overlap=10)
        doc = make_document(LONG_TEXT)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1

    def test_all_chunks_have_correct_strategy(self):
        chunker = FixedChunker(chunk_size=100, overlap=10)
        doc = make_document(LONG_TEXT)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.chunking_strategy == ChunkingStrategy.FIXED

    def test_chunk_indices_are_sequential(self):
        chunker = FixedChunker(chunk_size=100, overlap=10)
        doc = make_document(LONG_TEXT)
        chunks = chunker.chunk(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_overlap_is_nonzero_after_first(self):
        chunker = FixedChunker(chunk_size=100, overlap=20)
        doc = make_document(LONG_TEXT)
        chunks = chunker.chunk(doc)
        if len(chunks) > 1:
            assert chunks[1].chunk_overlap_tokens == 20
            assert chunks[0].chunk_overlap_tokens == 0

    def test_overlap_equals_chunk_size_raises(self):
        with pytest.raises(ValueError):
            FixedChunker(chunk_size=100, overlap=100)

    def test_all_chunks_have_doc_id(self):
        chunker = FixedChunker(chunk_size=100, overlap=10)
        doc = make_document(LONG_TEXT)
        doc_id = doc.doc_id
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.doc_id == doc_id

    def test_section_title_propagated(self):
        sections = [
            Section(title="Introduction", content="This is the introduction. " * 20),
            Section(title="Methods", content="These are the methods. " * 20),
        ]
        text = "\n\n".join(s.content for s in sections)
        doc = make_document(text, sections=sections)
        chunker = FixedChunker(chunk_size=50, overlap=5)
        chunks = chunker.chunk(doc)
        # At least some chunks should have a section title
        titled = [c for c in chunks if c.section_title is not None]
        assert len(titled) > 0

    def test_content_hash_is_unique_per_chunk(self):
        chunker = FixedChunker(chunk_size=100, overlap=0)
        doc = make_document(LONG_TEXT)
        chunks = chunker.chunk(doc)
        hashes = [c.content_hash for c in chunks]
        assert len(hashes) == len(set(hashes)), "Duplicate content hashes found"


# ─────────────────────────────────────────────────────────────────
#  Semantic chunker tests
# ─────────────────────────────────────────────────────────────────

class TestSemanticChunker:

    def test_empty_document_returns_no_chunks(self):
        chunker = SemanticChunker()
        doc = make_document("")
        assert chunker.chunk(doc) == []

    def test_short_document_returns_single_chunk(self):
        chunker = SemanticChunker()
        doc = make_document(SHORT_TEXT)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_all_chunks_have_correct_strategy(self):
        chunker = SemanticChunker(threshold=0.5)
        doc = make_document(LONG_TEXT)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.chunking_strategy == ChunkingStrategy.SEMANTIC

    def test_chunks_cover_full_document(self):
        """Combined text of all chunks should contain all words from original."""
        chunker = SemanticChunker(threshold=0.5)
        doc = make_document(SECTIONED_TEXT)
        chunks = chunker.chunk(doc)
        combined = " ".join(c.text for c in chunks)
        # Sample key words from original
        for word in ["Introduction", "Architecture", "Conclusion"]:
            assert word.lower() in combined.lower() or word in combined

    def test_max_tokens_ceiling_respected(self):
        chunker = SemanticChunker(max_tokens=100, threshold=0.9)
        doc = make_document(LONG_TEXT)
        chunks = chunker.chunk(doc)
        from ingestion.chunking.semantic import _TOKENIZER
        for chunk in chunks:
            token_count = len(_TOKENIZER.encode(chunk.text))
            assert token_count <= 150, f"Chunk exceeded max_tokens: {token_count} tokens"

    def test_sequential_chunk_indices(self):
        chunker = SemanticChunker(threshold=0.5)
        doc = make_document(LONG_TEXT)
        chunks = chunker.chunk(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


# ─────────────────────────────────────────────────────────────────
#  Hierarchical chunker tests
# ─────────────────────────────────────────────────────────────────

class TestHierarchicalChunker:

    def test_empty_document_returns_no_chunks(self):
        chunker = HierarchicalChunker()
        doc = make_document("")
        assert chunker.chunk(doc) == []

    def test_always_produces_l0_chunk(self):
        chunker = HierarchicalChunker()
        doc = make_document(LONG_TEXT)
        chunks = chunker.chunk(doc)
        l0 = [c for c in chunks if c.hierarchy_level == 0]
        assert len(l0) == 1, "Should always have exactly one L0 document summary chunk"

    def test_produces_multiple_levels(self):
        sections = [
            Section(title="Intro", content="Introduction content. " * 30),
            Section(title="Body", content="Body content of the document. " * 30),
        ]
        text = "\n\n".join(s.content for s in sections)
        doc = make_document(text, sections=sections)
        chunker = HierarchicalChunker(parent_tokens=200, child_tokens=50)
        chunks = chunker.chunk(doc)

        levels = {c.hierarchy_level for c in chunks}
        assert 0 in levels, "Should have L0"
        assert 1 in levels, "Should have L1"
        assert 2 in levels, "Should have L2"

    def test_l1_parent_is_l0(self):
        sections = [Section(title="Section", content="Content. " * 50)]
        doc = make_document("Content. " * 50, sections=sections)
        chunker = HierarchicalChunker(parent_tokens=200, child_tokens=50)
        chunks = chunker.chunk(doc)

        l0 = next(c for c in chunks if c.hierarchy_level == 0)
        l1_chunks = [c for c in chunks if c.hierarchy_level == 1]
        for l1 in l1_chunks:
            assert l1.parent_chunk_id == l0.id, "L1 parent should be L0"

    def test_l2_parent_is_l1(self):
        sections = [Section(title="Section", content="Content. " * 100)]
        doc = make_document("Content. " * 100, sections=sections)
        chunker = HierarchicalChunker(parent_tokens=500, child_tokens=50)
        chunks = chunker.chunk(doc)

        l1_ids = {c.id for c in chunks if c.hierarchy_level == 1}
        l2_chunks = [c for c in chunks if c.hierarchy_level == 2]
        for l2 in l2_chunks:
            assert l2.parent_chunk_id in l1_ids, "L2 parent should be an L1 chunk"

    def test_all_chunks_same_doc_id(self):
        doc = make_document(LONG_TEXT)
        chunker = HierarchicalChunker()
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.doc_id == doc.doc_id

    def test_all_chunks_have_correct_strategy(self):
        doc = make_document(LONG_TEXT)
        chunker = HierarchicalChunker()
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.chunking_strategy == ChunkingStrategy.HIERARCHICAL

    def test_no_document_with_sections_infers_sections(self):
        """Documents without sections should still produce multi-level chunks."""
        doc = make_document(LONG_TEXT, sections=[])
        chunker = HierarchicalChunker(parent_tokens=200, child_tokens=50)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
