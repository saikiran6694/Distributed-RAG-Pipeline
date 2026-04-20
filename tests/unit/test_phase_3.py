"""
tests/unit/test_phase3.py

Unit tests for Phase 3 components:
  - PromptBuilder: token budgeting, citation numbering, chunk formatting
  - QueryDecomposer: fallback behavior, response parsing
  - GeneratorChunk: SSE serialization
"""

from __future__ import annotations

import json
import pytest

from query.context_expander import ExpandedChunk
from query.prompt_builder import PromptBuilder, ConversationTurn
from query.generator import GeneratorChunk
from query.decomposer import QueryDecomposer


# ── Helpers ───────────────────────────────────────────────────────

def make_chunk(
    chunk_id="c1", doc_id="d1", text="Some retrieved text.",
    score=0.9, source_url="http://docs.example.com/page",
    section_title="Introduction", hierarchy_level=0,
    parent_text=None, chunk_index=0,
) -> ExpandedChunk:
    return ExpandedChunk(
        chunk_id=chunk_id, doc_id=doc_id, text=text, score=score,
        source_url=source_url, section_title=section_title,
        parent_text=parent_text, page_number=None,
        hierarchy_level=hierarchy_level, chunk_index=chunk_index,
        dense_rank=1, sparse_rank=1,
    )


# ── PromptBuilder ─────────────────────────────────────────────────

class TestPromptBuilder:

    def test_empty_chunks_returns_no_context_message(self):
        builder = PromptBuilder()
        prompt = builder.build("What is RAG?", [])
        assert "No relevant context" in prompt.user
        assert prompt.chunks_used == 0
        assert prompt.citations == []

    def test_single_chunk_included(self):
        builder = PromptBuilder()
        chunk = make_chunk(text="RAG stands for Retrieval-Augmented Generation.")
        prompt = builder.build("What is RAG?", [chunk])
        assert prompt.chunks_used == 1
        assert "[1]" in prompt.user
        assert "RAG stands for" in prompt.user

    def test_citation_index_matches_passage_number(self):
        builder = PromptBuilder()
        chunks = [
            make_chunk(chunk_id="c1", text="First passage content."),
            make_chunk(chunk_id="c2", text="Second passage content."),
        ]
        prompt = builder.build("test query", chunks)
        assert prompt.citations[0]["index"] == 1
        assert prompt.citations[1]["index"] == 2

    def test_source_url_in_citation(self):
        builder = PromptBuilder()
        chunk = make_chunk(source_url="http://example.com/doc")
        prompt = builder.build("query", [chunk])
        assert prompt.citations[0]["source_url"] == "http://example.com/doc"

    def test_section_title_included_in_passage(self):
        builder = PromptBuilder()
        chunk = make_chunk(section_title="Refund Policy", text="Refunds take 5 days.")
        prompt = builder.build("refund?", [chunk])
        assert "Refund Policy" in prompt.user

    def test_parent_context_included_for_l2_chunks(self):
        builder = PromptBuilder()
        chunk = make_chunk(
            hierarchy_level=2,
            parent_text="This section covers the company refund policy in detail.",
            text="Refunds are processed within 5-7 business days.",
        )
        prompt = builder.build("refund?", [chunk])
        assert "Context:" in prompt.user
        assert "refund policy" in prompt.user.lower()

    def test_parent_context_not_included_for_l0_chunks(self):
        builder = PromptBuilder()
        chunk = make_chunk(
            hierarchy_level=0,
            parent_text="Some parent text that should not appear.",
            text="Document summary content.",
        )
        prompt = builder.build("query", [chunk])
        assert "Some parent text" not in prompt.user

    def test_token_budget_limits_chunks(self):
        builder = PromptBuilder(max_context_tokens=500)
        # Create many chunks — only some should fit
        chunks = [
            make_chunk(chunk_id=f"c{i}", text="word " * 100)
            for i in range(20)
        ]
        prompt = builder.build("query", chunks)
        assert prompt.chunks_used < 20
        assert prompt.chunks_total == 20

    def test_build_messages_includes_system_and_user(self):
        builder = PromptBuilder()
        chunk = make_chunk()
        prompt = builder.build("query", [chunk])
        messages = builder.build_messages(prompt)
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    def test_build_messages_includes_history(self):
        builder = PromptBuilder()
        prompt = builder.build("follow-up question", [make_chunk()])
        history = [
            ConversationTurn(role="user", content="previous question"),
            ConversationTurn(role="assistant", content="previous answer"),
        ]
        messages = builder.build_messages(prompt, history)
        assert len(messages) == 4   # system + 2 history + user
        assert messages[1]["content"] == "previous question"
        assert messages[2]["content"] == "previous answer"

    def test_token_estimate_is_positive(self):
        builder = PromptBuilder()
        assert builder._estimate_tokens("hello world") > 0

    def test_trim_to_tokens_shortens_long_text(self):
        builder = PromptBuilder()
        long_text = "word " * 500
        trimmed = builder._trim_to_tokens(long_text, 10)
        assert len(trimmed) < len(long_text)
        assert trimmed.endswith("…")


# ── GeneratorChunk SSE ────────────────────────────────────────────

class TestGeneratorChunk:

    def test_sse_format_valid_json(self):
        chunk = GeneratorChunk(delta="hello")
        sse = chunk.to_sse()
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        data = json.loads(sse[6:].strip())
        assert data["delta"] == "hello"
        assert data["done"] is False

    def test_final_chunk_includes_citations(self):
        citations = [{"index": 1, "source_url": "http://x.com", "section_title": None, "score": 0.9}]
        chunk = GeneratorChunk(delta="", done=True, citations=citations)
        sse = chunk.to_sse()
        data = json.loads(sse[6:].strip())
        assert data["done"] is True
        assert len(data["citations"]) == 1

    def test_non_final_chunk_has_empty_citations(self):
        chunk = GeneratorChunk(delta="some text", done=False)
        data = json.loads(chunk.to_sse()[6:].strip())
        assert data["citations"] == []


# ── QueryDecomposer fallback ──────────────────────────────────────

class TestQueryDecomposerFallback:
    """Test fallback behavior without an LLM (parse logic only)."""

    def test_parse_valid_json(self):
        decomposer = QueryDecomposer()
        result = decomposer._parse_response(
            '{"queries": ["What is RAG?", "How does retrieval work?"]}',
            "original query"
        )
        assert result == ["What is RAG?", "How does retrieval work?"]

    def test_parse_invalid_json_returns_original(self):
        decomposer = QueryDecomposer()
        result = decomposer._parse_response("not valid json at all", "original query")
        assert result == ["original query"]

    def test_parse_empty_queries_returns_original(self):
        decomposer = QueryDecomposer()
        result = decomposer._parse_response('{"queries": []}', "original query")
        assert result == ["original query"]

    def test_parse_strips_markdown_fences(self):
        decomposer = QueryDecomposer()
        raw = '```json\n{"queries": ["sub query one"]}\n```'
        result = decomposer._parse_response(raw, "original")
        assert result == ["sub query one"]

    def test_parse_filters_empty_strings(self):
        decomposer = QueryDecomposer()
        result = decomposer._parse_response(
            '{"queries": ["valid query", "", "  "]}',
            "original"
        )
        assert result == ["valid query"]

    @pytest.mark.asyncio
    async def test_short_query_skips_llm(self):
        """Queries with 6 or fewer words return immediately without LLM call."""
        decomposer = QueryDecomposer()
        result = await decomposer.decompose("What is RAG?")
        assert result == ["What is RAG?"]