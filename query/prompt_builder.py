"""
Assembles retrieved, expanded chunks into a structured LLM prompt.

Responsibilities:
  - Token budget management: never exceed the context window
  - Citation numbering: each chunk gets a [1], [2] reference
  - Hierarchy-aware formatting: L1 context shown as preamble for L2 chunks
  - System prompt injection
  - Conversation history support (multi-turn)

Token budget strategy:
  Total context window (e.g. 128k for gpt-4o) minus:
    - System prompt tokens (~300)
    - Query tokens
    - Response budget (reserved for generation, e.g. 1500)
    = Available for retrieved chunks

  Chunks are included highest-score-first until budget is exhausted.
  Each chunk is trimmed if it alone exceeds the remaining budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from query.context_expander import ExpandedChunk
from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Approximate token budget constants
_SYSTEM_PROMPT_TOKENS  = 350
_RESPONSE_BUDGET       = 1500
_CHARS_PER_TOKEN       = 4    # rough approximation, avoids tiktoken network dep


_SYSTEM_PROMPT = """\
You are a precise question-answering assistant. Answer the user's question \
using ONLY the provided context passages. Each passage is labelled [1], [2], etc.

Rules:
- Cite your sources inline using [N] notation, e.g. "The refund window is 30 days [1]."
- If multiple passages support a claim, cite all of them: [1][3]
- If the context does not contain enough information to answer, say so clearly
- Do not make up facts or draw on knowledge outside the provided passages
- Keep your answer concise and well-structured
"""


@dataclass
class BuiltPrompt:
    """
    Ready-to-send prompt for the LLM.
    """
    system:        str
    user:          str                    # query + context passages
    citations:     list[dict]             # [{index, source_url, section_title, score}]
    chunks_used:   int
    chunks_total:  int                    # how many were available before trimming
    tokens_used:   int                    # estimated


@dataclass
class ConversationTurn:
    role:    str   # "user" | "assistant"
    content: str


class PromptBuilder:
    """
    Builds LLM prompts from retrieved chunks with token budget management.

    Usage:
        builder = PromptBuilder(max_context_tokens=8000)
        prompt = builder.build(query, expanded_chunks)
    """

    def __init__(self, max_context_tokens: int = 8000):
        self._max_context = max_context_tokens

    def build(
        self,
        query:   str,
        chunks:  list[ExpandedChunk],
        history: list[ConversationTurn] | None = None,
    ) -> BuiltPrompt:
        """
        Build a complete prompt from query + retrieved chunks.

        Chunks are included highest-score-first within the token budget.
        Each chunk is formatted with its citation number and section context.
        """
        if not chunks:
            return BuiltPrompt(
                system=_SYSTEM_PROMPT,
                user=f"Question: {query}\n\nNo relevant context was found.",
                citations=[],
                chunks_used=0,
                chunks_total=0,
                tokens_used=self._estimate_tokens(_SYSTEM_PROMPT + query),
            )

        # Token budget for context passages
        query_tokens   = self._estimate_tokens(query)
        history_tokens = sum(self._estimate_tokens(t.content) for t in (history or []))
        available = (
            self._max_context
            - _SYSTEM_PROMPT_TOKENS
            - query_tokens
            - history_tokens
            - _RESPONSE_BUDGET
        )
        available = max(available, 500)   # always allow at least a few chunks

        # Build context passages within budget
        passages:  list[str]  = []
        citations: list[dict] = []
        tokens_used = 0

        for i, chunk in enumerate(chunks, start=1):
            passage = self._format_chunk(i, chunk)
            chunk_tokens = self._estimate_tokens(passage)

            if tokens_used + chunk_tokens > available:
                # Trim this chunk to fit remaining budget
                remaining = available - tokens_used
                if remaining < 50:
                    break   # not worth adding a tiny fragment
                passage = self._trim_to_tokens(passage, remaining)
                chunk_tokens = self._estimate_tokens(passage)

            passages.append(passage)
            citations.append({
                "index":         i,
                "source_url":    chunk.source_url,
                "section_title": chunk.section_title,
                "score":         chunk.score,
                "chunk_id":      chunk.chunk_id,
            })
            tokens_used += chunk_tokens

            if tokens_used >= available:
                break

        context_block = "\n\n".join(passages)
        user_message  = f"Context passages:\n\n{context_block}\n\n---\n\nQuestion: {query}"

        return BuiltPrompt(
            system=_SYSTEM_PROMPT,
            user=user_message,
            citations=citations,
            chunks_used=len(passages),
            chunks_total=len(chunks),
            tokens_used=tokens_used + _SYSTEM_PROMPT_TOKENS + query_tokens,
        )

    def build_messages(
        self,
        prompt:  BuiltPrompt,
        history: list[ConversationTurn] | None = None,
    ) -> list[dict]:
        """
        Build the messages list for the LLM API call.
        Includes conversation history for multi-turn support.
        """
        messages = [{"role": "system", "content": prompt.system}]
        for turn in (history or []):
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": prompt.user})
        return messages

    # ─────────────────────────────────────────────────────────
    #  Formatting helpers
    # ─────────────────────────────────────────────────────────

    def _format_chunk(self, index: int, chunk: ExpandedChunk) -> str:
        """
        Format a single chunk as a numbered passage.

        For L2 (paragraph) chunks: prepend L1 parent context so the
        passage makes sense without surrounding text.

        Example output:
            [1] Section: Refund Policy
            Context: Our return and refund policy covers all purchases made...
            ---
            Refunds are processed within 5-7 business days of receiving the item.
        """
        lines = [f"[{index}]"]

        if chunk.section_title:
            lines.append(f"Section: {chunk.section_title}")

        if chunk.parent_text and chunk.hierarchy_level == 2:
            # Trim parent context to ~200 tokens to save budget
            parent = self._trim_to_tokens(chunk.parent_text.strip(), 200)
            lines.append(f"Context: {parent}")
            lines.append("---")

        lines.append(chunk.text.strip())

        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token estimate: 1 token ≈ 4 characters.
        """
        return max(1, len(text) // _CHARS_PER_TOKEN)

    def _trim_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Trim text to approximately max_tokens tokens.
        """
        max_chars = max_tokens * _CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text
        # Trim at word boundary
        trimmed = text[:max_chars]
        last_space = trimmed.rfind(" ")
        if last_space > max_chars * 0.8:
            trimmed = trimmed[:last_space]
        return trimmed + "…"