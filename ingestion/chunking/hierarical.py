"""
Hierarchical chunker: produces a three-level chunk tree.

  L0 — Document summary  (one per document, ~200 tokens)
  L1 — Section chunks    (one per heading, ~1500 tokens)
  L2 — Paragraph chunks  (fine-grained, ~300 tokens, with overlap)

Parent-child relationships are stored in Postgres (parent_chunk_id).
At query time, the retriever fetches L2 chunks but the prompt builder
includes the L1 parent for context, solving the "orphaned chunk" problem
where a retrieved paragraph makes no sense without its surrounding section.

Best for: long-form documents (books, reports, legal docs, manuals).
"""

from __future__ import annotations

import logging
import textwrap
from uuid import uuid4

import tiktoken

from shared.config import get_settings
from shared.models import Chunk, ChunkingStrategy, ParsedDocument, Section

logger = logging.getLogger(__name__)
settings = get_settings()

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _TOKENIZER.decode(tokens[:max_tokens])


class HierarchicalChunker:
    """
    Three-level document chunker.

    Produces a parent-child tree stored in:
      - Qdrant: vectors for every level (all are searchable)
      - Postgres: parent_chunk_id FK enabling context expansion

    Args:
        parent_tokens: Target L1 section chunk size (default from config)
        child_tokens:  Target L2 paragraph chunk size (default from config)
        child_overlap: Token overlap between sibling L2 chunks
    """

    def __init__(
        self,
        parent_tokens: int | None = None,
        child_tokens:  int | None = None,
        child_overlap: int | None = None,
    ):
        self.parent_tokens = parent_tokens or settings.CHUNK_HIER_PARENT_TOKENS
        self.child_tokens  = child_tokens  or settings.CHUNK_HIER_CHILD_TOKENS
        # Overlap is 10% of child size by default
        self.child_overlap = child_overlap or max(30, self.child_tokens // 10)

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """
        Produce all chunks across all levels.
        Returns flattened list ordered: L0, then L1s and their L2 children.
        """
        if not document.raw_text.strip():
            return []

        all_chunks: list[Chunk] = []
        chunk_index = 0

        # ── Level 0: Document summary ─────────────────────────
        l0_summary = self._summarize_document(document)
        l0_chunk = Chunk(
            id=uuid4(),
            doc_id=document.doc_id,
            chunk_index=chunk_index,
            text=l0_summary,
            section_title="Document summary",
            source_url=document.source_url,
            hierarchy_level=0,
            chunking_strategy=ChunkingStrategy.HIERARCHICAL,
        )
        all_chunks.append(l0_chunk)
        chunk_index += 1

        # ── Levels 1 & 2: Sections and paragraphs ─────────────
        sections = document.sections if document.sections else self._infer_sections(document)

        for section in sections:
            # Level 1: Section chunk
            l1_text = self._build_l1_text(section)
            l1_chunk = Chunk(
                id=uuid4(),
                doc_id=document.doc_id,
                chunk_index=chunk_index,
                text=_truncate_to_tokens(l1_text, self.parent_tokens),
                section_title=section.title,
                source_url=document.source_url,
                parent_chunk_id=l0_chunk.id,    # L1 parent is L0 document
                hierarchy_level=1,
                chunking_strategy=ChunkingStrategy.HIERARCHICAL,
            )
            all_chunks.append(l1_chunk)
            chunk_index += 1

            # Level 2: Paragraph chunks (children of L1)
            l2_chunks = self._split_into_paragraphs(
                text=section.content,
                doc_id=document.doc_id,
                source_url=document.source_url,
                section_title=section.title,
                parent_id=l1_chunk.id,
                start_index=chunk_index,
            )
            all_chunks.extend(l2_chunks)
            chunk_index += len(l2_chunks)

        logger.debug(
            "Hierarchical chunker produced %d chunks (L0=%d, L1=%d, L2=%d) for doc_id=%s",
            len(all_chunks),
            1,
            len(sections),
            sum(1 for c in all_chunks if c.hierarchy_level == 2),
            document.doc_id,
        )
        return all_chunks

    # ─────────────────────────────────────────────────────────
    #  Level builders
    # ─────────────────────────────────────────────────────────

    def _summarize_document(self, document: ParsedDocument) -> str:
        """
        Build L0 document summary.
        For now: title + first N tokens of document text.
        In Phase 3 this can be upgraded to LLM-generated summary.
        """
        parts = []
        if document.title:
            parts.append(f"Title: {document.title}")
        if document.author:
            parts.append(f"Author: {document.author}")

        # Extract section titles as a structural overview
        if document.sections:
            titles = [s.title for s in document.sections[:10] if s.title]
            if titles:
                parts.append("Sections: " + "; ".join(titles))

        # First 200 tokens of main text as content preview
        preview_tokens = 200
        preview = _truncate_to_tokens(document.raw_text, preview_tokens)
        parts.append(preview)

        return "\n\n".join(parts)

    def _build_l1_text(self, section: Section) -> str:
        """Build L1 section chunk: title + full section content."""
        return f"{section.title}\n\n{section.content}"

    def _split_into_paragraphs(
        self,
        text: str,
        doc_id,
        source_url: str,
        section_title: str | None,
        parent_id,
        start_index: int,
    ) -> list[Chunk]:
        """
        Split section content into L2 paragraph chunks with overlap.
        Uses token-aware sliding window on the encoded text.
        """
        if not text.strip():
            return []

        tokens = _TOKENIZER.encode(text)
        chunks: list[Chunk] = []
        start = 0
        local_index = 0

        while start < len(tokens):
            end = min(start + self.child_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = _TOKENIZER.decode(chunk_tokens).strip()

            if chunk_text:
                chunks.append(Chunk(
                    id=uuid4(),
                    doc_id=doc_id,
                    chunk_index=start_index + local_index,
                    text=chunk_text,
                    section_title=section_title,
                    source_url=source_url,
                    parent_chunk_id=parent_id,
                    hierarchy_level=2,
                    chunking_strategy=ChunkingStrategy.HIERARCHICAL,
                    chunk_overlap_tokens=self.child_overlap if local_index > 0 else 0,
                ))
                local_index += 1

            start += self.child_tokens - self.child_overlap

        return chunks

    def _infer_sections(self, document: ParsedDocument) -> list[Section]:
        """
        Fallback when document has no heading structure.
        Splits raw text into ~parent_tokens sized sections.
        """
        tokens = _TOKENIZER.encode(document.raw_text)
        sections: list[Section] = []
        start = 0
        idx = 0

        while start < len(tokens):
            end = min(start + self.parent_tokens, len(tokens))
            text = _TOKENIZER.decode(tokens[start:end])
            sections.append(Section(
                title=f"Section {idx + 1}",
                content=text,
            ))
            start = end
            idx += 1

        return sections