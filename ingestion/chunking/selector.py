"""
Selects the right chunking strategy for a given document type.

Strategy rationale:
  PDF / DOCX  → Hierarchical
    Long-form, structured documents with headings. The L0/L1/L2 tree
    preserves section context and enables parent-context expansion at
    query time. Most important for recall quality on long documents.

  HTML / Markdown → Semantic
    Narrative web content and documentation. Semantic chunking splits
    at topic boundaries rather than arbitrary token counts, producing
    coherent chunks that each cover one idea.

  TXT / Code → Fixed
    Uniform, unstructured content without clear topic boundaries.
    Fixed-size with overlap is fast and predictable.
"""

from __future__ import annotations

import logging

from ingestion.chunking.fixed import FixedChunker
from ingestion.chunking.hierarical import HierarchicalChunker
from ingestion.chunking.semantic import SemanticChunker
from shared.config import get_settings
from shared.models import DocType

logger = logging.getLogger(__name__)
settings = get_settings()

# DocType → strategy name
_STRATEGY_MAP: dict[str, str] = {
    DocType.PDF:      "hierarchical",
    DocType.DOCX:     "hierarchical",
    DocType.HTML:     "semantic",
    DocType.MARKDOWN: "semantic",
    DocType.TXT:      "fixed",
    DocType.CODE:     "fixed",
    DocType.UNKNOWN:  "fixed",
}


class ChunkingStrategySelector:
    """
    Returns the appropriate chunker instance for a given document type.

    Usage:
        selector = ChunkingStrategySelector()
        chunker  = selector.get(doc_type)
        chunks   = chunker.chunk(parsed_document)
    """

    def get(self, doc_type: str | DocType):
        """Return the right chunker for this document type."""
        # Normalise to string value
        dt = doc_type.value if hasattr(doc_type, "value") else str(doc_type)
        strategy = _STRATEGY_MAP.get(dt, "fixed")

        if strategy == "hierarchical":
            logger.debug("Using HierarchicalChunker for doc_type=%s", dt)
            return HierarchicalChunker()

        if strategy == "semantic":
            logger.debug("Using SemanticChunker for doc_type=%s", dt)
            return SemanticChunker()

        logger.debug("Using FixedChunker for doc_type=%s", dt)
        return FixedChunker()

    def strategy_name(self, doc_type: str | DocType) -> str:
        dt = doc_type.value if hasattr(doc_type, "value") else str(doc_type)
        return _STRATEGY_MAP.get(dt, "fixed")