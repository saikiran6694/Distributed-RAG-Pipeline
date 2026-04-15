"""
All Pydantic models used across the ingestion pipeline.
These are the contracts between stages — change carefully.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


# ─────────────────────────────────────────────────────────────────
#  Enums (mirror the Postgres ENUM types)
# ─────────────────────────────────────────────────────────────────

class DocType(str, Enum):
    PDF      = "pdf"
    HTML     = "html"
    DOCX     = "docx"
    MARKDOWN = "markdown"
    TXT      = "txt"
    CODE     = "code"
    UNKNOWN  = "unknown"


class DocStatus(str, Enum):
    PENDING           = "pending"
    QUEUED            = "queued"
    PARSING           = "parsing"
    CHUNKING          = "chunking"
    EMBEDDING         = "embedding"
    PARTIALLY_INDEXED = "partially_indexed"
    INDEXED           = "indexed"
    FAILED            = "failed"
    DELETED           = "deleted"


class ChunkStatus(str, Enum):
    PENDING_VECTOR = "pending_vector"
    INDEXED        = "indexed"
    STALE          = "stale"
    FAILED         = "failed"


class ChunkingStrategy(str, Enum):
    FIXED        = "fixed"
    SEMANTIC     = "semantic"
    HIERARCHICAL = "hierarchical"


# ─────────────────────────────────────────────────────────────────
#  Kafka message: raw-documents topic
# ─────────────────────────────────────────────────────────────────

class DocumentIngestionMessage(BaseModel):
    """
    Published to Kafka topic 'raw-documents' by the intake validator.
    Workers consume this message to begin parsing.
    """
    message_id:   UUID     = Field(default_factory=uuid4)
    doc_id:       UUID
    source_url:   str
    source_type:  str                           # 's3', 'http', 'notion', 'local'
    doc_type:     DocType
    byte_size:    int
    content_hash: str                           # SHA-256 of raw file bytes
    is_update:    bool     = False              # True = delete old chunks first
    previous_hash: str | None = None
    enqueued_at:  datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retry_count:  int      = 0
    schema_version: int    = 1

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────────────────────────────
#  Parser output: normalized document
# ─────────────────────────────────────────────────────────────────

class Section(BaseModel):
    """A heading-delimited section within a parsed document."""
    title:       str
    content:     str
    start_page:  int | None = None
    end_page:    int | None = None
    level:       int        = 1          # heading level (h1=1, h2=2, …)


class ParsedDocument(BaseModel):
    """
    Output of any parser worker.
    Uniform contract regardless of input format (PDF/HTML/DOCX/MD).
    """
    doc_id:       UUID
    source_url:   str
    doc_type:     DocType
    raw_text:     str
    sections:     list[Section]          = Field(default_factory=list)
    tables:       list[str]              = Field(default_factory=list)  # markdown-serialized
    page_count:   int | None             = None
    title:        str | None             = None
    author:       str | None             = None
    metadata:     dict[str, Any]         = Field(default_factory=dict)
    parse_errors: list[str]              = Field(default_factory=list)  # non-fatal errors
    parsed_at:    datetime               = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field
    @property
    def content_hash(self) -> str:
        """SHA-256 of the normalized text content."""
        return hashlib.sha256(self.raw_text.encode()).hexdigest()

    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.raw_text)

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────────────────────────────
#  Chunker output: individual chunk
# ─────────────────────────────────────────────────────────────────

class Chunk(BaseModel):
    """
    A single text chunk ready for embedding.
    id is derived from content_hash for idempotent Qdrant upserts.
    """
    id:                   UUID
    doc_id:               UUID
    chunk_index:          int
    text:                 str

    # Structural metadata (stored in Qdrant payload + Postgres)
    section_title:        str | None    = None
    page_number:          int | None    = None
    source_url:           str
    parent_chunk_id:      UUID | None   = None
    hierarchy_level:      int           = 0      # 0=doc, 1=section, 2=paragraph

    # Provenance
    chunking_strategy:    ChunkingStrategy
    chunk_overlap_tokens: int           = 0

    # Set after embedding
    embedding_model:      str | None    = None
    embedding_model_version: str | None = None
    embedding_dim:        int | None    = None
    vector:               list[float] | None = None

    schema_version:       int           = 1

    @computed_field
    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.text)

    @model_validator(mode="before")
    @classmethod
    def derive_id_from_hash(cls, values: dict) -> dict:
        """Derive chunk UUID from content hash for idempotency."""
        if "id" not in values and "text" in values:
            text_hash = hashlib.sha256(values["text"].encode()).hexdigest()
            values["id"] = UUID(text_hash[:32])
        return values

    def to_qdrant_payload(self) -> dict[str, Any]:
        """Serialize to Qdrant point payload (no vector — stored separately)."""
        return {
            "doc_id":                 str(self.doc_id),
            "chunk_index":            self.chunk_index,
            "section_title":          self.section_title,
            "page_number":            self.page_number,
            "source_url":             self.source_url,
            "parent_chunk_id":        str(self.parent_chunk_id) if self.parent_chunk_id else None,
            "hierarchy_level":        self.hierarchy_level,
            "chunking_strategy":      self.chunking_strategy,
            "embedding_model":        self.embedding_model,
            "embedding_model_version": self.embedding_model_version,
            "content_hash":           self.content_hash,
            "char_count":             self.char_count,
            "schema_version":         self.schema_version,
        }

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────────────────────────────
#  DLQ event
# ─────────────────────────────────────────────────────────────────

class DLQEvent(BaseModel):
    doc_id:          UUID | None = None
    kafka_topic:     str
    kafka_partition: int | None  = None
    kafka_offset:    int | None  = None
    error_type:      str
    error_message:   str
    payload:         dict[str, Any]
    retry_count:     int         = 0
