"""
Extracts structured metadata from a document using a single LLM call.
Runs after StorageWriter.store() completes — non-blocking enrichment.

Extracts:
  - title        — inferred or confirmed document title
  - summary      — 2-3 sentence summary
  - topics       — list of subject areas ["distributed systems", "Raft"]
  - entities     — named entities ["Apache Kafka", "DCTCP", "Google"]
  - doc_category — research_paper | documentation | article | report | other
  - language     — ISO 639-1 code ("en", "fr", etc.)
  - key_questions — 3 questions this document can answer

Design:
  - One structured LLM call with JSON response format
  - Pydantic v2 validates and coerces the LLM output
  - Falls back gracefully — tagging failure never breaks ingestion
  - Uses the fastest available LLM backend (groq > openai > ollama)
  - Input is truncated to first ~3000 tokens to keep latency low
"""

from __future__ import annotations

import json
import logging
import re
import time

from pydantic import BaseModel, Field, field_validator

from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Max chars of document text to pass to the tagger (~3000 tokens)
_MAX_INPUT_CHARS = 12_000

_SYSTEM_PROMPT = """\
You are a document metadata extractor. Given a document excerpt, extract structured
metadata and return ONLY valid JSON with no markdown, no explanation, nothing else.

Required JSON format:
{
  "title": "<document title, infer if not explicit>",
  "summary": "<2-3 sentence summary of the document's main content>",
  "topics": ["<topic 1>", "<topic 2>", "<topic 3>"],
  "entities": ["<entity 1>", "<entity 2>"],
  "doc_category": "<one of: research_paper, documentation, article, report, book_chapter, other>",
  "language": "<ISO 639-1 code, e.g. en>",
  "key_questions": [
    "<question this document answers>",
    "<question this document answers>",
    "<question this document answers>"
  ]
}

Rules:
- topics: 3-6 subject areas, lowercase, general to specific
- entities: proper nouns only — tools, companies, algorithms, people, protocols
- key_questions: real questions a user might search for that this doc answers
- doc_category: pick the single best fit
- summary: factual, no opinions, covers the main argument or content
"""


class DocumentTags(BaseModel):
    """
    Structured metadata extracted from a document by LLM.
    Stored as JSONB in the documents table.
    """

    title: str = Field(default="Untitled")
    summary: str = Field(default="")
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    doc_category: str = Field(default="other")
    language: str = Field(default="en")
    key_questions: list[str] = Field(default_factory=list)

    @field_validator("topics", "entities", "key_questions", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v or []

    @field_validator("doc_category", mode="before")
    @classmethod
    def normalise_category(cls, v):
        valid = {"research_paper", "documentation", "article", "report", "book_chapter", "other"}
        v = str(v).lower().strip()
        return v if v in valid else "other"

    @field_validator("language", mode="before")
    @classmethod
    def normalise_language(cls, v):
        return str(v).lower()[:2] if v else "en"

    @field_validator("topics", "entities", mode="after")
    @classmethod
    def lowercase_and_dedupe(cls, v):
        seen, result = set(), []
        for item in v:
            key = item.lower().strip()
            if key and key not in seen:
                seen.add(key)
                result.append(item.strip())
        return result[:10]  # cap at 10 items

    @field_validator("key_questions", mode="after")
    @classmethod
    def cap_questions(cls, v):
        return v[:5]  # max 5 questions

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "summary": self.summary,
            "topics": self.topics,
            "entities": self.entities,
            "doc_category": self.doc_category,
            "language": self.language,
            "key_questions": self.key_questions,
        }


class AutoTagger:
    """
    Extracts DocumentTags from raw document text using a single LLM call.

    Usage:
        tagger = AutoTagger()
        tags = await tagger.tag(raw_text, existing_title="My Doc")
        # tags.topics → ["distributed systems", "consensus"]
        # tags.summary → "This paper presents..."
    """

    async def tag(
        self,
        raw_text: str,
        existing_title: str | None = None,
    ) -> DocumentTags | None:
        """
        Extract tags from document text.
        Returns None if tagging fails — caller should handle gracefully.
        """
        if not raw_text or len(raw_text.strip()) < 50:
            return None

        t0 = time.monotonic()
        try:
            # Truncate to keep LLM call fast
            text = raw_text[:_MAX_INPUT_CHARS]
            title_hint = (
                f"Note: The document may be titled '{existing_title}'.\n\n"
                if existing_title
                else ""
            )
            user_message = f"{title_hint}{text}"

            raw = await self._call_llm(user_message)
            tags = self._parse(raw)

            # Use existing title if LLM produced a generic one
            if existing_title and tags.title in ("Untitled", "", "Document"):
                tags.title = existing_title

            elapsed = (time.monotonic() - t0) * 1000
            logger.info(
                "AutoTagger: doc='%s' category=%s topics=%d entities=%d latency=%.0fms",
                tags.title[:50],
                tags.doc_category,
                len(tags.topics),
                len(tags.entities),
                elapsed,
            )
            return tags

        except Exception as e:
            logger.warning("AutoTagger failed (non-critical): %s", e)
            return None

    # ── LLM backends ─────────────────────────────────────────────

    async def _call_llm(self, user_message: str) -> str:
        if settings.GROQ_API_KEY:
            return await self._call_groq(user_message)
        if settings.OPENAI_API_KEY:
            return await self._call_openai(user_message)
        return await self._call_ollama(user_message)

    async def _call_groq(self, user_message: str) -> str:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        response = await client.chat.completions.create(
            model=settings.GROQ_LLAMA_MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=600,
        )
        return response.choices[0].message.content

    async def _call_openai(self, user_message: str) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    async def _call_ollama(self, user_message: str) -> str:
        import httpx

        async with httpx.AsyncClient(base_url=settings.OLLAMA_BASE_URL, timeout=30) as client:
            response = await client.post(
                "/api/chat",
                json={
                    "model": settings.OLLAMA_MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "options": {"temperature": 0},
                },
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

    # ── Response parsing ──────────────────────────────────────────

    def _parse(self, raw: str) -> DocumentTags:
        """Parse and validate LLM JSON response through Pydantic."""
        try:
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(clean)
            return DocumentTags(**data)
        except Exception as e:
            logger.debug("AutoTagger parse failed: %s | raw=%r", e, raw[:200])
            return DocumentTags()  # empty tags rather than crashing
