"""
LLM-based query decomposition.

Replaces the regex heuristic from Phase 2 with an actual LLM call.
Detects compound queries and splits them into focused sub-queries that
each retrieve well from the vector index.

Examples:
  "What is the refund policy and how long does shipping take?"
  → ["What is the refund policy?", "How long does shipping take?"]

  "Compare Python and JavaScript for backend development"
  → ["What are Python's strengths for backend development?",
     "What are JavaScript's strengths for backend development?"]

  "What is RAG?"
  → ["What is RAG?"]   ← single, no decomposition needed

Design: one LLM call with a strict JSON response schema.
Falls back to the original query if the LLM call fails.
"""

from __future__ import annotations

import json
import logging
import re

from shared.config import get_settings
from shared.telemetry import traced_span

logger = logging.getLogger(__name__)
settings = get_settings()

_SYSTEM_PROMPT = """\
You are a query decomposition engine for a RAG system.

Your job: decide if a user query should be split into multiple focused
sub-queries for better retrieval, or kept as-is.

Rules:
- Split compound queries joined by "and", "also", "as well as", or multiple "?"
- Split comparison queries ("compare X and Y") into one query per subject
- Do NOT split simple single-topic queries
- Each sub-query must be self-contained and retrievable on its own
- Maximum 3 sub-queries — if more would be needed, keep the original
- Always return valid JSON, nothing else

Response format (strict JSON, no markdown):
{"queries": ["sub-query 1", "sub-query 2"]}

If no decomposition is needed:
{"queries": ["original query"]}
"""


class QueryDecomposer:
    """
    Decomposes complex queries into focused sub-queries using an LLM.

    Falls back gracefully to the original query if:
      - LLM call fails
      - Response is not valid JSON
      - LLM returns more than 3 sub-queries (safety limit)
    """

    def __init__(self):
        self._backend = settings.EMBED_BACKEND   # reuse same backend choice

    async def decompose(self, query: str) -> list[str]:
        """
        Returns a list of sub-queries. Always contains at least the original query.
        """
        # Short queries — skip LLM call entirely
        if len(query.split()) <= 6:
            return [query]

        with traced_span("decomposer.decompose", {"query_len": len(query)}):
            try:
                sub_queries = await self._call_llm(query)
                if not sub_queries or len(sub_queries) > 3:
                    return [query]
                logger.debug(
                    "Decomposed %r → %d sub-queries", query[:60], len(sub_queries)
                )
                return sub_queries
            except Exception as e:
                logger.warning("Query decomposition failed, using original: %s", e)
                return [query]

    async def _call_llm(self, query: str) -> list[str]:
        """Call the configured LLM backend for decomposition."""
        if settings.EMBED_BACKEND == "openai" and settings.OPENAI_API_KEY:
            return await self._call_openai(query)
        return await self._call_ollama(query)

    async def _call_openai(self, query: str) -> list[str]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        return self._parse_response(raw, query)

    async def _call_ollama(self, query: str) -> list[str]:
        import httpx
        async with httpx.AsyncClient(base_url=settings.OLLAMA_BASE_URL, timeout=15) as client:
            response = await client.post("/api/chat", json={
                "model": "llama3.2",
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                "stream": False,
                "options": {"temperature": 0},
            })
            response.raise_for_status()
            raw = response.json()["message"]["content"]
            return self._parse_response(raw, query)

    def _parse_response(self, raw: str, original: str) -> list[str]:
        """Parse JSON response, falling back to original query on any error."""
        try:
            # Strip markdown code fences if present
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(clean)
            queries = data.get("queries", [])
            # Filter empty strings, enforce non-empty list
            queries = [q.strip() for q in queries if q.strip()]
            return queries if queries else [original]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("Failed to parse decomposition response: %s | raw: %r", e, raw[:100])
            return [original]