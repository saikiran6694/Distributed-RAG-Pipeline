"""
Scores the groundedness of a generated answer against retrieved chunks.

Option A: Always returns a confidence score (0-100%) shown in the UI.
Option C: Only surfaces a warning + specific claim when confidence is low.

Design:
  - One structured LLM call (fast model — gpt-4o-mini / llama-3.1-8b-instant)
  - Passes the full answer + all citation chunks as context
  - Returns: overall_confidence (0-100), ungrounded_claim (str | None)
  - Falls back gracefully — if LLM call fails, returns None (UI shows nothing)
  - Runs AFTER generation, does not block the streaming response

LLM is asked to evaluate:
  1. What fraction of claims in the answer are directly supported by chunks?
  2. Is there any specific claim that appears to be fabricated or inferred
     beyond what the chunks support?
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_SYSTEM_PROMPT = """\
You are a factual grounding evaluator for a RAG (Retrieval-Augmented Generation) system.

You will be given:
1. An AI-generated answer
2. The source passages that were retrieved to generate it

Your job: evaluate how well the answer is grounded in the source passages.

Guidelines:
- First, internally identify all major factual claims in the answer
- A claim is "grounded" if it is explicitly stated or directly implied by the passages
- A claim is "ungrounded" if any part of it is not supported by the passages
- If a claim is partially supported, treat it as ungrounded
- Paraphrasing is allowed if meaning is preserved
- “Reasonable inference” must follow directly from the text and not require outside knowledge
- If passages conflict, claims should be marked ungrounded unless the answer reflects that uncertainty
- Do NOT evaluate completeness, only grounding

Respond ONLY with valid JSON:
{
  "overall_confidence": <integer 0-100>,
  "reasoning": "<one sentence explaining the score>",
  "ungrounded_claim": "<most significant unsupported claim, or null if fully grounded>"
}

Scoring guide:
  90-100: All claims directly supported by passages
  70-89: Minor inference, mostly supported
  50-69: Some unsupported extensions
  0-49: Significant hallucination
"""

# Threshold below which we show the warning banner (Option C)
CONFIDENCE_WARNING_THRESHOLD = 70


@dataclass
class GroundednessResult:
    """Result of hallucination detection."""

    overall_confidence: int  # 0-100
    reasoning: str  # one-sentence explanation
    ungrounded_claim: str | None  # specific problematic claim, or None
    latency_ms: float
    skipped: bool = False  # True if detection was skipped/failed

    @property
    def show_warning(self) -> bool:
        """Whether to show the warning banner (Option C)."""
        return not self.skipped and self.overall_confidence < CONFIDENCE_WARNING_THRESHOLD

    @property
    def confidence_label(self) -> str:
        """
        Human-readable label for the confidence score.
        """
        if self.overall_confidence >= 90:
            return "Very High"
        if self.overall_confidence >= 75:
            return "High"
        if self.overall_confidence >= 60:
            return "Moderate"
        if self.overall_confidence >= 40:
            return "Low"
        return "Very Low"

    def to_dict(self) -> dict:
        return {
            "overall_confidence": self.overall_confidence,
            "confidence_label": self.confidence_label,
            "reasoning": self.reasoning,
            "ungrounded_claim": self.ungrounded_claim,
            "show_warning": self.show_warning,
            "skipped": self.skipped,
        }


class HallucinationDetector:
    """
    Evaluates how well a generated answer is grounded in retrieved chunks.

    Usage:
        detector = HallucinationDetector()
        result = await detector.check(answer, citations)
        # result.overall_confidence → 0-100
        # result.show_warning       → bool (Option C trigger)
        # result.ungrounded_claim   → str | None
    """

    async def check(
        self,
        answer: str,
        citations: list[dict],  # list of citation dicts from BuiltPrompt
    ) -> GroundednessResult:
        """
        Score the answer against the citation chunks.

        Falls back to a skipped result on any failure — hallucination
        detection is non-critical and must never break the main response.
        """
        if not answer.strip() or not citations:
            return self._skipped()

        # Very short answers don't need checking
        if len(answer.split()) < 15:
            return GroundednessResult(
                overall_confidence=95,
                reasoning="Answer is too short to meaningfully evaluate.",
                ungrounded_claim=None,
                latency_ms=0,
            )

        t0 = time.monotonic()
        try:
            result = await self._call_llm(answer, citations)
            result.latency_ms = (time.monotonic() - t0) * 1000
            logger.debug(
                "Hallucination check: confidence=%d%% latency=%.0fms warning=%s",
                result.overall_confidence,
                result.latency_ms,
                result.show_warning,
            )
            return result
        except Exception as e:
            logger.warning("Hallucination detection failed (non-critical): %s", e)
            return self._skipped()

    async def _call_llm(self, answer: str, citations: list[dict]) -> GroundednessResult:
        """Build prompt and call the LLM."""
        user_message = self._build_prompt(answer, citations)

        if settings.OPENAI_API_KEY:
            raw = await self._call_openai(user_message)
        elif settings.GROQ_API_KEY:
            raw = await self._call_groq(user_message)
        else:
            raw = await self._call_ollama(user_message)

        return self._parse_response(raw)

    def _build_prompt(self, answer: str, citations: list[dict]) -> str:
        """Build the user message with answer + citation passages."""
        lines = ["## Generated Answer\n", answer.strip(), "\n## Source Passages\n"]
        for c in citations:
            idx = c.get("index", "?")
            source = c.get("source_url", "unknown")
            section = c.get("section_title", "")
            # chunk_text is the passage text — stored in citation dict
            text = c.get("text", c.get("chunk_text", ""))
            header = f"[{idx}] {section or source}"
            lines.append(f"{header}\n{text[:800]}\n")

        return "\n".join(lines)

    async def _call_openai(self, user_message: str) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

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
            max_tokens=300,
        )
        return response.choices[0].message.content

    async def _call_ollama(self, user_message: str) -> str:
        import httpx

        async with httpx.AsyncClient(base_url=settings.OLLAMA_BASE_URL, timeout=15) as client:
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

    def _parse_response(self, raw: str) -> GroundednessResult:
        try:
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(clean)

            confidence = int(data.get("overall_confidence", 0))
            confidence = max(0, min(100, confidence))  # clamp to 0-100

            return GroundednessResult(
                overall_confidence=confidence,
                reasoning=data.get("reasoning", ""),
                ungrounded_claim=data.get("ungrounded_claim") or None,
                latency_ms=0,
            )
        except Exception as e:
            logger.debug("Failed to parse hallucination response: %s | raw=%r", e, raw[:100])
            return self._skipped()

    def _skipped(self) -> GroundednessResult:
        return GroundednessResult(
            overall_confidence=0,
            reasoning="",
            ungrounded_claim=None,
            latency_ms=0,
            skipped=True,
        )
