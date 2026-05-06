"""
LLM generator: takes a built prompt and streams the response.

Supports:
  - OpenAI (gpt-4o, gpt-4o-mini) — streaming via SSE
  - Ollama (llama3.2, mistral, etc.) — streaming via chunked HTTP
  - Dry-run mode for testing without an LLM

Yields GeneratorChunk objects for Server-Sent Events streaming.
The final chunk carries citations and usage stats.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from query.prompt_builder import BuiltPrompt
from shared.config import get_settings
from shared.telemetry import traced_span

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class GeneratorChunk:
    """
    A single streamed chunk from the LLM.
    delta: the new text fragment (empty string on final chunk)
    done: True only on the last chunk
    citations: populated only on the final chunk
    """

    delta: str
    done: bool = False
    citations: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    groundedness: dict | None = field(default=None)

    def to_sse(self) -> str:
        """Format as a Server-Sent Event string."""
        import json

        data = {
            "delta": self.delta,
            "done": self.done,
            "citations": self.citations if self.done else [],
            "groundedness": self.groundedness if self.done else None,
        }
        if self.done and self.usage:
            data["usage"] = self.usage
        return f"data: {json.dumps(data)}\n\n"


class LLMGenerator:
    """
    Streams LLM responses from a BuiltPrompt.

    Usage:
        generator = LLMGenerator()
        async for chunk in generator.stream(prompt):
            yield chunk.to_sse()
    """

    async def stream(
        self,
        prompt: BuiltPrompt,
        history: list[dict] | None = None,
    ) -> AsyncIterator[GeneratorChunk]:
        """
        Stream response chunks. Always ends with a chunk where done=True.
        """
        from query.prompt_builder import PromptBuilder

        builder = PromptBuilder()
        messages = builder.build_messages(prompt, history)

        t0 = time.monotonic()

        with traced_span("generator.stream", {"backend": settings.EMBED_BACKEND}):
            try:
                if settings.EMBED_BACKEND == "openai" and settings.OPENAI_API_KEY:
                    async for chunk in self._stream_openai(messages, prompt, t0):
                        yield chunk
                elif settings.GROQ_API_KEY:
                    async for chunk in self._stream_groq(messages, prompt, t0):
                        yield chunk
                else:
                    async for chunk in self._stream_ollama(messages, prompt, t0):
                        yield chunk
            except Exception as e:
                logger.error("LLM generation failed: %s", e)
                yield GeneratorChunk(
                    delta=f"\n\n[Generation failed: {e}]",
                    done=True,
                    citations=prompt.citations,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )

    # ─────────────────────────────────────────────────────────
    #  OpenAI streaming
    # ─────────────────────────────────────────────────────────

    async def _stream_openai(
        self,
        messages: list[dict],
        prompt: BuiltPrompt,
        t0: float,
    ) -> AsyncIterator[GeneratorChunk]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=_RESPONSE_BUDGET,
            stream=True,
        )

        usage = {}
        async for event in stream:
            delta = event.choices[0].delta.content or ""
            finish = event.choices[0].finish_reason

            if finish == "stop":
                if hasattr(event, "usage") and event.usage:
                    usage = {
                        "prompt_tokens": event.usage.prompt_tokens,
                        "completion_tokens": event.usage.completion_tokens,
                    }
                yield GeneratorChunk(
                    delta="",
                    done=True,
                    citations=prompt.citations,
                    usage=usage,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
                return

            if delta:
                yield GeneratorChunk(delta=delta)

        # Safety: ensure done is always emitted
        yield GeneratorChunk(
            delta="",
            done=True,
            citations=prompt.citations,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # ─────────────────────────────────────────────────────────
    #  Ollama streaming
    # ─────────────────────────────────────────────────────────

    async def _stream_ollama(
        self,
        messages: list[dict],
        prompt: BuiltPrompt,
        t0: float,
    ) -> AsyncIterator[GeneratorChunk]:
        import json as _json

        import httpx

        async with httpx.AsyncClient(
            base_url=settings.OLLAMA_BASE_URL,
            timeout=120,
        ) as client:
            async with client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": settings.OLLAMA_MODEL_NAME,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": 0.2, "num_predict": _RESPONSE_BUDGET},
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue

                    delta = event.get("message", {}).get("content", "")
                    done = event.get("done", False)

                    if done:
                        yield GeneratorChunk(
                            delta="",
                            done=True,
                            citations=prompt.citations,
                            usage={
                                "prompt_tokens": event.get("prompt_eval_count", 0),
                                "completion_tokens": event.get("eval_count", 0),
                            },
                            latency_ms=(time.monotonic() - t0) * 1000,
                        )
                        return

                    if delta:
                        yield GeneratorChunk(delta=delta)

        yield GeneratorChunk(
            delta="",
            done=True,
            citations=prompt.citations,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # ─────────────────────────────────────────────────────────
    #  GROQ streaming
    # ─────────────────────────────────────────────────────────

    async def _stream_groq(
        self,
        messages: list[dict],
        prompt: BuiltPrompt,
        t0: float,
    ) -> AsyncIterator[GeneratorChunk]:
        """
        Stream via Groq's OpenAI-compatible API.
        Groq runs open-source models (Llama, Mixtral) on custom hardware —
        typically 10-20x faster than Ollama on CPU.

        """
        from groq import AsyncGroq

        client = AsyncGroq(api_key=settings.GROQ_API_KEY)

        response_stream = await client.chat.completions.create(
            model=settings.GROQ_LLAMA_MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=_RESPONSE_BUDGET,
            stream=True,
        )

        async for event in response_stream:
            delta = event.choices[0].delta.content or ""
            finish = event.choices[0].finish_reason

            if finish == "stop":
                # Groq sends usage on the last chunk's x_groq field
                x_groq = getattr(event, "x_groq", None)

                if x_groq and hasattr(x_groq, "usage"):
                    usage = {
                        "prompt_tokens": x_groq.usage.prompt_tokens,
                        "completion_tokens": x_groq.usage.completion_tokens,
                    }
                yield GeneratorChunk(
                    delta="",
                    done=True,
                    citations=prompt.citations,
                    usage=usage,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
                return

            if delta:
                yield GeneratorChunk(delta=delta)

        # Safety: ensure done is always emitted
        yield GeneratorChunk(
            delta="",
            done=True,
            citations=prompt.citations,
            latency_ms=(time.monotonic() - t0) * 1000,
        )


# Response budget constant (shared with prompt_builder)
_RESPONSE_BUDGET = 1500
