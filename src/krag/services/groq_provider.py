"""Groq-backed LLM provider (production path).

Implements the krag LLMProvider protocol over Groq's OpenAI-compatible
endpoint using the Secure Vault surrogate pattern (see groq_auth.py).
Records prompt/completion/reasoning tokens and cost per call.

Cost: the key used here is on Groq's free tier, so billed cost is $0.00.
The tracker records tokens honestly and labels the pricing basis; it does
NOT invent list-price multiplications.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from krag.services.groq_auth import (
    GROQ_DEFAULT_MODEL,
    extract_content,
    groq_available,
    groq_chat,
)
from krag.services.llm import LLMProvider, LLMResult

logger = logging.getLogger(__name__)

PRICING_BASIS = "groq-free-tier-billed-0.00"


class GroqLLMProvider:
    """LLMProvider over Groq. Credential resolved transiently per call."""

    name = f"groq:{GROQ_DEFAULT_MODEL}"

    def __init__(self, model: str = GROQ_DEFAULT_MODEL, timeout: int = 120) -> None:
        self._model = model
        self._timeout = timeout
        self.name = f"groq:{model}"
        if not groq_available():
            raise RuntimeError(
                "Groq credential not available (Secure Vault entry custom.groq missing)"
            )

    def generate(self, prompt: str, max_tokens: int = 1024, system: str | None = None) -> LLMResult:
        t0 = time.perf_counter()
        system_prompt = system or (
            "Answer ONLY from the provided context. Cite every factual "
            "claim as [chunk_id]. If the context is insufficient, reply "
            "exactly: INSUFFICIENT_EVIDENCE"
        )
        content, usage, finish_reason = self._chat(prompt, system_prompt, max_tokens)
        # gpt-oss can spend the whole budget on reasoning and return empty
        # content with finish_reason="length": retry once with a larger
        # budget before surfacing an explicit empty-content failure.
        if not content and finish_reason == "length" and max_tokens < 4096:
            logger.warning(
                "groq %s empty content (finish=length); retrying with larger budget",
                self._model,
            )
            content, usage, finish_reason = self._chat(
                prompt, system_prompt, min(max_tokens * 2, 4096)
            )
        latency_ms = (time.perf_counter() - t0) * 1000
        tokens_in = int(usage.get("prompt_tokens", 0))
        tokens_out = int(usage.get("completion_tokens", 0))
        reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0))
        logger.debug(
            "groq %s: in=%d out=%d reasoning=%d %.0fms finish=%s",
            self._model,
            tokens_in,
            tokens_out,
            reasoning,
            latency_ms,
            finish_reason,
        )
        return LLMResult(
            text=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            reasoning_tokens=reasoning,
            latency_ms=latency_ms,
            model=self.name,
            empty_content=not content,
        )

    def _chat(
        self, prompt: str, system_prompt: str, max_tokens: int
    ) -> tuple[str, dict[str, Any], str]:
        response = groq_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=self._model,
            max_tokens=max_tokens,
            timeout=self._timeout,
        )
        return extract_content(response)


def build_groq_provider(model: str = GROQ_DEFAULT_MODEL) -> LLMProvider | None:
    """Return a Groq provider, or None if no credential is available."""
    try:
        return GroqLLMProvider(model)
    except RuntimeError as exc:
        logger.info("Groq unavailable, degrading: %s", exc)
        return None
