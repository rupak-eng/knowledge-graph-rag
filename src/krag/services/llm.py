"""LLM provider abstraction.

``LLMProvider`` is the only interface the QA/answer layer talks to.
Implementations:
* ``StubLLMProvider`` — deterministic, offline, $0. Labels itself honestly.
* ``OpenAICompatibleProvider`` — any OpenAI-compatible /chat/completions
  endpoint (OpenAI, Claude via proxy, Ollama, vLLM...). Configured from env.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Protocol

from pydantic import BaseModel

from krag.config import Settings

logger = logging.getLogger(__name__)


class LLMResult(BaseModel):
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    # Reasoning/thinking tokens billed inside completion_tokens by some
    # providers (e.g. gpt-oss on Groq). Tracked separately for honesty.
    reasoning_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    # True when the model returned empty content (e.g. finish_reason=length
    # after spending the budget on reasoning); caller decides how to handle.
    empty_content: bool = False


class LLMProvider(Protocol):
    name: str

    def generate(
        self, prompt: str, max_tokens: int = 1024, system: str | None = None
    ) -> LLMResult: ...


class StubLLMProvider:
    """Deterministic extractive answerer for offline tests/benchmarks.

    It does NOT hallucinate: it answers by quoting the retrieved context and
    attaching the chunk IDs it was given. Honest about being a stub.
    """

    name = "stub-extractive-v1"

    def generate(self, prompt: str, max_tokens: int = 1024, system: str | None = None) -> LLMResult:
        t0 = time.perf_counter()
        contexts = _parse_contexts(prompt)
        question = _parse_question(prompt)
        answer = _extractive_answer(question, contexts, max_tokens)
        return LLMResult(
            text=answer,
            tokens_in=len(prompt.split()),
            tokens_out=len(answer.split()),
            latency_ms=(time.perf_counter() - t0) * 1000,
            model=self.name,
        )


def _parse_contexts(prompt: str) -> list[tuple[str, str]]:
    """Parse [chunk_id] blocks out of the assembled context section."""
    contexts: list[tuple[str, str]] = []
    for m in re.finditer(
        r"\[chunk_id=(?P<cid>[^\]]+)\]\s*\n(?P<body>.*?)(?=\n\[chunk_id=|\Z)",
        prompt,
        re.DOTALL,
    ):
        contexts.append((m.group("cid").strip(), m.group("body").strip()))
    return contexts


def _parse_question(prompt: str) -> str:
    m = re.search(r"QUESTION:\s*(.+?)(?:\n|$)", prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extractive_answer(question: str, contexts: list[tuple[str, str]], max_tokens: int) -> str:
    if not contexts:
        return "INSUFFICIENT_EVIDENCE"
    qwords = {
        w.lower()
        for w in re.findall(r"[A-Za-z0-9$%.,]+", question)
        if len(w) > 3 and w.lower() not in _STOPWORDS
    }
    scored: list[tuple[int, str, str]] = []
    for cid, body in contexts:
        body_words = set(re.findall(r"[a-z0-9$%.,]+", body.lower()))
        overlap = len(qwords & body_words)
        # money figures / proper nouns get a boost — they answer "how much/who"
        boost = 2 if re.search(r"\$[\d,]+", body) else 0
        scored.append((overlap + boost, cid, body))
    scored.sort(key=lambda t: t[0], reverse=True)
    parts: list[str] = []
    used_tokens = 0
    for score, cid, body in scored[:4]:
        if score == 0:
            continue
        sentence = body.split(".")[0].strip()
        if len(sentence) > 400:
            sentence = sentence[:400] + "..."
        part = f"{sentence}. [{cid}]"
        used_tokens += len(part.split())
        if used_tokens > max_tokens:
            break
        parts.append(part)
    if not parts:
        return "INSUFFICIENT_EVIDENCE"
    return " ".join(parts)


_STOPWORDS = frozenset(
    "what which when where who whom whose why how much many does did are was were the "
    "a an and or of in on for to with by from as at that this these those it its their "
    "there their report company fiscal year 2024 does have has had than then".split()
)


class OpenAICompatibleProvider:
    """Any OpenAI-compatible chat completions endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60) -> None:
        import httpx

        self.name = f"openai-compatible:{model}"
        self._model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def generate(self, prompt: str, max_tokens: int = 1024, system: str | None = None) -> LLMResult:
        t0 = time.perf_counter()
        resp = self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {
                        "role": "system",
                        "content": system
                        or (
                            "Answer ONLY from the provided context. Cite every claim "
                            "as [chunk_id]. If the context is insufficient, reply "
                            "exactly: INSUFFICIENT_EVIDENCE"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        return LLMResult(
            text=text,
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            latency_ms=(time.perf_counter() - t0) * 1000,
            model=self._model,
        )


def build_llm_provider(base_url: str, api_key: str, model: str, timeout: int = 60) -> LLMProvider:
    if base_url and model:
        return OpenAICompatibleProvider(base_url, api_key, model, timeout)
    logger.info("No LLM configured; using deterministic stub provider")
    return StubLLMProvider()


def build_production_llm(settings: Settings) -> LLMProvider:
    """Production LLM resolution order: Groq (Vault) -> explicit config -> stub.

    The Groq credential is resolved transiently at runtime and never persisted.
    """
    from krag.services.groq_provider import build_groq_provider

    choice = getattr(settings, "groq_model_choice", "default")
    if choice == "quality":
        groq_model = getattr(settings, "groq_quality_model", "openai/gpt-oss-120b")
    else:
        groq_model = getattr(settings, "groq_default_model", "openai/gpt-oss-20b")

    groq = build_groq_provider(groq_model)
    if groq is not None:
        return groq
    # explicit OpenAI-compatible config as second choice
    base_url = getattr(settings, "llm_base_url", "")
    model = getattr(settings, "llm_model", "")
    if base_url and model:
        return OpenAICompatibleProvider(
            base_url,
            getattr(settings, "llm_api_key", ""),
            model,
            int(getattr(settings, "llm_timeout_seconds", 60)),
        )
    logger.info("No LLM credential available; using deterministic stub provider")
    return StubLLMProvider()
