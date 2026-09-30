"""Cost accounting: every query's spend is measured, never guessed.

Pricing table covers common OpenAI-compatible models; unknown models record
tokens with $0 and are labeled accordingly. The stub provider is always $0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# USD per 1K tokens (input, output). Sourced from public provider pricing pages;
# update when pricing changes. Values are informational — the tracker only
# multiplies tokens actually observed.
PRICE_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-5-haiku": (0.0008, 0.004),
    "stub-extractive-v1": (0.0, 0.0),
    # Groq key used in this project is on the free tier: billed $0.00.
    # Recorded as 0.0 deliberately — see benchmark report pricing notes.
    "groq:openai/gpt-oss-20b": (0.0, 0.0),
    "groq:openai/gpt-oss-120b": (0.0, 0.0),
}


@dataclass
class CostTracker:
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    llm_cost_usd: float = 0.0
    llm_latency_ms: float = 0.0
    embedding_texts: int = 0
    embedding_latency_ms: float = 0.0
    model_label: str = "stub-extractive-v1"
    events: list[dict[str, object]] = field(default_factory=list)

    def record_llm(self, tokens_in: int, tokens_out: int, model: str, latency_ms: float) -> None:
        self.llm_tokens_in += tokens_in
        self.llm_tokens_out += tokens_out
        self.llm_latency_ms += latency_ms
        self.model_label = model
        price_in, price_out = PRICE_PER_1K.get(self._base_model(model), (0.0, 0.0))
        cost = tokens_in / 1000 * price_in + tokens_out / 1000 * price_out
        self.llm_cost_usd += cost
        self.events.append(
            {
                "type": "llm",
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": round(cost, 6),
                "latency_ms": round(latency_ms, 1),
                "ts": time.time(),
            }
        )

    def record_embedding(self, n_texts: int, latency_ms: float) -> None:
        self.embedding_texts += n_texts
        self.embedding_latency_ms += latency_ms
        self.events.append(
            {
                "type": "embedding",
                "texts": n_texts,
                "latency_ms": round(latency_ms, 1),
                "ts": time.time(),
            }
        )

    def query_cost_usd(self) -> float:
        return round(self.llm_cost_usd, 6)

    def summary(self) -> dict[str, object]:
        return {
            "model": self.model_label,
            "llm_tokens_in": self.llm_tokens_in,
            "llm_tokens_out": self.llm_tokens_out,
            "llm_cost_usd": round(self.llm_cost_usd, 6),
            "llm_latency_ms": round(self.llm_latency_ms, 1),
            "embedding_texts": self.embedding_texts,
            "embedding_latency_ms": round(self.embedding_latency_ms, 1),
        }

    @staticmethod
    def _base_model(model: str) -> str:
        # "openai-compatible:gpt-4o-mini" -> "gpt-4o-mini"
        return model.split(":")[-1]
