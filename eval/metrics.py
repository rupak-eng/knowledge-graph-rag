"""Evaluation metrics: token-F1, recall@k, MRR, latency percentiles.

All metrics are computed from measured runs. Nothing here is estimated.
"""

from __future__ import annotations

import re
import statistics


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between predicted and gold answer (SQuAD-style)."""
    pred_tokens = normalize(prediction)
    gold_tokens = normalize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = 0
    gold_counts: dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    for t in pred_tokens:
        if gold_counts.get(t, 0) > 0:
            common += 1
            gold_counts[t] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 1.0
    topk = set(retrieved_ids[:k])
    return len(topk & set(gold_ids)) / len(gold_ids)


def mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    gold = set(gold_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "n": 0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "p50": round(statistics.median(ordered), 1),
        "p95": round(ordered[min(n - 1, int(0.95 * n))], 1),
        "mean": round(statistics.mean(ordered), 1),
        "n": n,
    }


def mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0
