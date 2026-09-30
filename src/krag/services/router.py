"""Retrieval router: decides vector / graph / hybrid per query.

v1 is a rules-based router with an explicit low-confidence fallback to
hybrid. The interface (RouteDecision with confidence + reasons) is the same
one an LLM router would return, so the router is swappable without touching
the QA pipeline. Every decision is logged with the question for later
analysis (see eval/routing_log.jsonl written by the benchmark runner).
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from krag.domain.schemas import RouteDecision
from krag.services.extractor import COMPANIES

logger = logging.getLogger(__name__)


class Router(Protocol):
    def route(self, question: str) -> RouteDecision: ...


# Question patterns that signal graph-friendly queries
GRAPH_CUES = [
    r"\bwhich\b.*\b(acquisitions?|subsidiaries|segments?|products?)\b",
    r"\b(acquisitions?|subsidiaries|segments?)\b.*\bdoes\b",
    r"\bwho\s+(is|was)\b.*\b(ceo|cfo|chief)\b",
    r"\bhow\b.*\b(relat|connect)\w*\b",
    r"\bcompare\b",
    r"\bbetween\b.*\band\b",
    r"\blist\b.*\b(all|the)\b",
    r"\bwhat\b.*\b(acquired|owns?|subsidiari)\w*\b",
    r"\brelationship\b",
]

VECTOR_CUES = [
    r"\bdefine\b|\bdefinition\b|\bwhat is\b|\bwhat does\b.*\bmean\b",
    r"\bhow much\b|\bwhat (was|were) the\b.*\b(revenue|income|profit|loss)\b",
    r"\bpolicy\b|\bpolicies\b",
    r"\bwhen\b.*\b(filed|reported|announced)\b",
]

HOP_CUES = [
    r"\b.*\b's\b.*\b(ceo|subsidiary|acquisition)\b",  # possessive chains
    r"\bwho\b.*\bleads\b.*\bacquired\b",
]


class RuleRouter:
    """Cheap, explainable router. Falls back to hybrid below confidence."""

    def __init__(self, low_confidence_threshold: float = 0.55) -> None:
        self._threshold = low_confidence_threshold

    def route(self, question: str) -> RouteDecision:
        q = question.lower()
        reasons: list[str] = []
        graph_score = 0.0
        vector_score = 0.0

        entities_found = self._find_entities(question)
        if entities_found:
            graph_score += 0.25
            reasons.append(f"mentions known entities: {', '.join(entities_found)}")

        for pat in GRAPH_CUES:
            if re.search(pat, q):
                graph_score += 0.35
                reasons.append(f"graph cue matched: {pat[:40]}...")
                break

        for pat in HOP_CUES:
            if re.search(pat, q):
                graph_score += 0.25
                reasons.append("multi-hop cue detected")
                break

        for pat in VECTOR_CUES:
            if re.search(pat, q):
                vector_score += 0.4
                reasons.append(f"vector cue matched: {pat[:40]}...")
                break

        # definition-style questions with no entities -> vector
        if re.match(r"^\s*what\s+is\b", q) and not entities_found:
            vector_score += 0.3
            reasons.append("definition-style question, no entities")

        total = graph_score + vector_score
        if total == 0:
            return RouteDecision(
                path="hybrid",
                confidence=0.5,
                reasons=["no strong cues; defaulting to hybrid"],
                entities_found=entities_found,
            )

        if graph_score > vector_score:
            confidence = min(0.95, 0.5 + (graph_score - vector_score))
            path = "graph" if confidence >= self._threshold else "hybrid"
            if path == "hybrid":
                reasons.append(
                    f"graph-leaning but confidence {confidence:.2f} < "
                    f"{self._threshold}; falling back to hybrid"
                )
            return RouteDecision(
                path=path, confidence=confidence, reasons=reasons,
                entities_found=entities_found,
            )
        confidence = min(0.95, 0.5 + (vector_score - graph_score))
        path = "vector" if confidence >= self._threshold else "hybrid"
        if path == "hybrid":
            reasons.append(
                f"vector-leaning but confidence {confidence:.2f} < "
                f"{self._threshold}; falling back to hybrid"
            )
        return RouteDecision(
            path=path, confidence=confidence, reasons=reasons,
            entities_found=entities_found,
        )

    @staticmethod
    def _find_entities(question: str) -> list[str]:
        q = question.lower()
        found: list[str] = []
        for canon, info in COMPANIES.items():
            names = [canon] + info["aliases"] + info["products"] + info["people"]
            for n in names:
                if n.lower() in q:
                    found.append(n)
                    break
        return found
