"""Grounded answer generation with validated citations.

Contract:
1. The generator may ONLY cite chunk IDs present in the retrieved set.
2. Every citation in the final answer is validated against that set;
   citations to non-retrieved chunks are stripped, and claims left without
   any valid citation trigger abstention rather than hallucination.
"""

from __future__ import annotations

import logging
import re
import time

from krag.domain.schemas import (
    Answer,
    Citation,
    RetrievedChunk,
    RetrievalResult,
)
from krag.services.cost import CostTracker
from krag.services.llm import LLMProvider

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[([A-Za-z0-9_#.\-]+)\]")

ANSWER_PROMPT = """You answer questions about SEC 10-K filings using ONLY the context below.

CONTEXT (each block is labeled with its chunk_id):
{contexts}

GRAPH FACTS (derived from the knowledge graph; evidence chunk IDs listed):
{graph_facts}

RULES:
- Every factual claim must end with a citation like [chunk_id].
- Use ONLY chunk IDs that appear in the context above.
- If the context does not contain the answer, reply exactly: INSUFFICIENT_EVIDENCE
- Be concise.

QUESTION:
{question}
"""


class AnswerService:
    def __init__(self, llm: LLMProvider, cost: CostTracker) -> None:
        self._llm = llm
        self._cost = cost

    def answer(
        self, question: str, retrieval: RetrievalResult
    ) -> Answer:
        t0 = time.perf_counter()
        retrieved_ids = {c.chunk_id for c in retrieval.chunks}

        if not retrieval.chunks:
            return self._abstain(question, retrieval, t0, "no chunks retrieved")

        prompt = ANSWER_PROMPT.format(
            contexts=self._format_contexts(retrieval.chunks),
            graph_facts=self._format_facts(retrieval),
            question=question,
        )
        result = self._llm.generate(prompt)
        self._cost.record_llm(
            result.tokens_in, result.tokens_out, result.model, result.latency_ms
        )

        text, citations = self._validate_citations(result.text, retrieval.chunks)

        abstained = text.strip() == "INSUFFICIENT_EVIDENCE" or not citations
        if not citations and text.strip() != "INSUFFICIENT_EVIDENCE":
            logger.warning("answer produced no valid citations; abstaining")
            return self._abstain(
                question, retrieval, t0, "no valid citations after validation"
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        return Answer(
            question=question,
            answer_text=text,
            citations=citations,
            contexts_used=len(retrieval.chunks),
            route=retrieval.route,
            latency_ms=latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=self._cost.query_cost_usd(),
            abstained=abstained,
        )

    # -- internals --------------------------------------------------------
    @staticmethod
    def _format_contexts(chunks: list[RetrievedChunk]) -> str:
        parts = []
        for c in chunks:
            parts.append(f"[chunk_id={c.chunk_id}]\n({c.doc_id} {c.section})\n{c.text}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_facts(retrieval: RetrievalResult) -> str:
        if not retrieval.graph_facts:
            return "(none)"
        return "\n".join(f"- {f.statement}" for f in retrieval.graph_facts[:20])

    def _validate_citations(
        self, text: str, chunks: list[RetrievedChunk]
    ) -> tuple[str, list[Citation]]:
        """Strip citations to chunks that were not retrieved; keep the rest.

        Returns (cleaned_text, citations). A citation that fails validation is
        removed from the text and never reported as valid — the answer cannot
        point at evidence it did not see.
        """
        chunk_by_id = {c.chunk_id: c for c in chunks}
        citations: list[Citation] = []
        seen: set[str] = set()

        def _replace(m: re.Match[str]) -> str:
            cid = m.group(1)
            if cid in chunk_by_id and cid not in seen:
                seen.add(cid)
                c = chunk_by_id[cid]
                citations.append(
                    Citation(chunk_id=cid, section=c.section, valid=True)
                )
                return m.group(0)
            logger.warning("stripping invalid citation [%s]", cid)
            return ""  # invalid citation: remove it

        cleaned = CITATION_RE.sub(_replace, text)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
        return cleaned, citations

    def _abstain(
        self, question: str, retrieval: RetrievalResult, t0: float, reason: str
    ) -> Answer:
        logger.info("abstaining on %r: %s", question[:60], reason)
        return Answer(
            question=question,
            answer_text="INSUFFICIENT_EVIDENCE",
            citations=[],
            contexts_used=len(retrieval.chunks),
            route=retrieval.route,
            latency_ms=(time.perf_counter() - t0) * 1000,
            abstained=True,
        )
