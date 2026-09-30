"""Tests: citation validation, abstention, chunker, API wiring."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from krag.config import Settings, reset_settings
from krag.domain.schemas import Chunk, RetrievalResult, RetrievedChunk, RouteDecision
from krag.services.answer import AnswerService
from krag.services.chunker import chunk_section, split_sections
from krag.services.cost import CostTracker
from krag.services.embeddings import StubEmbeddingProvider
from krag.services.llm import StubLLMProvider
from krag.services.qa import QASystem
from krag.services.router import RuleRouter
from krag.storage.graph_store import InMemoryGraphStore
from krag.storage.vector_store import InMemoryVectorStore


@pytest.fixture()
def answer_service() -> AnswerService:
    return AnswerService(StubLLMProvider(), CostTracker())


def _retrieval(chunks: list[RetrievedChunk]) -> RetrievalResult:
    return RetrievalResult(
        chunks=chunks,
        route=RouteDecision(path="vector", confidence=0.8, reasons=["test"]),
        latency_ms=1.0,
    )


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        doc_id="aapl",
        section="ITEM 1",
        text=text,
        score=0.9,
        source="vector",
    )


def test_citations_validated_against_retrieved_set(
    answer_service: AnswerService,
) -> None:
    chunks = [_chunk("aapl#c00001", "Apple reported iPhone revenue of $200 billion.")]
    text, citations = answer_service._validate_citations(
        "Apple iPhone revenue was $200 billion. [aapl#c00001] Also see [msft#c99999].",
        chunks,
    )
    assert [c.chunk_id for c in citations] == ["aapl#c00001"]
    assert all(c.valid for c in citations)
    assert "[msft#c99999]" not in text  # invalid citation stripped


def test_answer_abstains_without_evidence(answer_service: AnswerService) -> None:
    ans = answer_service.answer("What is the moon made of?", _retrieval([]))
    assert ans.abstained
    assert ans.answer_text == "INSUFFICIENT_EVIDENCE"


def test_answer_cites_retrieved_chunks(answer_service: AnswerService) -> None:
    chunks = [
        _chunk("aapl#c00001", "Apple iPhone revenue was $200 billion in fiscal 2024."),
        _chunk("aapl#c00002", "Apple Services revenue grew year over year."),
    ]
    ans = answer_service.answer("What was Apple iPhone revenue?", _retrieval(chunks))
    assert not ans.abstained
    assert ans.citations
    retrieved = {c.chunk_id for c in chunks}
    assert all(c.chunk_id in retrieved and c.valid for c in ans.citations)


def test_chunker_sections_and_ids() -> None:
    text = "Intro line\nITEM 1. Business\nApple sells iPhone.\nITEM 1A. Risk Factors\nRisks exist."
    sections = split_sections(text)
    assert sections[0][0] == "FRONT MATTER"
    assert sections[1][0].startswith("ITEM 1")
    assert sections[2][0].startswith("ITEM 1A")
    chunks, _ = chunk_section("aapl", "ITEM 1", "x" * 4000, 0)
    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "aapl#c00000"
    assert all(len(c.text) <= 1500 for c in chunks)


@pytest.fixture()
def qa_system(monkeypatch: pytest.MonkeyPatch) -> QASystem:
    reset_settings()
    settings = Settings(
        graph_backend="memory",
        vector_backend="memory",
        llm_base_url="",
        llm_model="",
    )
    qa = QASystem(
        settings,
        graph=InMemoryGraphStore(),
        vector=InMemoryVectorStore(),
        embedder=StubEmbeddingProvider(dim=settings.embedding_dim),
        llm=StubLLMProvider(),
        router=RuleRouter(),
    )
    # seed one chunk so /ask has evidence
    chunk = Chunk(
        chunk_id="aapl#c00000",
        doc_id="aapl",
        index=0,
        section="ITEM 1",
        text="Apple iPhone revenue was $200 billion in fiscal 2024.",
    )
    qa.vector.add([chunk], qa.embedder.embed([chunk.text]))
    return qa


def test_api_health_and_ask(qa_system: QASystem, monkeypatch: pytest.MonkeyPatch) -> None:
    import krag.api.main as main

    monkeypatch.setattr(main, "_qa", qa_system)
    client = TestClient(main.app)
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["vector_chunks"] == 1

    resp = client.post("/ask", json={"question": "What was Apple iPhone revenue?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"]
    assert body["route"]["path"] in ("vector", "graph", "hybrid")
    assert body["latency_ms"] >= 0


def test_api_ask_rejects_short_question(
    qa_system: QASystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    import krag.api.main as main

    monkeypatch.setattr(main, "_qa", qa_system)
    client = TestClient(main.app)
    resp = client.post("/ask", json={"question": "hi"})
    assert resp.status_code == 422
