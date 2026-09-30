"""FastAPI application: /health, /ask, /graph/entity."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from krag.config import get_settings
from krag.domain.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
)
from krag.services.qa import QASystem

logger = logging.getLogger(__name__)

_qa: QASystem | None = None


def get_qa() -> QASystem:
    global _qa
    if _qa is None:
        _qa = QASystem(get_settings())
    return _qa


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    logging.basicConfig(level=get_settings().log_level)
    logger.info("starting knowledge-graph-rag api")
    yield
    global _qa
    if _qa is not None:
        _qa.close()
        _qa = None


app = FastAPI(
    title="Knowledge Graph RAG",
    description=(
        "Hybrid vector + knowledge-graph RAG over SEC 10-K filings. "
        "Retrieval routing, grounded answers, validated citations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    qa = get_qa()
    return HealthResponse(
        status="ok",
        graph_backend=get_settings().graph_backend,
        vector_backend=get_settings().vector_backend,
        llm_provider=qa.llm.name,
        graph_entities=qa.graph.entity_count(),
        graph_relations=qa.graph.relation_count(),
        vector_chunks=qa.vector.count(),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    qa = get_qa()
    try:
        answer = qa.ask(req.question, top_k=req.top_k, force_route=req.force_route)
    except Exception as exc:  # noqa: BLE001 - surface as 500 with message
        logger.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(
        question=answer.question,
        answer=answer.answer_text,
        citations=answer.citations,
        route=answer.route,
        latency_ms=round(answer.latency_ms, 1),
        cost_usd=answer.cost_usd,
        abstained=answer.abstained,
    )


@app.get("/graph/entity/{entity_id}")
def entity_detail(entity_id: str) -> dict[str, Any]:
    """Inspect a graph node: canonical name, aliases, 1-hop facts."""
    qa = get_qa()
    entity = qa.graph.get_entity(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="entity not found")
    rows = qa.graph.run_template("entity_facts", {"entity_id": entity_id})
    return {
        "entity_id": entity.entity_id,
        "name": entity.name,
        "type": entity.entity_type.value,
        "aliases": entity.aliases,
        "source_chunks": entity.source_chunk_ids[:20],
        "facts": rows[:25],
    }


@app.get("/graph/search")
def entity_search(name: str) -> JSONResponse:
    qa = get_qa()
    entity = qa.graph.find_by_name(name)
    if entity is None:
        return JSONResponse({"found": False})
    return JSONResponse(
        {
            "found": True,
            "entity_id": entity.entity_id,
            "name": entity.name,
            "type": entity.entity_type.value,
            "aliases": entity.aliases,
        }
    )
