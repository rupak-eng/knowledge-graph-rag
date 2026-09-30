"""Pydantic domain models: chunks, entities, relations, retrieval, answers."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from krag.domain.ontology import EntityType, RelationType, relation_allowed


def chunk_id_for(doc_id: str, index: int) -> str:
    """Deterministic shared chunk ID used by BOTH Neo4j and pgvector.

    This shared key is the whole trick of the hybrid design: a graph path can
    be resolved back to source text, and a retrieved passage can be expanded
    into its graph neighborhood.
    """
    return f"{doc_id}#c{index:05d}"


class Document(BaseModel):
    doc_id: str
    title: str
    source: str  # e.g. "SEC EDGAR 10-K FY2024"
    filing_date: str = ""
    content_hash: str = ""  # sha256 of raw text; ingestion skips unchanged docs


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    index: int
    section: str = ""  # e.g. "ITEM 1. Business"
    text: str
    entity_ids: list[str] = Field(default_factory=list)  # entities mentioned

    @field_validator("text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunk text must not be empty")
        return v


class Entity(BaseModel):
    """A resolved, canonical entity node."""

    entity_id: str  # stable id derived from normalized name + type
    name: str  # canonical display name
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_chunk_ids: list[str] = Field(default_factory=list)

    @staticmethod
    def make_id(name: str, entity_type: EntityType) -> str:
        norm = Entity.normalize(name)
        digest = hashlib.sha256(f"{entity_type.value}|{norm}".encode()).hexdigest()[:16]
        return f"ent_{digest}"

    @staticmethod
    def normalize(name: str) -> str:
        n = name.strip().lower()
        for suffix in (
            " inc.", " inc", " corp.", " corp", " corporation", " ltd.", " ltd",
            " llc", " plc", " co.", " company",
        ):
            if n.endswith(suffix):
                n = n[: -len(suffix)].strip(" ,.")
        return " ".join(n.split())


class Relation(BaseModel):
    src_id: str
    src_type: EntityType
    rel: RelationType
    dst_id: str
    dst_type: EntityType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("dst_type")
    @classmethod
    def _allowed(cls, v: EntityType, info: Any) -> EntityType:
        src = info.data.get("src_type")
        rel = info.data.get("rel")
        if src is not None and rel is not None and not relation_allowed(src, rel, v):
            raise ValueError(f"relation not in ontology: {src} -{rel}-> {v}")
        return v


# ---------------- Extraction ----------------


class ExtractedEntity(BaseModel):
    """Raw extractor output, before resolution."""

    name: str
    entity_type: EntityType
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ExtractedRelation(BaseModel):
    src_name: str
    src_type: EntityType
    rel: RelationType
    dst_name: str
    dst_type: EntityType
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence: str = ""  # the sentence that justified this relation


class ExtractionResult(BaseModel):
    chunk_id: str
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


# ---------------- Retrieval ----------------

RoutePath = Literal["vector", "graph", "hybrid"]


class RouteDecision(BaseModel):
    path: RoutePath
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    entities_found: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    section: str
    text: str
    score: float
    source: Literal["vector", "graph"]


class GraphFact(BaseModel):
    """A graph path rendered as a readable statement for the answer generator."""

    statement: str  # e.g. "Apple --HAS_SEGMENT--> iPhone (from chunk aapl#c00042)"
    chunk_ids: list[str]
    path_length: int


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    graph_facts: list[GraphFact] = Field(default_factory=list)
    route: RouteDecision
    latency_ms: float = 0.0


# ---------------- Answers ----------------


class Citation(BaseModel):
    chunk_id: str
    section: str = ""
    valid: bool = True  # False if the cited chunk was NOT in the retrieved set


class Answer(BaseModel):
    question: str
    answer_text: str
    citations: list[Citation]
    contexts_used: int
    route: RouteDecision
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    abstained: bool = False  # True when evidence was insufficient


# ---------------- API ----------------


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=50)
    force_route: RoutePath | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    route: RouteDecision
    latency_ms: float
    cost_usd: float
    abstained: bool


class HealthResponse(BaseModel):
    status: str
    graph_backend: str
    vector_backend: str
    llm_provider: str
    graph_entities: int = 0
    graph_relations: int = 0
    vector_chunks: int = 0
