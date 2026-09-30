"""Hybrid retriever: vector search + graph traversal, merged and deduplicated.

The shared chunk_id key is what makes the hybrid real:
* graph -> text: graph facts carry source chunk IDs; resolve them to chunks.
* text -> graph: vector hits carry entity_ids; expand their graph neighborhood.
"""

from __future__ import annotations

import logging
import time

from krag.domain.schemas import (
    GraphFact,
    RetrievalResult,
    RetrievedChunk,
    RouteDecision,
)
from krag.services.cypher_templates import execute_in_memory, get_template  # noqa: F401
from krag.services.embeddings import EmbeddingProvider
from krag.services.router import Router
from krag.storage.graph_store import GraphStore
from krag.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        embedder: EmbeddingProvider,
        router: Router,
        top_k_vector: int = 8,
        top_k_graph: int = 12,
        max_hops: int = 3,
    ) -> None:
        self._vector = vector_store
        self._graph = graph_store
        self._embedder = embedder
        self._router = router
        self._top_k_vector = top_k_vector
        self._top_k_graph = top_k_graph
        self._max_hops = max_hops

    def retrieve(
        self, question: str, top_k: int | None = None, force_route: str | None = None
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        route: RouteDecision = self._router.route(question)
        if force_route is not None:
            route = route.model_copy(update={"path": force_route})
            route.reasons.append(f"route overridden to {force_route} by caller")

        k = top_k or self._top_k_vector
        chunks: list[RetrievedChunk] = []
        graph_facts: list[GraphFact] = []

        if route.path in ("vector", "hybrid"):
            chunks.extend(self._vector_search(question, k))
        if route.path in ("graph", "hybrid"):
            facts, gchunks = self._graph_search(route, question)
            graph_facts.extend(facts)
            chunks.extend(gchunks)

        # Fallback: graph routing with zero graph evidence (unknown entities,
        # empty graph) degrades to vector rather than abstaining silently.
        if route.path == "graph" and not chunks and not graph_facts:
            route.reasons.append("graph path yielded no evidence; falling back to vector search")
            route = route.model_copy(update={"path": "hybrid"})
            chunks.extend(self._vector_search(question, k))

        chunks = self._dedupe(chunks)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "retrieve route=%s chunks=%d facts=%d latency=%.1fms",
            route.path,
            len(chunks),
            len(graph_facts),
            latency_ms,
        )
        return RetrievalResult(
            chunks=chunks[: max(k, self._top_k_graph)],
            graph_facts=graph_facts,
            route=route,
            latency_ms=latency_ms,
        )

    # -- vector path -----------------------------------------------------
    def _vector_search(self, question: str, k: int) -> list[RetrievedChunk]:
        qvec = self._embedder.embed([question])[0]
        return self._vector.search(qvec, top_k=k)

    # -- graph path ------------------------------------------------------
    def _graph_search(
        self, route: RouteDecision, question: str
    ) -> tuple[list[GraphFact], list[RetrievedChunk]]:
        facts: list[GraphFact] = []
        chunks: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()

        for name in route.entities_found:
            entity = self._graph.find_by_name(name)
            if entity is None:
                continue
            rows = self._graph.run_template(
                "entity_neighborhood",
                {
                    "entity_id": entity.entity_id,
                    "max_hops": min(self._max_hops, 2),
                    "limit": self._top_k_graph,
                },
            )
            for row in rows:
                chunk_ids = [c for c in row.get("chunks", []) if c]
                statement = (
                    f"{row['src']} --{row['rel']}--> {row['dst']} "
                    f"(evidence: {', '.join(chunk_ids[:3])})"
                )
                facts.append(
                    GraphFact(
                        statement=statement,
                        chunk_ids=chunk_ids,
                        path_length=1,
                    )
                )
                for cid in chunk_ids:
                    if cid in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(cid)
                    rc = self._vector.get(cid)
                    if rc is not None:
                        rc = rc.model_copy(update={"source": "graph"})
                        chunks.append(rc)

        # multi-entity questions: try a path between the first two entities
        if len(route.entities_found) >= 2:
            e1 = self._graph.find_by_name(route.entities_found[0])
            e2 = self._graph.find_by_name(route.entities_found[1])
            if e1 is not None and e2 is not None:
                rows = self._graph.run_template(
                    "path_between",
                    {
                        "src_id": e1.entity_id,
                        "dst_id": e2.entity_id,
                        "max_hops": self._max_hops,
                    },
                )
                for row in rows:
                    nodes = row.get("nodes", [])
                    rels = row.get("rels", [])
                    chunk_lists = row.get("chunks", [])
                    flat: list[str] = [c for cl in chunk_lists for c in (cl or [])]
                    pairs = zip(nodes, rels + [""], strict=True)
                    hops = " -> ".join(f"{n}-[{r}]->" for n, r in pairs).rstrip("->")
                    facts.append(
                        GraphFact(
                            statement=f"PATH: {hops}{nodes[-1] if nodes else ''} "
                            f"(evidence: {', '.join(flat[:3])})",
                            chunk_ids=flat,
                            path_length=len(rels),
                        )
                    )
                    for cid in flat:
                        if cid in seen_chunk_ids:
                            continue
                        seen_chunk_ids.add(cid)
                        rc = self._vector.get(cid)
                        if rc is not None:
                            chunks.append(rc.model_copy(update={"source": "graph"}))
        return facts, chunks

    @staticmethod
    def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen: set[str] = set()
        out: list[RetrievedChunk] = []
        # vector hits first (they carry the similarity score ordering)
        for c in sorted(chunks, key=lambda c: (c.source != "vector", -c.score)):
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                out.append(c)
        return out
