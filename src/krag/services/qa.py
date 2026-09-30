"""QA pipeline: route -> retrieve -> generate grounded answer."""

from __future__ import annotations

import logging
import time

from krag.config import Settings
from krag.domain.schemas import Answer, RoutePath
from krag.services.answer import AnswerService
from krag.services.cost import CostTracker
from krag.services.embeddings import EmbeddingProvider
from krag.services.llm import LLMProvider, build_llm_provider
from krag.services.retriever import HybridRetriever
from krag.services.router import Router, RuleRouter
from krag.storage.graph_store import GraphStore, build_graph_store
from krag.storage.vector_store import VectorStore, build_vector_store
from krag.services.ingest import _default_embedder

logger = logging.getLogger(__name__)


class QASystem:
    """Assembled QA system. Owns stores, retriever, answer service, cost."""

    def __init__(
        self,
        settings: Settings,
        graph: GraphStore | None = None,
        vector: VectorStore | None = None,
        embedder: EmbeddingProvider | None = None,
        llm: LLMProvider | None = None,
        router: Router | None = None,
    ) -> None:
        self.settings = settings
        self.graph = graph or build_graph_store(
            settings.neo4j_uri, settings.neo4j_user,
            settings.neo4j_password, settings.graph_backend,
        )
        self.vector = vector or build_vector_store(
            settings.database_url, settings.embedding_dim, settings.vector_backend
        )
        self.embedder = embedder or _default_embedder(settings)
        self.llm = llm or build_llm_provider(
            settings.llm_base_url, settings.llm_api_key,
            settings.llm_model, settings.llm_timeout_seconds,
        )
        self.router = router or RuleRouter(settings.router_low_confidence_threshold)
        self.retriever = HybridRetriever(
            vector_store=self.vector,
            graph_store=self.graph,
            embedder=self.embedder,
            router=self.router,
            top_k_vector=settings.qa_top_k_vector,
            top_k_graph=settings.qa_top_k_graph,
            max_hops=settings.qa_max_hops,
        )
        self.cost = CostTracker()
        self.answers = AnswerService(self.llm, self.cost)

    def ask(
        self, question: str, top_k: int | None = None, force_route: RoutePath | None = None
    ) -> Answer:
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(question, top_k=top_k, force_route=force_route)
        answer = self.answers.answer(question, retrieval)
        # retrieval latency is included inside answer.latency_ms; surface total
        answer.latency_ms = (time.perf_counter() - t0) * 1000
        return answer

    def close(self) -> None:
        self.graph.close()
        self.vector.close()
