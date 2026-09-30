"""Idempotent ingestion pipeline: 10-K HTML -> chunks -> entities -> graph+vector.

Idempotency contract: re-running ingestion over unchanged documents changes
nothing. Document content hashes are persisted in ``data/processed/manifest.json``;
documents whose hash matches are skipped entirely, and graph writes use MERGE
semantics while vector writes use upsert-on-chunk_id.

Usage: python -m krag.services.ingest --data-dir data [--reingest]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from krag.config import Settings, get_settings
from krag.domain.ontology import relation_allowed
from krag.domain.schemas import (
    Chunk,
    Entity,
    ExtractedRelation,
    ExtractionResult,
    Relation,
)
from krag.services.chunker import chunk_document, parse_10k_html
from krag.services.embeddings import (
    EmbeddingProvider,
    SentenceTransformerProvider,
    StubEmbeddingProvider,
)
from krag.services.extractor import Extractor, LLMExtractor, SpacyRuleExtractor
from krag.services.llm import LLMProvider, StubLLMProvider, build_production_llm
from krag.services.resolver import EntityResolver
from krag.storage.graph_store import GraphStore, build_graph_store
from krag.storage.vector_store import VectorStore, build_vector_store

logger = logging.getLogger(__name__)

FILINGS = [
    ("aapl", "aapl-20240928.htm", "Apple Inc. FY2024 10-K"),
    ("msft", "msft-20240630.htm", "Microsoft Corporation FY2024 10-K"),
    ("nvda", "nvda-20240128.htm", "NVIDIA Corporation FY2024 10-K"),
]


def build_ingest_stack(
    settings: Settings | None = None,
    extractor: Extractor | None = None,
    embedder: EmbeddingProvider | None = None,
    llm: LLMProvider | None = None,
) -> tuple[GraphStore, VectorStore, EmbeddingProvider, Extractor]:
    s = settings or get_settings()
    graph = build_graph_store(s.neo4j_uri, s.neo4j_user, s.neo4j_password, s.graph_backend)
    vector = build_vector_store(s.database_url, s.embedding_dim, s.vector_backend)
    emb = embedder or _default_embedder(s)
    llm_provider = llm or build_production_llm(s)
    ext = extractor or _default_extractor(llm_provider)
    return graph, vector, emb, ext


def _default_embedder(s: Settings) -> EmbeddingProvider:
    # Offline workaround: prefer an explicit local snapshot path, then the
    # HF hub cache from a previous download, before attempting a download
    # (the VM's egress proxy mangles huggingface_hub redirects).
    candidates: list[str] = []
    explicit = getattr(s, "embedding_model_path", "")
    if explicit:
        candidates.append(explicit)
    candidates.append(
        "/home/hatch/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
        "/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    for path in candidates:
        try:
            import os

            if path and os.path.isdir(path):
                return SentenceTransformerProvider(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("local embedding snapshot %s failed: %s", path, exc)
    try:
        return SentenceTransformerProvider(s.embedding_model)
    except Exception as exc:  # noqa: BLE001 - offline fallback
        logger.warning("sentence-transformers unavailable (%s); using stub", exc)
        return StubEmbeddingProvider(dim=getattr(s, "embedding_dim", 384))


def _default_extractor(llm: LLMProvider) -> Extractor:
    if not isinstance(llm, StubLLMProvider):
        logger.info("LLM configured; using LLM extractor")
        return LLMExtractor(llm)
    try:
        return SpacyRuleExtractor()
    except Exception as exc:  # noqa: BLE001 - offline fallback
        logger.warning("spaCy model unavailable (%s); extraction disabled", exc)
        return _NullExtractor()


class _NullExtractor:
    def extract(self, chunk: Chunk) -> ExtractionResult:
        cid = chunk.chunk_id
        return ExtractionResult(chunk_id=cid, entities=[], relations=[])


def ingest(
    data_dir: Path,
    reingest: bool = False,
    extractor_name: str = "spacy",
    graph_backend: str = "real",
    vector_backend: str = "real",
) -> dict[str, object]:
    settings = get_settings()
    settings.graph_backend = graph_backend
    settings.vector_backend = vector_backend
    graph, vector, embedder, extractor = build_ingest_stack(
        settings, extractor=_choose_extractor(settings, extractor_name)
    )
    manifest_path = data_dir / "processed" / "manifest.json"
    manifest: dict[str, str] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )

    t0 = time.perf_counter()
    stats: dict[str, Any] = {
        "documents": 0,
        "chunks": 0,
        "entities": 0,
        "relations": 0,
        "skipped_unchanged": 0,
    }
    resolver = EntityResolver()

    for doc_id, filename, title in FILINGS:
        path = data_dir / "raw" / filename
        if not path.exists():
            logger.warning("missing filing %s; skipping", path)
            continue
        doc, clean_text = parse_10k_html(path, doc_id, title)
        if not reingest and manifest.get(doc_id) == doc.content_hash:
            logger.info("unchanged %s; skipping", doc_id)
            stats["skipped_unchanged"] += 1
            continue

        chunks = chunk_document(doc, clean_text)
        stats["documents"] += 1
        stats["chunks"] += len(chunks)

        # 1. extract + resolve entities per chunk
        raw_relations: dict[str, list[ExtractedRelation]] = {}
        for chunk in chunks:
            result = extractor.extract(chunk)
            for extracted in result.entities:
                resolver.resolve(extracted, chunk.chunk_id)
            raw_relations[chunk.chunk_id] = result.relations

        for entity in resolver.entities():
            graph.merge_entity(entity)

        # 2. relations: map extracted names -> resolved ids via a fresh lookup
        name_to_id: dict[tuple[str, str], str] = {}
        for entity in resolver.entities():
            name_to_id[(entity.entity_type.value, Entity.normalize(entity.name))] = entity.entity_id
            for alias in entity.aliases:
                name_to_id.setdefault(
                    (entity.entity_type.value, Entity.normalize(alias)), entity.entity_id
                )

        rel_count = 0
        for chunk in chunks:
            for rr in raw_relations.get(chunk.chunk_id, []):
                src_id = name_to_id.get((rr.src_type.value, Entity.normalize(rr.src_name)))
                dst_id = name_to_id.get((rr.dst_type.value, Entity.normalize(rr.dst_name)))
                if src_id is None or dst_id is None:
                    continue
                if not relation_allowed(rr.src_type, rr.rel, rr.dst_type):
                    continue
                graph.merge_relation(
                    Relation(
                        src_id=src_id,
                        src_type=rr.src_type,
                        rel=rr.rel,
                        dst_id=dst_id,
                        dst_type=rr.dst_type,
                        confidence=rr.confidence,
                        source_chunk_ids=[chunk.chunk_id],
                    )
                )
                rel_count += 1

        # 3. vector index (upsert on chunk_id -> idempotent)
        texts = [c.text for c in chunks]
        embeddings = embedder.embed(texts)
        vector.add(chunks, embeddings)

        manifest[doc_id] = doc.content_hash
        stats["relations"] += rel_count
        logger.info("ingested %s: %d chunks, %d relations", doc_id, len(chunks), rel_count)

    stats["entities"] = graph.entity_count()
    stats["relations_total"] = graph.relation_count()
    stats["resolver"] = resolver.stats()
    stats["vector_chunks"] = vector.count()
    stats["elapsed_s"] = round(time.perf_counter() - t0, 1)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    graph.close()
    vector.close()
    logger.info("ingest stats: %s", stats)
    return stats


def _choose_extractor(settings: Settings, name: str) -> Extractor:
    """Select the extraction backend for a corpus run.

    - ``spacy``: deterministic NER + rule relations (v1 baseline).
    - ``llm``: production LLM extractor (Groq when configured, else stub —
      the stub produces no entities, so this is only useful with Groq).
    """
    if name == "llm":
        llm_provider = build_production_llm(settings)
        return LLMExtractor(llm_provider)
    try:
        return SpacyRuleExtractor()
    except Exception as exc:  # noqa: BLE001 - offline fallback
        logger.warning("spaCy model unavailable (%s); extraction disabled", exc)
        return _NullExtractor()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reingest", action="store_true")
    parser.add_argument(
        "--extractor",
        choices=["spacy", "llm"],
        default="spacy",
        help="extraction backend: spacy (deterministic v1) or llm",
    )
    parser.add_argument(
        "--graph-backend",
        choices=["real", "memory"],
        default="real",
        help="Neo4j (real) or in-memory graph store",
    )
    parser.add_argument(
        "--vector-backend",
        choices=["real", "memory"],
        default="real",
        help="pgvector (real) or in-memory vector store",
    )
    args = parser.parse_args()
    stats = ingest(
        args.data_dir,
        reingest=args.reingest,
        extractor_name=args.extractor,
        graph_backend=args.graph_backend,
        vector_backend=args.vector_backend,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
