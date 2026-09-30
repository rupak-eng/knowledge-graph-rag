"""Idempotent ingestion pipeline: 10-K HTML -> chunks -> entities -> graph+vector.

Idempotency contract: re-running ingestion over unchanged documents changes
nothing. Document content hashes are persisted in ``data/processed/manifest.json``;
documents whose hash matches are skipped entirely, and graph writes use MERGE
semantics while vector writes use upsert-on-chunk_id.

Usage: python -m krag.services.ingest --data-dir data [--reingest]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

from krag.domain.ontology import relation_allowed
from krag.domain.schemas import Entity, ExtractedRelation, Relation, chunk_id_for
from krag.services.chunker import chunk_document, parse_10k_html
from krag.services.embeddings import (
    EmbeddingProvider,
    SentenceTransformerProvider,
    StubEmbeddingProvider,
)
from krag.services.extractor import Extractor, LLMExtractor, SpacyRuleExtractor
from krag.services.llm import LLMProvider, StubLLMProvider, build_llm_provider
from krag.services.resolver import EntityResolver
from krag.storage.graph_store import GraphStore, build_graph_store
from krag.storage.vector_store import VectorStore, build_vector_store
from krag.config import get_settings

logger = logging.getLogger(__name__)

FILINGS = [
    ("aapl", "aapl-20240928_htm.xml", "Apple Inc. FY2024 10-K"),
    ("msft", "msft-20240630.htm", "Microsoft Corporation FY2024 10-K"),
    ("nvda", "nvda-20240128.htm", "NVIDIA Corporation FY2024 10-K"),
]


def build_ingest_stack(
    settings: object = None,
    extractor: Extractor | None = None,
    embedder: EmbeddingProvider | None = None,
    llm: LLMProvider | None = None,
) -> tuple[GraphStore, VectorStore, EmbeddingProvider, Extractor]:
    s = settings or get_settings()
    graph = build_graph_store(s.neo4j_uri, s.neo4j_user, s.neo4j_password, s.graph_backend)  # type: ignore[attr-defined]
    vector = build_vector_store(s.database_url, s.embedding_dim, s.vector_backend)  # type: ignore[attr-defined]
    emb = embedder or _default_embedder(s)  # type: ignore[attr-defined]
    llm_provider = llm or build_llm_provider(
        s.llm_base_url, s.llm_api_key, s.llm_model, s.llm_timeout_seconds  # type: ignore[attr-defined]
    )
    ext = extractor or _default_extractor(llm_provider)
    return graph, vector, emb, ext


def _default_embedder(s: object) -> EmbeddingProvider:
    try:
        return SentenceTransformerProvider(getattr(s, "embedding_model"))
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
    def extract(self, chunk: object) -> object:  # type: ignore[override]
        from krag.domain.schemas import ExtractionResult

        cid = getattr(chunk, "chunk_id", "unknown")
        return ExtractionResult(chunk_id=cid, entities=[], relations=[])


def ingest(data_dir: Path, reingest: bool = False) -> dict[str, object]:
    settings = get_settings()
    graph, vector, embedder, extractor = build_ingest_stack(settings)
    manifest_path = data_dir / "processed" / "manifest.json"
    manifest: dict[str, str] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )

    t0 = time.perf_counter()
    stats = {
        "documents": 0, "chunks": 0, "entities": 0, "relations": 0,
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
            name_to_id[(entity.entity_type.value, Entity.normalize(entity.name))] = (
                entity.entity_id
            )
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
                        src_id=src_id, src_type=rr.src_type, rel=rr.rel,
                        dst_id=dst_id, dst_type=rr.dst_type,
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
        logger.info(
            "ingested %s: %d chunks, %d relations", doc_id, len(chunks), rel_count
        )

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


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reingest", action="store_true")
    args = parser.parse_args()
    stats = ingest(args.data_dir, reingest=args.reingest)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
