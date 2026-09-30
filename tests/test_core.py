"""Unit tests: ontology, chunking, resolution, stores, router, citations."""

from __future__ import annotations

import pytest

from krag.domain.ontology import EntityType, RelationType, relation_allowed
from krag.domain.schemas import Chunk, Entity, Relation, chunk_id_for
from krag.services.cypher_templates import (
    execute_in_memory,
    get_template,
    template_params,
)
from krag.services.resolver import EntityResolver
from krag.services.router import RuleRouter
from krag.storage.graph_store import InMemoryGraphStore
from krag.storage.vector_store import InMemoryVectorStore
from krag.services.embeddings import StubEmbeddingProvider


def test_chunk_ids_deterministic_and_shared() -> None:
    assert chunk_id_for("aapl", 3) == chunk_id_for("aapl", 3)
    assert chunk_id_for("aapl", 3) != chunk_id_for("msft", 3)


def test_ontology_allows_expected_triples() -> None:
    assert relation_allowed(
        EntityType.COMPANY, RelationType.HAS_SEGMENT, EntityType.SEGMENT
    )
    assert not relation_allowed(
        EntityType.PERSON, RelationType.HAS_SEGMENT, EntityType.SEGMENT
    )


def test_relation_rejects_off_ontology() -> None:
    with pytest.raises(ValueError):
        Relation(
            src_id="a", src_type=EntityType.PERSON, rel=RelationType.HAS_SEGMENT,
            dst_id="b", dst_type=EntityType.SEGMENT,
        )


def test_resolver_merges_aliases() -> None:
    from krag.domain.schemas import ExtractedEntity

    r = EntityResolver()
    e1 = r.resolve(
        ExtractedEntity(name="Apple Inc.", entity_type=EntityType.COMPANY), "c1"
    )
    e2 = r.resolve(
        ExtractedEntity(name="Apple", entity_type=EntityType.COMPANY), "c2"
    )
    assert e1.entity_id == e2.entity_id
    assert "Apple" in e2.aliases or e2.name == "Apple"
    assert set(e2.source_chunk_ids) == {"c1", "c2"}
    # different types do not merge
    e3 = r.resolve(
        ExtractedEntity(name="Apple", entity_type=EntityType.PRODUCT), "c3"
    )
    assert e3.entity_id != e1.entity_id


def test_resolver_does_not_merge_distinct_subsidiaries() -> None:
    from krag.domain.schemas import ExtractedEntity

    r = EntityResolver()
    e1 = r.resolve(
        ExtractedEntity(name="Apple Operations International",
                        entity_type=EntityType.SUBSIDIARY), "c1"
    )
    e2 = r.resolve(
        ExtractedEntity(name="Apple Operations Europe",
                        entity_type=EntityType.SUBSIDIARY), "c2"
    )
    assert e1.entity_id != e2.entity_id


def test_graph_merge_idempotent() -> None:
    g = InMemoryGraphStore()
    ent = Entity(
        entity_id=Entity.make_id("Apple", EntityType.COMPANY),
        name="Apple", entity_type=EntityType.COMPANY,
        aliases=["Apple Inc."], source_chunk_ids=["aapl#c00001"],
    )
    g.merge_entity(ent)
    g.merge_entity(ent.model_copy(update={"source_chunk_ids": ["aapl#c00002"]}))
    assert g.entity_count() == 1
    got = g.get_entity(ent.entity_id)
    assert got is not None
    assert set(got.source_chunk_ids) == {"aapl#c00001", "aapl#c00002"}

    seg = Entity(
        entity_id=Entity.make_id("iPhone", EntityType.PRODUCT),
        name="iPhone", entity_type=EntityType.PRODUCT,
        source_chunk_ids=["aapl#c00001"],
    )
    g.merge_entity(seg)
    rel = Relation(
        src_id=ent.entity_id, src_type=EntityType.COMPANY,
        rel=RelationType.SELLS_PRODUCT,
        dst_id=seg.entity_id, dst_type=EntityType.PRODUCT,
        source_chunk_ids=["aapl#c00001"],
    )
    g.merge_relation(rel)
    g.merge_relation(rel.model_copy(update={"source_chunk_ids": ["aapl#c00005"]}))
    assert g.relation_count() == 1


def test_cypher_templates_parameterized() -> None:
    for tid in ("entity_facts", "entity_neighborhood", "path_between",
                "relation_outgoing", "acquisitions_by_company"):
        q = get_template(tid)
        assert "$" in q  # parameterized, no f-string interpolation
        assert "{" not in q.replace("{entity_id:", "").replace("{rel:", "")
        assert template_params(tid)
    with pytest.raises(ValueError):
        get_template("drop_database")


def test_cypher_template_in_memory_execution() -> None:
    g = InMemoryGraphStore()
    apple = Entity(
        entity_id=Entity.make_id("Apple", EntityType.COMPANY),
        name="Apple", entity_type=EntityType.COMPANY,
        source_chunk_ids=["aapl#c00001"],
    )
    iphone = Entity(
        entity_id=Entity.make_id("iPhone", EntityType.PRODUCT),
        name="iPhone", entity_type=EntityType.PRODUCT,
        source_chunk_ids=["aapl#c00001"],
    )
    g.merge_entity(apple)
    g.merge_entity(iphone)
    g.merge_relation(Relation(
        src_id=apple.entity_id, src_type=EntityType.COMPANY,
        rel=RelationType.SELLS_PRODUCT, dst_id=iphone.entity_id,
        dst_type=EntityType.PRODUCT, source_chunk_ids=["aapl#c00001"],
    ))
    rows = execute_in_memory(
        "entity_facts", {"entity_id": apple.entity_id}, g
    )
    assert len(rows) == 1
    assert rows[0]["dst"] == "iPhone"
    assert rows[0]["chunks"] == ["aapl#c00001"]


def test_vector_store_search_orders_by_similarity() -> None:
    vs = InMemoryVectorStore()
    emb = StubEmbeddingProvider(dim=16)
    chunks = [
        Chunk(chunk_id=f"doc#c{i:05d}", doc_id="doc", index=i, text=t)
        for i, t in enumerate(["apple iphone revenue", "microsoft azure cloud", "nvidia gpu chips"])
    ]
    vecs = emb.embed([c.text for c in chunks])
    vs.add(chunks, vecs)
    assert vs.count() == 3
    q = emb.embed(["apple iphone revenue"])[0]
    hits = vs.search(q, top_k=2)
    assert hits[0].chunk_id == "doc#c00000"
    assert hits[0].score > hits[1].score


def test_router_routes_by_cues() -> None:
    router = RuleRouter()
    d = router.route("Which subsidiaries does Microsoft own?")
    assert d.path in ("graph", "hybrid")
    d = router.route("What is the definition of revenue recognition?")
    assert d.path in ("vector", "hybrid")
    d = router.route("Who is the CEO of Apple and what products does Apple sell?")
    assert d.path in ("graph", "hybrid")
    assert 0.0 <= d.confidence <= 1.0
    assert d.reasons
