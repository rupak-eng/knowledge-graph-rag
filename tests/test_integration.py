"""Integration tests against REAL Neo4j + pgvector.

These run only when the services are reachable AND the required env vars are
set (local dev / docker-compose). CI runs the unit suite against in-memory
backends instead.

Required env (never committed — tests fail closed without them):
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATABASE_URL

Isolation guarantees:
- Neo4j fixtures use a unique ``TEST_`` name prefix and are deleted by prefix
  in teardown. The corpus graph is never cleared or touched.
- pgvector fixtures use a dedicated ``test_chunks`` table. The production
  ``chunks`` table is never created, truncated, or written.
"""

from __future__ import annotations

import os
import socket
import uuid

import pytest

from krag.domain.ontology import EntityType, RelationType
from krag.domain.schemas import Chunk, Entity, Relation
from krag.services.embeddings import StubEmbeddingProvider
from krag.storage.graph_store import Neo4jGraphStore
from krag.storage.vector_store import PgVectorStore


def _tcp_open(host: str, port: int) -> bool:
    try:
        socket.create_connection((host, port), timeout=2).close()
        return True
    except OSError:
        return False


def _env(name: str) -> str | None:
    return os.environ.get(name)


NEO4J_ENV = all(_env(k) for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"))
PG_ENV = bool(_env("DATABASE_URL"))

needs_neo4j = pytest.mark.skipif(
    not (_tcp_open("127.0.0.1", 7687) and NEO4J_ENV),
    reason="Neo4j not reachable or NEO4J_* env vars unset",
)
needs_pg = pytest.mark.skipif(
    not (_tcp_open("127.0.0.1", 5432) and PG_ENV),
    reason="Postgres not reachable or DATABASE_URL unset",
)

# Unique per-run prefix so parallel/leftover fixtures never collide.
TEST_PREFIX = f"KRAGTEST_{uuid.uuid4().hex[:8]}_"


def _test_entity(name: str, etype: EntityType, chunk: str) -> Entity:
    pname = TEST_PREFIX + name
    return Entity(
        entity_id=Entity.make_id(pname, etype),
        name=pname,
        entity_type=etype,
        aliases=[],
        source_chunk_ids=[chunk],
    )


@needs_neo4j
class TestNeo4jGraphStore:
    @pytest.fixture()
    def store(self) -> Neo4jGraphStore:
        uri, user, password = _env("NEO4J_URI"), _env("NEO4J_USER"), _env("NEO4J_PASSWORD")
        assert uri and user and password
        s = Neo4jGraphStore(uri, user, password)
        s.delete_by_name_prefix(TEST_PREFIX)  # clean slate for our prefix only
        yield s
        s.delete_by_name_prefix(TEST_PREFIX)
        s.close()

    def test_merge_entity_idempotent(self, store: Neo4jGraphStore) -> None:
        apple = _test_entity("Apple", EntityType.COMPANY, "aapl#c00001")
        store.merge_entity(apple)
        store.merge_entity(apple.model_copy(update={"source_chunk_ids": ["aapl#c00002"]}))
        got = store.get_entity(apple.entity_id)
        assert got is not None
        assert set(got.source_chunk_ids) == {"aapl#c00001", "aapl#c00002"}

    def test_merge_relation_idempotent(self, store: Neo4jGraphStore) -> None:
        apple = _test_entity("Apple", EntityType.COMPANY, "aapl#c00001")
        iphone = _test_entity("iPhone", EntityType.PRODUCT, "aapl#c00001")
        store.merge_entity(apple)
        store.merge_entity(iphone)
        rel = Relation(
            src_id=apple.entity_id,
            src_type=EntityType.COMPANY,
            rel=RelationType.SELLS_PRODUCT,
            dst_id=iphone.entity_id,
            dst_type=EntityType.PRODUCT,
            source_chunk_ids=["aapl#c00001"],
        )
        store.merge_relation(rel)
        store.merge_relation(rel.model_copy(update={"source_chunk_ids": ["aapl#c00007"]}))
        rows = store.run_template("entity_facts", {"entity_id": apple.entity_id})
        assert len(rows) == 1
        assert set(rows[0]["chunks"]) == {"aapl#c00001", "aapl#c00007"}

    def test_find_by_name_case_insensitive(self, store: Neo4jGraphStore) -> None:
        apple = _test_entity("Apple", EntityType.COMPANY, "aapl#c00001")
        store.merge_entity(apple)
        assert store.find_by_name(apple.name.lower()) is not None
        assert store.find_by_name(apple.name.upper()) is not None
        assert store.find_by_name(TEST_PREFIX + "nonexistent-corp") is None

    def test_parameterized_template_execution(self, store: Neo4jGraphStore) -> None:
        apple = _test_entity("Apple", EntityType.COMPANY, "aapl#c00001")
        iphone = _test_entity("iPhone", EntityType.PRODUCT, "aapl#c00001")
        store.merge_entity(apple)
        store.merge_entity(iphone)
        store.merge_relation(
            Relation(
                src_id=apple.entity_id,
                src_type=EntityType.COMPANY,
                rel=RelationType.SELLS_PRODUCT,
                dst_id=iphone.entity_id,
                dst_type=EntityType.PRODUCT,
                source_chunk_ids=["aapl#c00001"],
            )
        )
        rows = store.run_template("entity_facts", {"entity_id": apple.entity_id})
        assert len(rows) == 1
        assert rows[0]["dst"] == iphone.name
        assert rows[0]["rel"] == "SELLS_PRODUCT"
        assert "aapl#c00001" in rows[0]["chunks"]

    def test_neighbors_multi_hop(self, store: Neo4jGraphStore) -> None:
        apple = _test_entity("Apple", EntityType.COMPANY, "aapl#c00001")
        iphone = _test_entity("iPhone", EntityType.PRODUCT, "aapl#c00001")
        tim = _test_entity("Tim Cook", EntityType.PERSON, "aapl#c00002")
        for e in (apple, iphone, tim):
            store.merge_entity(e)
        store.merge_relation(
            Relation(
                src_id=apple.entity_id,
                src_type=EntityType.COMPANY,
                rel=RelationType.SELLS_PRODUCT,
                dst_id=iphone.entity_id,
                dst_type=EntityType.PRODUCT,
                source_chunk_ids=["aapl#c00001"],
            )
        )
        store.merge_relation(
            Relation(
                src_id=apple.entity_id,
                src_type=EntityType.COMPANY,
                rel=RelationType.LED_BY,
                dst_id=tim.entity_id,
                dst_type=EntityType.PERSON,
                source_chunk_ids=["aapl#c00002"],
            )
        )
        n1 = store.neighbors(apple.entity_id, max_hops=1)
        assert len(n1) == 2
        n2 = store.neighbors(apple.entity_id, max_hops=2)
        assert len(n2) >= 2

    def test_prefix_cleanup_leaves_corpus_untouched(self, store: Neo4jGraphStore) -> None:
        """Deleting our prefix must not remove corpus entities."""
        before = store.entity_count()
        apple = _test_entity("Apple", EntityType.COMPANY, "aapl#c00001")
        store.merge_entity(apple)
        assert store.entity_count() == before + 1
        assert store.delete_by_name_prefix(TEST_PREFIX) == 1
        assert store.entity_count() == before


@needs_pg
class TestPgVectorStore:
    @pytest.fixture()
    def store(self) -> PgVectorStore:
        url = _env("DATABASE_URL")
        assert url
        s = PgVectorStore(url, dim=16, table="test_chunks")
        s.clear()  # only touches test_chunks
        yield s
        s.clear()
        s.close()

    def test_add_search_upsert(self, store: PgVectorStore) -> None:
        emb = StubEmbeddingProvider(dim=16)
        chunks = [
            Chunk(
                chunk_id="t#c00000",
                doc_id="t",
                index=0,
                section="S1",
                text="apple iphone revenue",
            ),
            Chunk(
                chunk_id="t#c00001",
                doc_id="t",
                index=1,
                section="S1",
                text="microsoft azure cloud",
            ),
        ]
        vecs = emb.embed([c.text for c in chunks])
        assert store.add(chunks, vecs) == 2
        assert store.count() == 2

        q = emb.embed(["apple iphone revenue"])[0]
        hits = store.search(q, top_k=2)
        assert hits[0].chunk_id == "t#c00000"
        assert hits[0].score >= hits[1].score

        # upsert on chunk_id: re-adding changes nothing in count
        store.add(chunks, vecs)
        assert store.count() == 2

        got = store.get("t#c00001")
        assert got is not None and got.doc_id == "t"
        assert store.get("missing") is None

    def test_production_table_untouched(self, store: PgVectorStore) -> None:
        """The test table must be separate from the production chunks table."""
        url = _env("DATABASE_URL")
        assert url
        prod = PgVectorStore(url, dim=384)
        prod_count = prod.count()
        prod.close()
        # writing to test_chunks must not change the production table
        emb = StubEmbeddingProvider(dim=16)
        c = Chunk(chunk_id="t#probe", doc_id="t", index=0, section="S", text="probe")
        store.add([c], emb.embed([c.text]))
        prod2 = PgVectorStore(url, dim=384)
        assert prod2.count() == prod_count
        prod2.close()
