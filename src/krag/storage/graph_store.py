"""Graph store: protocol + Neo4j backend (primary) + in-memory backend (tests).

All Cypher is parameterized ($params) — the model layer never emits raw Cypher.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from krag.domain.ontology import EntityType, RelationType
from krag.domain.schemas import Entity, Relation

logger = logging.getLogger(__name__)


class GraphStore(Protocol):
    """Storage interface for the entity graph."""

    def merge_entity(self, entity: Entity) -> Entity: ...
    def merge_relation(self, relation: Relation) -> None: ...
    def get_entity(self, entity_id: str) -> Entity | None: ...
    def find_by_name(self, name: str) -> Entity | None: ...
    def neighbors(
        self, entity_id: str, max_hops: int = 2, limit: int = 50
    ) -> list[tuple[Entity, RelationType, Entity, list[str]]]: ...
    def run_template(self, template_id: str, params: dict[str, Any]) -> list[dict[str, Any]]: ...
    def entity_count(self) -> int: ...
    def relation_count(self) -> int: ...
    def clear(self) -> None: ...
    def close(self) -> None: ...


class InMemoryGraphStore:
    """Dict-backed graph for unit tests and offline benchmarks."""

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._by_norm: dict[str, str] = {}  # (type|normalized_name) -> entity_id
        self._relations: list[Relation] = []
        self._adj: dict[str, list[int]] = {}  # entity_id -> relation indexes

    def merge_entity(self, entity: Entity) -> Entity:
        key = f"{entity.entity_type.value}|{Entity.normalize(entity.name)}"
        existing_id = self._by_norm.get(key)
        if existing_id is not None:
            existing = self._entities[existing_id]
            merged_aliases = list(dict.fromkeys(existing.aliases + entity.aliases + [entity.name]))
            merged_chunks = list(dict.fromkeys(existing.source_chunk_ids + entity.source_chunk_ids))
            merged = existing.model_copy(
                update={"aliases": merged_aliases, "source_chunk_ids": merged_chunks}
            )
            self._entities[existing_id] = merged
            return merged
        self._entities[entity.entity_id] = entity
        self._by_norm[key] = entity.entity_id
        self._adj.setdefault(entity.entity_id, [])
        return entity

    def merge_relation(self, relation: Relation) -> None:
        for i, r in enumerate(self._relations):
            if (r.src_id, r.rel, r.dst_id) == (relation.src_id, relation.rel, relation.dst_id):
                merged = list(dict.fromkeys(r.source_chunk_ids + relation.source_chunk_ids))
                self._relations[i] = r.model_copy(update={"source_chunk_ids": merged})
                return
        self._relations.append(relation)
        self._adj.setdefault(relation.src_id, []).append(len(self._relations) - 1)
        self._adj.setdefault(relation.dst_id, []).append(len(self._relations) - 1)

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def find_by_name(self, name: str) -> Entity | None:
        norm = Entity.normalize(name)
        for ent in self._entities.values():
            if Entity.normalize(ent.name) == norm or norm in {
                Entity.normalize(a) for a in ent.aliases
            }:
                return ent
        return None

    def neighbors(
        self, entity_id: str, max_hops: int = 2, limit: int = 50
    ) -> list[tuple[Entity, RelationType, Entity, list[str]]]:
        out: list[tuple[Entity, RelationType, Entity, list[str]]] = []
        seen: set[str] = {entity_id}
        frontier = [entity_id]
        for _ in range(max_hops):
            nxt: list[str] = []
            for eid in frontier:
                for ri in self._adj.get(eid, []):
                    rel = self._relations[ri]
                    other_id = rel.dst_id if rel.src_id == eid else rel.src_id
                    if other_id in seen:
                        continue
                    seen.add(other_id)
                    src = self._entities.get(rel.src_id)
                    dst = self._entities.get(rel.dst_id)
                    if src is None or dst is None:
                        continue
                    out.append((src, rel.rel, dst, rel.source_chunk_ids))
                    if len(out) >= limit:
                        return out
                    nxt.append(other_id)
            frontier = nxt
        return out

    def run_template(self, template_id: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        # Templates are implemented in krag.services.cypher_templates and share
        # this executor for the in-memory backend.
        from krag.services import cypher_templates

        return cypher_templates.execute_in_memory(template_id, params, self)

    def entity_count(self) -> int:
        return len(self._entities)

    def relation_count(self) -> int:
        return len(self._relations)

    def clear(self) -> None:
        self._entities.clear()
        self._by_norm.clear()
        self._relations.clear()
        self._adj.clear()

    def close(self) -> None:
        pass


class Neo4jGraphStore:
    """Production backend using the official neo4j driver.

    Writes use MERGE so ingestion is idempotent; every edge carries the source
    chunk IDs so facts can be cited back to text.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        with self._driver.session() as s:
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE")
        logger.info("Connected to Neo4j at %s", uri)

    def merge_entity(self, entity: Entity) -> Entity:
        # No APOC dependency: read-then-write union keeps this working on
        # stock neo4j:*-community images.
        with self._driver.session() as s:
            existing = s.run(
                "MATCH (e:Entity {entity_id: $eid}) "
                "RETURN e.aliases AS aliases, e.source_chunk_ids AS chunks",
                eid=entity.entity_id,
            ).single()
            if existing is None:
                s.run(
                    "CREATE (e:Entity {entity_id: $eid, name: $name, "
                    "entity_type: $etype, aliases: $aliases, source_chunk_ids: $chunks})",
                    eid=entity.entity_id,
                    name=entity.name,
                    etype=entity.entity_type.value,
                    aliases=entity.aliases,
                    chunks=entity.source_chunk_ids,
                )
                return entity
            aliases = list(
                dict.fromkeys(list(existing["aliases"] or []) + entity.aliases + [entity.name])
            )
            chunks = list(
                dict.fromkeys(list(existing["chunks"] or []) + entity.source_chunk_ids)
            )
            s.run(
                "MATCH (e:Entity {entity_id: $eid}) "
                "SET e.aliases = $aliases, e.source_chunk_ids = $chunks",
                eid=entity.entity_id,
                aliases=aliases,
                chunks=chunks,
            )
            return entity.model_copy(
                update={"aliases": aliases, "source_chunk_ids": chunks}
            )

    def merge_relation(self, relation: Relation) -> None:
        with self._driver.session() as s:
            existing = s.run(
                "MATCH (a:Entity {entity_id: $src})-[r:REL {rel: $rel}]->"
                "(b:Entity {entity_id: $dst}) RETURN r.source_chunk_ids AS chunks",
                src=relation.src_id,
                dst=relation.dst_id,
                rel=relation.rel.value,
            ).single()
            if existing is None:
                s.run(
                    "MATCH (a:Entity {entity_id: $src}), (b:Entity {entity_id: $dst}) "
                    "CREATE (a)-[r:REL {rel: $rel, source_chunk_ids: $chunks}]->(b)",
                    src=relation.src_id,
                    dst=relation.dst_id,
                    rel=relation.rel.value,
                    chunks=relation.source_chunk_ids,
                )
            else:
                chunks = list(
                    dict.fromkeys(
                        list(existing["chunks"] or []) + relation.source_chunk_ids
                    )
                )
                s.run(
                    "MATCH (a:Entity {entity_id: $src})-[r:REL {rel: $rel}]->"
                    "(b:Entity {entity_id: $dst}) SET r.source_chunk_ids = $chunks",
                    src=relation.src_id,
                    dst=relation.dst_id,
                    rel=relation.rel.value,
                    chunks=chunks,
                )

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._driver.session() as s:
            rec = s.run(
                "MATCH (e:Entity {entity_id: $eid}) RETURN e", eid=entity_id
            ).single()
        return _row_to_entity(rec["e"]) if rec else None

    def find_by_name(self, name: str) -> Entity | None:
        with self._driver.session() as s:
            rec = s.run(
                "MATCH (e:Entity) WHERE toLower(e.name) = toLower($name) RETURN e LIMIT 1",
                name=name,
            ).single()
        return _row_to_entity(rec["e"]) if rec else None

    def neighbors(
        self, entity_id: str, max_hops: int = 2, limit: int = 50
    ) -> list[tuple[Entity, RelationType, Entity, list[str]]]:
        with self._driver.session() as s:
            rows = s.run(
                """
                MATCH (a:Entity {entity_id: $eid})-[r:REL*1..$hops]-(b:Entity)
                RETURN a, r, b LIMIT $limit
                """,
                eid=entity_id,
                hops=max_hops,
                limit=limit,
            ).data()
        out = []
        for row in rows:
            # r is a path; take the first hop for the fact rendering
            rel = row["r"][0] if isinstance(row["r"], list) else row["r"]
            out.append(
                (
                    _row_to_entity(row["a"]),
                    RelationType(rel["rel"]),
                    _row_to_entity(row["b"]),
                    list(rel.get("source_chunk_ids", [])),
                )
            )
        return out

    def run_template(self, template_id: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        from krag.services import cypher_templates

        query = cypher_templates.get_template(template_id)
        with self._driver.session() as s:
            return s.run(query, **params).data()

    def entity_count(self) -> int:
        with self._driver.session() as s:
            return s.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]  # type: ignore[index]

    def relation_count(self) -> int:
        with self._driver.session() as s:
            return s.run("MATCH ()-[r:REL]->() RETURN count(r) AS c").single()["c"]  # type: ignore[index]

    def clear(self) -> None:
        with self._driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        self._driver.close()


def _row_to_entity(row: Any) -> Entity:
    return Entity(
        entity_id=row["entity_id"],
        name=row["name"],
        entity_type=EntityType(row["entity_type"]),
        aliases=list(row.get("aliases", []) or []),
        source_chunk_ids=list(row.get("source_chunk_ids", []) or []),
    )


def build_graph_store(uri: str, user: str, password: str, backend: str) -> GraphStore:
    if backend == "real":
        return Neo4jGraphStore(uri, user, password)
    if backend == "memory":
        return InMemoryGraphStore()
    raise ValueError(f"unknown GRAPH_BACKEND: {backend}")
