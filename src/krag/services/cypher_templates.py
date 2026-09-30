"""Parameterized Cypher template library.

The model NEVER emits raw Cypher against the database — a security control
and a reproducibility guarantee. The QA layer picks a template by query type
and fills parameters only. Each template declares its required params.

Templates are stored once here and executed by both backends:
* Neo4j: the Cypher string runs via the driver with $params.
* In-memory: ``execute_in_memory`` implements the same semantics over the
  dict-backed store (used in unit tests / offline eval).
"""

from __future__ import annotations

from typing import Any

TEMPLATES: dict[str, dict[str, Any]] = {
    "entity_facts": {
        "description": "All 1-hop facts about one entity (both directions).",
        "params": ["entity_id"],
        "cypher": """
            MATCH (e:Entity {entity_id: $entity_id})-[r:REL]-(other:Entity)
            RETURN e.name AS src, r.rel AS rel, other.name AS dst,
                   r.source_chunk_ids AS chunks
            LIMIT 50
        """,
    },
    "entity_neighborhood": {
        "description": "N-hop neighborhood facts around an entity.",
        "params": ["entity_id", "max_hops", "limit"],
        "cypher": """
            MATCH (e:Entity {entity_id: $entity_id})-[r:REL*1..$max_hops]-(other:Entity)
            WITH e, r, other LIMIT $limit
            RETURN e.name AS src, [x IN r | x.rel][0] AS rel, other.name AS dst,
                   [x IN r | x.source_chunk_ids][0] AS chunks
        """,
    },
    "path_between": {
        "description": "Shortest path (<= max_hops) between two entities.",
        "params": ["src_id", "dst_id", "max_hops"],
        "cypher": """
            MATCH p = shortestPath(
              (a:Entity {entity_id: $src_id})-[r:REL*1..$max_hops]-(b:Entity {entity_id: $dst_id})
            )
            RETURN [n IN nodes(p) | n.name] AS nodes,
                   [x IN relationships(p) | x.rel] AS rels,
                   [x IN relationships(p) | x.source_chunk_ids] AS chunks
        """,
    },
    "relation_outgoing": {
        "description": "All outgoing edges of a given relation type from an entity.",
        "params": ["entity_id", "rel"],
        "cypher": """
            MATCH (e:Entity {entity_id: $entity_id})-[r:REL {rel: $rel}]->(other:Entity)
            RETURN other.name AS dst, other.entity_type AS dst_type,
                   r.source_chunk_ids AS chunks
        """,
    },
    "acquisitions_by_company": {
        "description": "Companies/acquisitions acquired by the given company.",
        "params": ["entity_id"],
        "cypher": """
            MATCH (e:Entity {entity_id: $entity_id})-[r:REL {rel: 'ACQUIRED'}]->(t:Entity)
            RETURN t.name AS target, t.entity_type AS target_type,
                   r.source_chunk_ids AS chunks
        """,
    },
}


def get_template(template_id: str) -> str:
    try:
        return str(TEMPLATES[template_id]["cypher"])
    except KeyError:
        raise ValueError(f"unknown Cypher template: {template_id}") from None


def template_params(template_id: str) -> list[str]:
    try:
        return list(TEMPLATES[template_id]["params"])
    except KeyError:
        raise ValueError(f"unknown Cypher template: {template_id}") from None


def execute_in_memory(
    template_id: str, params: dict[str, Any], store: Any
) -> list[dict[str, Any]]:
    """Same semantics as the Cypher templates, over the in-memory store.

    ``store`` is an InMemoryGraphStore (imported lazily to avoid a cycle).
    """
    missing = [p for p in template_params(template_id) if p not in params]
    if missing:
        raise ValueError(f"template {template_id} missing params: {missing}")

    if template_id == "entity_facts":
        ent = store.get_entity(params["entity_id"])
        if ent is None:
            return []
        return [
            {"src": s.name, "rel": rel.value, "dst": d.name, "chunks": chunks}
            for s, rel, d, chunks in store.neighbors(params["entity_id"], 1, 50)
        ]

    if template_id == "entity_neighborhood":
        hops = int(params.get("max_hops", 2))
        limit = int(params.get("limit", 50))
        ent = store.get_entity(params["entity_id"])
        if ent is None:
            return []
        return [
            {"src": s.name, "rel": rel.value, "dst": d.name, "chunks": chunks}
            for s, rel, d, chunks in store.neighbors(params["entity_id"], hops, limit)
        ]

    if template_id == "path_between":
        path = _bfs_path(
            store, params["src_id"], params["dst_id"], int(params.get("max_hops", 3))
        )
        if path is None:
            return []
        names = [store.get_entity(eid).name for eid in path[0]]  # type: ignore[union-attr]
        return [{"nodes": names, "rels": path[1], "chunks": path[2]}]

    if template_id == "relation_outgoing":
        ent = store.get_entity(params["entity_id"])
        if ent is None:
            return []
        rows = []
        for s, rel, d, chunks in store.neighbors(params["entity_id"], 1, 200):
            if s.entity_id == params["entity_id"] and rel.value == params["rel"]:
                rows.append(
                    {"dst": d.name, "dst_type": d.entity_type.value, "chunks": chunks}
                )
        return rows

    if template_id == "acquisitions_by_company":
        return execute_in_memory(
            "relation_outgoing",
            {"entity_id": params["entity_id"], "rel": "ACQUIRED"},
            store,
        )

    raise ValueError(f"unknown Cypher template: {template_id}")


def _bfs_path(
    store: Any, src_id: str, dst_id: str, max_hops: int
) -> tuple[list[str], list[str], list[list[str]]] | None:
    from collections import deque

    queue: deque[tuple[str, list[str], list[str], list[list[str]]]] = deque()
    queue.append((src_id, [src_id], [], []))
    visited = {src_id}
    while queue:
        eid, eids, rels, chunks = queue.popleft()
        if eid == dst_id and len(eids) > 1:
            return eids, rels, chunks
        if len(eids) - 1 >= max_hops:
            continue
        for s, rel, d, ch in store.neighbors(eid, 1, 200):
            nxt = d.entity_id if s.entity_id == eid else s.entity_id
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, eids + [nxt], rels + [rel.value], chunks + [ch]))
    return None
