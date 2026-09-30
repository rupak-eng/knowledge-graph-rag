"""Entity resolution: collapse duplicate mentions into canonical nodes.

"Acme Corp", "Acme Corporation" and "ACME" must become ONE node, or the graph
is unqueryable. Strategy (in order):

1. Normalize (lowercase, strip corporate suffixes/punctuation).
2. Exact normalized match within the same entity type -> merge.
3. Alias match: a mention equal to a stored alias -> merge.
4. High-confidence substring containment for long names (e.g. "Apple Inc." vs
   "Apple") -> merge, aliases recorded.

Embedding-similarity merging is intentionally NOT in v1: on 10-K text it
merges distinct subsidiaries ("Apple Operations International" vs
"Apple Operations Europe" are 0.93 cosine but different entities), which is
worse than a duplicate node. Documented in README "What didn't work" if a
tuned threshold experiment fails — the experiment is left to eval/.
"""

from __future__ import annotations

import logging

from krag.domain.ontology import EntityType
from krag.domain.schemas import Entity, ExtractedEntity

logger = logging.getLogger(__name__)


class EntityResolver:
    """Stateful resolver: feed it extractions, get canonical entities."""

    def __init__(self) -> None:
        self._canonical: dict[str, Entity] = {}  # entity_id -> Entity
        self._index: dict[tuple[str, str], str] = {}  # (type, normalized) -> entity_id

    def resolve(self, extracted: ExtractedEntity, chunk_id: str) -> Entity:
        norm = Entity.normalize(extracted.name)
        key = (extracted.entity_type.value, norm)

        entity_id = self._index.get(key)
        if entity_id is None:
            entity_id = self._alias_lookup(extracted.name, extracted.entity_type)
        if entity_id is None:
            entity_id = self._containment_lookup(norm, extracted.entity_type)

        if entity_id is None:
            entity = Entity(
                entity_id=Entity.make_id(extracted.name, extracted.entity_type),
                name=extracted.name.strip(),
                entity_type=extracted.entity_type,
                aliases=[],
                confidence=extracted.confidence,
                source_chunk_ids=[chunk_id],
            )
            self._canonical[entity.entity_id] = entity
            self._index[key] = entity.entity_id
            return entity

        entity = self._canonical[entity_id]
        aliases = list(entity.aliases)
        if extracted.name.strip() not in aliases and extracted.name.strip() != entity.name:
            aliases.append(extracted.name.strip())
        chunks = list(dict.fromkeys(entity.source_chunk_ids + [chunk_id]))
        merged = entity.model_copy(
            update={
                "aliases": aliases,
                "source_chunk_ids": chunks,
                "confidence": max(entity.confidence, extracted.confidence),
            }
        )
        self._canonical[entity_id] = merged
        self._index[key] = entity_id
        return merged

    def _alias_lookup(self, name: str, etype: EntityType) -> str | None:
        norm = Entity.normalize(name)
        for eid, ent in self._canonical.items():
            if ent.entity_type != etype:
                continue
            if norm in {Entity.normalize(a) for a in ent.aliases}:
                return eid
        return None

    def _containment_lookup(self, norm: str, etype: EntityType) -> str | None:
        # Only for names long enough that containment is meaningful; prevents
        # "Apple" swallowing "Apple Watch" (different types anyway) and
        # avoids merging distinct short names.
        if len(norm) < 8:
            return None
        for (t, existing_norm), eid in self._index.items():
            if t != etype.value or len(existing_norm) < 8:
                continue
            if norm in existing_norm or existing_norm in norm:
                # require shared token prefix to avoid "Apple Operations
                # International" == "Apple Operations Europe" merges
                if norm.split()[0] == existing_norm.split()[0]:
                    return eid
        return None

    def entities(self) -> list[Entity]:
        return list(self._canonical.values())

    def stats(self) -> dict[str, int]:
        return {
            "canonical_entities": len(self._canonical),
            "aliases": sum(len(e.aliases) for e in self._canonical.values()),
        }
