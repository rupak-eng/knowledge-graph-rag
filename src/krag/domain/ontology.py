"""Controlled ontology: the fixed set of entity and relation types.

Everything the extractor may emit and everything the graph stores must come
from these enums. An open-ended "extract all entities" prompt is exactly what
we avoid: it produces graphs nobody can query.
"""

from __future__ import annotations

from enum import StrEnum


class EntityType(StrEnum):
    COMPANY = "Company"
    SUBSIDIARY = "Subsidiary"
    PERSON = "Person"
    PRODUCT = "Product"
    SEGMENT = "Segment"
    ACQUISITION = "Acquisition"
    METRIC = "Metric"
    GEOGRAPHY = "Geography"
    TECHNOLOGY = "Technology"
    RISK = "Risk"


class RelationType(StrEnum):
    HAS_SEGMENT = "HAS_SEGMENT"  # Company -> Segment
    HAS_SUBSIDIARY = "HAS_SUBSIDIARY"  # Company -> Subsidiary
    ACQUIRED = "ACQUIRED"  # Company -> Company|Acquisition
    SELLS_PRODUCT = "SELLS_PRODUCT"  # Company -> Product
    LED_BY = "LED_BY"  # Company -> Person
    REPORTS_METRIC = "REPORTS_METRIC"  # Company|Segment -> Metric
    OPERATES_IN = "OPERATES_IN"  # Company -> Geography
    COMPETES_WITH = "COMPETES_WITH"  # Company -> Company
    DEPENDS_ON = "DEPENDS_ON"  # Company|Product -> Company|Technology
    PARTNERS_WITH = "PARTNERS_WITH"  # Company -> Company
    FACES_RISK = "FACES_RISK"  # Company -> Risk
    USES_TECHNOLOGY = "USES_TECHNOLOGY"  # Company|Product -> Technology


# Allowed (source_type, relation, target_type) triples. The extractor and the
# graph store both validate against this table.
ALLOWED_RELATIONS: frozenset[tuple[EntityType, RelationType, EntityType]] = frozenset(
    {
        (EntityType.COMPANY, RelationType.HAS_SEGMENT, EntityType.SEGMENT),
        (EntityType.COMPANY, RelationType.HAS_SUBSIDIARY, EntityType.SUBSIDIARY),
        (EntityType.COMPANY, RelationType.ACQUIRED, EntityType.COMPANY),
        (EntityType.COMPANY, RelationType.ACQUIRED, EntityType.ACQUISITION),
        (EntityType.COMPANY, RelationType.SELLS_PRODUCT, EntityType.PRODUCT),
        (EntityType.COMPANY, RelationType.LED_BY, EntityType.PERSON),
        (EntityType.COMPANY, RelationType.REPORTS_METRIC, EntityType.METRIC),
        (EntityType.SEGMENT, RelationType.REPORTS_METRIC, EntityType.METRIC),
        (EntityType.COMPANY, RelationType.OPERATES_IN, EntityType.GEOGRAPHY),
        (EntityType.COMPANY, RelationType.COMPETES_WITH, EntityType.COMPANY),
        (EntityType.COMPANY, RelationType.DEPENDS_ON, EntityType.COMPANY),
        (EntityType.COMPANY, RelationType.DEPENDS_ON, EntityType.TECHNOLOGY),
        (EntityType.PRODUCT, RelationType.DEPENDS_ON, EntityType.TECHNOLOGY),
        (EntityType.COMPANY, RelationType.PARTNERS_WITH, EntityType.COMPANY),
        (EntityType.COMPANY, RelationType.FACES_RISK, EntityType.RISK),
        (EntityType.COMPANY, RelationType.USES_TECHNOLOGY, EntityType.TECHNOLOGY),
        (EntityType.PRODUCT, RelationType.USES_TECHNOLOGY, EntityType.TECHNOLOGY),
    }
)


def relation_allowed(src: EntityType, rel: RelationType, dst: EntityType) -> bool:
    return (src, rel, dst) in ALLOWED_RELATIONS
