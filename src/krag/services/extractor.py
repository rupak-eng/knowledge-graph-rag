"""Entity/relation extraction.

Two implementations behind one interface:

* ``SpacyRuleExtractor`` (default): spaCy NER mapped onto the controlled
  ontology plus dependency/regex relation rules, all validated by Pydantic.
  Deterministic, offline, $0 — a legitimate engineering choice for v1, and
  every output is measurable.
* ``LLMExtractor``: strict-schema JSON extraction via an OpenAI-compatible
  LLM, with Pydantic validation + retry. Used when an LLM is configured.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from krag.domain.ontology import EntityType, RelationType
from krag.domain.schemas import (
    Chunk,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)

logger = logging.getLogger(__name__)


class Extractor(Protocol):
    def extract(self, chunk: Chunk) -> ExtractionResult: ...


# ---------------------------------------------------------------------------
# Seed knowledge: canonical companies, segments, products, people, subsidiaries.
# Sourced from the FY2024 10-Ks themselves (verified during corpus inspection).
# ---------------------------------------------------------------------------

COMPANIES: dict[str, dict[str, list[str]]] = {
    "Apple": {
        "aliases": ["Apple Inc.", "Apple Computer"],
        "segments": ["Products", "Services"],
        "products": ["iPhone", "Mac", "iPad", "Apple Watch", "AirPods", "Apple TV",
                     "Apple Music", "iCloud"],
        "people": ["Tim Cook"],
        "subsidiaries": ["Braeburn Capital", "Apple Operations International",
                         "Apple Operations Europe"],
    },
    "Microsoft": {
        "aliases": ["Microsoft Corporation"],
        "segments": ["Productivity and Business Processes", "Intelligent Cloud",
                     "More Personal Computing"],
        "products": ["Azure", "Microsoft 365", "Office", "Windows", "Xbox",
                     "Surface", "LinkedIn", "GitHub", "Dynamics 365", "Teams"],
        "people": ["Satya Nadella"],
        "subsidiaries": ["LinkedIn Corporation", "GitHub, Inc.", "Nuance Communications",
                         "Activision Blizzard"],
    },
    "NVIDIA": {
        "aliases": ["NVIDIA Corporation"],
        "segments": ["Compute & Networking", "Graphics"],
        "products": ["GPU", "CUDA", "DGX", "GeForce", "H100", "Blackwell",
                     "Omniverse", "DRIVE"],
        "people": ["Jensen Huang"],
        "subsidiaries": ["Mellanox Technologies"],
    },
}

TECHNOLOGIES = [
    "artificial intelligence", "machine learning", "generative AI", "large language model",
    "cloud computing", "semiconductor", "GPU", "CPU", "5G", "augmented reality",
    "virtual reality", "quantum computing", "cybersecurity",
]

SPACY_TO_ONTOLOGY = {
    "PERSON": EntityType.PERSON,
    "PRODUCT": EntityType.PRODUCT,
    "GPE": EntityType.GEOGRAPHY,
    "LOC": EntityType.GEOGRAPHY,
    # ORG is disambiguated by _classify_org below
}

_COMPANY_LOOKUP: dict[str, str] = {}
for _canon, _info in COMPANIES.items():
    _COMPANY_LOOKUP[_canon.lower()] = _canon
    for _a in _info["aliases"]:
        _COMPANY_LOOKUP[_a.lower().rstrip(".")] = _canon

MONEY_RE = re.compile(r"\$\s?[\d,]+(?:\.\d+)?\s*(billion|million|trillion)?", re.IGNORECASE)


def _classify_org(name: str, sentence: str) -> EntityType:
    low = name.lower().rstrip(".")
    if low in _COMPANY_LOOKUP:
        return EntityType.COMPANY
    for info in COMPANIES.values():
        if name in info["subsidiaries"] or low in {s.lower() for s in info["subsidiaries"]}:
            return EntityType.SUBSIDIARY
    if "subsidiary" in sentence.lower() and ("acquir" in sentence.lower() or "subsidiari" in sentence.lower()):
        return EntityType.SUBSIDIARY
    return EntityType.COMPANY


class SpacyRuleExtractor:
    def __init__(self, model: str = "en_core_web_sm") -> None:
        import spacy

        self._nlp = spacy.load(model)
        logger.info("Loaded spaCy model %s", model)

    # -- public ---------------------------------------------------------
    def extract(self, chunk: Chunk) -> ExtractionResult:
        doc = self._nlp(chunk.text[:20000])  # guard against pathological chunks
        entities: dict[str, ExtractedEntity] = {}
        relations: list[ExtractedRelation] = []

        def add_entity(name: str, etype: EntityType, conf: float) -> None:
            key = f"{etype.value}|{name.lower()}"
            if key not in entities or entities[key].confidence < conf:
                entities[key] = ExtractedEntity(
                    name=name.strip(), entity_type=etype, confidence=conf
                )

        # 1. NER pass -----------------------------------------------------
        for sent in doc.sents:
            sent_text = sent.text
            for ent in sent.ents:
                label = ent.label_
                if label == "ORG":
                    add_entity(ent.text, _classify_org(ent.text, sent_text), 0.8)
                elif label in SPACY_TO_ONTOLOGY:
                    add_entity(ent.text, SPACY_TO_ONTOLOGY[label], 0.8)
            self._extract_relations(sent_text, add_entity, relations)

        # 2. Seed-knowledge pass: canonical names spaCy may miss -----------
        low_text = chunk.text.lower()
        for canon, info in COMPANIES.items():
            for seg in info["segments"]:
                if seg.lower() in low_text:
                    add_entity(seg, EntityType.SEGMENT, 0.9)
                    relations.append(
                        ExtractedRelation(
                            src_name=canon, src_type=EntityType.COMPANY,
                            rel=RelationType.HAS_SEGMENT,
                            dst_name=seg, dst_type=EntityType.SEGMENT,
                            confidence=0.9,
                            evidence=self._evidence(chunk.text, seg),
                        )
                    )
            for prod in info["products"]:
                if re.search(rf"\b{re.escape(prod)}\b", chunk.text, re.IGNORECASE):
                    add_entity(prod, EntityType.PRODUCT, 0.85)
            for person in info["people"]:
                if person.lower() in low_text:
                    add_entity(person, EntityType.PERSON, 0.95)
            for sub in info["subsidiaries"]:
                if sub.lower().rstrip(".") in low_text:
                    add_entity(sub, EntityType.SUBSIDIARY, 0.9)

        result = ExtractionResult(
            chunk_id=chunk.chunk_id,
            entities=list(entities.values()),
            relations=relations,
        )
        return self._dedupe(result)

    # -- relation rules ---------------------------------------------------
    def _extract_relations(
        self,
        sent: str,
        add_entity: object,  # callable
        relations: list[ExtractedRelation],
    ) -> None:
        low = sent.lower()

        # ACQUIRED: "X acquired Y"
        m = re.search(
            r"([A-Z][\w&.,'’\- ]+?)\s+(?:acquired|completed the acquisition of)\s+"
            r"([A-Z][\w&.,'’\- ]+?)(?:\.|,| for | in \d)",
            sent,
        )
        if m:
            relations.append(
                ExtractedRelation(
                    src_name=m.group(1).strip(), src_type=EntityType.COMPANY,
                    rel=RelationType.ACQUIRED,
                    dst_name=m.group(2).strip(), dst_type=EntityType.COMPANY,
                    confidence=0.75, evidence=sent.strip(),
                )
            )

        # LED_BY: "X, CEO of Y" / "Y CEO X" / "X is the CEO"
        m = re.search(
            r"([A-Z][a-z]+ [A-Z][a-z]+)[,.]?\s+(?:is |was |serves as |has served as )?"
            r"(?:the |our )?(Chief Executive Officer|CEO|CFO|Chief Financial Officer)",
            sent,
        )
        if m:
            person, role = m.group(1), m.group(2)
            company = self._company_in_context(sent)
            if company:
                relations.append(
                    ExtractedRelation(
                        src_name=company, src_type=EntityType.COMPANY,
                        rel=RelationType.LED_BY,
                        dst_name=person, dst_type=EntityType.PERSON,
                        confidence=0.85, evidence=sent.strip(),
                    )
                )

        # COMPETES_WITH
        if re.search(r"\bcompet\w*\s+with\b", low):
            company = self._company_in_context(sent)
            for ent_like in re.findall(r"with ([A-Z][\w&.,'’\- ]+?)(?:\.|,|;| and )", sent):
                if company and ent_like.strip().lower() != company.lower():
                    relations.append(
                        ExtractedRelation(
                            src_name=company, src_type=EntityType.COMPANY,
                            rel=RelationType.COMPETES_WITH,
                            dst_name=ent_like.strip(), dst_type=EntityType.COMPANY,
                            confidence=0.6, evidence=sent.strip(),
                        )
                    )

        # FACES_RISK: risk-factor language
        if re.search(r"\brisk factors?\b", low) or "could be adversely affected" in low:
            company = self._company_in_context(sent)
            if company:
                risk = sent.strip()[:160]
                relations.append(
                    ExtractedRelation(
                        src_name=company, src_type=EntityType.COMPANY,
                        rel=RelationType.FACES_RISK,
                        dst_name=risk, dst_type=EntityType.RISK,
                        confidence=0.6, evidence=sent.strip(),
                    )
                )

        # REPORTS_METRIC: money figures near a known company/segment
        for money in MONEY_RE.finditer(sent):
            company = self._company_in_context(sent)
            if company:
                metric_name = f"{money.group(0).strip()} ({sent.strip()[:120]}...)"
                relations.append(
                    ExtractedRelation(
                        src_name=company, src_type=EntityType.COMPANY,
                        rel=RelationType.REPORTS_METRIC,
                        dst_name=metric_name, dst_type=EntityType.METRIC,
                        confidence=0.55, evidence=sent.strip(),
                    )
                )
                break  # one metric per sentence keeps the graph queryable

        # USES_TECHNOLOGY
        for tech in TECHNOLOGIES:
            if tech in low:
                company = self._company_in_context(sent)
                if company:
                    relations.append(
                        ExtractedRelation(
                            src_name=company, src_type=EntityType.COMPANY,
                            rel=RelationType.USES_TECHNOLOGY,
                            dst_name=tech, dst_type=EntityType.TECHNOLOGY,
                            confidence=0.6, evidence=sent.strip(),
                        )
                    )
                    break

    def _company_in_context(self, sent: str) -> str | None:
        low = sent.lower()
        for canon in COMPANIES:
            if canon.lower() in low:
                return canon
        return None

    @staticmethod
    def _evidence(text: str, needle: str) -> str:
        idx = text.lower().find(needle.lower())
        if idx < 0:
            return text[:160]
        return text[max(0, idx - 60) : idx + 160].strip()

    @staticmethod
    def _dedupe(result: ExtractionResult) -> ExtractionResult:
        seen: set[tuple[str, str, str, str]] = set()
        deduped: list[ExtractedRelation] = []
        for r in result.relations:
            key = (r.src_name.lower(), r.rel.value, r.dst_name.lower(), r.evidence[:40])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return result.model_copy(update={"relations": deduped})


# ---------------------------------------------------------------------------
# LLM extractor (production path when an LLM provider is configured)
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You extract entities and relations from SEC filing text into a strict schema.
Entity types: Company, Subsidiary, Person, Product, Segment, Acquisition, Metric, Geography, Technology, Risk.
Relation types: HAS_SEGMENT, HAS_SUBSIDIARY, ACQUIRED, SELLS_PRODUCT, LED_BY, REPORTS_METRIC,
OPERATES_IN, COMPETES_WITH, DEPENDS_ON, PARTNERS_WITH, FACES_RISK, USES_TECHNOLOGY.
Respond with JSON only: {"entities": [{"name": str, "entity_type": str, "confidence": float}],
"relations": [{"src_name": str, "src_type": str, "rel": str, "dst_name": str, "dst_type": str,
"confidence": float, "evidence": str}]}. Emit only relations supported by the text."""


class LLMExtractor:
    """Strict-schema LLM extraction with Pydantic validation and retry."""

    def __init__(self, llm: object, max_retries: int = 2) -> None:
        self._llm = llm  # krag.services.llm.LLMProvider
        self._max_retries = max_retries

    def extract(self, chunk: Chunk) -> ExtractionResult:
        last_err: Exception | None = None
        for _ in range(self._max_retries + 1):
            try:
                raw = self._llm.generate(  # type: ignore[union-attr]
                    EXTRACTION_SYSTEM_PROMPT
                    + "\n\nTEXT:\n"
                    + chunk.text[:6000]
                )
                data = json.loads(_strip_code_fences(raw.text))
                entities = [ExtractedEntity(**e) for e in data.get("entities", [])]
                relations = [ExtractedRelation(**r) for r in data.get("relations", [])]
                return ExtractionResult(
                    chunk_id=chunk.chunk_id, entities=entities, relations=relations
                )
            except Exception as exc:  # noqa: BLE001 - retry then surface
                last_err = exc
                logger.warning("LLM extraction failed, retrying: %s", exc)
        raise RuntimeError(f"LLM extraction failed after retries: {last_err}")


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()
