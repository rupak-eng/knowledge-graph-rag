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
from krag.services.llm import LLMProvider

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
        "products": [
            "iPhone",
            "Mac",
            "iPad",
            "Apple Watch",
            "AirPods",
            "Apple TV",
            "Apple Music",
            "iCloud",
        ],
        "people": ["Tim Cook"],
        "subsidiaries": [
            "Braeburn Capital",
            "Apple Operations International",
            "Apple Operations Europe",
        ],
    },
    "Microsoft": {
        "aliases": ["Microsoft Corporation"],
        "segments": [
            "Productivity and Business Processes",
            "Intelligent Cloud",
            "More Personal Computing",
        ],
        "products": [
            "Azure",
            "Microsoft 365",
            "Office",
            "Windows",
            "Xbox",
            "Surface",
            "LinkedIn",
            "GitHub",
            "Dynamics 365",
            "Teams",
        ],
        "people": ["Satya Nadella"],
        "subsidiaries": [
            "LinkedIn Corporation",
            "GitHub, Inc.",
            "Nuance Communications",
            "Activision Blizzard",
        ],
    },
    "NVIDIA": {
        "aliases": ["NVIDIA Corporation"],
        "segments": ["Compute & Networking", "Graphics"],
        "products": ["GPU", "CUDA", "DGX", "GeForce", "H100", "Blackwell", "Omniverse", "DRIVE"],
        "people": ["Jensen Huang"],
        "subsidiaries": ["Mellanox Technologies"],
    },
}

DOC_COMPANY = {"aapl": "Apple", "msft": "Microsoft", "nvda": "NVIDIA"}

TECHNOLOGIES = [
    "artificial intelligence",
    "machine learning",
    "generative AI",
    "large language model",
    "cloud computing",
    "semiconductor",
    "GPU",
    "CPU",
    "5G",
    "augmented reality",
    "virtual reality",
    "quantum computing",
    "cybersecurity",
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


def _classify_org(name: str, sentence: str) -> EntityType:
    low = name.lower().rstrip(".")
    if low in _COMPANY_LOOKUP:
        return EntityType.COMPANY
    for info in COMPANIES.values():
        if name in info["subsidiaries"] or low in {s.lower() for s in info["subsidiaries"]}:
            return EntityType.SUBSIDIARY
    if "subsidiary" in sentence.lower() and (
        "acquir" in sentence.lower() or "subsidiari" in sentence.lower()
    ):
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
            name = " ".join(name.strip().split())  # collapse whitespace/newlines
            if not name or len(name) > 120:
                return
            key = f"{etype.value}|{name.lower()}"
            if key not in entities or entities[key].confidence < conf:
                entities[key] = ExtractedEntity(name=name, entity_type=etype, confidence=conf)

        def relate(
            src_name: str,
            src_type: EntityType,
            rel: RelationType,
            dst_name: str,
            dst_type: EntityType,
            confidence: float,
            evidence: str,
        ) -> None:
            # Every relation endpoint is emitted as an entity, so the
            # ingestion resolver can always map both ends to canonical ids.
            add_entity(src_name, src_type, confidence)
            add_entity(dst_name, dst_type, confidence)
            relations.append(
                ExtractedRelation(
                    src_name=" ".join(src_name.strip().split()),
                    src_type=src_type,
                    rel=rel,
                    dst_name=" ".join(dst_name.strip().split()),
                    dst_type=dst_type,
                    confidence=confidence,
                    evidence=evidence.strip()[:400],
                )
            )

        filing_company = DOC_COMPANY.get(chunk.doc_id)

        # 1. NER pass -----------------------------------------------------
        for sent in doc.sents:
            sent_text = sent.text
            for ent in sent.ents:
                label = ent.label_
                if label == "ORG":
                    add_entity(ent.text, _classify_org(ent.text, sent_text), 0.8)
                elif label in SPACY_TO_ONTOLOGY:
                    add_entity(ent.text, SPACY_TO_ONTOLOGY[label], 0.8)
            self._extract_relations(sent_text, relate, filing_company)

        # 2. Seed-knowledge pass: canonical names spaCy may miss -----------
        # Relations here are grounded in the chunk actually mentioning the
        # name; the filing company is the subject.
        low_text = chunk.text.lower()
        for canon, info in COMPANIES.items():
            mentioned = canon.lower() in low_text or any(
                a.lower().rstrip(".") in low_text for a in info["aliases"]
            )
            for seg in info["segments"]:
                if seg.lower() in low_text:
                    relate(
                        canon,
                        EntityType.COMPANY,
                        RelationType.HAS_SEGMENT,
                        seg,
                        EntityType.SEGMENT,
                        0.9,
                        self._evidence(chunk.text, seg),
                    )
            for prod in info["products"]:
                if re.search(rf"\b{re.escape(prod)}\b", chunk.text, re.IGNORECASE):
                    if mentioned or (filing_company == canon):
                        relate(
                            canon,
                            EntityType.COMPANY,
                            RelationType.SELLS_PRODUCT,
                            prod,
                            EntityType.PRODUCT,
                            0.85,
                            self._evidence(chunk.text, prod),
                        )
                    else:
                        add_entity(prod, EntityType.PRODUCT, 0.85)
            for person in info["people"]:
                if person.lower() in low_text:
                    add_entity(person, EntityType.PERSON, 0.95)
                    if mentioned or (filing_company == canon):
                        relate(
                            canon,
                            EntityType.COMPANY,
                            RelationType.LED_BY,
                            person,
                            EntityType.PERSON,
                            0.9,
                            self._evidence(chunk.text, person),
                        )
            for sub in info["subsidiaries"]:
                if sub.lower().rstrip(".") in low_text:
                    if mentioned or (filing_company == canon):
                        relate(
                            canon,
                            EntityType.COMPANY,
                            RelationType.HAS_SUBSIDIARY,
                            sub,
                            EntityType.SUBSIDIARY,
                            0.9,
                            self._evidence(chunk.text, sub),
                        )
                    else:
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
        relate: object,  # relate(src_name, src_type, rel, dst_name, dst_type, conf, evidence)
        filing_company: str | None,
    ) -> None:
        low = sent.lower()

        def subject() -> str | None:
            # Prefer a company named in the sentence; fall back to the filing
            # company only when the sentence uses "the Company"/"we"/"our".
            named = self._company_in_context(sent)
            if named:
                return named
            if filing_company and re.search(r"\b(the company|we|our|us)\b", low):
                return filing_company
            return None

        # ACQUIRED: "X acquired Y" — acquirer must be a known company or the
        # filing company (the raw regex otherwise captures date fragments).
        m = re.search(
            r"([A-Z][\w&.,'’\- ]+?)\s+(?:acquired|completed the acquisition of)\s+"
            r"([A-Z][\w&.,'’\- ]+?)(?:\.|,| for | in \d)",
            sent,
        )
        if m:
            acquirer = m.group(1).strip()
            known = self._company_in_context(acquirer)
            if known is None and filing_company:
                # sentence is in the filing company's 10-K about its own deal
                known = filing_company
            acquired = m.group(2).strip().rstrip(",.;")
            words = [w for w in acquired.split() if w not in {"&", "of", "the"}]
            proper = all(w[0].isupper() for w in words if w)
            looks_corporate = bool(
                re.search(r"\b(Inc|LLC|Corp|Corporation|Ltd|Company|Technologies)\b\.?", acquired)
            )
            if 1 <= len(words) <= 6 and proper and (len(words) >= 2 or looks_corporate):
                relate(
                    known or acquirer,
                    EntityType.COMPANY,
                    RelationType.ACQUIRED,  # type: ignore[operator]
                    acquired,
                    EntityType.COMPANY,
                    0.75,
                    sent.strip(),
                )

        # LED_BY: "X, CEO of Y" / "Y CEO X" / "X is the CEO"
        m = re.search(
            r"([A-Z][a-z]+ [A-Z][a-z]+)[,.]?\s+(?:is |was |serves as |has served as )?"
            r"(?:the |our )?(Chief Executive Officer|CEO|CFO|Chief Financial Officer)",
            sent,
        )
        if m:
            person = m.group(1)
            # guard: the "name" must not be title words ("Executive Officer")
            if re.search(r"\b(Executive|Officer|Chief|Financial|Principal)\b", person):
                person = ""
            company = subject()
            if company and person:
                relate(
                    company,
                    EntityType.COMPANY,
                    RelationType.LED_BY,  # type: ignore[operator]
                    person,
                    EntityType.PERSON,
                    0.85,
                    sent.strip(),
                )

        # COMPETES_WITH — capture the leading run of capitalized words after
        # "with" (the competitor name), stopping at lowercase filler.
        # Single generic words ("AI") are rejected; real competitors are
        # multi-word or carry a corporate suffix / known-company name.
        if re.search(r"\bcompet\w*\s+with\b", low):
            company = subject()
            for m in re.finditer(r"\bwith\s+([A-Z][\w&.,'’\-]*(?:\s+[A-Z][\w&.,'’\-]*)*)", sent):
                cand = m.group(1).strip().rstrip(",.;")
                words = [w for w in cand.split() if w not in {"&", "of", "the"}]
                looks_corporate = bool(
                    re.search(r"\b(Inc|LLC|Corp|Corporation|Ltd|Company|Technologies)\b\.?", cand)
                ) or any(
                    cand.lower() == alias.lower().rstrip(".")
                    for canon, info in COMPANIES.items()
                    for alias in [canon] + info.get("aliases", [])
                )
                if (
                    company
                    and cand.lower() != company.lower()
                    and 1 <= len(words) <= 5
                    and len(cand) <= 60
                    and (len(words) >= 2 or looks_corporate)
                ):
                    relate(
                        company,
                        EntityType.COMPANY,
                        RelationType.COMPETES_WITH,  # type: ignore[operator]
                        cand,
                        EntityType.COMPANY,
                        0.6,
                        sent.strip(),
                    )

        # FACES_RISK: risk-factor language; risk entity is a cleaned summary
        if re.search(r"\brisk factors?\b", low) or "could be adversely affected" in low:
            company = subject()
            if company:
                risk = " ".join(sent.strip().split())[:100]
                relate(
                    company,
                    EntityType.COMPANY,
                    RelationType.FACES_RISK,  # type: ignore[operator]
                    risk,
                    EntityType.RISK,
                    0.6,
                    sent.strip(),
                )

        # USES_TECHNOLOGY
        for tech in TECHNOLOGIES:
            if tech in low:
                company = subject()
                if company:
                    relate(
                        company,
                        EntityType.COMPANY,
                        RelationType.USES_TECHNOLOGY,  # type: ignore[operator]
                        tech.title(),
                        EntityType.TECHNOLOGY,
                        0.6,
                        sent.strip(),
                    )
                    break

        # OPERATES_IN: geography mentions near the filing company
        # (kept conservative: explicit "operations in X" phrasing)
        m = re.search(r"\boperations? in ([A-Z][a-z]+(?: [A-Z][a-z]+)?)", sent)
        if m:
            company = subject()
            if company:
                relate(
                    company,
                    EntityType.COMPANY,
                    RelationType.OPERATES_IN,  # type: ignore[operator]
                    m.group(1),
                    EntityType.GEOGRAPHY,
                    0.6,
                    sent.strip(),
                )

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

EXTRACTION_SYSTEM_PROMPT = """You extract entities and relations from SEC filing text.
Entity types: Company, Subsidiary, Person, Product, Segment, Acquisition,
Metric, Geography, Technology, Risk.
Relation types: HAS_SEGMENT, HAS_SUBSIDIARY, ACQUIRED, SELLS_PRODUCT, LED_BY,
REPORTS_METRIC, OPERATES_IN, COMPETES_WITH, DEPENDS_ON, PARTNERS_WITH,
FACES_RISK, USES_TECHNOLOGY.
Respond with JSON only: {"entities": [{"name": str, "entity_type": str,
"confidence": float}], "relations": [{"src_name": str, "src_type": str,
"rel": str, "dst_name": str, "dst_type": str, "confidence": float,
"evidence": str}]}. Emit only relations supported by the text."""


class LLMExtractor:
    """Strict-schema LLM extraction with Pydantic validation and retry."""

    def __init__(self, llm: LLMProvider, max_retries: int = 2) -> None:
        self._llm = llm
        self._max_retries = max_retries

    def extract(self, chunk: Chunk) -> ExtractionResult:
        last_err: Exception | None = None
        for _ in range(self._max_retries + 1):
            try:
                raw = self._llm.generate(
                    "TEXT:\n" + chunk.text[:6000],
                    system=EXTRACTION_SYSTEM_PROMPT,
                )
                if getattr(raw, "empty_content", False):
                    raise ValueError("LLM returned empty content")
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
