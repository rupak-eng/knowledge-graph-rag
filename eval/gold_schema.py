"""Gold QA dataset schema for the knowledge-graph RAG benchmark.

Each item is hand-verified against the actual FY2024 10-K chunk text.
Fields:
- id: stable identifier (q001, q002, ...)
- question: natural-language question
- answer: verified correct answer (concise; must appear in or follow from chunks)
- chunk_ids: supporting chunk IDs that contain the answer evidence
- hops: 1 | 2 | 3 — reasoning hops required
- category: vector | graph | hybrid — which retrieval path should shine
- notes: verification notes (optional)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GoldQA(BaseModel):
    id: str
    question: str
    answer: str
    chunk_ids: list[str] = Field(min_length=1)
    hops: int = Field(ge=1, le=3)
    category: str  # vector | graph | hybrid
    notes: str = ""
