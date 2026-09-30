# Knowledge Graph RAG for Enterprise Data

Hybrid retrieval system combining **vector search** (pgvector) with a **knowledge graph** (Neo4j) over real SEC 10-K filings. Built to answer: does graph-augmented retrieval actually beat vector-only on multi-hop questions?

**Corpus:** FY2024 10-Ks for Apple (AAPL), Microsoft (MSFT), NVIDIA (NVDA) — 828 chunks, 982 graph entities, 68 relations.

## Benchmark Results

**36 hand-verified questions** (16× 1-hop, 14× 2-hop, 6× 3-hop) over the same ingested corpus. Full results: [`bench/results/benchmark_stub.json`](bench/results/benchmark_stub.json).

### Deterministic/Stub Provider

| System | Token-F1 | Recall@5 | Recall@10 | MRR | p50 latency | p95 latency |
|--------|----------|----------|-----------|-----|-------------|-------------|
| Vector-only | 0.051 | 0.514 | 0.639 | 0.492 | 38ms | 89ms |
| Hybrid | 0.053 | 0.514 | 0.625 | 0.488 | 37ms | 70ms |

**By hop (Recall@5):**

| Hop | n | Vector-only | Hybrid |
|-----|---|-------------|--------|
| 1 | 16 | 0.625 | 0.625 |
| 2 | 14 | 0.464 | 0.464 |
| 3 | 6 | 0.333 | 0.333 |

**Finding:** With the deterministic stub provider, hybrid retrieval achieves **parity** with vector-only, not improvement. The rule-based router correctly identifies metric-seeking questions for vector search, but the graph adds little on this question set. Token-F1 is low because the stub generates placeholder text — retrieval metrics (recall/MRR) are the meaningful comparison.

**Cost:** $0.00 (stub). 66k/73k tokens in, ~4k out.

### Real Provider (Groq)

**Status:** Blocked by Groq free-tier daily token limit (200k TPD). The 72-question benchmark (36 × 2 systems) exceeds the daily budget. Pending limit reset.

The pipeline is verified working with Groq `openai/gpt-oss-20b` (see demo evidence). Real-provider benchmark will be added when tokens are available.

## Architecture

```
Question → Router → Vector Search (pgvector) ←→ Graph Search (Neo4j)
                    ↓
            Hybrid Retrieval → Grounded Answer (Groq/Stub)
                    ↓
            Citation Validation → Answer + Sources
```

**Key design decisions:**

1. **No raw Cypher generation.** The LLM never writes Cypher. Code selects from parameterized, validated templates (`entity_neighborhood`, 1/2/3-hop fixed patterns).

2. **Controlled ontology.** 9 relation types: `HAS_SEGMENT`, `SELLS_PRODUCT`, `HAS_SUBSIDIARY`, `ACQUIRED`, `LED_BY`, `COMPETES_WITH`, `OPERATES_IN`, `PARTNERS_WITH`, `USES_TECHNOLOGY`. All triples validated against the ontology.

3. **Deterministic ingestion.** Chunk IDs are content hashes (`{doc}#c{NNNNN}`). Re-running ingestion is idempotent — unchanged documents are skipped.

4. **Entity resolution.** spaCy NER + rule-based extractors, with alias merging ("Apple Inc." → "Apple").

## Quick Start

### Prerequisites

- Python 3.12+
- Neo4j (bolt://localhost:7687)
- PostgreSQL 16 + pgvector (localhost:5432)

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Configure

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="..."
export DATABASE_URL="postgresql://.../..."
export EMBEDDING_MODEL_PATH="/path/to/all-MiniLM-L6-v2"
```

### Ingest

```bash
# Download SEC filings (writes data/raw/ + SHA256 manifest)
python scripts/download_sec.py --out data/raw

# Ingest into Neo4j + pgvector
python -m krag.services.ingest --data-dir data --extractor spacy \
  --graph-backend real --vector-backend real
```

### Ask

```bash
# CLI demo
python scripts/demo_queries.py --llm stub --out demo/demo_run.json

# API server
uvicorn krag.api.main:app --host 127.0.0.1 --port 8000

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Apple'"'"'s reportable segments?"}'
```

### Benchmark

```bash
# Deterministic stub (no API costs)
python -m eval.run_benchmark --llm stub --out bench/results/benchmark_stub.json

# Real provider (requires Groq credential, respects rate limits)
python -m eval.run_benchmark --llm groq --groq-model openai/gpt-oss-20b \
  --out bench/results/benchmark_groq20b.json
```

## Project Structure

```
src/krag/
  domain/       # Pydantic schemas, ontology, controlled triples
  storage/      # Neo4j + pgvector adapters (real + in-memory)
  services/     # Extractor, resolver, router, retriever, QA, LLM
  api/          # FastAPI app
eval/
  gold_qa.json      # 36 hand-verified questions
  run_benchmark.py  # Vector vs hybrid comparison
  metrics.py        # token-F1, recall@k, MRR
bench/results/      # Committed benchmark JSON
demo/               # Demo transcripts
scripts/            # SEC downloader, demo runner
```

## API Endpoints

- `GET /health` — service status, corpus counts
- `POST /ask` — `{question}` → grounded answer with citations
- `GET /graph/entity/{id}` — entity details + relations
- `GET /graph/search?name=...` — entity search

## Testing

```bash
# Unit + integration (requires Neo4j + PostgreSQL)
pytest tests/ -v

# Lint + typecheck
ruff check src/ && ruff format --check src/
mypy src/
```

**Current status:** 24 tests pass against real Neo4j + pgvector. Integration tests use isolated `KRAGTEST_*` prefixes and never touch production data.

## Limitations & Honest Notes

1. **Hybrid ≈ vector-only** on the current 36-question set with stub provider. The graph helps for entity-relationship questions but the corpus is dominated by metric questions best served by vector search.

2. **Groq benchmark pending** due to free-tier daily token limits.

3. **Extraction is rule-based** (spaCy + regex). LLM extraction (Groq 20b/120b) is implemented but not yet benchmarked against the rule-based baseline.

4. **No Docker daemon** in this environment — `docker-compose.yml` is provided but `docker compose up` was not executed here.

## License

MIT
