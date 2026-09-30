"""Benchmark runner: hybrid GraphRAG vs vector-only baseline.

Runs the gold question set twice over the SAME ingested corpus:
  1. baseline: force_route="vector"  (plain vector RAG)
  2. hybrid:   router decides (vector / graph / hybrid)

Reports per-hop accuracy (token-F1 vs gold), retrieval recall@k / MRR,
latency p50/p95, and measured cost. Writes:
  bench/results/benchmark.json   - full per-question results + aggregates
  bench/results/routing_log.jsonl - every routing decision
  eval/sample_outputs.jsonl       - >=10 real outputs in the shared eval schema
    {"input","output","contexts":[{"text","source","chunk_id"}],"expected","metadata"}

Usage: python -m eval.run_benchmark --data-dir data --out bench/results/benchmark.json
       [--llm stub|groq] [--groq-model openai/gpt-oss-20b]

Benchmark protocol (binding):
  1. Run the deterministic/stub benchmark FIRST, commit results.
  2. Run the real-provider (Groq) benchmark, label every table.
Tables are labeled "Deterministic/Stub" vs "Real Provider: Groq <model>".
Each report records provider/model, dataset size, UTC timestamp, per-item and
p50/p95 latency, prompt/completion tokens, and cost. Nothing is extrapolated.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.metrics import mean, mrr, percentiles, recall_at_k, token_f1  # noqa: E402
from krag.config import get_settings  # noqa: E402
from krag.services.groq_provider import build_groq_provider  # noqa: E402
from krag.services.llm import StubLLMProvider  # noqa: E402
from krag.services.qa import QASystem  # noqa: E402

logger = logging.getLogger(__name__)

GOLD_PATH = Path(__file__).parent / "gold_qa.json"


def load_gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text())


def run_system(
    qa: QASystem, gold: list[dict], force_route: str | None, tag: str
) -> tuple[list[dict], list[dict]]:
    results: list[dict] = []
    routing_log: list[dict] = []
    for item in gold:
        q = item["question"]
        t0 = time.perf_counter()
        try:
            answer = qa.ask(q, force_route=force_route)  # type: ignore[arg-type]
            error = None
        except Exception as exc:  # noqa: BLE001 - record failures honestly
            logger.exception("question failed: %s", q[:60])
            error = str(exc)
            answer = None
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_ids: list[str] = []
        retrieved_chunks: list = []
        if answer is not None:
            retrieval = qa.retriever.retrieve(q, force_route=force_route)  # type: ignore[arg-type]
            retrieved_chunks = retrieval.chunks
            retrieved_ids = [c.chunk_id for c in retrieval.chunks]

        gold_ids: list[str] = item.get("chunk_ids", [])
        row = {
            "id": item["id"],
            "hop": item["hops"],
            "question": q,
            "gold_answer": item["answer"],
            "predicted": answer.answer_text if answer else f"ERROR: {error}",
            "token_f1": token_f1(answer.answer_text, item["answer"]) if answer else 0.0,
            "recall@5": recall_at_k(retrieved_ids, gold_ids, 5),
            "recall@10": recall_at_k(retrieved_ids, gold_ids, 10),
            "mrr": mrr(retrieved_ids, gold_ids),
            "latency_ms": round(latency_ms, 1),
            "route": answer.route.path if answer else "error",
            "route_confidence": answer.route.confidence if answer else 0.0,
            "citations": [c.chunk_id for c in answer.citations] if answer else [],
            "citation_valid_rate": (
                sum(1 for c in answer.citations if c.valid) / len(answer.citations)
                if answer and answer.citations
                else (1.0 if answer and not answer.citations and answer.abstained else 0.0)
            ),
            "abstained": answer.abstained if answer else True,
            "cost_usd": answer.cost_usd if answer else 0.0,
            "tokens_in": answer.tokens_in if answer else 0,
            "tokens_out": answer.tokens_out if answer else 0,
            "llm_model": qa.llm.name,
            "retrieved_chunks": [
                {
                    "text": c.text,
                    "source": c.source,
                    "chunk_id": c.chunk_id,
                    "section": c.section,
                }
                for c in retrieved_chunks
            ],
        }
        results.append(row)
        routing_log.append(
            {
                "system": tag,
                "id": item["id"],
                "question": q,
                "route": row["route"],
                "confidence": row["route_confidence"],
                "reasons": answer.route.reasons if answer else [],
            }
        )
    return results, routing_log


def aggregate(results: list[dict]) -> dict:
    by_hop: dict[str, list[dict]] = {}
    for r in results:
        by_hop.setdefault(str(r["hop"]), []).append(r)
    hops = {}
    for hop, rows in sorted(by_hop.items()):
        hops[hop] = {
            "n": len(rows),
            "token_f1": mean([r["token_f1"] for r in rows]),
            "recall@5": mean([r["recall@5"] for r in rows]),
            "recall@10": mean([r["recall@10"] for r in rows]),
            "mrr": mean([r["mrr"] for r in rows]),
            "abstain_rate": mean([1.0 if r["abstained"] else 0.0 for r in rows]),
        }
    return {
        "n": len(results),
        "by_hop": hops,
        "token_f1": mean([r["token_f1"] for r in results]),
        "recall@5": mean([r["recall@5"] for r in results]),
        "recall@10": mean([r["recall@10"] for r in results]),
        "mrr": mean([r["mrr"] for r in results]),
        "latency_ms": percentiles([r["latency_ms"] for r in results]),
        "abstain_rate": mean([1.0 if r["abstained"] else 0.0 for r in results]),
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 6),
        "total_tokens_in": sum(r["tokens_in"] for r in results),
        "total_tokens_out": sum(r["tokens_out"] for r in results),
        "citation_valid_rate": mean([r["citation_valid_rate"] for r in results]),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("bench/results/benchmark.json"))
    parser.add_argument(
        "--llm",
        choices=["stub", "groq"],
        default="stub",
        help="answer-generation provider: deterministic stub or Groq",
    )
    parser.add_argument(
        "--groq-model",
        default="openai/gpt-oss-20b",
        help="Groq model id (recorded verbatim in every row)",
    )
    args = parser.parse_args()

    settings = get_settings()
    gold = load_gold()
    logger.info("loaded %d gold questions", len(gold))

    if args.llm == "groq":
        llm = build_groq_provider(args.groq_model)
        if llm is None:
            raise SystemExit("Groq credential unavailable; cannot run real-provider benchmark")
        llm_label = f"Real Provider: Groq {args.groq_model}"
    else:
        llm = StubLLMProvider()
        llm_label = "Deterministic/Stub"

    qa = QASystem(settings, llm=llm)
    try:
        vec_results, vec_log = run_system(qa, gold, force_route="vector", tag="vector-only")
        hybrid_results, hybrid_log = run_system(qa, gold, force_route=None, tag="hybrid")
        corpus_counts = {
            "vector_chunks": qa.vector.count(),
            "graph_entities": qa.graph.entity_count(),
            "graph_relations": qa.graph.relation_count(),
        }
    finally:
        qa.close()

    utc_now = datetime.datetime.now(datetime.UTC).isoformat()
    report = {
        "generated_at_utc": utc_now,
        "label": llm_label,
        "provider": "groq" if args.llm == "groq" else "stub",
        "llm_model": qa.llm.name,
        "dataset_size": len(gold),
        "pricing_note": (
            "Groq free tier: billed cost $0.00; tokens recorded from API usage. "
            if args.llm == "groq"
            else "Deterministic stub provider: $0.00 by construction."
        ),
        "environment": {
            "graph_backend": settings.graph_backend,
            "vector_backend": settings.vector_backend,
            "llm_provider": qa.llm.name,
            "embedding_model": settings.embedding_model,
        },
        "corpus": {
            "documents": ["aapl", "msft", "nvda"],
            **corpus_counts,
        },
        "vector_only": aggregate(vec_results),
        "hybrid": aggregate(hybrid_results),
        "per_question": {"vector_only": vec_results, "hybrid": hybrid_results},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    with open(args.out.parent / "routing_log.jsonl", "w") as f:
        for row in vec_log + hybrid_log:
            f.write(json.dumps(row) + "\n")

    # shared eval schema for the sibling LLM-eval project (>=10 real outputs)
    sample_path = Path(__file__).parent / "sample_outputs.jsonl"
    with open(sample_path, "w") as f:
        for row in hybrid_results[: max(10, len(hybrid_results) // 2)]:
            f.write(
                json.dumps(
                    {
                        "input": row["question"],
                        "output": row["predicted"],
                        "contexts": [
                            {
                                "text": c["text"],
                                "source": f"{c['source']}:{c['section']}",
                                "chunk_id": c["chunk_id"],
                            }
                            for c in row["retrieved_chunks"][:6]
                        ],
                        "expected": row["gold_answer"],
                        "metadata": {
                            "hop": row["hop"],
                            "route": row["route"],
                            "token_f1": row["token_f1"],
                            "system": "knowledge-graph-rag/hybrid",
                        },
                    }
                )
                + "\n"
            )

    print(
        json.dumps(
            {
                "vector_only": report["vector_only"]["token_f1"],
                "hybrid": report["hybrid"]["token_f1"],
                "by_hop": {
                    h: {
                        "vector_only": report["vector_only"]["by_hop"][h]["token_f1"],
                        "hybrid": report["hybrid"]["by_hop"][h]["token_f1"],
                    }
                    for h in report["hybrid"]["by_hop"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
