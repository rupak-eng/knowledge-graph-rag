"""End-to-end demo: ask real questions, capture request/response transcripts.

Usage: python scripts/demo_queries.py --data-dir data --out demo/demo_run.json
       [--llm stub|groq]

Writes one JSON transcript per question under demo/, suitable for the README.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from krag.config import get_settings  # noqa: E402
from krag.services.groq_provider import build_groq_provider  # noqa: E402
from krag.services.llm import StubLLMProvider  # noqa: E402
from krag.services.qa import QASystem  # noqa: E402

DEMO_QUESTIONS = [
    ("1-hop", "What are Apple's reportable segments?"),
    ("1-hop", "Who is the CEO of NVIDIA?"),
    ("2-hop", "Which company acquired Activision Blizzard and what are its reportable segments?"),
    ("2-hop", "What products does Microsoft sell through its Intelligent Cloud segment?"),
    ("3-hop", "Which subsidiaries are mentioned in connection with Apple's Services segment?"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("demo/demo_run.json"))
    parser.add_argument("--llm", choices=["stub", "groq"], default="stub")
    args = parser.parse_args()

    settings = get_settings()
    llm = build_groq_provider() if args.llm == "groq" else StubLLMProvider()
    if args.llm == "groq" and llm is None:
        raise SystemExit("Groq unavailable")
    label = f"Real Provider: Groq {llm.name}" if args.llm == "groq" else "Deterministic/Stub"

    qa = QASystem(settings, llm=llm)
    try:
        transcript = []
        for hop, question in DEMO_QUESTIONS:
            t0 = time.perf_counter()
            answer = qa.ask(question)
            transcript.append(
                {
                    "hop": hop,
                    "request": {"question": question},
                    "response": {
                        "answer": answer.answer_text,
                        "citations": [
                            {"chunk_id": c.chunk_id, "section": c.section, "valid": c.valid}
                            for c in answer.citations
                        ],
                        "route": {
                            "path": answer.route.path,
                            "confidence": round(answer.route.confidence, 2),
                            "reasons": answer.route.reasons,
                        },
                        "latency_ms": round(answer.latency_ms, 1),
                        "cost_usd": answer.cost_usd,
                        "abstained": answer.abstained,
                    },
                }
            )
            print(
                f"[{hop}] {question[:60]}... -> route={answer.route.path} "
                f"citations={len(answer.citations)} "
                f"{(time.perf_counter() - t0) * 1000:.0f}ms"
            )
        out = {
            "label": label,
            "llm": qa.llm.name,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "transcript": transcript,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")
    finally:
        qa.close()


if __name__ == "__main__":
    main()
