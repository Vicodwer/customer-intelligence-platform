from __future__ import annotations

import json
import time
from pathlib import Path

from src.rag.answer import answer_complaint_question
from src.rag.build_index import build_rag_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

RAG_MONITOR_JSON = PROCESSED_DIR / "rag_monitoring_metrics.json"
MONITORING_REPORT_MD = DOCS_DIR / "monitoring_report.md"


MONITORING_QUERIES = [
    {
        "question": "credit card billing dispute account problem",
        "top_k": 3,
        "min_score": 0.01,
    },
    {
        "question": "identity theft someone opened account",
        "top_k": 3,
        "min_score": 0.01,
    },
    {
        "question": "credit reporting error incorrect information",
        "top_k": 3,
        "min_score": 0.01,
    },
    {
        "question": "zzzz qwerty unrelated nonsense",
        "top_k": 3,
        "min_score": 0.8,
    },
]


def estimate_token_count(text: str) -> int:
    return max(1, len(str(text).split()))


def run_rag_monitoring(rebuild_index: bool = False) -> dict:
    if rebuild_index:
        build_rag_index(max_features=5000)

    records = []

    for item in MONITORING_QUERIES:
        start = time.perf_counter()

        result = answer_complaint_question(
            question=item["question"],
            top_k=item["top_k"],
            min_score=item["min_score"],
        )

        latency_ms = (time.perf_counter() - start) * 1000

        scores = [
            float(evidence.get("score", 0.0))
            for evidence in result.get("retrieved_evidence", [])
        ]

        records.append(
            {
                "question": item["question"],
                "refused": result["refused"],
                "evidence_count": len(result["evidence_ids"]),
                "top_score": max(scores) if scores else 0.0,
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "latency_ms": latency_ms,
                "estimated_answer_tokens": estimate_token_count(result["answer"]),
                "prompt_version": result["prompt_version"],
                "index_version": result["index_version"],
            }
        )

    request_count = len(records)
    refusal_count = sum(1 for record in records if record["refused"])
    empty_retrieval_count = sum(1 for record in records if record["evidence_count"] == 0)

    payload = {
        "report_type": "rag_monitoring",
        "request_count": request_count,
        "retrieval_hit_rate": (
            (request_count - empty_retrieval_count) / request_count
            if request_count
            else 0.0
        ),
        "empty_retrieval_count": empty_retrieval_count,
        "refusal_count": refusal_count,
        "refusal_rate": refusal_count / request_count if request_count else 0.0,
        "avg_top_k_score": (
            sum(record["top_score"] for record in records) / request_count
            if request_count
            else 0.0
        ),
        "avg_latency_ms": (
            sum(record["latency_ms"] for record in records) / request_count
            if request_count
            else 0.0
        ),
        "avg_estimated_answer_tokens": (
            sum(record["estimated_answer_tokens"] for record in records) / request_count
            if request_count
            else 0.0
        ),
        "records": records,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAG_MONITOR_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def append_rag_monitoring_markdown(payload: dict) -> None:
    existing = ""

    if MONITORING_REPORT_MD.exists():
        existing = MONITORING_REPORT_MD.read_text(encoding="utf-8")

    content = f"""{existing}

## RAG monitoring

- Request count: {payload["request_count"]}
- Retrieval hit rate: {payload["retrieval_hit_rate"]:.2%}
- Empty retrieval count: {payload["empty_retrieval_count"]}
- Refusal count: {payload["refusal_count"]}
- Refusal rate: {payload["refusal_rate"]:.2%}
- Average top-k score: {payload["avg_top_k_score"]:.4f}
- Average latency ms: {payload["avg_latency_ms"]:.2f}
- Average estimated answer tokens: {payload["avg_estimated_answer_tokens"]:.2f}

### RAG monitoring interpretation

The RAG monitor checks whether retrieval is returning evidence, whether the refusal rule is active, and whether latency remains acceptable for a small local MVP.
"""

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    MONITORING_REPORT_MD.write_text(content, encoding="utf-8")


def main() -> None:
    payload = run_rag_monitoring(rebuild_index=True)
    append_rag_monitoring_markdown(payload)

    print(f"Saved RAG monitoring metrics: {RAG_MONITOR_JSON}")
    print(f"Updated monitoring report: {MONITORING_REPORT_MD}")
    print(f"Retrieval hit rate: {payload['retrieval_hit_rate']:.2%}")
    print(f"Refusal rate: {payload['refusal_rate']:.2%}")


if __name__ == "__main__":
    main()