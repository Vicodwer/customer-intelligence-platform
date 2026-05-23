from __future__ import annotations

import argparse
import json
from collections import Counter

from src.rag.retrieve import DEFAULT_MIN_SCORE, DEFAULT_TOP_K, retrieve_complaints


PROMPT_VERSION = "deterministic-complaint-answer-v1"


def _top_counts(items: list[str], limit: int = 3) -> list[tuple[str, int]]:
    cleaned = [str(item).strip() for item in items if str(item).strip() and str(item) != "nan"]
    return Counter(cleaned).most_common(limit)


def _format_top_counts(label: str, counts: list[tuple[str, int]]) -> str:
    if not counts:
        return f"{label}: not enough metadata available"

    formatted = ", ".join([f"{value} ({count})" for value, count in counts])
    return f"{label}: {formatted}"


def build_grounded_summary(question: str, results: list[dict]) -> str:
    evidence_ids = [str(item["complaint_id"]) for item in results]

    products = _top_counts([item.get("product") for item in results])
    issues = _top_counts([item.get("issue") for item in results])
    companies = _top_counts([item.get("company") for item in results])

    top_score = max(float(item.get("score", 0.0)) for item in results)

    lines = [
        f"Based on {len(results)} retrieved complaint records, the complaint intelligence summary is:",
        _format_top_counts("Top products", products),
        _format_top_counts("Top issues", issues),
        _format_top_counts("Companies appearing in the retrieved evidence", companies),
        f"Highest retrieval score: {top_score:.4f}",
        f"Cited complaint IDs: {', '.join(evidence_ids)}",
        "This is grounded complaint intelligence only, not legal or financial advice.",
    ]

    return "\n".join(lines)


def answer_complaint_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    product: str | None = None,
    company: str | None = None,
    issue: str | None = None,
    state: str | None = None,
) -> dict:
    retrieval = retrieve_complaints(
        query=question,
        top_k=top_k,
        min_score=min_score,
        product=product,
        company=company,
        issue=issue,
        state=state,
    )

    results = retrieval["results"]
    evidence_ids = [item["complaint_id"] for item in results]

    if not retrieval["evidence_sufficient"]:
        return {
            "question": question,
            "answer": (
                "I do not have enough retrieved complaint evidence to answer this reliably. "
                "Try a more specific question or lower the retrieval threshold for exploration."
            ),
            "refused": True,
            "evidence_ids": [],
            "retrieved_evidence": [],
            "evidence_sufficiency_note": (
                f"Insufficient evidence: no retrieved complaint record crossed "
                f"the similarity threshold min_score={min_score}."
            ),
            "prompt_version": PROMPT_VERSION,
            "index_version": retrieval.get("index_version"),
            "retrieval": {
                "top_k": top_k,
                "min_score": min_score,
                "result_count": 0,
                "filters": retrieval.get("filters", {}),
            },
        }

    answer = build_grounded_summary(question=question, results=results)

    return {
        "question": question,
        "answer": answer,
        "refused": False,
        "evidence_ids": evidence_ids,
        "retrieved_evidence": results,
        "evidence_sufficiency_note": (
            f"Sufficient evidence: {len(results)} complaint records crossed "
            f"the similarity threshold min_score={min_score}."
        ),
        "prompt_version": PROMPT_VERSION,
        "index_version": retrieval.get("index_version"),
        "retrieval": {
            "top_k": top_k,
            "min_score": min_score,
            "result_count": len(results),
            "filters": retrieval.get("filters", {}),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer a complaint intelligence question.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--product", default=None)
    parser.add_argument("--company", default=None)
    parser.add_argument("--issue", default=None)
    parser.add_argument("--state", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output = answer_complaint_question(
        question=args.question,
        top_k=args.top_k,
        min_score=args.min_score,
        product=args.product,
        company=args.company,
        issue=args.issue,
        state=args.state,
    )

    print(json.dumps(output, indent=2))