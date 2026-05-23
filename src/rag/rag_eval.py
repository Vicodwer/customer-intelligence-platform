from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.rag.answer import answer_complaint_question
from src.rag.build_index import RAG_INDEX_PATH, build_rag_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

RAG_EVAL_RESULTS_PATH = PROCESSED_DIR / "rag_eval_results.json"
RAG_REPORT_PATH = DOCS_DIR / "rag_report.md"


EVAL_QUESTIONS = [
    {
        "id": "rag_eval_001",
        "question": "What complaints mention credit card billing disputes and account problems?",
        "top_k": 3,
        "min_score": 0.01,
        "expected_refused": False,
        "expected_terms": ["credit", "card"],
        "expected_evidence_rule": "At least one retrieved complaint should cite a credit/card-related record.",
    },
    {
        "id": "rag_eval_002",
        "question": "What complaints discuss identity theft or someone opening accounts?",
        "top_k": 3,
        "min_score": 0.01,
        "expected_refused": False,
        "expected_terms": ["identity", "account"],
        "expected_evidence_rule": "At least one retrieved complaint should be related to identity theft or account opening.",
    },
    {
        "id": "rag_eval_003",
        "question": "What complaints involve credit reporting errors?",
        "top_k": 3,
        "min_score": 0.01,
        "expected_refused": False,
        "expected_terms": ["credit", "report"],
        "expected_evidence_rule": "At least one retrieved complaint should mention credit reporting or report errors.",
    },
    {
        "id": "rag_eval_004",
        "question": "What complaints mention debt collection calls or attempts to collect a debt?",
        "top_k": 3,
        "min_score": 0.01,
        "expected_refused": False,
        "expected_terms": ["debt", "collection"],
        "expected_evidence_rule": "At least one retrieved complaint should be related to debt collection.",
    },
    {
        "id": "rag_eval_005",
        "question": "What complaints mention mortgage payment or foreclosure problems?",
        "top_k": 3,
        "min_score": 0.01,
        "expected_refused": False,
        "expected_terms": ["mortgage", "payment"],
        "expected_evidence_rule": "At least one retrieved complaint should involve mortgage or payment issues.",
    },
    {
        "id": "rag_eval_006",
        "question": "What complaints mention bank account fees or checking account problems?",
        "top_k": 3,
        "min_score": 0.01,
        "expected_refused": False,
        "expected_terms": ["account", "fee"],
        "expected_evidence_rule": "At least one retrieved complaint should discuss account or fee problems.",
    },
    {
        "id": "rag_eval_007",
        "question": "What complaints mention student loan repayment problems?",
        "top_k": 3,
        "min_score": 0.01,
        "expected_refused": False,
        "expected_terms": ["student", "loan"],
        "expected_evidence_rule": "At least one retrieved complaint should relate to student loans.",
    },
    {
        "id": "rag_eval_008",
        "question": "What complaints mention fraud, unauthorized charges, or suspicious transactions?",
        "top_k": 3,
        "min_score": 0.01,
        "expected_refused": False,
        "expected_terms": ["fraud", "unauthorized"],
        "expected_evidence_rule": "At least one retrieved complaint should discuss fraud or unauthorized transactions.",
    },
    {
        "id": "rag_eval_009",
        "question": "zzzz qwerty unrelated nonsense",
        "top_k": 3,
        "min_score": 0.8,
        "expected_refused": True,
        "expected_terms": [],
        "expected_evidence_rule": "Should refuse because no evidence should cross the high similarity threshold.",
    },
    {
        "id": "rag_eval_010",
        "question": "purple elephant spaceship banana volcano",
        "top_k": 3,
        "min_score": 0.8,
        "expected_refused": True,
        "expected_terms": [],
        "expected_evidence_rule": "Should refuse because the query is unrelated to complaint evidence.",
    },
]


def _contains_any_expected_term(answer_payload: dict, expected_terms: list[str]) -> bool:
    if not expected_terms:
        return True

    searchable_text_parts = [
        answer_payload.get("answer", ""),
    ]

    for item in answer_payload.get("retrieved_evidence", []):
        searchable_text_parts.append(str(item.get("product", "")))
        searchable_text_parts.append(str(item.get("issue", "")))
        searchable_text_parts.append(str(item.get("evidence_preview", "")))

    searchable_text = " ".join(searchable_text_parts).lower()

    return any(term.lower() in searchable_text for term in expected_terms)


def evaluate_one_case(case: dict) -> dict:
    answer_payload = answer_complaint_question(
        question=case["question"],
        top_k=case["top_k"],
        min_score=case["min_score"],
    )

    evidence_ids = answer_payload.get("evidence_ids", [])
    refused = bool(answer_payload.get("refused"))

    refused_check = refused == case["expected_refused"]
    evidence_check = True

    if not case["expected_refused"]:
        evidence_check = len(evidence_ids) > 0

    term_check = _contains_any_expected_term(
        answer_payload=answer_payload,
        expected_terms=case["expected_terms"],
    )

    passed = refused_check and evidence_check and term_check

    if passed:
        note = "PASS: behavior, evidence, and expected terms matched."
    else:
        failed_parts = []

        if not refused_check:
            failed_parts.append(
                f"refused={refused}, expected_refused={case['expected_refused']}"
            )

        if not evidence_check:
            failed_parts.append("expected evidence IDs but none were returned")

        if not term_check:
            failed_parts.append(
                f"expected one of terms={case['expected_terms']} in answer/evidence"
            )

        note = "FAIL: " + "; ".join(failed_parts)

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_refused": case["expected_refused"],
        "actual_refused": refused,
        "expected_terms": case["expected_terms"],
        "expected_evidence_rule": case["expected_evidence_rule"],
        "evidence_ids": evidence_ids,
        "evidence_count": len(evidence_ids),
        "evidence_sufficiency_note": answer_payload.get("evidence_sufficiency_note"),
        "prompt_version": answer_payload.get("prompt_version"),
        "index_version": answer_payload.get("index_version"),
        "passed": passed,
        "note": note,
    }


def write_rag_report(payload: dict) -> None:
    rows = []

    for result in payload["results"]:
        rows.append(
            "| {id} | {passed} | {refused} | {count} | {ids} | {note} |".format(
                id=result["id"],
                passed=result["passed"],
                refused=result["actual_refused"],
                count=result["evidence_count"],
                ids=", ".join(result["evidence_ids"][:5]),
                note=result["note"].replace("|", "/"),
            )
        )

    table = "\n".join(rows)

    content = f"""# RAG Evaluation Report

## Purpose

Evaluate whether the complaint intelligence assistant returns grounded answers with cited evidence IDs and refuses when retrieval is weak.

## Summary

- Created at UTC: {payload["created_at_utc"]}
- Total tests: {payload["summary"]["total_tests"]}
- Passed: {payload["summary"]["passed"]}
- Failed: {payload["summary"]["failed"]}
- Pass rate: {payload["summary"]["pass_rate"]:.2%}
- Average evidence count: {payload["summary"]["avg_evidence_count"]:.2f}
- Refusal count: {payload["summary"]["refusal_count"]}

## Evaluation table

| ID | Passed | Refused | Evidence Count | Evidence IDs | Note |
|---|---:|---:|---:|---|---|
{table}

## Failure cases

{payload["failure_notes"]}

## Known limitations

- This MVP uses TF-IDF retrieval instead of dense embeddings.
- The answer generator is deterministic and summary-based, not a generative LLM.
- Evidence relevance is checked with simple expected-term rules.
- Later hardening should add semantic relevance scoring, adversarial tests, and latency/token metrics.
"""

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    RAG_REPORT_PATH.write_text(content, encoding="utf-8")


def run_rag_eval(rebuild_index: bool = False) -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if rebuild_index or not RAG_INDEX_PATH.exists():
        build_rag_index(max_features=5000)

    results = [evaluate_one_case(case) for case in EVAL_QUESTIONS]

    total_tests = len(results)
    passed_count = sum(1 for result in results if result["passed"])
    failed_count = total_tests - passed_count
    refusal_count = sum(1 for result in results if result["actual_refused"])
    avg_evidence_count = sum(result["evidence_count"] for result in results) / total_tests

    failures = [result for result in results if not result["passed"]]

    if failures:
        failure_notes = "\n".join(
            [
                f"- {result['id']}: {result['note']}"
                for result in failures
            ]
        )
    else:
        failure_notes = "No failing cases in this run."

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_tests": total_tests,
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": passed_count / total_tests,
            "avg_evidence_count": avg_evidence_count,
            "refusal_count": refusal_count,
        },
        "results": results,
        "failure_notes": failure_notes,
    }

    RAG_EVAL_RESULTS_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    write_rag_report(payload)

    print(f"Saved RAG eval results: {RAG_EVAL_RESULTS_PATH}")
    print(f"Saved RAG report: {RAG_REPORT_PATH}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Pass rate: {passed_count / total_tests:.2%}")
    print(f"Refusal count: {refusal_count}")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG evaluation tests.")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_rag_eval(rebuild_index=args.rebuild_index)