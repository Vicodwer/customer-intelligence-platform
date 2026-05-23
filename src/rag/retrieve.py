from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAG_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "rag"

RAG_INDEX_PATH = RAG_ARTIFACT_DIR / "complaint_tfidf_index.joblib"
RAG_DOCSTORE_PATH = RAG_ARTIFACT_DIR / "complaint_docstore.csv"
RAG_METADATA_PATH = RAG_ARTIFACT_DIR / "rag_index_metadata.json"

DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.05


def load_rag_artifacts() -> tuple[dict[str, Any], pd.DataFrame, dict]:
    if not RAG_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing RAG index: {RAG_INDEX_PATH}. "
            "Run: python -m src.rag.build_index"
        )

    if not RAG_DOCSTORE_PATH.exists():
        raise FileNotFoundError(
            f"Missing RAG docstore: {RAG_DOCSTORE_PATH}. "
            "Run: python -m src.rag.build_index"
        )

    artifact = joblib.load(RAG_INDEX_PATH)
    docstore = pd.read_csv(RAG_DOCSTORE_PATH)

    metadata = {}
    if RAG_METADATA_PATH.exists():
        metadata = json.loads(RAG_METADATA_PATH.read_text(encoding="utf-8"))

    return artifact, docstore, metadata


def apply_filters(
    rows: list[dict],
    product: str | None = None,
    company: str | None = None,
    issue: str | None = None,
    state: str | None = None,
) -> list[dict]:
    filtered_rows = rows

    filters = {
        "product": product,
        "company": company,
        "issue": issue,
        "state": state,
    }

    for column, value in filters.items():
        if not value:
            continue

        value_lower = value.lower()

        filtered_rows = [
            row
            for row in filtered_rows
            if value_lower in str(row.get(column, "")).lower()
        ]

    return filtered_rows


def retrieve_complaints(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    product: str | None = None,
    company: str | None = None,
    issue: str | None = None,
    state: str | None = None,
) -> dict:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    artifact, docstore, metadata = load_rag_artifacts()

    vectorizer = artifact["vectorizer"]
    nearest_neighbors = artifact["nearest_neighbors"]

    query_vector = vectorizer.transform([query.strip()])

    n_neighbors = min(max(top_k * 10, top_k), len(docstore))

    distances, indices = nearest_neighbors.kneighbors(
        query_vector,
        n_neighbors=n_neighbors,
    )

    candidate_rows: list[dict] = []

    for distance, index in zip(distances[0], indices[0]):
        score = 1.0 - float(distance)
        row = docstore.iloc[int(index)].to_dict()
        row["_score"] = score
        candidate_rows.append(row)

    candidate_rows = apply_filters(
        candidate_rows,
        product=product,
        company=company,
        issue=issue,
        state=state,
    )

    candidate_rows = [
        row
        for row in candidate_rows
        if float(row.get("_score", 0.0)) >= min_score
    ]

    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: float(row.get("_score", 0.0)),
        reverse=True,
    )[:top_k]

    results = []

    for row in candidate_rows:
        narrative = str(row.get("clean_narrative", ""))

        results.append(
            {
                "complaint_id": str(row.get("complaint_id")),
                "score": round(float(row.get("_score", 0.0)), 4),
                "product": row.get("product"),
                "issue": row.get("issue"),
                "company": row.get("company"),
                "state": row.get("state"),
                "date_received": row.get("date_received"),
                "evidence_preview": narrative[:350],
            }
        )

    return {
        "query": query,
        "top_k": top_k,
        "min_score": min_score,
        "filters": {
            "product": product,
            "company": company,
            "issue": issue,
            "state": state,
        },
        "index_version": metadata.get("index_version"),
        "result_count": len(results),
        "results": results,
        "evidence_sufficient": len(results) > 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve complaint evidence.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--product", default=None)
    parser.add_argument("--company", default=None)
    parser.add_argument("--issue", default=None)
    parser.add_argument("--state", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output = retrieve_complaints(
        query=args.query,
        top_k=args.top_k,
        min_score=args.min_score,
        product=args.product,
        company=args.company,
        issue=args.issue,
        state=args.state,
    )

    print(json.dumps(output, indent=2))