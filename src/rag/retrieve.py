
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
from functools import lru_cache
import numpy as np

def _mtime_ns(path: Path) -> int:
    """File modification time in ns, or 0 if the file does not exist."""
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return 0


@lru_cache(maxsize=1)
def _load_rag_artifacts_cached(
    index_mtime_ns: int,
    docstore_mtime_ns: int,
    metadata_mtime_ns: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict]:
    # The mtime arguments are only the cache key: if build_index rewrites
    # a file, its mtime changes, the key changes, and we reload once.
    artifact = joblib.load(RAG_INDEX_PATH)
    docstore = pd.read_csv(RAG_DOCSTORE_PATH)

    metadata: dict = {}
    if RAG_METADATA_PATH.exists():
        metadata = json.loads(RAG_METADATA_PATH.read_text(encoding="utf-8"))

    return artifact, docstore, metadata


def clear_rag_cache() -> None:
    _load_rag_artifacts_cached.cache_clear()


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

    return _load_rag_artifacts_cached(
        _mtime_ns(RAG_INDEX_PATH),
        _mtime_ns(RAG_DOCSTORE_PATH),
        _mtime_ns(RAG_METADATA_PATH),
    )


def build_filter_mask(
    docstore: pd.DataFrame,
    product: str | None = None,
    company: str | None = None,
    issue: str | None = None,
    state: str | None = None,
) -> np.ndarray:
    """Boolean mask of docstore rows that match every filter given."""
    mask = np.ones(len(docstore), dtype=bool)

    filters = {
        "product": product,
        "company": company,
        "issue": issue,
        "state": state,
    }

    for column, value in filters.items():
        if not value:
            continue

        if column not in docstore.columns:
            return np.zeros(len(docstore), dtype=bool)

        matches = (
            docstore[column]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(value.lower(), regex=False)
            .to_numpy()
        )
        mask &= matches

    return mask

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
    matrix = artifact["matrix"]

    query_vector = vectorizer.transform([query.strip()])

    
    scores = (matrix @ query_vector.T).toarray().ravel()

    
    mask = build_filter_mask(
        docstore,
        product=product,
        company=company,
        issue=issue,
        state=state,
    )

    eligible = np.flatnonzero(mask & (scores >= min_score))
    ranked = eligible[np.argsort(-scores[eligible], kind="stable")][:top_k]

    results = []

    for index in ranked:
        row = docstore.iloc[int(index)].to_dict()
        narrative = str(row.get("clean_narrative", ""))

        results.append(
            {
                "complaint_id": str(row.get("complaint_id")),
                "score": round(float(scores[index]), 4),
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