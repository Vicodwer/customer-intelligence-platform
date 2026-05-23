from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAG_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "rag"

COMPLAINTS_CLEAN_CSV = PROCESSED_DIR / "complaints_clean.csv"

RAG_INDEX_PATH = RAG_ARTIFACT_DIR / "complaint_tfidf_index.joblib"
RAG_DOCSTORE_PATH = RAG_ARTIFACT_DIR / "complaint_docstore.csv"
RAG_METADATA_PATH = RAG_ARTIFACT_DIR / "rag_index_metadata.json"


TEXT_COLUMN = "clean_narrative"

DOCSTORE_COLUMNS = [
    "complaint_id",
    "date_received",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "company",
    "state",
    "company_response",
    "timely",
    "submitted_via",
    "clean_narrative",
]


def load_clean_complaints(path: Path = COMPLAINTS_CLEAN_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing clean complaints file: {path}. "
            "Run: python -m src.data_pipeline.features"
        )

    df = pd.read_csv(path)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Missing required text column: {TEXT_COLUMN}")

    df = df.dropna(subset=[TEXT_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df = df[df[TEXT_COLUMN].str.len() >= 30].copy()
    df = df.drop_duplicates(subset=["complaint_id"]).reset_index(drop=True)

    return df


def build_rag_index(max_features: int = 20000) -> dict:
    RAG_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean_complaints()

    for column in DOCSTORE_COLUMNS:
        if column not in df.columns:
            df[column] = None

    docstore = df[DOCSTORE_COLUMNS].copy()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
    )

    matrix = vectorizer.fit_transform(docstore[TEXT_COLUMN].fillna("").astype(str))

    nearest_neighbors = NearestNeighbors(
        n_neighbors=5,
        metric="cosine",
        algorithm="brute",
    )
    nearest_neighbors.fit(matrix)

    artifact = {
        "vectorizer": vectorizer,
        "nearest_neighbors": nearest_neighbors,
        "matrix": matrix,
        "text_column": TEXT_COLUMN,
    }

    joblib.dump(artifact, RAG_INDEX_PATH)
    docstore.to_csv(RAG_DOCSTORE_PATH, index=False)

    index_version = datetime.now(timezone.utc).strftime("rag-index-%Y%m%dT%H%M%SZ")

    metadata = {
        "index_version": index_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "index_type": "tfidf_nearest_neighbors",
        "doc_count": int(len(docstore)),
        "text_column": TEXT_COLUMN,
        "max_features": max_features,
        "index_path": str(RAG_INDEX_PATH.relative_to(PROJECT_ROOT)),
        "docstore_path": str(RAG_DOCSTORE_PATH.relative_to(PROJECT_ROOT)),
    }

    RAG_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved RAG index: {RAG_INDEX_PATH}")
    print(f"Saved RAG docstore: {RAG_DOCSTORE_PATH}")
    print(f"Saved RAG metadata: {RAG_METADATA_PATH}")
    print(f"Indexed complaint documents: {len(docstore)}")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build complaint retrieval index.")
    parser.add_argument("--max-features", type=int, default=20000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_rag_index(max_features=args.max_features)

  