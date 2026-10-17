"""Dense (embedding) index for the complaint corpus.

Build once (takes a few minutes the first time):

    python -m src.rag.dense

Embeddings are stored next to the TF-IDF index. They are tied to the *content*
of the corpus (a fingerprint of every complaint id and text), not to the index
version, so rebuilding the TF-IDF index with identical data does not throw
the embeddings away, while changed data is detected and refused.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.rag.embedder import OnnxEmbedder, get_embedder
from src.rag.retrieve import RAG_ARTIFACT_DIR, RAG_DOCSTORE_PATH, load_rag_artifacts


EMBEDDINGS_PATH = RAG_ARTIFACT_DIR / "complaint_embeddings.npy"
EMBEDDINGS_META_PATH = RAG_ARTIFACT_DIR / "complaint_embeddings.json"

TEXT_COLUMN = "clean_narrative"


def corpus_fingerprint(docstore: pd.DataFrame) -> str:
    """Changes if any complaint is added, removed, edited or reordered."""
    digest = hashlib.sha256()

    ids = docstore["complaint_id"].astype(str)
    texts = docstore[TEXT_COLUMN].fillna("").astype(str)

    for complaint_id, text in zip(ids, texts):
        digest.update(complaint_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        digest.update(b"\x01")

    return digest.hexdigest()


class DenseIndex:
    def __init__(self, embeddings: np.ndarray, ids: list[str]) -> None:
        if embeddings.ndim != 2 or embeddings.shape[0] != len(ids):
            raise ValueError(
                f"{embeddings.shape[0] if embeddings.ndim == 2 else '?'} embeddings "
                f"for {len(ids)} documents."
            )

        self.embeddings = embeddings
        self.ids = ids

    def top(self, query_vector: np.ndarray, depth: int) -> list[tuple[int, float]]:
        """Best `depth` documents as (row index, cosine similarity)."""
        scores = self.embeddings @ query_vector
        ranked = np.argsort(-scores, kind="stable")[:depth]

        return [(int(row), float(scores[row])) for row in ranked]


# ---------- build ----------


def build_dense_index(embedder: OnnxEmbedder | None = None) -> dict[str, Any]:
    _, docstore, _ = load_rag_artifacts()

    embedder = embedder or get_embedder()

    texts = docstore[TEXT_COLUMN].fillna("").astype(str).tolist()
    embeddings = embedder.encode_documents(texts)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": embedder.model_id,
        "model_revision": embedder.revision,
        "precision": "fp32",
        "max_length": embedder.max_length,
        "query_prefix": embedder.query_prefix,
        "doc_count": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "corpus_fingerprint": corpus_fingerprint(docstore),
    }

    EMBEDDINGS_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    clear_dense_cache()

    return metadata


# ---------- load ----------


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return -1


@lru_cache(maxsize=2)
def _load_cached(embeddings_mtime: int, meta_mtime: int, docstore_mtime: int) -> DenseIndex:
    if embeddings_mtime < 0 or meta_mtime < 0:
        raise FileNotFoundError(
            "Dense embeddings are not built yet. Run: python -m src.rag.dense"
        )

    _, docstore, _ = load_rag_artifacts()
    metadata = json.loads(EMBEDDINGS_META_PATH.read_text(encoding="utf-8"))

    if metadata.get("corpus_fingerprint") != corpus_fingerprint(docstore):
        raise RuntimeError(
            "The saved embeddings do not match the current complaints "
            "(the data changed since they were built). Run: python -m src.rag.dense"
        )

    embeddings = np.load(EMBEDDINGS_PATH)
    ids = docstore["complaint_id"].astype(str).tolist()

    return DenseIndex(embeddings, ids)


def get_dense_index() -> DenseIndex:
    return _load_cached(
        _mtime_ns(EMBEDDINGS_PATH),
        _mtime_ns(EMBEDDINGS_META_PATH),
        _mtime_ns(RAG_DOCSTORE_PATH),
    )


def clear_dense_cache() -> None:
    _load_cached.cache_clear()


# ---------- search ----------


def search_dense(query: str, depth: int, embedder: OnnxEmbedder | None = None) -> list[str]:
    """Complaint ids of the `depth` most similar complaints, best first."""
    index = get_dense_index()
    embedder = embedder or get_embedder()

    return [index.ids[row] for row, _ in index.top(embedder.encode_query(query), depth)]


if __name__ == "__main__":
    info = build_dense_index()

    print(f"Saved embeddings: {EMBEDDINGS_PATH}")
    print(
        f"{info['doc_count']} documents, {info['dimension']} dimensions, "
        f"model {info['model']} @ {info['model_revision']}"
    )