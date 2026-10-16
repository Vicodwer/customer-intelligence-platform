"""Okapi BM25 ranking on a sparse term-count matrix.

Implemented directly on scikit-learn and scipy so the project needs no extra
dependency. Scores are the standard BM25 sum over query terms:

    idf(t)   = ln(1 + (N - df + 0.5) / (df + 0.5))
    score(d) = sum_t idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))

All document weights are precomputed once, so a query is one sparse
matrix-vector product.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        if not texts:
            raise ValueError("Cannot build a BM25 index from zero documents.")

        if k1 < 0 or not 0.0 <= b <= 1.0:
            raise ValueError("Require k1 >= 0 and 0 <= b <= 1.")

        self.k1 = k1
        self.b = b

        # Same tokenisation as the TF-IDF index (lowercase, English stop words),
        # but unigrams only and no minimum document frequency.
        self.vectorizer = CountVectorizer(lowercase=True, stop_words="english")

        counts = self.vectorizer.fit_transform(texts).tocsr().astype(np.float64)

        n_docs = counts.shape[0]
        doc_len = np.asarray(counts.sum(axis=1)).ravel()
        avg_len = float(doc_len.mean()) or 1.0
        doc_freq = np.asarray((counts > 0).sum(axis=0)).ravel()

        self.idf = np.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        # Weight of every stored (document, term) cell, computed in one pass.
        row_of_cell = np.repeat(np.arange(n_docs), np.diff(counts.indptr))
        tf = counts.data
        length_norm = k1 * (1.0 - b + b * doc_len[row_of_cell] / avg_len)
        weights = self.idf[counts.indices] * tf * (k1 + 1.0) / (tf + length_norm)

        self.weights = sparse.csr_matrix(
            (weights, counts.indices, counts.indptr),
            shape=counts.shape,
        )

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for the query."""
        query_vector = self.vectorizer.transform([query]).tocsr()

        # Each distinct query term counts once, however often it is repeated.
        query_vector.data[:] = 1.0

        return (self.weights @ query_vector.T).toarray().ravel()

    def top(self, query: str, depth: int) -> list[tuple[int, float]]:
        """Best `depth` documents as (row index, score). Zero scores are dropped."""
        scores = self.scores(query)

        positive = np.flatnonzero(scores > 0)
        ranked = positive[np.argsort(-scores[positive], kind="stable")][:depth]

        return [(int(index), float(scores[index])) for index in ranked]