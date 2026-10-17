"""Reciprocal Rank Fusion: merge several ranked lists into one.

    score(doc) = sum over lists of 1 / (k + rank of doc in that list)

Only ranks are used, never raw scores, so BM25 scores and cosine similarities
(which live on different scales) can be combined without any rescaling.
A document ranked well by more than one list rises to the top.
"""

from __future__ import annotations

from typing import Sequence


RRF_K = 60  # the value from the original RRF paper; larger k flattens the rank bonus


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = RRF_K,
    depth: int | None = None,
) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive.")

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}

    for ranking in rankings:
        counted: set[str] = set()

        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in counted:  # a repeat inside one list counts once
                continue

            counted.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(doc_id, len(first_seen))

    # Equal scores keep the order in which the documents were first met,
    # so the result is the same on every run.
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], first_seen[doc_id]))

    return ordered if depth is None else ordered[:depth]