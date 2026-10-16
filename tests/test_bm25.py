import math

import pytest

from src.rag.bm25 import BM25Index
from src.rag.build_index import build_rag_index
from src.rag import eval_retrieval as ev


DOCS = [
    "apple banana",
    "apple cherry cherry",
    "grape melon banana banana banana",
]


def reference_scores(docs: list[str], query: str, k1: float, b: float) -> list[float]:
    """Slow, obviously-correct BM25 written independently of the sparse version."""
    tokens = [doc.split() for doc in docs]
    n_docs = len(tokens)
    avg_len = sum(len(t) for t in tokens) / n_docs

    scores = []

    for doc_tokens in tokens:
        score = 0.0

        for term in set(query.split()):
            doc_freq = sum(term in other for other in tokens)
            idf = math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            tf = doc_tokens.count(term)

            score += idf * tf * (k1 + 1) / (
                tf + k1 * (1 - b + b * len(doc_tokens) / avg_len)
            )

        scores.append(score)

    return scores


def test_scores_match_reference_implementation():
    index = BM25Index(DOCS, k1=1.5, b=0.75)

    actual = index.scores("apple cherry")
    expected = reference_scores(DOCS, "apple cherry", k1=1.5, b=0.75)

    assert list(actual) == pytest.approx(expected)


def test_scores_match_reference_with_other_parameters():
    index = BM25Index(DOCS, k1=1.2, b=0.4)

    actual = index.scores("banana grape")
    expected = reference_scores(DOCS, "banana grape", k1=1.2, b=0.4)

    assert list(actual) == pytest.approx(expected)


def test_best_matching_document_ranks_first():
    index = BM25Index(DOCS)

    top = index.top("cherry", depth=3)

    assert top[0][0] == 1


def test_documents_without_a_query_word_are_not_returned():
    index = BM25Index(DOCS)

    rows = [row for row, _ in index.top("grape", depth=10)]

    assert rows == [2]
    assert index.top("zzzzzz unknownword", depth=10) == []


def test_repeating_a_term_has_diminishing_returns():
    few = BM25Index(["apple filler", "other words here"]).scores("apple")[0]
    some = BM25Index(["apple " * 10 + "filler", "other words here"]).scores("apple")[0]
    many = BM25Index(["apple " * 20 + "filler", "other words here"]).scores("apple")[0]

    assert few < some < many
    # Doubling the count from 10 to 20 adds far less than doubling the score.
    assert many < 1.5 * some


def test_shorter_document_scores_higher_for_the_same_term_count():
    docs = ["apple banana", "apple " + " ".join(["filler"] * 30)]
    scores = BM25Index(docs, b=0.75).scores("apple")

    assert scores[0] > scores[1]


def test_length_normalisation_can_be_switched_off():
    docs = ["apple banana", "apple " + " ".join(["filler"] * 30)]
    scores = BM25Index(docs, b=0.0).scores("apple")

    assert scores[0] == pytest.approx(scores[1])


def test_repeated_query_words_do_not_change_the_ranking_scores():
    index = BM25Index(DOCS)

    assert list(index.scores("cherry")) == pytest.approx(list(index.scores("cherry cherry cherry")))


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        BM25Index([])

    with pytest.raises(ValueError):
        BM25Index(DOCS, b=1.5)


# ---------- inside the evaluation harness ----------


def test_bm25_is_registered_after_the_baseline():
    assert list(ev.RETRIEVERS)[:2] == ["tfidf", "bm25"]


def test_bm25_beats_random_on_the_golden_set():
    build_rag_index(max_features=5000)

    result = ev.run_evaluation(["tfidf", "bm25"])
    summary = result["retrievers"]["bm25"]["summary"]

    assert summary["lift_over_random"] >= 5
    assert summary["hit_rate_at_k"] >= 0.5
    assert "bm25" in result["comparisons"]