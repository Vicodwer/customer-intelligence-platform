import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import HashingVectorizer

from src.rag import dense
from src.rag import eval_retrieval as ev
from src.rag.build_index import build_rag_index


class FakeEmbedder:
    """Bag-of-words embedder, so similar wording gives similar vectors."""

    model_id = "fake/model"
    revision = "test"
    max_length = 512
    query_prefix = ""

    def __init__(self):
        self._hasher = HashingVectorizer(
            n_features=512, alternate_sign=False, norm="l2", stop_words="english"
        )

    def encode_documents(self, texts):
        return self._hasher.transform(list(texts)).toarray().astype(np.float32)

    def encode_query(self, query):
        return self.encode_documents([query])[0]


def _docstore(texts):
    return pd.DataFrame(
        {
            "complaint_id": [str(i) for i in range(len(texts))],
            "clean_narrative": texts,
        }
    )


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point the dense index at a temp folder so real embeddings are never touched."""
    monkeypatch.setattr(dense, "EMBEDDINGS_PATH", tmp_path / "e.npy")
    monkeypatch.setattr(dense, "EMBEDDINGS_META_PATH", tmp_path / "e.json")
    monkeypatch.setattr(dense, "RAG_DOCSTORE_PATH", tmp_path / "docstore.csv")
    monkeypatch.setattr(dense, "get_embedder", lambda: FakeEmbedder())
    dense.clear_dense_cache()

    yield tmp_path

    dense.clear_dense_cache()


def _use_docstore(monkeypatch, docstore):
    monkeypatch.setattr(dense, "load_rag_artifacts", lambda: ({}, docstore, {}))
    dense.clear_dense_cache()


TEXTS = [
    "my credit card was charged a late fee i never agreed to",
    "the mortgage servicer lost my payment and reported me late",
    "debt collector keeps calling my workplace about an old loan",
]


def test_top_returns_the_most_similar_document_first():
    vectors = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    index = dense.DenseIndex(vectors, ["a", "b", "c"])

    top = index.top(np.array([0, 1], dtype=np.float32), depth=2)

    assert [row for row, _ in top] == [1, 2]
    assert top[0][1] == pytest.approx(1.0)


def test_mismatched_ids_are_rejected():
    with pytest.raises(ValueError):
        dense.DenseIndex(np.zeros((3, 2), dtype=np.float32), ["a", "b"])


def test_fingerprint_changes_when_text_or_order_changes():
    base = dense.corpus_fingerprint(_docstore(TEXTS))

    assert base == dense.corpus_fingerprint(_docstore(TEXTS))
    assert base != dense.corpus_fingerprint(_docstore([TEXTS[0] + "!", *TEXTS[1:]]))
    assert base != dense.corpus_fingerprint(_docstore(list(reversed(TEXTS))))


def test_build_then_search_finds_the_matching_complaint(isolated, monkeypatch):
    _use_docstore(monkeypatch, _docstore(TEXTS))

    info = dense.build_dense_index(FakeEmbedder())
    result = dense.search_dense("debt collector calling my work", depth=3, embedder=FakeEmbedder())

    assert info["doc_count"] == 3
    assert result[0] == "2"


def test_search_before_building_explains_what_to_run(isolated, monkeypatch):
    _use_docstore(monkeypatch, _docstore(TEXTS))

    with pytest.raises(FileNotFoundError, match="python -m src.rag.dense"):
        dense.search_dense("anything", depth=3, embedder=FakeEmbedder())


def test_embeddings_for_changed_data_are_refused(isolated, monkeypatch):
    _use_docstore(monkeypatch, _docstore(TEXTS))
    dense.build_dense_index(FakeEmbedder())

    _use_docstore(monkeypatch, _docstore([TEXTS[0], TEXTS[1], "completely different text now"]))

    with pytest.raises(RuntimeError, match="do not match"):
        dense.search_dense("anything", depth=3, embedder=FakeEmbedder())


# ---------- inside the evaluation harness ----------


def test_dense_and_hybrid_are_registered_but_not_run_by_default():
    assert {"dense", "hybrid"} <= set(ev.RETRIEVERS)
    assert ev.DEFAULT_RETRIEVERS == ["tfidf", "bm25"]


def test_harness_runs_dense_and_hybrid_end_to_end(isolated):
    build_rag_index(max_features=5000)  # real corpus, only the embeddings are fake
    dense.clear_dense_cache()
    dense.build_dense_index(FakeEmbedder())

    result = ev.run_evaluation(["bm25", "dense", "hybrid"])

    assert set(result["retrievers"]) == {"bm25", "dense", "hybrid"}
    assert set(result["comparisons"]) == {"dense", "hybrid"}
    assert result["retrievers"]["hybrid"]["summary"]["precision_at_k"] > 0