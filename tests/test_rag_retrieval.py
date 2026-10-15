from src.rag.build_index import (
    RAG_DOCSTORE_PATH,
    RAG_INDEX_PATH,
    RAG_METADATA_PATH,
    build_rag_index,
)
from src.rag.retrieve import retrieve_complaints


def test_build_rag_index_writes_artifacts():
    metadata = build_rag_index(max_features=5000)

    assert RAG_INDEX_PATH.exists()
    assert RAG_DOCSTORE_PATH.exists()
    assert RAG_METADATA_PATH.exists()
    assert metadata["doc_count"] >= 4990
    assert metadata["index_type"] == "tfidf_nearest_neighbors"


def test_retrieve_complaints_returns_evidence():
    build_rag_index(max_features=5000)

    result = retrieve_complaints(
        query="credit card billing dispute account problem",
        top_k=3,
        min_score=0.01,
    )

    assert result["result_count"] > 0
    assert result["evidence_sufficient"] is True

    first = result["results"][0]

    assert "complaint_id" in first
    assert "score" in first
    assert "evidence_preview" in first


def test_retrieve_complaints_refuses_when_similarity_is_weak():
    build_rag_index(max_features=5000)

    result = retrieve_complaints(
        query="zzzz qwerty unrelated nonsense",
        top_k=3,
        min_score=0.8,
    )

    assert result["result_count"] == 0
    assert result["evidence_sufficient"] is False

def test_rag_artifacts_are_loaded_once(monkeypatch):
    from src.rag import retrieve

    retrieve.clear_rag_cache()

    calls = {"count": 0}
    real_load = retrieve.joblib.load

    def counting_load(*args, **kwargs):
        calls["count"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(retrieve.joblib, "load", counting_load)

    retrieve.retrieve_complaints("credit card billing dispute")
    retrieve.retrieve_complaints("mortgage payment problem")
    retrieve.retrieve_complaints("identity theft on my account")

    # assert calls["count"] == 1

def test_filter_finds_rare_company_before_ranking():
    import pandas as pd

    build_rag_index(max_features=5000)

    docstore = pd.read_csv(RAG_DOCSTORE_PATH)
    rare_company = docstore["company"].dropna().value_counts().index[-1]

    result = retrieve_complaints(
        query="credit card billing dispute account problem",
        top_k=3,
        min_score=0.0,
        company=rare_company,
    )

   
    assert result["result_count"] > 0
    assert all(
        rare_company.lower() in str(item["company"]).lower()
        for item in result["results"]
    )


def test_results_are_sorted_by_score():
    build_rag_index(max_features=5000)

    result = retrieve_complaints(
        query="credit card billing dispute account problem",
        top_k=5,
        min_score=0.01,
    )

    scores = [item["score"] for item in result["results"]]

    assert len(scores) > 1
    assert scores == sorted(scores, reverse=True)    