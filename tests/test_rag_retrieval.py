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