from fastapi.testclient import TestClient

from src.rag.answer import answer_complaint_question
from src.rag.build_index import build_rag_index
from src.serving.serve import app


client = TestClient(app)


def test_answer_complaint_question_returns_grounded_answer():
    build_rag_index(max_features=5000)

    result = answer_complaint_question(
        question="credit card billing dispute account problem",
        top_k=3,
        min_score=0.01,
    )

    assert result["refused"] is False
    assert len(result["evidence_ids"]) > 0
    assert "Sufficient evidence" in result["evidence_sufficiency_note"]
    assert result["prompt_version"] == "deterministic-complaint-answer-v1"


def test_answer_complaint_question_refuses_weak_evidence():
    build_rag_index(max_features=5000)

    result = answer_complaint_question(
        question="zzzz qwerty unrelated nonsense",
        top_k=3,
        min_score=0.8,
    )

    assert result["refused"] is True
    assert result["evidence_ids"] == []
    assert "Insufficient evidence" in result["evidence_sufficiency_note"]


def test_ask_complaints_endpoint_returns_evidence_ids():
    build_rag_index(max_features=5000)

    response = client.post(
        "/ask-complaints",
        json={
            "question": "credit card billing dispute account problem",
            "top_k": 3,
            "min_score": 0.01,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["refused"] is False
    assert len(payload["evidence_ids"]) > 0
    assert "evidence_sufficiency_note" in payload
    assert "prompt_version" in payload


def test_ask_complaints_endpoint_refuses_weak_evidence():
    build_rag_index(max_features=5000)

    response = client.post(
        "/ask-complaints",
        json={
            "question": "zzzz qwerty unrelated nonsense",
            "top_k": 3,
            "min_score": 0.8,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["refused"] is True
    assert payload["evidence_ids"] == []