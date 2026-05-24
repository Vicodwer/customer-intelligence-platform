from fastapi.testclient import TestClient

from src.rag.build_index import build_rag_index
from src.serving.serve import app


client = TestClient(app)


VALID_CUSTOMER = {
    "age": 35,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191.0,
}


def test_customer_intel_endpoint_combines_prediction_and_complaints():
    build_rag_index(max_features=5000)

    response = client.post(
        "/customer-intel",
        json={
            "customer": VALID_CUSTOMER,
            "complaint_question": "credit card billing dispute account problem",
            "product": "Credit card",
            "top_k": 3,
            "min_score": 0.01,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert "conversion" in payload
    assert "complaint_intelligence" in payload
    assert "integration_note" in payload

    conversion = payload["conversion"]
    complaint_intel = payload["complaint_intelligence"]

    assert 0.0 <= conversion["probability"] <= 1.0
    assert conversion["conversion_band"] in {"low", "medium", "high"}
    assert "model_version" in conversion

    assert complaint_intel["refused"] is False
    assert len(complaint_intel["evidence_ids"]) > 0
    assert "evidence_sufficiency_note" in complaint_intel


def test_customer_intel_endpoint_rejects_invalid_customer():
    build_rag_index(max_features=5000)

    invalid_customer = {
        **VALID_CUSTOMER,
        "age": 5,
    }

    response = client.post(
        "/customer-intel",
        json={
            "customer": invalid_customer,
            "complaint_question": "credit card billing dispute account problem",
            "top_k": 3,
            "min_score": 0.01,
        },
    )

    assert response.status_code == 422