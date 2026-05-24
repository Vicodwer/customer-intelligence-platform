from fastapi.testclient import TestClient

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


def test_batch_score_endpoint_scores_multiple_customers():
    response = client.post(
        "/batch-score",
        json={
            "customers": [
                VALID_CUSTOMER,
                VALID_CUSTOMER,
            ]
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["scored_count"] == 2
    assert "model_version" in payload
    assert "model_type" in payload
    assert "band_counts" in payload
    assert len(payload["predictions"]) == 2

    for item in payload["predictions"]:
        assert item["prediction"] in {0, 1}
        assert 0.0 <= item["probability"] <= 1.0
        assert item["conversion_band"] in {"low", "medium", "high"}


def test_batch_score_endpoint_rejects_empty_batch():
    response = client.post(
        "/batch-score",
        json={"customers": []},
    )

    assert response.status_code == 422


def test_batch_score_endpoint_rejects_invalid_customer_in_batch():
    invalid_customer = {
        **VALID_CUSTOMER,
        "age": 5,
    }

    response = client.post(
        "/batch-score",
        json={"customers": [invalid_customer]},
    )

    assert response.status_code == 422