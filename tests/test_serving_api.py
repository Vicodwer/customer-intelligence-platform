from fastapi.testclient import TestClient

from src.serving.serve import app


client = TestClient(app)


VALID_PAYLOAD = {
    "customer": {
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
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] in {"ok", "degraded"}
    assert "app_version" in payload
    assert "model_loaded" in payload


def test_predict_endpoint_valid_payload():
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    payload = response.json()

    assert payload["prediction"] in {0, 1}
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["decision"] in {"likely_to_convert", "unlikely_to_convert"}
    assert payload["conversion_band"] in {"low", "medium", "high"}
    assert "model_version" in payload


def test_predict_endpoint_rejects_invalid_payload():
    invalid_payload = {
        "customer": {
            **VALID_PAYLOAD["customer"],
            "age": 5,
        }
    }

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422