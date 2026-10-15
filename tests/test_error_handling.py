import re

from fastapi.testclient import TestClient

from src.serving import serve

client = TestClient(serve.app, raise_server_exceptions=False)

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


def test_every_response_has_a_request_id():
    response = client.get("/health/live")

    assert re.fullmatch(r"[0-9a-f]{12}", response.headers["X-Request-ID"])


def test_client_supplied_request_id_is_echoed():
    response = client.get("/health/live", headers={"X-Request-ID": "abc-123"})

    assert response.headers["X-Request-ID"] == "abc-123"


def test_unsafe_request_id_is_replaced():
    response = client.get(
        "/health/live",
        headers={"X-Request-ID": "bad id with spaces!"},
    )

    assert re.fullmatch(r"[0-9a-f]{12}", response.headers["X-Request-ID"])


def test_prediction_error_is_generic_and_carries_reference(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail C:\\models\\private")

    monkeypatch.setattr(serve, "load_model_artifact", boom)

    response = client.post("/predict", json={"customer": VALID_CUSTOMER})

    assert response.status_code == 500

    detail = response.json()["detail"]

    assert "secret" not in detail
    assert response.headers["X-Request-ID"] in detail


def test_missing_rag_index_does_not_leak_file_paths(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("Missing RAG index: C:\\secret\\folder\\index.joblib")

    monkeypatch.setattr(serve, "answer_complaint_question", boom)

    response = client.post(
        "/ask-complaints",
        json={"question": "credit card billing dispute"},
    )

    assert response.status_code == 503
    assert "secret" not in response.json()["detail"]


def test_responses_do_not_expose_model_path():
    predict_response = client.post("/predict", json={"customer": VALID_CUSTOMER})

    assert predict_response.status_code == 200
    assert "model_path" not in predict_response.json()
    assert "model_path" not in client.get("/health").json()