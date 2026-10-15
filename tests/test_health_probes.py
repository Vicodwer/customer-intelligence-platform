from fastapi.testclient import TestClient

from src.serving import serve

client = TestClient(serve.app)


def test_live_is_always_ok():
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_ready_returns_200_when_model_and_index_exist():
    response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True
    assert body["rag_index_loaded"] is True
    assert body["problems"] == []


def test_ready_returns_503_when_model_missing(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("no model")

    monkeypatch.setattr(serve, "load_model_artifact", boom)

    response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["model_loaded"] is False
    assert "model_unavailable" in body["problems"]


def test_ready_returns_503_when_rag_index_missing(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("no index")

    monkeypatch.setattr(serve, "load_rag_artifacts", boom)

    response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["rag_index_loaded"] is False
    assert "rag_index_unavailable" in body["problems"]