from pathlib import Path


def test_dockerfile_exists_and_runs_api():
    path = Path("Dockerfile")
    assert path.exists()

    content = path.read_text(encoding="utf-8")

    assert "FROM python:3.11-slim" in content
    assert "pip install" in content
    assert "uvicorn src.serving.serve:app" in content
    assert "python -m src.data_pipeline.ingest" in content
    assert "python -m src.training.evaluate" in content
    assert "python -m src.rag.build_index" in content


def test_dockerignore_exists_and_excludes_heavy_artifacts():
    path = Path(".dockerignore")
    assert path.exists()

    content = path.read_text(encoding="utf-8")

    required_patterns = [
        ".venv",
        "data/raw/*",
        "data/processed/*",
        "models/*",
        "artifacts/*",
    ]

    for pattern in required_patterns:
        assert pattern in content


def test_docker_compose_exists():
    path = Path("docker-compose.yml")
    assert path.exists()

    content = path.read_text(encoding="utf-8")

    assert "customer-intel-api" in content
    assert "8000:8000" in content