from pathlib import Path


def test_github_actions_ci_workflow_exists():
    path = Path(".github/workflows/ci.yml")
    assert path.exists(), "Missing GitHub Actions workflow: .github/workflows/ci.yml"

    content = path.read_text(encoding="utf-8")

    required_commands = [
        "python -m src.data_pipeline.ingest",
        "python -m src.data_pipeline.validate",
        "python -m src.data_pipeline.features",
        "python -m src.training.train",
        "python -m src.training.evaluate",
        "python -m src.rag.build_index",
        "python -m src.rag.rag_eval",
        "pytest -q",
    ]

    for command in required_commands:
        assert command in content, f"CI workflow missing command: {command}"