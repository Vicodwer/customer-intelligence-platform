from pathlib import Path


def test_readme_exists_and_documents_core_commands():
    readme = Path("README.md")
    assert readme.exists()

    content = readme.read_text(encoding="utf-8")

    required_sections = [
        "Customer Intelligence Platform",
        "Fresh clone setup",
        "Run the full local pipeline",
        "GET `/health`",
        "POST `/predict`",
        "POST `/ask-complaints`",
        "CI/CD",
        "Known limitations",
    ]

    for section in required_sections:
        assert section in content, f"README missing section: {section}"

    required_commands = [
        "python -m src.data_pipeline.ingest",
        "python -m src.data_pipeline.validate",
        "python -m src.data_pipeline.features",
        "python -m src.training.train",
        "python -m src.training.evaluate",
        "python -m src.rag.build_index",
        "python -m src.rag.rag_eval",
        "uvicorn src.serving.serve:app --reload",
        "pytest -q",
    ]

    for command in required_commands:
        assert command in content, f"README missing command: {command}"