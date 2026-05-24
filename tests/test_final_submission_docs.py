from pathlib import Path


def test_demo_script_exists():
    path = Path("docs/demo_script.md")
    assert path.exists()

    content = path.read_text(encoding="utf-8")

    required_terms = [
        "Demo Script",
        "/health",
        "/predict",
        "/ask-complaints",
        "/customer-intel",
        "/batch-score",
        "Docker",
        "GitHub Actions",
    ]

    for term in required_terms:
        assert term in content, f"demo_script.md missing: {term}"


def test_submission_checklist_exists():
    path = Path("docs/submission_checklist.md")
    assert path.exists()

    content = path.read_text(encoding="utf-8")

    required_terms = [
        "Submission Checklist",
        "FastAPI",
        "RAG",
        "ML drift report",
        "Docker",
        "CI passing",
    ]

    for term in required_terms:
        assert term in content, f"submission_checklist.md missing: {term}"


def test_reflection_exists():
    path = Path("docs/reflection.md")
    assert path.exists()

    content = path.read_text(encoding="utf-8")

    required_terms = [
        "Reflection",
        "What was built",
        "Key technical decisions",
        "Current limitations",
        "Final outcome",
    ]

    for term in required_terms:
        assert term in content, f"reflection.md missing: {term}"