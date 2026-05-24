from pathlib import Path


def test_architecture_doc_exists_and_mentions_core_components():
    path = Path("docs/architecture.md")
    assert path.exists()

    content = path.read_text(encoding="utf-8")

    required_terms = [
        "Customer Intelligence Platform",
        "ML lane",
        "RAG lane",
        "FastAPI",
        "/predict",
        "/batch-score",
        "/ask-complaints",
        "/customer-intel",
        "GitHub Actions",
        "Docker",
    ]

    for term in required_terms:
        assert term in content, f"architecture.md missing: {term}"


def test_decision_log_exists_and_records_key_decisions():
    path = Path("docs/decision_log.md")
    assert path.exists()

    content = path.read_text(encoding="utf-8")

    required_terms = [
        "Exclude `duration`",
        "logistic regression",
        "random forest",
        "promotion gate",
        "TF-IDF",
        "refusal",
        "Docker",
    ]

    for term in required_terms:
        assert term in content, f"decision_log.md missing: {term}"