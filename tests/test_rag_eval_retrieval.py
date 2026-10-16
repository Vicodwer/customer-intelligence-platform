import pandas as pd
import pytest

from src.rag import eval_retrieval as ev
from src.rag.build_index import build_rag_index
from src.rag.retrieve import load_rag_artifacts


def _tiny_docstore() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "complaint_id": ["1", "2", "3", "4", "5", "6"],
            "product": ["P", "P", "P", "Q", "Q", "Q"],
            "issue": ["I1", "I1", "I1", "I2", "I2", "I3"],
        }
    )


def _query(**overrides) -> dict:
    query = {
        "id": "x",
        "question": "anything",
        "product": "P",
        "issue": "I1",
        "strength": "strong",
    }
    query.update(overrides)
    return query


# ---------- metric maths (no index needed) ----------


def test_metrics_are_computed_correctly():
    # Relevant docs are 1, 2, 3. Top five returned: 4, 1, 5, 2, 6.
    def retriever(query: str, depth: int) -> list[str]:
        return ["4", "1", "5", "2", "6", "3"][:depth]

    result = ev.evaluate(retriever, [_query()], _tiny_docstore(), k=5, depth=10)
    row = result["per_query"][0]

    assert row["precision_at_k"] == pytest.approx(2 / 5)
    assert row["hit_at_k"] is True
    assert row["reciprocal_rank"] == pytest.approx(1 / 2)
    assert row["random_precision"] == pytest.approx(3 / 6)
    assert result["summary"]["lift_over_random"] == pytest.approx(0.4 / 0.5)


def test_returning_too_few_results_is_penalised():
    def retriever(query: str, depth: int) -> list[str]:
        return ["1"]

    row = ev.evaluate(retriever, [_query()], _tiny_docstore(), k=5)["per_query"][0]

    assert row["precision_at_k"] == pytest.approx(1 / 5)


def test_no_results_scores_zero():
    def retriever(query: str, depth: int) -> list[str]:
        return []

    row = ev.evaluate(retriever, [_query()], _tiny_docstore())["per_query"][0]

    assert row["precision_at_k"] == 0
    assert row["hit_at_k"] is False
    assert row["reciprocal_rank"] == 0


def test_unknown_label_is_rejected():
    def retriever(query: str, depth: int) -> list[str]:
        return []

    with pytest.raises(ValueError):
        ev.evaluate(retriever, [_query(issue="Typo")], _tiny_docstore())


def test_category_name_variants_are_merged():
    assert ev.normalize_product("Credit card") == "Credit card or prepaid card"
    assert (
        ev.normalize_product(
            "Credit reporting, credit repair services, or other personal consumer reports"
        )
        == "Credit reporting"
    )
    assert (
        ev.normalize_issue("Dealing with my lender or servicer")
        == "Dealing with your lender or servicer"
    )


# ---------- golden file ----------


def test_golden_file_is_well_formed():
    queries = ev.load_golden_queries()

    assert len(queries) >= 25
    assert {query["strength"] for query in queries} <= set(ev.VALID_STRENGTHS)


# ---------- against the real index ----------


@pytest.fixture(scope="module")
def real_results() -> dict:
    build_rag_index(max_features=5000)
    return ev.run_evaluation(["tfidf"])["retrievers"]["tfidf"]


def test_every_golden_label_exists_in_the_index():
    build_rag_index(max_features=5000)
    _, docstore, _ = load_rag_artifacts()
    _, group_sizes = ev.build_group_lookup(docstore)

    for query in ev.load_golden_queries():
        key = (query["product"], query["issue"])

        assert group_sizes.get(key, 0) >= 10, f"{query['id']}: {key} too small or missing"


def test_tfidf_baseline_beats_random_by_a_wide_margin(real_results):
    summary = real_results["summary"]

    # Floors sit well below the measured baseline so they catch a real
    # regression (e.g. a broken index) without failing on tiny numeric drift.
    assert summary["lift_over_random"] >= 5
    assert summary["hit_rate_at_k"] >= 0.5
    assert summary["precision_at_k"] >= 0.2


def test_report_renders_all_sections(real_results):
    text = ev.render_report(
        {
            "index_version": "test",
            "doc_count": 5000,
            "query_count": 29,
            "retrievers": {"tfidf": real_results},
        }
    )

    assert "# Retrieval evaluation" in text
    assert "Limits of this evaluation" in text
    assert "| tfidf |" in text