from src.rag.rag_eval import (
    RAG_EVAL_RESULTS_PATH,
    RAG_REPORT_PATH,
    run_rag_eval,
)


def test_rag_eval_writes_results_and_report():
    payload = run_rag_eval(rebuild_index=True)

    assert RAG_EVAL_RESULTS_PATH.exists()
    assert RAG_REPORT_PATH.exists()

    assert payload["summary"]["total_tests"] == 10
    assert payload["summary"]["passed"] >= 8
    assert payload["summary"]["pass_rate"] >= 0.8

    for result in payload["results"]:
        assert "question" in result
        assert "evidence_ids" in result
        assert "expected_evidence_rule" in result
        assert "passed" in result
        assert "note" in result