from monitoring.ml_drift import ML_DRIFT_JSON, run_ml_drift_report
from monitoring.rag_monitor import RAG_MONITOR_JSON, run_rag_monitoring


def test_ml_drift_report_is_generated():
    payload = run_ml_drift_report()

    assert ML_DRIFT_JSON.exists()
    assert payload["report_type"] == "ml_drift"
    assert "drift_detected" in payload
    assert "features" in payload
    assert len(payload["features"]) > 0


def test_rag_monitoring_metrics_are_generated():
    payload = run_rag_monitoring(rebuild_index=True)

    assert RAG_MONITOR_JSON.exists()
    assert payload["report_type"] == "rag_monitoring"
    assert payload["request_count"] >= 4
    assert 0.0 <= payload["retrieval_hit_rate"] <= 1.0
    assert 0.0 <= payload["refusal_rate"] <= 1.0
    assert payload["avg_latency_ms"] >= 0.0