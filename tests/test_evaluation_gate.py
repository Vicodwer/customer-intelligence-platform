import json
from pathlib import Path

from src.training.evaluate import (
    CANDIDATE_MODEL_PATH,
    CHAMPION_MODEL_PATH,
    PROMOTION_REPORT_PATH,
    run_promotion_gate,
)


def test_promotion_gate_writes_artifacts_and_blocks_worse_model():
    report = run_promotion_gate()

    assert CANDIDATE_MODEL_PATH.exists()
    assert CHAMPION_MODEL_PATH.exists()
    assert PROMOTION_REPORT_PATH.exists()

    saved_report = json.loads(
        Path(PROMOTION_REPORT_PATH).read_text(encoding="utf-8")
    )

    assert "promotion_decision" in saved_report
    assert "worse_model_block_decision" in saved_report

    assert saved_report["worse_model_block_decision"]["promoted"] is False
    assert report["worse_model_block_decision"]["promoted"] is False

    assert saved_report["champion_source"] in {
        "candidate_random_forest",
        "baseline_logistic_regression",
    }

    candidate_metrics = saved_report["models"]["candidate_random_forest"]["metrics"]

    assert 0.0 <= candidate_metrics["roc_auc"] <= 1.0
    assert 0.0 <= candidate_metrics["pr_auc"] <= 1.0
    assert 0.0 <= candidate_metrics["f1"] <= 1.0