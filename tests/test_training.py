import json
from pathlib import Path

import joblib

from src.training.train import (
    BASELINE_MODEL_PATH,
    MODEL_METADATA_PATH,
    train_baseline_model,
)


def test_train_baseline_model_writes_artifacts():
    metadata = train_baseline_model()

    assert BASELINE_MODEL_PATH.exists()
    assert MODEL_METADATA_PATH.exists()

    saved_metadata = json.loads(Path(MODEL_METADATA_PATH).read_text(encoding="utf-8"))

    assert saved_metadata["model_type"] == "logistic_regression_baseline"
    assert "logistic_regression" in saved_metadata["metrics"]
    assert "dummy_prior" in saved_metadata["metrics"]

    logistic_metrics = saved_metadata["metrics"]["logistic_regression"]

    assert 0.0 <= logistic_metrics["roc_auc"] <= 1.0
    assert 0.0 <= logistic_metrics["pr_auc"] <= 1.0
    assert 0.0 <= logistic_metrics["f1"] <= 1.0

    artifact = joblib.load(BASELINE_MODEL_PATH)

    assert "model" in artifact
    assert "model_version" in artifact
    assert "threshold" in artifact

    assert metadata["model_version"] == saved_metadata["model_version"]