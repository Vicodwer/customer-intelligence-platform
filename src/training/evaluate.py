from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.training.train import (
    BASELINE_MODEL_PATH,
    CATEGORICAL_FEATURES,
    DOCS_DIR,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    build_logistic_model,
    build_preprocessor,
    evaluate_binary_classifier,
    load_training_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_MODEL_PATH = MODELS_DIR / "candidate_model.joblib"
CHAMPION_MODEL_PATH = MODELS_DIR / "champion_model.joblib"
PROMOTION_REPORT_PATH = MODELS_DIR / "promotion_report.json"
EVALUATION_PREDICTIONS_PATH = PROCESSED_DIR / "evaluation_predictions.csv"
EVALUATION_REPORT_MD = DOCS_DIR / "promotion_report.md"

PROMOTION_RULES = {
    "min_pr_auc_improvement": 0.01,
    "max_f1_drop": 0.02,
    "max_roc_auc_drop": 0.01,
}


def build_candidate_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=8,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_deliberately_worse_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                DummyClassifier(
                    strategy="most_frequent",
                    random_state=42,
                ),
            ),
        ]
    )


def decide_promotion(
    baseline_metrics: dict,
    candidate_metrics: dict,
    rules: dict = PROMOTION_RULES,
) -> dict:
    pr_auc_delta = (
        candidate_metrics["pr_auc"] - baseline_metrics["pr_auc"]
    )

    f1_delta = (
        candidate_metrics["f1"] - baseline_metrics["f1"]
    )

    roc_auc_delta = (
        candidate_metrics["roc_auc"] - baseline_metrics["roc_auc"]
    )

    checks = {
        "pr_auc_improvement_check": pr_auc_delta >= rules["min_pr_auc_improvement"],
        "f1_drop_check": f1_delta >= -rules["max_f1_drop"],
        "roc_auc_drop_check": roc_auc_delta >= -rules["max_roc_auc_drop"],
    }

    promoted = all(checks.values())

    return {
        "promoted": promoted,
        "checks": checks,
        "deltas": {
            "pr_auc_delta": float(pr_auc_delta),
            "f1_delta": float(f1_delta),
            "roc_auc_delta": float(roc_auc_delta),
        },
        "rules": rules,
    }


def save_model_artifact(
    model,
    path: Path,
    model_version: str,
    model_type: str,
    threshold: float,
) -> None:
    artifact = {
        "model": model,
        "model_version": model_version,
        "model_type": model_type,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "threshold": threshold,
    }

    joblib.dump(artifact, path)


def write_promotion_markdown(report: dict) -> None:
    baseline = report["models"]["baseline_logistic_regression"]["metrics"]
    candidate = report["models"]["candidate_random_forest"]["metrics"]
    worse = report["models"]["deliberately_worse_dummy"]["metrics"]

    decision = report["promotion_decision"]
    worse_decision = report["worse_model_block_decision"]

    content = f"""# Promotion Gate Report

## Purpose

Compare the current baseline model with an improved candidate model using a relative promotion gate.

## Promotion rules

- PR-AUC must improve by at least {PROMOTION_RULES["min_pr_auc_improvement"]}
- F1 may drop by no more than {PROMOTION_RULES["max_f1_drop"]}
- ROC-AUC may drop by no more than {PROMOTION_RULES["max_roc_auc_drop"]}

## Metrics

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Baseline logistic regression | {baseline["roc_auc"]:.4f} | {baseline["pr_auc"]:.4f} | {baseline["f1"]:.4f} | {baseline["precision"]:.4f} | {baseline["recall"]:.4f} | {baseline["brier_score"]:.4f} |
| Candidate random forest | {candidate["roc_auc"]:.4f} | {candidate["pr_auc"]:.4f} | {candidate["f1"]:.4f} | {candidate["precision"]:.4f} | {candidate["recall"]:.4f} | {candidate["brier_score"]:.4f} |
| Deliberately worse dummy | {worse["roc_auc"]:.4f} | {worse["pr_auc"]:.4f} | {worse["f1"]:.4f} | {worse["precision"]:.4f} | {worse["recall"]:.4f} | {worse["brier_score"]:.4f} |

## Candidate promotion decision

Promoted: {decision["promoted"]}

Reason:
- PR-AUC delta: {decision["deltas"]["pr_auc_delta"]:.4f}
- F1 delta: {decision["deltas"]["f1_delta"]:.4f}
- ROC-AUC delta: {decision["deltas"]["roc_auc_delta"]:.4f}

Checks:
- PR-AUC improvement check: {decision["checks"]["pr_auc_improvement_check"]}
- F1 drop check: {decision["checks"]["f1_drop_check"]}
- ROC-AUC drop check: {decision["checks"]["roc_auc_drop_check"]}

## Deliberately worse model block decision

Blocked: {not worse_decision["promoted"]}

Reason:
- PR-AUC delta: {worse_decision["deltas"]["pr_auc_delta"]:.4f}
- F1 delta: {worse_decision["deltas"]["f1_delta"]:.4f}
- ROC-AUC delta: {worse_decision["deltas"]["roc_auc_delta"]:.4f}

## Champion model

Selected champion artifact:

{report["champion_model_path"]}
"""

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_REPORT_MD.write_text(content, encoding="utf-8")


def run_promotion_gate() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    baseline_model = build_logistic_model()
    baseline_model.fit(X_train, y_train)
    baseline_probabilities = baseline_model.predict_proba(X_test)[:, 1]
    baseline_metrics = evaluate_binary_classifier(y_test, baseline_probabilities)

    candidate_model = build_candidate_model()
    candidate_model.fit(X_train, y_train)
    candidate_probabilities = candidate_model.predict_proba(X_test)[:, 1]
    candidate_metrics = evaluate_binary_classifier(y_test, candidate_probabilities)

    worse_model = build_deliberately_worse_model()
    worse_model.fit(X_train, y_train)
    worse_probabilities = worse_model.predict_proba(X_test)[:, 1]
    worse_metrics = evaluate_binary_classifier(y_test, worse_probabilities)

    candidate_decision = decide_promotion(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
    )

    worse_decision = decide_promotion(
        baseline_metrics=baseline_metrics,
        candidate_metrics=worse_metrics,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    candidate_version = datetime.now(timezone.utc).strftime("candidate-%Y%m%dT%H%M%SZ")
    champion_version = datetime.now(timezone.utc).strftime("champion-%Y%m%dT%H%M%SZ")

    save_model_artifact(
        model=candidate_model,
        path=CANDIDATE_MODEL_PATH,
        model_version=candidate_version,
        model_type="random_forest_candidate",
        threshold=candidate_metrics["threshold"],
    )

    if candidate_decision["promoted"]:
        save_model_artifact(
            model=candidate_model,
            path=CHAMPION_MODEL_PATH,
            model_version=champion_version,
            model_type="random_forest_champion",
            threshold=candidate_metrics["threshold"],
        )
        champion_source = "candidate_random_forest"
    else:
        save_model_artifact(
            model=baseline_model,
            path=CHAMPION_MODEL_PATH,
            model_version=champion_version,
            model_type="logistic_regression_champion",
            threshold=baseline_metrics["threshold"],
        )
        champion_source = "baseline_logistic_regression"

    predictions_df = X_test.copy()
    predictions_df["actual"] = y_test.values
    predictions_df["baseline_probability"] = baseline_probabilities
    predictions_df["candidate_probability"] = candidate_probabilities
    predictions_df["worse_probability"] = worse_probabilities
    predictions_df.to_csv(EVALUATION_PREDICTIONS_PATH, index=False)

    report = {
        "created_at_utc": created_at,
        "promotion_rules": PROMOTION_RULES,
        "models": {
            "baseline_logistic_regression": {
                "metrics": baseline_metrics,
            },
            "candidate_random_forest": {
                "artifact_path": str(CANDIDATE_MODEL_PATH.relative_to(PROJECT_ROOT)),
                "metrics": candidate_metrics,
            },
            "deliberately_worse_dummy": {
                "metrics": worse_metrics,
            },
        },
        "promotion_decision": candidate_decision,
        "worse_model_block_decision": worse_decision,
        "champion_source": champion_source,
        "champion_model_path": str(CHAMPION_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "evaluation_predictions_path": str(
            EVALUATION_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)
        ),
    }

    PROMOTION_REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    write_promotion_markdown(report)

    if not BASELINE_MODEL_PATH.exists():
        save_model_artifact(
            model=baseline_model,
            path=BASELINE_MODEL_PATH,
            model_version="baseline-from-promotion-gate",
            model_type="logistic_regression_baseline",
            threshold=baseline_metrics["threshold"],
        )

    if not MODEL_METADATA_PATH.exists():
        MODEL_METADATA_PATH.write_text(
            json.dumps(
                {
                    "model_type": "logistic_regression_baseline",
                    "metrics": {
                        "logistic_regression": baseline_metrics,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"Saved candidate model: {CANDIDATE_MODEL_PATH}")
    print(f"Saved champion model: {CHAMPION_MODEL_PATH}")
    print(f"Saved promotion report: {PROMOTION_REPORT_PATH}")
    print(f"Saved promotion markdown: {EVALUATION_REPORT_MD}")
    print(f"Saved evaluation predictions: {EVALUATION_PREDICTIONS_PATH}")

    print("\nBaseline metrics:")
    for key, value in baseline_metrics.items():
        print(f"  {key}: {value}")

    print("\nCandidate metrics:")
    for key, value in candidate_metrics.items():
        print(f"  {key}: {value}")

    print("\nCandidate promoted:")
    print(f"  {candidate_decision['promoted']}")

    print("\nDeliberately worse model blocked:")
    print(f"  {not worse_decision['promoted']}")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate models and run promotion gate.")
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    run_promotion_gate()