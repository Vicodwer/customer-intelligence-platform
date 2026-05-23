from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

FEATURES_PATH = PROCESSED_DIR / "bank_features.csv"
TARGET_PATH = PROCESSED_DIR / "bank_target.csv"
FEATURE_METADATA_PATH = PROCESSED_DIR / "feature_metadata.json"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
BASELINE_PREDICTIONS_PATH = PROCESSED_DIR / "baseline_predictions.csv"
MODEL_REPORT_PATH = DOCS_DIR / "model_report.md"

TARGET_COLUMN = "y"

NUMERIC_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing features file: {FEATURES_PATH}. "
            "Run: python -m src.data_pipeline.features"
        )

    if not TARGET_PATH.exists():
        raise FileNotFoundError(
            f"Missing target file: {TARGET_PATH}. "
            "Run: python -m src.data_pipeline.features"
        )

    X = pd.read_csv(FEATURES_PATH)
    y = pd.read_csv(TARGET_PATH)[TARGET_COLUMN]

    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_logistic_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def build_dummy_model() -> DummyClassifier:
    return DummyClassifier(strategy="prior", random_state=42)


def evaluate_binary_classifier(
    y_true: pd.Series,
    probabilities,
    threshold: float = 0.5,
) -> dict:
    y_pred = [1 if probability >= threshold else 0 for probability in probabilities]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    return metrics


def write_model_report(metadata: dict) -> None:
    logistic = metadata["metrics"]["logistic_regression"]
    dummy = metadata["metrics"]["dummy_prior"]

    confusion_matrix_text = json.dumps(logistic["confusion_matrix"], indent=2)

    content = f"""# Model Report

## Model purpose

Predict whether a contacted customer will subscribe to a term deposit.

## Dataset

- Source: UCI Bank Marketing sample
- Training rows: {metadata["split"]["train_rows"]}
- Test rows: {metadata["split"]["test_rows"]}
- Target: `{TARGET_COLUMN}`

## Feature decision

The `duration` column is intentionally excluded because it is only known after the call ends. Keeping it would create train-serving leakage.

## Baseline comparison

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Dummy prior | {dummy["roc_auc"]:.4f} | {dummy["pr_auc"]:.4f} | {dummy["f1"]:.4f} | {dummy["precision"]:.4f} | {dummy["recall"]:.4f} | {dummy["brier_score"]:.4f} |
| Logistic regression | {logistic["roc_auc"]:.4f} | {logistic["pr_auc"]:.4f} | {logistic["f1"]:.4f} | {logistic["precision"]:.4f} | {logistic["recall"]:.4f} | {logistic["brier_score"]:.4f} |

## Confusion matrix at threshold 0.5

{confusion_matrix_text}

## Current decision

The logistic regression pipeline is saved as the current baseline serving candidate. A stricter promotion gate will be added in the next modelling step.
"""

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_REPORT_PATH.write_text(content, encoding="utf-8")


def train_baseline_model() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    dummy_model = build_dummy_model()
    dummy_model.fit(X_train, y_train)
    dummy_probabilities = dummy_model.predict_proba(X_test)[:, 1]

    logistic_model = build_logistic_model()
    logistic_model.fit(X_train, y_train)
    logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]

    dummy_metrics = evaluate_binary_classifier(y_test, dummy_probabilities)
    logistic_metrics = evaluate_binary_classifier(y_test, logistic_probabilities)

    predictions_df = X_test.copy()
    predictions_df["actual"] = y_test.values
    predictions_df["predicted_probability"] = logistic_probabilities
    predictions_df["predicted_label"] = (
        predictions_df["predicted_probability"] >= logistic_metrics["threshold"]
    ).astype(int)
    predictions_df.to_csv(BASELINE_PREDICTIONS_PATH, index=False)

    model_version = datetime.now(timezone.utc).strftime("baseline-%Y%m%dT%H%M%SZ")

    artifact = {
        "model": logistic_model,
        "model_version": model_version,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "threshold": logistic_metrics["threshold"],
    }

    joblib.dump(artifact, BASELINE_MODEL_PATH)

    metadata = {
        "model_version": model_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "logistic_regression_baseline",
        "artifact_path": str(BASELINE_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "prediction_sample_path": str(BASELINE_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
        "feature_metadata_path": str(FEATURE_METADATA_PATH.relative_to(PROJECT_ROOT)),
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_size": 0.2,
            "random_state": 42,
            "stratified": True,
        },
        "metrics": {
            "dummy_prior": dummy_metrics,
            "logistic_regression": logistic_metrics,
        },
    }

    MODEL_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    write_model_report(metadata)

    print(f"Saved model artifact: {BASELINE_MODEL_PATH}")
    print(f"Saved model metadata: {MODEL_METADATA_PATH}")
    print(f"Saved predictions: {BASELINE_PREDICTIONS_PATH}")
    print(f"Saved model report: {MODEL_REPORT_PATH}")

    print("\nLogistic regression metrics:")
    for key, value in logistic_metrics.items():
        print(f"  {key}: {value}")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline campaign model.")
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    train_baseline_model()