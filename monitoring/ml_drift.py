from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
MONITORING_DIR = PROJECT_ROOT / "monitoring"

BANK_FEATURES_CSV = PROCESSED_DIR / "bank_features.csv"
ML_DRIFT_JSON = PROCESSED_DIR / "ml_drift_report.json"
MONITORING_REPORT_MD = DOCS_DIR / "monitoring_report.md"


NUMERIC_COLUMNS = [
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

CATEGORICAL_COLUMNS = [
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


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
) -> float:
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    if expected.empty or actual.empty:
        return 0.0

    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(expected.quantile(quantiles).to_numpy())

    if len(breakpoints) < 3:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def categorical_drift_score(reference: pd.Series, current: pd.Series) -> float:
    reference_dist = reference.fillna("missing").astype(str).value_counts(normalize=True)
    current_dist = current.fillna("missing").astype(str).value_counts(normalize=True)

    categories = sorted(set(reference_dist.index) | set(current_dist.index))

    score = 0.0

    for category in categories:
        ref_pct = max(float(reference_dist.get(category, 0.0)), 0.0001)
        cur_pct = max(float(current_dist.get(category, 0.0)), 0.0001)
        score += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)

    return float(score)


def simulate_current_batch(reference_df: pd.DataFrame) -> pd.DataFrame:
    current = reference_df.sample(n=min(1000, len(reference_df)), random_state=99).copy()

    if "age" in current.columns:
        current["age"] = current["age"] + 8

    if "campaign" in current.columns:
        current["campaign"] = current["campaign"] + 2

    if "month" in current.columns:
        current["month"] = "nov"

    return current


def run_ml_drift_report() -> dict:
    if not BANK_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Missing features file: {BANK_FEATURES_CSV}. "
            "Run python -m src.data_pipeline.features first."
        )

    df = pd.read_csv(BANK_FEATURES_CSV)

    reference = df.sample(n=min(1000, len(df)), random_state=42)
    current = simulate_current_batch(df)

    feature_reports = []

    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            continue

        psi = population_stability_index(
            reference[column].to_numpy(),
            current[column].to_numpy(),
        )

        feature_reports.append(
            {
                "feature": column,
                "type": "numeric",
                "drift_score": psi,
                "drifted": psi >= 0.2,
            }
        )

    for column in CATEGORICAL_COLUMNS:
        if column not in df.columns:
            continue

        score = categorical_drift_score(reference[column], current[column])

        feature_reports.append(
            {
                "feature": column,
                "type": "categorical",
                "drift_score": score,
                "drifted": score >= 0.2,
            }
        )

    drifted_features = [item for item in feature_reports if item["drifted"]]

    payload = {
        "report_type": "ml_drift",
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "drift_threshold": 0.2,
        "drifted_feature_count": len(drifted_features),
        "drift_detected": len(drifted_features) > 0,
        "features": feature_reports,
        "recommended_action": (
            "Review feature shift and consider retraining trigger."
            if drifted_features
            else "No material drift detected in this simulated batch."
        ),
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ML_DRIFT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def append_ml_monitoring_markdown(payload: dict) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    drifted = [
        item for item in payload["features"]
        if item["drifted"]
    ]

    drifted_lines = "\n".join(
        [
            f"- {item['feature']} ({item['type']}): {item['drift_score']:.4f}"
            for item in drifted
        ]
    ) or "No drifted features."

    content = f"""# Monitoring Report

## ML drift monitoring

- Reference rows: {payload["reference_rows"]}
- Current rows: {payload["current_rows"]}
- Drift threshold: {payload["drift_threshold"]}
- Drift detected: {payload["drift_detected"]}
- Drifted feature count: {payload["drifted_feature_count"]}

### Drifted features

{drifted_lines}

### Recommended action

{payload["recommended_action"]}
"""

    MONITORING_REPORT_MD.write_text(content, encoding="utf-8")


def main() -> None:
    payload = run_ml_drift_report()
    append_ml_monitoring_markdown(payload)

    print(f"Saved ML drift report: {ML_DRIFT_JSON}")
    print(f"Updated monitoring report: {MONITORING_REPORT_MD}")
    print(f"Drift detected: {payload['drift_detected']}")


if __name__ == "__main__":
    main()