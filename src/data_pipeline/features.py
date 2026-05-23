from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
PROCESSED_DIR = DATA_DIR / "processed"

BANK_SAMPLE_CSV = SAMPLES_DIR / "bank_marketing_sample.csv"
COMPLAINT_SAMPLE_CSV = SAMPLES_DIR / "cfpb_complaints_sample.csv"

BANK_FEATURES_CSV = PROCESSED_DIR / "bank_features.csv"
BANK_TARGET_CSV = PROCESSED_DIR / "bank_target.csv"
COMPLAINTS_CLEAN_CSV = PROCESSED_DIR / "complaints_clean.csv"
FEATURE_METADATA_JSON = PROCESSED_DIR / "feature_metadata.json"


TARGET_COLUMN = "y"

# Important:
# We intentionally drop "duration" for production-style campaign prediction.
# Duration is only known after a phone call ends, so using it would create leakage.
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

BANK_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

COMPLAINT_METADATA_COLUMNS = [
    "complaint_id",
    "date_received",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "company",
    "state",
    "company_response",
    "timely",
    "submitted_via",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_bank_sample(path: Path = BANK_SAMPLE_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Bank sample not found: {path}")

    return pd.read_csv(path)


def load_complaints_sample(path: Path = COMPLAINT_SAMPLE_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Complaint sample not found: {path}")

    return pd.read_csv(path)


def clean_bank_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for column in CATEGORICAL_FEATURES:
        if column in cleaned.columns:
            cleaned[column] = (
                cleaned[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": "unknown", "": "unknown"})
            )

    for column in NUMERIC_FEATURES:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = (
            cleaned[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0})
        )

    return cleaned


def split_bank_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    missing_features = set(BANK_FEATURE_COLUMNS) - set(df.columns)

    if missing_features:
        raise ValueError(f"Missing bank feature columns: {sorted(missing_features)}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    cleaned = clean_bank_dataframe(df)

    X = cleaned[BANK_FEATURE_COLUMNS].copy()
    y = cleaned[TARGET_COLUMN].copy()

    if y.isna().any():
        raise ValueError("Target column contains values outside expected yes/no mapping.")

    return X, y.astype(int)


def clean_text_for_rag(text: str) -> str:
    if pd.isna(text):
        return ""

    cleaned = str(text)

    # Light PII-style redaction. We keep meaning but avoid exposing obvious contact details.
    cleaned = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", cleaned)
    cleaned = re.sub(r"\b(?:\+?\d[\d\-\s().]{7,}\d)\b", "[PHONE]", cleaned)

    # Normalize whitespace.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def clean_complaints_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {"complaint_id", "consumer_complaint_narrative", "product", "issue", "company"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing complaint columns: {sorted(missing)}")

    cleaned = df.copy()

    cleaned["clean_narrative"] = cleaned["consumer_complaint_narrative"].apply(
        clean_text_for_rag
    )

    cleaned = cleaned[cleaned["clean_narrative"].str.len() >= 30].copy()

    for column in COMPLAINT_METADATA_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = None

    output_columns = COMPLAINT_METADATA_COLUMNS + [
        "clean_narrative",
    ]

    cleaned = cleaned[output_columns]
    cleaned = cleaned.drop_duplicates(subset=["complaint_id"])
    cleaned = cleaned.reset_index(drop=True)

    return cleaned


def write_feature_metadata(
    bank_input_path: Path = BANK_SAMPLE_CSV,
    complaints_input_path: Path = COMPLAINT_SAMPLE_CSV,
) -> Path:
    metadata = {
        "target_column": TARGET_COLUMN,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "bank_feature_columns": BANK_FEATURE_COLUMNS,
        "dropped_columns": {
            "duration": "Dropped to avoid leakage because call duration is only known after the contact ends."
        },
        "complaint_text_column": "clean_narrative",
        "complaint_metadata_columns": COMPLAINT_METADATA_COLUMNS,
        "input_hashes": {
            "bank_sample_sha256": file_sha256(bank_input_path),
            "complaints_sample_sha256": file_sha256(complaints_input_path),
        },
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    FEATURE_METADATA_JSON.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return FEATURE_METADATA_JSON


def run_feature_engineering() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    bank_df = load_bank_sample()
    complaints_df = load_complaints_sample()

    X_bank, y_bank = split_bank_features_target(bank_df)
    complaints_clean = clean_complaints_dataframe(complaints_df)

    X_bank.to_csv(BANK_FEATURES_CSV, index=False)
    y_bank.to_frame(name=TARGET_COLUMN).to_csv(BANK_TARGET_CSV, index=False)
    complaints_clean.to_csv(COMPLAINTS_CLEAN_CSV, index=False)

    metadata_path = write_feature_metadata()

    print(f"Saved bank features: {BANK_FEATURES_CSV} rows={len(X_bank)} cols={X_bank.shape[1]}")
    print(f"Saved bank target: {BANK_TARGET_CSV} rows={len(y_bank)}")
    print(
        f"Saved clean complaints: {COMPLAINTS_CLEAN_CSV} "
        f"rows={len(complaints_clean)}"
    )
    print(f"Saved feature metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reusable project features.")
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    run_feature_engineering()