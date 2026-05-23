from pathlib import Path

import pandas as pd

from src.data_pipeline.features import (
    BANK_FEATURE_COLUMNS,
    BANK_FEATURES_CSV,
    BANK_TARGET_CSV,
    COMPLAINTS_CLEAN_CSV,
    clean_text_for_rag,
    run_feature_engineering,
    split_bank_features_target,
)


def test_split_bank_features_target_excludes_duration():
    df = pd.read_csv("data/samples/bank_marketing_sample.csv")

    X, y = split_bank_features_target(df)

    assert "duration" not in X.columns
    assert list(X.columns) == BANK_FEATURE_COLUMNS
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y)


def test_clean_text_for_rag_redacts_email_and_phone():
    text = "My email is person@example.com and phone is +91 98765 43210."
    cleaned = clean_text_for_rag(text)

    assert "person@example.com" not in cleaned
    assert "+91 98765 43210" not in cleaned
    assert "[EMAIL]" in cleaned
    assert "[PHONE]" in cleaned


def test_run_feature_engineering_writes_outputs():
    run_feature_engineering()

    assert Path(BANK_FEATURES_CSV).exists()
    assert Path(BANK_TARGET_CSV).exists()
    assert Path(COMPLAINTS_CLEAN_CSV).exists()

    bank_features = pd.read_csv(BANK_FEATURES_CSV)
    bank_target = pd.read_csv(BANK_TARGET_CSV)
    complaints = pd.read_csv(COMPLAINTS_CLEAN_CSV)

    assert len(bank_features) == 5000
    assert len(bank_target) == 5000
    assert len(complaints) >= 4990
    assert "clean_narrative" in complaints.columns
    assert complaints["clean_narrative"].str.len().min() >= 30