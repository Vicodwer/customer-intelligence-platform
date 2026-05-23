from pathlib import Path

import pandas as pd


def test_bank_sample_exists_after_ingestion():
    path = Path("data/samples/bank_marketing_sample.csv")
    assert path.exists(), "Run: python -m src.data_pipeline.ingest"

    df = pd.read_csv(path)
    assert not df.empty
    assert "y" in df.columns
    assert len(df) == 5000


def test_complaints_sample_exists_after_ingestion():
    path = Path("data/samples/cfpb_complaints_sample.csv")
    assert path.exists(), "Run: python -m src.data_pipeline.ingest"

    df = pd.read_csv(path)
    assert not df.empty
    assert len(df) == 5000

    required_columns = {
        "complaint_id",
        "product",
        "issue",
        "company",
        "consumer_complaint_narrative",
    }

    missing = required_columns - set(df.columns)
    assert not missing, f"Missing complaint columns: {missing}"


def test_data_readme_exists():
    path = Path("data/README.md")
    assert path.exists()