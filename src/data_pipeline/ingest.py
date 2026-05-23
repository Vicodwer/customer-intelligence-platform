from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import certifi
import pandas as pd
import requests
import urllib3


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLES_DIR = DATA_DIR / "samples"

BANK_ZIP_URLS = [
    "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    "https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
]

# Hugging Face-hosted complaint sample.
# This avoids the broken Socrata URL and avoids downloading the huge CFPB zip.
HF_COMPLAINT_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_COMPLAINT_DATASET = "aciborowska/customers-complaints"

BANK_RAW_CSV = RAW_DIR / "bank-additional-full.csv"
BANK_SAMPLE_CSV = SAMPLES_DIR / "bank_marketing_sample.csv"

COMPLAINT_RAW_JSON = RAW_DIR / "complaints_sample_hf.json"
COMPLAINT_SAMPLE_CSV = SAMPLES_DIR / "cfpb_complaints_sample.csv"

DATA_README = DATA_DIR / "README.md"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def download_file_with_fallback(
    urls: list[str],
    output_path: Path,
    force: bool = False,
) -> None:
    if output_path.exists() and not force and output_path.stat().st_size > 1000:
        print(f"Using existing download: {output_path}")
        return

    headers = {"User-Agent": "customer-intelligence-platform/0.1"}
    last_error: Exception | None = None

    for url in urls:
        try:
            print(f"Trying download: {url}")

            response = requests.get(
                url,
                headers=headers,
                timeout=90,
            )
            response.raise_for_status()

            output_path.write_bytes(response.content)

            if output_path.stat().st_size < 1000:
                raise ValueError(
                    f"Downloaded file is too small: {output_path.stat().st_size} bytes"
                )

            print(f"Downloaded successfully: {output_path}")
            return

        except Exception as exc:
            print(f"Download failed for {url}: {exc}")
            last_error = exc

    raise RuntimeError(f"All download URLs failed. Last error: {last_error}")


def extract_bank_csv_from_zip(downloaded_zip: Path) -> Path:
    """
    Supports both UCI formats:

    1. Direct bank-additional.zip containing:
       bank-additional/bank-additional-full.csv

    2. New UCI bundle bank+marketing.zip containing:
       bank-additional.zip
    """
    with zipfile.ZipFile(downloaded_zip, "r") as outer_zip:
        outer_names = outer_zip.namelist()

        direct_csv_candidates = [
            name
            for name in outer_names
            if name.endswith("bank-additional-full.csv")
        ]

        if direct_csv_candidates:
            csv_member = direct_csv_candidates[0]
            outer_zip.extract(csv_member, RAW_DIR)
            return RAW_DIR / csv_member

        nested_zip_candidates = [
            name
            for name in outer_names
            if name.endswith("bank-additional.zip")
        ]

        if not nested_zip_candidates:
            raise FileNotFoundError(
                "Could not find bank-additional-full.csv or bank-additional.zip "
                f"inside {downloaded_zip}. Found: {outer_names}"
            )

        nested_zip_name = nested_zip_candidates[0]
        nested_zip_path = RAW_DIR / "bank-additional.zip"
        nested_zip_path.write_bytes(outer_zip.read(nested_zip_name))

    with zipfile.ZipFile(nested_zip_path, "r") as inner_zip:
        inner_names = inner_zip.namelist()

        csv_candidates = [
            name
            for name in inner_names
            if name.endswith("bank-additional-full.csv")
        ]

        if not csv_candidates:
            raise FileNotFoundError(
                f"Could not find bank-additional-full.csv inside nested zip. "
                f"Found: {inner_names}"
            )

        csv_member = csv_candidates[0]
        inner_zip.extract(csv_member, RAW_DIR)
        return RAW_DIR / csv_member


def download_bank_marketing(
    force: bool = False,
    sample_size: int = 5000,
) -> pd.DataFrame:
    ensure_dirs()

    bank_download_zip = RAW_DIR / "bank-marketing-download.zip"

    download_file_with_fallback(
        urls=BANK_ZIP_URLS,
        output_path=bank_download_zip,
        force=force,
    )

    extracted_csv = extract_bank_csv_from_zip(bank_download_zip)

    if not extracted_csv.exists():
        raise FileNotFoundError(
            f"Expected file not found after unzip: {extracted_csv}"
        )

    df = pd.read_csv(extracted_csv, sep=";")
    df.to_csv(BANK_RAW_CSV, index=False)

    sample_n = min(sample_size, len(df))
    sample_df = df.sample(n=sample_n, random_state=42)
    sample_df.to_csv(BANK_SAMPLE_CSV, index=False)

    print(f"Saved raw bank data: {BANK_RAW_CSV} rows={len(df)}")
    print(f"Saved bank sample: {BANK_SAMPLE_CSV} rows={len(sample_df)}")

    return df


def normalize_hf_complaint_record(record: dict) -> dict:
    """
    Normalize Hugging Face dataset columns into the column names
    our project expects for the RAG lane.
    """
    return {
        "complaint_id": record.get("Complaint_ID"),
        "date_received": record.get("Date_received"),
        "product": record.get("Product"),
        "sub_product": record.get("Sub_product"),
        "issue": record.get("Issue"),
        "sub_issue": record.get("Sub_issue"),
        "company": record.get("Company"),
        "state": record.get("State"),
        "consumer_complaint_narrative": record.get("Consumer_complaint_narrative"),
        "company_response": record.get("Company response to consumer"),
        "timely": record.get("Timely_response?"),
        "submitted_via": record.get("Submitted_via"),
    }


def fetch_complaints_from_huggingface(
    sample_size: int = 5000,
    page_size: int = 100,
    max_retries: int = 3,
    verify_ssl: bool = True,
    min_narrative_chars: int = 30,
) -> pd.DataFrame:
    ensure_dirs()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "customer-intelligence-platform/0.1",
            "Accept": "application/json",
        }
    )

    if verify_ssl:
        session.verify = certifi.where()
    else:
        session.verify = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        print(
            "WARNING: SSL verification is disabled only for downloading public complaint sample data."
        )

    all_records: list[dict] = []
    raw_pages: list[dict] = []

    # Fetch extra rows because some narratives are too short and will be filtered out.
    max_fetch_rows = sample_size + 1000

    print(
        f"Fetching complaint sample from Hugging Face dataset={HF_COMPLAINT_DATASET} "
        f"target_size={sample_size}, page_size={page_size}, "
        f"min_narrative_chars={min_narrative_chars}"
    )

    for offset in range(0, max_fetch_rows, page_size):
        current_size = min(page_size, max_fetch_rows - offset)

        params = {
            "dataset": HF_COMPLAINT_DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": current_size,
        }

        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                print(
                    f"Fetching HF complaint page offset={offset}, "
                    f"size={current_size}, attempt={attempt}"
                )

                response = session.get(
                    HF_COMPLAINT_ROWS_URL,
                    params=params,
                    timeout=(10, 120),
                )
                response.raise_for_status()

                payload = response.json()
                raw_pages.append(payload)

                rows = payload.get("rows", [])

                if not rows:
                    print(f"No more complaint records returned at offset={offset}")
                    break

                for item in rows:
                    row = item.get("row", {})
                    normalized = normalize_hf_complaint_record(row)

                    narrative = str(
                        normalized.get("consumer_complaint_narrative") or ""
                    ).strip()

                    if len(narrative) < min_narrative_chars:
                        continue

                    normalized["consumer_complaint_narrative"] = narrative
                    all_records.append(normalized)

                break

            except Exception as exc:
                last_error = exc
                wait_seconds = attempt * 3

                print(
                    f"HF complaint page failed offset={offset}, attempt={attempt}. "
                    f"Retrying in {wait_seconds}s. Error: {exc}"
                )

                time.sleep(wait_seconds)

        else:
            raise RuntimeError(
                f"Failed to fetch HF complaint page at offset={offset} "
                f"after {max_retries} attempts. Last error: {last_error}"
            )

        if len(all_records) >= sample_size:
            break

    with COMPLAINT_RAW_JSON.open("w", encoding="utf-8") as f:
        json.dump(raw_pages, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(all_records)

    if df.empty:
        raise ValueError("Complaint sample ingestion returned zero records.")

    df = df.dropna(subset=["consumer_complaint_narrative"])
    df = df.drop_duplicates(subset=["complaint_id"])

    df["consumer_complaint_narrative"] = (
        df["consumer_complaint_narrative"].astype(str).str.strip()
    )

    df = df[df["consumer_complaint_narrative"].str.len() >= min_narrative_chars]
    df = df.head(sample_size)

    if len(df) < sample_size:
        raise ValueError(
            f"Only collected {len(df)} valid complaint rows after filtering. "
            f"Requested {sample_size}. Try increasing max_fetch_rows."
        )

    df.to_csv(COMPLAINT_SAMPLE_CSV, index=False)

    print(f"Saved complaint sample: {COMPLAINT_SAMPLE_CSV} rows={len(df)}")

    return df


def write_data_readme() -> None:
    content = """# Data

This project uses two public datasets / public dataset samples.

## ML lane: UCI Bank Marketing

Purpose: predict whether a contacted customer subscribes to a term deposit.

Local files:
- raw: `data/raw/bank-additional-full.csv`
- sample: `data/samples/bank_marketing_sample.csv`

Target column:
- `y`

## LLM/RAG lane: Consumer complaint narratives

Purpose: complaint intelligence over public complaint narratives with cited evidence.

Local files:
- raw API JSON sample: `data/raw/complaints_sample_hf.json`
- sample CSV: `data/samples/cfpb_complaints_sample.csv`

Important handling rule:
- Do not commit full datasets.
- Do not show full raw complaint narratives in demo screenshots unless cleaned/redacted.
- Keep only small samples in Git.
"""

    DATA_README.write_text(content, encoding="utf-8")
    print(f"Wrote {DATA_README}")


def run_ingestion(
    bank_sample_size: int,
    complaints_sample_size: int,
    force: bool,
    allow_insecure_downloads: bool,
) -> None:
    bank_df = download_bank_marketing(
        force=force,
        sample_size=bank_sample_size,
    )

    complaints_df = fetch_complaints_from_huggingface(
        sample_size=complaints_sample_size,
        verify_ssl=not allow_insecure_downloads,
    )

    write_data_readme()

    print("\nIngestion complete.")
    print(f"Bank rows: {len(bank_df)}")
    print(f"Complaint rows: {len(complaints_df)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and sample project datasets."
    )

    parser.add_argument(
        "--bank-sample-size",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--complaints-sample-size",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download bank data even if it already exists.",
    )

    parser.add_argument(
        "--allow-insecure-downloads",
        action="store_true",
        help=(
            "Disable SSL verification only for public dataset downloads. "
            "Use only if local certificate verification fails."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_ingestion(
        bank_sample_size=args.bank_sample_size,
        complaints_sample_size=args.complaints_sample_size,
        force=args.force,
        allow_insecure_downloads=args.allow_insecure_downloads,
    )