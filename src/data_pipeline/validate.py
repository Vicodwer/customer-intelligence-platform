from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
PROCESSED_DIR = DATA_DIR / "processed"

BANK_SAMPLE_CSV = SAMPLES_DIR / "bank_marketing_sample.csv"
COMPLAINT_SAMPLE_CSV = SAMPLES_DIR / "cfpb_complaints_sample.csv"

VALIDATION_REPORT_PATH = PROCESSED_DIR / "validation_report.json"


BANK_REQUIRED_COLUMNS = {
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "y",
}

COMPLAINT_REQUIRED_COLUMNS = {
    "complaint_id",
    "date_received",
    "product",
    "issue",
    "company",
    "consumer_complaint_narrative",
}


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    details: str


@dataclass
class DatasetValidationResult:
    dataset: str
    row_count: int
    checks: list[ValidationCheck]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def _missing_columns(df: pd.DataFrame, required_columns: set[str]) -> set[str]:
    return required_columns - set(df.columns)


def _check(name: str, passed: bool, details: str) -> ValidationCheck:
    return ValidationCheck(name=name, passed=bool(passed), details=details)


def validate_bank_marketing(path: Path = BANK_SAMPLE_CSV) -> DatasetValidationResult:
    if not path.exists():
        raise FileNotFoundError(f"Bank sample not found: {path}")

    df = pd.read_csv(path)
    checks: list[ValidationCheck] = []

    missing_cols = _missing_columns(df, BANK_REQUIRED_COLUMNS)
    checks.append(
        _check(
            name="bank_schema_required_columns",
            passed=not missing_cols,
            details=f"Missing columns: {sorted(missing_cols)}",
        )
    )

    checks.append(
        _check(
            name="bank_non_empty",
            passed=len(df) > 0,
            details=f"Rows: {len(df)}",
        )
    )

    checks.append(
        _check(
            name="bank_no_duplicate_rows",
            passed=df.duplicated().sum() == 0,
            details=f"Duplicate rows: {int(df.duplicated().sum())}",
        )
    )

    required_nulls = df[list(BANK_REQUIRED_COLUMNS & set(df.columns))].isna().sum().sum()
    checks.append(
        _check(
            name="bank_no_missing_required_values",
            passed=required_nulls == 0,
            details=f"Missing required values: {int(required_nulls)}",
        )
    )

    # Business rule 1: target must be binary yes/no.
    if "y" in df.columns:
        invalid_targets = sorted(set(df["y"].dropna().unique()) - {"yes", "no"})
        checks.append(
            _check(
                name="bank_target_is_yes_no",
                passed=not invalid_targets,
                details=f"Invalid target values: {invalid_targets}",
            )
        )

    # Business rule 2: customer age should be plausible.
    if "age" in df.columns:
        bad_age_count = int((~df["age"].between(16, 100)).sum())
        checks.append(
            _check(
                name="bank_age_plausible_range",
                passed=bad_age_count == 0,
                details=f"Rows with age outside 16-100: {bad_age_count}",
            )
        )

    # Business rule 3: call duration cannot be negative.
    if "duration" in df.columns:
        bad_duration_count = int((df["duration"] < 0).sum())
        checks.append(
            _check(
                name="bank_duration_non_negative",
                passed=bad_duration_count == 0,
                details=f"Rows with negative duration: {bad_duration_count}",
            )
        )

    # Business rule 4: campaign contact count should be at least 1.
    if "campaign" in df.columns:
        bad_campaign_count = int((df["campaign"] < 1).sum())
        checks.append(
            _check(
                name="bank_campaign_at_least_one",
                passed=bad_campaign_count == 0,
                details=f"Rows with campaign < 1: {bad_campaign_count}",
            )
        )

    # Business rule 5: previous contacts cannot be negative.
    if "previous" in df.columns:
        bad_previous_count = int((df["previous"] < 0).sum())
        checks.append(
            _check(
                name="bank_previous_non_negative",
                passed=bad_previous_count == 0,
                details=f"Rows with previous < 0: {bad_previous_count}",
            )
        )

    # Business rule 6: pdays is 999 for not previously contacted, otherwise non-negative.
    if "pdays" in df.columns:
        bad_pdays_count = int((df["pdays"] < 0).sum())
        checks.append(
            _check(
                name="bank_pdays_non_negative_or_999",
                passed=bad_pdays_count == 0,
                details=f"Rows with pdays < 0: {bad_pdays_count}",
            )
        )

    return DatasetValidationResult(
        dataset="bank_marketing",
        row_count=len(df),
        checks=checks,
    )


def validate_complaints(path: Path = COMPLAINT_SAMPLE_CSV) -> DatasetValidationResult:
    if not path.exists():
        raise FileNotFoundError(f"Complaint sample not found: {path}")

    df = pd.read_csv(path)
    checks: list[ValidationCheck] = []

    missing_cols = _missing_columns(df, COMPLAINT_REQUIRED_COLUMNS)
    checks.append(
        _check(
            name="complaints_schema_required_columns",
            passed=not missing_cols,
            details=f"Missing columns: {sorted(missing_cols)}",
        )
    )

    checks.append(
        _check(
            name="complaints_non_empty",
            passed=len(df) > 0,
            details=f"Rows: {len(df)}",
        )
    )

    if "complaint_id" in df.columns:
        duplicate_ids = int(df["complaint_id"].duplicated().sum())
        checks.append(
            _check(
                name="complaints_no_duplicate_ids",
                passed=duplicate_ids == 0,
                details=f"Duplicate complaint_id values: {duplicate_ids}",
            )
        )

    required_present = list(COMPLAINT_REQUIRED_COLUMNS & set(df.columns))
    required_nulls = df[required_present].isna().sum().sum()
    checks.append(
        _check(
            name="complaints_no_missing_required_values",
            passed=required_nulls == 0,
            details=f"Missing required values: {int(required_nulls)}",
        )
    )

    # Business rule 7: complaint narrative should be long enough to retrieve useful evidence.
    if "consumer_complaint_narrative" in df.columns:
        narrative_lengths = df["consumer_complaint_narrative"].fillna("").str.len()
        short_narratives = int((narrative_lengths < 30).sum())
        checks.append(
            _check(
                name="complaints_narrative_min_length",
                passed=short_narratives == 0,
                details=f"Narratives shorter than 30 characters: {short_narratives}",
            )
        )

    # Business rule 8: product should not be blank.
    if "product" in df.columns:
        blank_products = int((df["product"].fillna("").str.strip() == "").sum())
        checks.append(
            _check(
                name="complaints_product_not_blank",
                passed=blank_products == 0,
                details=f"Blank product rows: {blank_products}",
            )
        )

    # Business rule 9: issue should not be blank.
    if "issue" in df.columns:
        blank_issues = int((df["issue"].fillna("").str.strip() == "").sum())
        checks.append(
            _check(
                name="complaints_issue_not_blank",
                passed=blank_issues == 0,
                details=f"Blank issue rows: {blank_issues}",
            )
        )

    # Business rule 10: company should not be blank.
    if "company" in df.columns:
        blank_companies = int((df["company"].fillna("").str.strip() == "").sum())
        checks.append(
            _check(
                name="complaints_company_not_blank",
                passed=blank_companies == 0,
                details=f"Blank company rows: {blank_companies}",
            )
        )

    return DatasetValidationResult(
        dataset="complaints",
        row_count=len(df),
        checks=checks,
    )


def write_validation_report(results: list[DatasetValidationResult]) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "passed": all(result.passed for result in results),
        "datasets": [
            {
                "dataset": result.dataset,
                "row_count": result.row_count,
                "passed": result.passed,
                "checks": [asdict(check) for check in result.checks],
            }
            for result in results
        ],
    }

    VALIDATION_REPORT_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return VALIDATION_REPORT_PATH


def run_validation() -> bool:
    results = [
        validate_bank_marketing(),
        validate_complaints(),
    ]

    report_path = write_validation_report(results)

    print(f"Validation report written to: {report_path}")

    for result in results:
        status = "PASSED" if result.passed else "FAILED"
        print(f"\n[{status}] {result.dataset} rows={result.row_count}")

        for check in result.checks:
            check_status = "PASS" if check.passed else "FAIL"
            print(f"  - {check_status}: {check.name} | {check.details}")

    all_passed = all(result.passed for result in results)

    if not all_passed:
        print("\nValidation failed.")
        return False

    print("\nValidation passed.")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate project sample datasets.")
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    ok = run_validation()
    raise SystemExit(0 if ok else 1)