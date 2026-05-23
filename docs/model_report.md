# Model Report

## Model purpose

Predict whether a contacted customer will subscribe to a term deposit.

## Dataset

- Source: UCI Bank Marketing sample
- Training rows: 4000
- Test rows: 1000
- Target: `y`

## Feature decision

The `duration` column is intentionally excluded because it is only known after the call ends. Keeping it would create train-serving leakage.

## Baseline comparison

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Dummy prior | 0.5000 | 0.1120 | 0.0000 | 0.0000 | 0.0000 | 0.0995 |
| Logistic regression | 0.7608 | 0.3644 | 0.4138 | 0.3188 | 0.5893 | 0.1702 |

## Confusion matrix at threshold 0.5

{
  "tn": 747,
  "fp": 141,
  "fn": 46,
  "tp": 66
}

## Current decision

The logistic regression pipeline is saved as the current baseline serving candidate. A stricter promotion gate will be added in the next modelling step.
