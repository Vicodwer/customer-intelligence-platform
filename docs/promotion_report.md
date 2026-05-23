# Promotion Gate Report

## Purpose

Compare the current baseline model with an improved candidate model using a relative promotion gate.

## Promotion rules

- PR-AUC must improve by at least 0.01
- F1 may drop by no more than 0.02
- ROC-AUC may drop by no more than 0.01

## Metrics

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Baseline logistic regression | 0.7608 | 0.3644 | 0.4138 | 0.3188 | 0.5893 | 0.1702 |
| Candidate random forest | 0.7844 | 0.4289 | 0.4483 | 0.3652 | 0.5804 | 0.1420 |
| Deliberately worse dummy | 0.5000 | 0.1120 | 0.0000 | 0.0000 | 0.0000 | 0.1120 |

## Candidate promotion decision

Promoted: True

Reason:
- PR-AUC delta: 0.0645
- F1 delta: 0.0345
- ROC-AUC delta: 0.0236

Checks:
- PR-AUC improvement check: True
- F1 drop check: True
- ROC-AUC drop check: True

## Deliberately worse model block decision

Blocked: True

Reason:
- PR-AUC delta: -0.2524
- F1 delta: -0.4138
- ROC-AUC delta: -0.2608

## Champion model

Selected champion artifact:

models\champion_model.joblib
