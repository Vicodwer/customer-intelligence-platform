# Architecture

## Customer Intelligence Platform

This project contains two main intelligence lanes:

1. **ML lane** for campaign conversion prediction.
2. **RAG lane** for complaint intelligence with cited evidence IDs.

Both are exposed through a FastAPI service.

---

## High-level architecture

```mermaid
flowchart TD
    A[Public Data Sources] --> B[Data Ingestion Scripts]

    B --> C1[Bank Marketing Sample]
    B --> C2[Complaint Narrative Sample]

    C1 --> D1[Data Validation]
    C2 --> D1

    D1 --> E1[Feature Engineering]
    E1 --> F1[ML Training]
    F1 --> G1[Promotion Gate]
    G1 --> H1[Champion Model Artifact]

    E1 --> F2[Complaint Clean Text]
    F2 --> G2[RAG Index Build]
    G2 --> H2[TF-IDF Index + Docstore]

    H1 --> I[FastAPI Service]
    H2 --> I

    I --> J1[/health]
    I --> J2[/predict]
    I --> J3[/batch-score]
    I --> J4[/ask-complaints]
    I --> J5[/customer-intel]

    K[GitHub Actions CI] --> B
    K --> D1
    K --> E1
    K --> F1
    K --> G1
    K --> G2
    K --> L[Tests + Reports]

    M[Docker Image] --> I
ML lane
Purpose

Predict whether a contacted customer is likely to subscribe to a term deposit.

Flow
ingest.py
  -> validate.py
  -> features.py
  -> train.py
  -> evaluate.py
  -> champion_model.joblib
  -> /predict and /batch-score
Model strategy

The project trains:

Dummy baseline
Logistic regression baseline
Random forest candidate
Champion model selected by promotion gate

The promotion gate compares candidate vs baseline using:

PR-AUC improvement
F1 drop limit
ROC-AUC drop limit

A deliberately worse dummy model is evaluated and blocked to demonstrate gate behavior.

RAG lane
Purpose

Answer complaint intelligence questions using retrieved evidence with complaint IDs.

Flow
ingest.py
  -> validate.py
  -> features.py
  -> build_index.py
  -> retrieve.py
  -> answer.py
  -> /ask-complaints and /customer-intel
Retrieval strategy

The MVP uses:

Cleaned complaint narratives
TF-IDF vectorizer
Nearest-neighbor search
Evidence previews
Complaint ID citations
Refusal when similarity is below threshold
API layer

FastAPI exposes:

Endpoint	Purpose
GET /health	Check app and model readiness
POST /predict	Score one customer
POST /batch-score	Score multiple customers
POST /ask-complaints	Ask complaint intelligence question
POST /customer-intel	Combine conversion prediction with complaint intelligence
CI/CD

GitHub Actions runs:

install dependencies
ingest sample data
validate data
build features
train baseline model
run promotion gate
build RAG index
run RAG evaluation
run tests
upload reports
Docker runtime

The Docker image runs the full startup pipeline before launching FastAPI:

ingest
validate
features
train
evaluate
build RAG index
start uvicorn

This makes the container self-contained for demo usage.

Generated artifacts

Most generated artifacts are ignored by Git:

data/raw/
data/processed/
models/
artifacts/

Important generated reports include:

docs/model_report.md
docs/promotion_report.md
docs/rag_report.md
docs/monitoring_report.md
Known architecture limitations
RAG uses TF-IDF instead of dense embeddings.
No managed vector database yet.
No cloud-hosted production environment yet.
No async job queue for batch scoring.
Docker startup performs full training/indexing, which is useful for reproducibility but slow for production.