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
