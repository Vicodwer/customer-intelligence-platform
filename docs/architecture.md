# Architecture

## Customer Intelligence Platform

This project is a production-minded Customer Intelligence Platform with two intelligence lanes:

1. **ML lane**: predicts campaign conversion for bank marketing customers.
2. **RAG lane**: answers complaint-intelligence questions with cited complaint evidence IDs.

The platform is exposed through a FastAPI backend API and is supported by CI, Docker, monitoring reports, and Azure deployment evidence.

---

## Architecture diagram artifacts

The architecture diagram is stored in two formats:

```text
docs/architecture.drawio
docs/architecture.png
architecture.drawio is the diagrams-as-code source file. It is XML-based and can be edited in draw.io / diagrams.net.
architecture.png is the exported image for README, reports, and demo slides.
High-level flow
Public Data Sources
  -> Data Ingestion
  -> Data Validation
  -> Feature Engineering
  -> ML lane
  -> RAG lane
  -> FastAPI Serving Layer
  -> API Endpoints

The project is script-first. Core tasks run through Python modules instead of notebooks only.

ML lane

The ML lane predicts whether a customer is likely to subscribe to a term deposit.

Pipeline:

src/data_pipeline/ingest.py
  -> src/data_pipeline/validate.py
  -> src/data_pipeline/features.py
  -> src/training/train.py
  -> src/training/evaluate.py
  -> models/champion_model.joblib
  -> /predict and /batch-score

Models used:

Dummy baseline
Logistic regression baseline
Random forest candidate
Champion model selected by promotion gate

The promotion gate checks PR-AUC, F1, and ROC-AUC before promoting a candidate model. A deliberately worse dummy model is also evaluated and blocked to demonstrate model governance.

The duration feature is excluded because it would cause train-serving leakage.

RAG lane

The RAG lane answers complaint intelligence questions using retrieved complaint evidence.

Pipeline:

src/data_pipeline/features.py
  -> src/rag/build_index.py
  -> src/rag/retrieve.py
  -> src/rag/answer.py
  -> /ask-complaints and /customer-intel

The MVP retrieval layer uses:

cleaned complaint narratives
TF-IDF vectorization
nearest-neighbor retrieval
evidence previews
cited complaint IDs
refusal when evidence is weak

This makes the RAG response grounded and testable.

FastAPI service

The serving layer is implemented in:

src/serving/serve.py

Available endpoints:

Endpoint	Purpose
GET /health	Checks API and model readiness
POST /predict	Scores one customer
POST /batch-score	Scores multiple customers
POST /ask-complaints	Answers complaint intelligence questions
POST /customer-intel	Combines ML conversion output with complaint intelligence

The project does not include a custom frontend. It is used through FastAPI Swagger UI at /docs, PowerShell, Postman, or a future frontend.

CI/CD

GitHub Actions runs the full validation pipeline:

install dependencies
ingest sample data
validate data
build features
train model
run promotion gate
build RAG index
run RAG evaluation
run tests
upload generated reports

Workflow file:

.github/workflows/ci.yml
Docker and Azure deployment

The project is Dockerized with:

Dockerfile
docker-compose.yml
.dockerignore

The Docker image was built locally and pushed to Azure Container Registry.

Azure deployment used:

Azure Container Registry
Azure Container Apps
Region: Southeast Asia

The deployed API successfully served /predict over HTTPS.

Monitoring

Monitoring scripts are stored in:

monitoring/ml_drift.py
monitoring/rag_monitor.py

Generated monitoring output includes:

data/processed/ml_drift_report.json
data/processed/rag_monitoring_metrics.json
docs/monitoring_report.md

The ML monitoring report checks drift scores.
The RAG monitoring report tracks retrieval hit rate, refusal rate, top score, latency, and estimated answer tokens.

Generated reports

Important reports:

docs/model_report.md
docs/promotion_report.md
docs/rag_report.md
docs/monitoring_report.md
Known limitations
RAG uses TF-IDF instead of dense embeddings.
The complaint answer generator is deterministic, not a full LLM.
Docker startup retrains and rebuilds artifacts for reproducibility, which is slower than a production image with prebuilt artifacts.
No custom frontend is included.
No SHAP dashboard or production auth layer is included yet.
Future improvements
Add dense embeddings with FAISS or Chroma.
Add strict LLM answer synthesis with citations.
Add SHAP or feature importance dashboard.
Add authentication and rate limiting.
Add a lightweight web frontend.
Split Docker build-time artifacts from runtime startup.
'@ | Set-Content docs\architecture.md -Encoding UTF8