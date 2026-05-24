@'
# Submission Checklist

## Core platform

- [x] GitHub repository exists.
- [x] Fresh-clone setup documented.
- [x] Script-first workflow.
- [x] Data ingestion script.
- [x] Data validation script.
- [x] Feature engineering script.
- [x] ML training script.
- [x] Model evaluation and promotion gate.
- [x] RAG index build script.
- [x] RAG retrieval script.
- [x] RAG answer script.
- [x] RAG evaluation script.
- [x] FastAPI serving layer.

## API endpoints

- [x] `GET /health`
- [x] `POST /predict`
- [x] `POST /batch-score`
- [x] `POST /ask-complaints`
- [x] `POST /customer-intel`

## ML lane

- [x] Baseline model.
- [x] Candidate model.
- [x] Champion model.
- [x] Promotion gate.
- [x] Deliberately worse model blocked.
- [x] Model report.

## RAG lane

- [x] Complaint text cleaning.
- [x] Retrieval index.
- [x] Cited evidence IDs.
- [x] Refusal when evidence is weak.
- [x] 10-question RAG evaluation.
- [x] RAG report.

## Monitoring

- [x] ML drift report.
- [x] RAG monitoring metrics.
- [x] Monitoring markdown report.

## CI/CD and deployment evidence

- [x] GitHub Actions workflow.
- [x] CI passing.
- [x] Dockerfile.
- [x] Docker Compose.
- [x] `.dockerignore`.
- [x] Docker image builds.
- [x] Docker container runs API.
- [x] Docker `/health` returns ok.

## Documentation

- [x] README.
- [x] Architecture document.
- [x] Decision log.
- [x] Demo script.
- [x] Submission checklist.
- [x] Reflection.

## Reports

- [x] `docs/model_report.md`
- [x] `docs/promotion_report.md`
- [x] `docs/rag_report.md`
- [x] `docs/monitoring_report.md`

## Known stretch items not yet complete

- [ ] Azure cloud deployment.
- [ ] Dense embeddings with FAISS or Chroma.
- [ ] True LLM answer generation.
- [ ] SHAP explainability.
- [ ] Auth/rate limiting.
- [ ] Production artifact registry.
'@ | Set-Content docs\submission_checklist.md -Encoding UTF8