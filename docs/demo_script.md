# Demo Script

## Demo goal

Show a production-minded Customer Intelligence Platform with:

1. Campaign conversion prediction.
2. Complaint intelligence with cited evidence.
3. Integrated customer intelligence.
4. CI, Docker, monitoring, and reproducible scripts.

Target duration: 5–8 minutes.

---

## 1. Repo and CI overview

Show the GitHub repository.

Mention:

- Script-first pipeline.
- GitHub Actions is passing.
- Tests run automatically.
- Reports are generated during CI.

Show:

```text
.github/workflows/ci.yml

Explain that CI runs:

ingest -> validate -> features -> train -> evaluate -> RAG index -> RAG eval -> tests
2. Data and validation

Show:

src/data_pipeline/ingest.py
src/data_pipeline/validate.py
src/data_pipeline/features.py

Mention:

Bank Marketing data is used for ML.
Complaint narratives are used for RAG.
Validation checks schema, missing values, duplicates, and business rules.
Feature engineering drops duration to avoid leakage.

Run or show command:

python -m src.data_pipeline.validate

Expected:

Validation passed.
3. ML lane

Show:

src/training/train.py
src/training/evaluate.py
docs/model_report.md
docs/promotion_report.md

Mention:

Logistic regression baseline.
Random forest candidate.
Promotion gate.
Deliberately worse model blocked.
Champion artifact generated.

Show generated artifact:

models/champion_model.joblib
4. RAG lane

Show:

src/rag/build_index.py
src/rag/retrieve.py
src/rag/answer.py
src/rag/rag_eval.py
docs/rag_report.md

Mention:

Complaint narratives are cleaned.
TF-IDF retrieval is used for MVP.
Answers cite complaint IDs.
Weak evidence triggers refusal.

Run:

python -m src.rag.answer --question "What complaints mention credit card billing disputes and account problems?" --top-k 3 --min-score 0.01

Show:

refused: false
evidence_ids: [...]
5. API demo

Start API:

uvicorn src.serving.serve:app --reload

Open:

http://127.0.0.1:8000/docs
Health
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method GET

Expected:

status: ok
model_loaded: True
Predict

Run /predict with the sample customer payload.

Show:

probability
decision
conversion_band
model_version
Ask complaints

Run /ask-complaints.

Show:

answer
evidence_ids
evidence_sufficiency_note
prompt_version
Customer intel

Run /customer-intel.

Show:

conversion
complaint_intelligence
integration_note
Batch score

Run /batch-score.

Show:

scored_count
band_counts
predictions
6. Monitoring

Show:

monitoring/ml_drift.py
monitoring/rag_monitor.py
docs/monitoring_report.md

Mention:

ML drift report includes feature drift scores and retraining recommendation.
RAG monitoring includes retrieval hit rate, refusal rate, score, latency, and estimated answer tokens.

Run:

python -m monitoring.ml_drift
python -m monitoring.rag_monitor
7. Docker demo

Show:

Dockerfile
docker-compose.yml
.dockerignore

Build:

docker build -t customer-intelligence-platform:latest .

Run on port 8001 if 8000 is busy:

docker run --rm -p 8001:8000 customer-intelligence-platform:latest

Test:

Invoke-RestMethod -Uri "http://127.0.0.1:8001/health" -Method GET

Expected:

status: ok
model_loaded: True
8. Final closing

Mention current limitations:

TF-IDF instead of dense embeddings.
Deterministic answer generation instead of full LLM generation.
Docker startup retrains for reproducibility, but production would use prebuilt artifacts.
Cloud deployment is the next step.

End with:

This MVP demonstrates a complete script-first customer intelligence platform with ML scoring, grounded complaint intelligence, monitoring, CI, Docker, and reproducible documentation.

---

## 18.2 Create `docs/submission_checklist.md`

```markdown
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