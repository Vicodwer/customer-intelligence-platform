# Customer Intelligence Platform

A production-minded mini-project that combines two customer intelligence services behind one reproducible pipeline:

1. **ML service**: predicts whether a contacted customer is likely to subscribe to a term deposit.
2. **RAG service**: answers complaint-intelligence questions over complaint narratives with cited evidence IDs.

Built with:

- FastAPI
- scikit-learn
- GitHub Actions
- local model artifacts
- TF-IDF retrieval for the MVP RAG layer
- scripted data, training, evaluation, and serving workflows

---

## Project status

Current MVP includes:

- Data ingestion
- Data validation
- Feature engineering
- Baseline ML model
- Improved candidate model
- Relative promotion gate
- FastAPI `/health`
- FastAPI `/predict`
- FastAPI `/ask-complaints`
- RAG retrieval with evidence IDs
- RAG refusal logic when evidence is weak
- 10-question RAG evaluation report
- GitHub Actions CI pipeline

---

## Repository structure

```text
customer-intelligence-platform/
data/
  raw/                 # ignored by Git
  processed/           # ignored by Git
  samples/             # small sample CSVs
src/
  data_pipeline/
    ingest.py
    validate.py
    features.py
  training/
    train.py
    evaluate.py
  serving/
    serve.py
  rag/
    build_index.py
    retrieve.py
    answer.py
    rag_eval.py
tests/
.github/workflows/
  ci.yml
pipelines/
app/
monitoring/
docs/
models/                # ignored by Git
artifacts/             # ignored by Git
```

---

## Data

### ML lane

Uses the UCI Bank Marketing dataset.

Target column:

```text
y
```

Target meaning:

- `yes`: customer subscribed to a term deposit
- `no`: customer did not subscribe

The `duration` feature is intentionally excluded because it is only known after the call ends and would create train-serving leakage.

### RAG lane

Uses a public complaint narrative sample for complaint intelligence.

Important handling rules:

- Full raw datasets are not committed.
- Generated raw and processed files are ignored.
- Complaint narratives should not be exposed in screenshots unless cleaned or redacted.
- RAG answers cite complaint IDs and include an evidence sufficiency note.

---

## Fresh clone setup

```powershell
git clone https://github.com/Vicodwer/customer-intelligence-platform.git
cd customer-intelligence-platform

python -m venv .venv
.\.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Run the full local pipeline

### 1. Ingest data

```powershell
python -m src.data_pipeline.ingest --bank-sample-size 5000 --complaints-sample-size 5000
```

### 2. Validate data

```powershell
python -m src.data_pipeline.validate
```

Expected result:

```text
Validation passed.
```

### 3. Build features

```powershell
python -m src.data_pipeline.features
```

Expected generated files:

```text
data/processed/bank_features.csv
data/processed/bank_target.csv
data/processed/complaints_clean.csv
data/processed/feature_metadata.json
```

### 4. Train baseline model

```powershell
python -m src.training.train
```

Expected generated files:

```text
models/baseline_model.joblib
models/model_metadata.json
data/processed/baseline_predictions.csv
docs/model_report.md
```

### 5. Run promotion gate

```powershell
python -m src.training.evaluate
```

Expected generated files:

```text
models/candidate_model.joblib
models/champion_model.joblib
models/promotion_report.json
docs/promotion_report.md
```

The promotion gate compares the logistic regression baseline with a random forest candidate and blocks a deliberately worse dummy model.

### 6. Build RAG index

```powershell
python -m src.rag.build_index --max-features 5000
```

Expected generated files:

```text
artifacts/rag/complaint_tfidf_index.joblib
artifacts/rag/complaint_docstore.csv
artifacts/rag/rag_index_metadata.json
```

### 7. Run RAG evaluation

```powershell
python -m src.rag.rag_eval --rebuild-index
```

Expected generated files:

```text
data/processed/rag_eval_results.json
docs/rag_report.md
```

### 8. Run tests

```powershell
pytest -q
```

---

## Run API

```powershell
uvicorn src.serving.serve:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

---

## API endpoints

### GET `/health`

Checks whether the API and model are ready.

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method GET
```

Example output:

```text
status        : ok
app_version   : 0.1.0
model_loaded  : True
model_version : champion-...
model_type    : random_forest_champion
model_path    : models\champion_model.joblib
```

---

### POST `/predict`

Predicts whether a customer is likely to convert.

```powershell
$body = @{
  customer = @{
    age = 35
    job = "admin."
    marital = "married"
    education = "university.degree"
    default = "no"
    housing = "yes"
    loan = "no"
    contact = "cellular"
    month = "may"
    day_of_week = "mon"
    campaign = 1
    pdays = 999
    previous = 0
    poutcome = "nonexistent"
    "emp.var.rate" = 1.1
    "cons.price.idx" = 93.994
    "cons.conf.idx" = -36.4
    euribor3m = 4.857
    "nr.employed" = 5191.0
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

Example output:

```text
prediction      : 0
probability     : 0.27807927421774736
threshold       : 0.5
decision        : unlikely_to_convert
conversion_band : low
model_version   : champion-...
model_type      : random_forest_champion
model_path      : models\champion_model.joblib
```

---

### POST `/ask-complaints`

Answers complaint intelligence questions with cited evidence IDs.

```powershell
$body = @{
  question = "What complaints mention credit card billing disputes and account problems?"
  top_k = 3
  min_score = 0.01
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/ask-complaints" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

Example output includes:

```text
refused                   : False
evidence_ids              : {2869476, 4952931, 7394845}
evidence_sufficiency_note : Sufficient evidence: 3 complaint records crossed the similarity threshold min_score=0.01.
prompt_version            : deterministic-complaint-answer-v1
index_version             : rag-index-...
```

The endpoint refuses when retrieval evidence is weak.

Example refusal test:

```powershell
$body = @{
  question = "zzzz qwerty unrelated nonsense"
  top_k = 3
  min_score = 0.8
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/ask-complaints" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

Expected behavior:

```text
refused      : True
evidence_ids : {}
```

---

## Model summary

The ML lane trains and compares:

- Dummy prior baseline
- Logistic regression baseline
- Random forest candidate
- Champion model selected through a promotion gate

The promotion gate checks:

- PR-AUC improvement
- F1 drop limit
- ROC-AUC drop limit

A deliberately worse dummy model is evaluated and blocked to demonstrate gate behavior.

Generated reports:

```text
docs/model_report.md
docs/promotion_report.md
```

---

## RAG summary

The current RAG MVP uses:

- Cleaned complaint narratives
- TF-IDF vectorization
- Nearest-neighbor retrieval
- Cited complaint IDs
- Evidence sufficiency note
- Refusal when similarity is weak
- 10-question RAG evaluation

Generated report:

```text
docs/rag_report.md
```

---

## CI/CD

GitHub Actions runs the full scripted pipeline:

- Install dependencies
- Ingest sample data
- Validate data
- Build features
- Train baseline model
- Run promotion gate
- Build RAG index
- Run RAG evaluation
- Run tests
- Upload generated reports

Workflow file:

```text
.github/workflows/ci.yml
```

Current CI status: passing.

---

## Important generated artifacts

These are generated locally or in CI and are mostly ignored by Git:

```text
data/raw/
data/processed/
models/
artifacts/
```

This keeps the repository lightweight and avoids committing raw datasets, trained binaries, or local artifacts.

---

## Known limitations

- RAG uses TF-IDF instead of dense embeddings or FAISS/Chroma.
- The complaint answer generator is deterministic and summary-based, not a generative LLM.
- No Docker image yet.
- No `/batch-score` endpoint yet.
- No drift monitoring report yet.
- No cloud deployment evidence yet.
- No `/customer-intel` integration endpoint yet.
- No SHAP or detailed governance audit yet.

---

## Next hardening steps

1. Add `/customer-intel` integration endpoint.
2. Add ML drift report.
3. Add RAG monitoring metrics.
4. Add Dockerfile and Docker Compose.
5. Add `/batch-score`.
6. Add deployment evidence.
7. Add architecture diagram.
8. Add demo recording.
9. Add `reflection.md`.