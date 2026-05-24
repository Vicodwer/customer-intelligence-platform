# Decision Log

## 001 — Use script-first workflow instead of notebooks

Decision: all core project stages run through Python modules.

Reason:
- The brief requires final execution through scripts, not notebooks alone.
- CI can validate scripts automatically.
- Fresh-clone reproducibility is easier.

Status: accepted.

---

## 002 — Exclude `duration` from ML features

Decision: drop the `duration` column from model features.

Reason:
- Call duration is only known after a call ends.
- Including it would cause train-serving leakage.
- The serving API cannot know duration before a campaign contact.

Status: accepted.

---

## 003 — Use logistic regression baseline

Decision: train logistic regression as the first meaningful baseline.

Reason:
- Fast to train.
- Interpretable.
- Works well with one-hot encoded categorical variables.
- Provides a stable baseline for promotion gate comparison.

Status: accepted.

---

## 004 — Use random forest as candidate model

Decision: train a random forest candidate and compare it against the logistic baseline.

Reason:
- Captures non-linear patterns.
- Provides a stronger candidate without heavy dependencies.
- Works locally and in CI.

Status: accepted.

---

## 005 — Add promotion gate

Decision: promote candidate only if relative metric checks pass.

Promotion rules:
- PR-AUC improvement must meet minimum threshold.
- F1 drop must remain within allowed limit.
- ROC-AUC drop must remain within allowed limit.

Reason:
- Prevents blindly replacing a baseline.
- Demonstrates model governance.
- Blocks deliberately worse model.

Status: accepted.

---

## 006 — Use TF-IDF retrieval for MVP RAG

Decision: use TF-IDF plus nearest neighbors for the first RAG index.

Reason:
- No external embedding API needed.
- Fast in CI.
- Deterministic and easy to debug.
- Sufficient for MVP evidence retrieval.

Tradeoff:
- Less semantically powerful than dense embeddings.

Status: accepted.

---

## 007 — Use deterministic complaint answer generation

Decision: generate complaint intelligence summaries deterministically from retrieved evidence.

Reason:
- Avoids hallucination in early MVP.
- Ensures every answer is grounded in retrieved complaint IDs.
- Easier to test refusal behavior.

Tradeoff:
- Less fluent than an LLM-generated response.

Status: accepted.

---

## 008 — Add refusal when retrieval evidence is weak

Decision: refuse complaint answers when no evidence crosses the similarity threshold.

Reason:
- Prevents unsupported answers.
- Makes RAG quality testable.
- Aligns with grounded-answer requirement.

Status: accepted.

---

## 009 — Keep raw/generated artifacts out of Git

Decision: ignore raw data, processed data, models, and RAG artifacts.

Reason:
- Keeps repo lightweight.
- Avoids committing large files.
- CI can regenerate artifacts.

Status: accepted.

---

## 010 — Docker container runs full demo pipeline at startup

Decision: Docker command runs ingestion, validation, feature generation, training, promotion, RAG index build, then starts API.

Reason:
- Makes container reproducible from source.
- Avoids needing committed model binaries.
- Good for demo and grading.

Tradeoff:
- Startup is slower than a production image with prebuilt artifacts.

Status: accepted.