# Data

This project uses two public datasets / public dataset samples.

## ML lane: UCI Bank Marketing

Purpose: predict whether a contacted customer subscribes to a term deposit.

Local files:
- raw: `data/raw/bank-additional-full.csv`
- sample: `data/samples/bank_marketing_sample.csv`

Target column:
- `y`

## LLM/RAG lane: Consumer complaint narratives

Purpose: complaint intelligence over public complaint narratives with cited evidence.

Local files:
- raw API JSON sample: `data/raw/complaints_sample_hf.json`
- sample CSV: `data/samples/cfpb_complaints_sample.csv`

Important handling rule:
- Do not commit full datasets.
- Do not show full raw complaint narratives in demo screenshots unless cleaned/redacted.
- Keep only small samples in Git.
