# RAG Evaluation Report

## Purpose

Evaluate whether the complaint intelligence assistant returns grounded answers with cited evidence IDs and refuses when retrieval is weak.

## Summary

- Created at UTC: 2026-05-23T18:19:46.516707+00:00
- Total tests: 10
- Passed: 10
- Failed: 0
- Pass rate: 100.00%
- Average evidence count: 2.40
- Refusal count: 2

## Evaluation table

| ID | Passed | Refused | Evidence Count | Evidence IDs | Note |
|---|---:|---:|---:|---|---|
| rag_eval_001 | True | False | 3 | 2869476, 4952931, 7394845 | PASS: behavior, evidence, and expected terms matched. |
| rag_eval_002 | True | False | 3 | 3338526, 3421794, 7082890 | PASS: behavior, evidence, and expected terms matched. |
| rag_eval_003 | True | False | 3 | 6264204, 7305436, 3046595 | PASS: behavior, evidence, and expected terms matched. |
| rag_eval_004 | True | False | 3 | 5591548, 2860146, 1532232 | PASS: behavior, evidence, and expected terms matched. |
| rag_eval_005 | True | False | 3 | 6586464, 3827591, 3955472 | PASS: behavior, evidence, and expected terms matched. |
| rag_eval_006 | True | False | 3 | 4494743, 4551140, 2557599 | PASS: behavior, evidence, and expected terms matched. |
| rag_eval_007 | True | False | 3 | 6719856, 3111059, 6830262 | PASS: behavior, evidence, and expected terms matched. |
| rag_eval_008 | True | False | 3 | 6069599, 3035332, 2635616 | PASS: behavior, evidence, and expected terms matched. |
| rag_eval_009 | True | True | 0 |  | PASS: behavior, evidence, and expected terms matched. |
| rag_eval_010 | True | True | 0 |  | PASS: behavior, evidence, and expected terms matched. |

## Failure cases

No failing cases in this run.

## Known limitations

- This MVP uses TF-IDF retrieval instead of dense embeddings.
- The answer generator is deterministic and summary-based, not a generative LLM.
- Evidence relevance is checked with simple expected-term rules.
- Later hardening should add semantic relevance scoring, adversarial tests, and latency/token metrics.
