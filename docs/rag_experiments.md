# RAG retrieval experiments

A running log of changes tested against the golden set in `data/eval/golden_queries.jsonl`
(29 queries). Each row is one measured run of `python -m src.rag.eval_retrieval`.
`docs/rag_eval.md` always holds the latest full report; this file keeps the history.

| # | Change | Docs | P@5 | Hit@5 | MRR@10 | Lift | Verdict |
|---|---|---|---|---|---|---|---|
| 0 | Baseline: TF-IDF on text as ingested | 5000 | 0.317 | 0.690 | 0.556 | 9.4x | reference |
| 1 | Strip standalone CFPB redaction tokens (`XXXX`, `XX/XX/2021`) | 4997 | 0.303 | 0.724 | 0.481 | 9.0x | no gain |
| 2 | Also strip redaction glued to words (`XXXXt`, `repXXXX`) | 4997 | 0.297 | 0.690 | 0.463 | 8.8x | no gain, kept |

## Experiment 1 and 2: removing CFPB redaction tokens

**Hypothesis.** 78% of narratives contain about 14 `XXXX` tokens each. They carry no
meaning, so removing them should make TF-IDF match on real content and improve retrieval.

**Result.** No improvement. Precision@5 and MRR@10 fell slightly and Hit@5 was flat or
up by one query. For experiment 1, precision@5 changed in 7 of 29 queries: better in 2,
worse in 5, unchanged in 22. With so few changed queries this cannot be told apart from
noise (a sign test on 2 versus 5 gives p of about 0.45).

**Why the change was kept anyway.** It is not justified by this metric. It is justified
by what comes next: the redaction tokens also appear in `evidence_preview` shown to
users, and they would be fed into embedding models and LLM prompts, where they waste
tokens and add noise. The change is cheap and the downstream benefit is plausible, but
it must be re-measured when dense retrieval and generation are added. If it hurts there,
it gets reverted.

**Side effect.** Three complaints fell below the 30-character minimum after stripping
and were dropped from the index (5000 to 4997).

**Lesson for the evaluation itself.** A 29-query set cannot resolve differences of a few
points. Before comparing retrievers, the golden set needs enough queries to detect
changes of about 0.03 in precision.