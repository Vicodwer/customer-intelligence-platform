# RAG retrieval experiments

A running log of changes tested against the golden set in `data/eval/golden_queries.jsonl`
(29 queries). Each row is one measured run of `python -m src.rag.eval_retrieval`.
`docs/rag_eval.md` always holds the latest full report; this file keeps the history.

| # | Change | Docs | P@5 | Hit@5 | MRR@10 | Lift | Verdict |
|---|---|---|---|---|---|---|---|
| 0 | Baseline: TF-IDF on text as ingested | 5000 | 0.317 | 0.690 | 0.556 | 9.4x | reference |
| 1 | Strip standalone CFPB redaction tokens (`XXXX`, `XX/XX/2021`) | 4997 | 0.303 | 0.724 | 0.481 | 9.0x | no gain |
| 2 | Also strip redaction glued to words (`XXXXt`, `repXXXX`) | 4997 | 0.297 | 0.690 | 0.463 | 8.8x | no gain, kept |
| 3 | BM25 (k1=1.5, b=0.75, unigrams) instead of TF-IDF, on the cleaned text | 4997 | 0.407 | 0.690 | 0.510 | 12.1x | better ranking, p=0.049 |

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

## Experiment 3: BM25 instead of TF-IDF

**Hypothesis.** BM25's term-frequency saturation and length normalisation suit
complaints better than TF-IDF cosine similarity, because complaints range from one
sentence to several paragraphs.

**Setup.** Okapi BM25 (`src/rag/bm25.py`) with the standard defaults `k1=1.5`, `b=0.75`,
unigrams, the same stop words, built on the same cleaned text as experiment 2. The
parameters were not tuned: 29 queries is too few to tune without overfitting. The
implementation is tested against an independent plain-Python version of the formula.

**Result.** Precision@5 rose from 0.297 to 0.407 and MRR@10 from 0.463 to 0.510.
Hit@5 did not change (0.690). Per query, BM25 was better on 13, worse on 4 and equal on
12 (sign test p = 0.049, borderline). The gain is not only from the broad queries:
over the 25 non-broad queries, P@5 went from about 0.27 to about 0.35.

**What it did not fix.** Both retrievers find nothing relevant in the top 5 for 9
queries, and 7 of those are the same (q02, q10, q11, q14, q15, q19, q21). BM25 rescued
two queries (q06, q18) and lost two (q04, q20). These shared failures are cases where
the question's wording differs from the complaint text, or the category label is
ambiguous. Matching on words cannot close that gap, which is the case for trying dense
(embedding) retrieval and a hybrid next.

**Caveats.** A p-value of 0.049 on 29 queries is suggestive, not conclusive. The two
pipelines also differ in more than the scoring formula (TF-IDF uses unigrams plus
bigrams and drops words seen in fewer than 2 documents; BM25 does neither), so this
compares whole retrievers, not formulas alone. Labels are noisy, as in every run here.