"""Retrieval evaluation harness for the complaint RAG layer.

Measures how well a retriever finds complaints from the right category.

Relevance is *category based*: a retrieved complaint counts as relevant to a
query if its (product, issue) pair equals the pair the query was written for.
These labels come from the CFPB data itself (the consumer picks the category),
so they are noisy. See docs/rag_eval.md for the limits of this method.

Usage:
    python -m src.rag.eval_retrieval
    python -m src.rag.eval_retrieval --write-report
    python -m src.rag.eval_retrieval --retriever tfidf bm25
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.rag.bm25 import BM25Index
from src.rag.dense import search_dense
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.retrieve import load_rag_artifacts, retrieve_complaints


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLDEN_PATH = PROJECT_ROOT / "data" / "eval" / "golden_queries.jsonl"
RESULTS_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "rag_retrieval_report.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "rag_eval.md"

TOP_K = 5
MRR_DEPTH = 10

REQUIRED_QUERY_KEYS = ("id", "question", "product", "issue", "strength")
VALID_STRENGTHS = ("strong", "mixed", "weak", "broad")

# The CFPB data uses several names for the same category. Merge them so the
# labels describe one group each.
PRODUCT_ALIASES = {
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit card": "Credit card or prepaid card",
}

ISSUE_ALIASES = {
    "Cont'd attempts collect debt not owed": "Attempts to collect debt not owed",
    "Dealing with my lender or servicer": "Dealing with your lender or servicer",
    "Problem with a company's investigation into an existing problem": (
        "Problem with a credit reporting company's investigation into an existing problem"
    ),
}

# A retriever takes (query, depth) and returns complaint IDs, best first.
Retriever = Callable[[str, int], list[str]]


def normalize_product(value: Any) -> str:
    text = str(value).strip()
    return PRODUCT_ALIASES.get(text, text)


def normalize_issue(value: Any) -> str:
    text = str(value).strip()
    return ISSUE_ALIASES.get(text, text)


def load_golden_queries(path: Path = GOLDEN_PATH) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing golden query file: {path}")

    queries: list[dict[str, str]] = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue

        query = json.loads(line)

        missing = [key for key in REQUIRED_QUERY_KEYS if key not in query]
        if missing:
            raise ValueError(f"{path.name} line {line_number}: missing keys {missing}")

        if query["strength"] not in VALID_STRENGTHS:
            raise ValueError(
                f"{path.name} line {line_number}: strength must be one of {VALID_STRENGTHS}"
            )

        queries.append(query)

    ids = [query["id"] for query in queries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: duplicate query ids")

    return queries


def build_group_lookup(
    docstore: pd.DataFrame,
) -> tuple[dict[str, tuple[str, str]], Counter]:
    """Map complaint_id -> (product, issue), and count documents per group."""
    groups = list(
        zip(
            docstore["product"].map(normalize_product),
            docstore["issue"].map(normalize_issue),
        )
    )
    ids = docstore["complaint_id"].astype(str).tolist()

    return dict(zip(ids, groups)), Counter(groups)


def tfidf_retriever(query: str, depth: int) -> list[str]:
    # min_score is just above zero so documents sharing no words with the
    # query are never returned (a retriever that finds nothing returns nothing).
    result = retrieve_complaints(query, top_k=depth, min_score=1e-9)
    return [item["complaint_id"] for item in result["results"]]


@lru_cache(maxsize=2)
def _bm25_for(index_version: str | None) -> tuple[BM25Index, tuple[str, ...]]:
    # Keyed on the index version, so rebuilding the RAG index rebuilds this too.
    _, docstore, _ = load_rag_artifacts()

    texts = docstore["clean_narrative"].fillna("").astype(str).tolist()
    ids = tuple(docstore["complaint_id"].astype(str))

    return BM25Index(texts), ids


def bm25_retriever(query: str, depth: int) -> list[str]:
    _, _, metadata = load_rag_artifacts()
    index, ids = _bm25_for(metadata.get("index_version"))

    return [ids[row] for row, _ in index.top(query, depth)]


# Add new retrievers here (dense embeddings, hybrid) to get a new row in the
# comparison table. The first entry is the baseline the others are compared to.
def dense_retriever(query: str, depth: int) -> list[str]:
      return search_dense(query, depth)


FUSION_DEPTH = 100


def hybrid_retriever(query: str, depth: int) -> list[str]:
      pool = max(depth, FUSION_DEPTH)

      return reciprocal_rank_fusion(
          [bm25_retriever(query, pool), dense_retriever(query, pool)],
          depth=depth,
      )


RETRIEVERS: dict[str, Retriever] = {
      "tfidf": tfidf_retriever,
      "bm25": bm25_retriever,
      "dense": dense_retriever,
      "hybrid": hybrid_retriever,
  }

DEFAULT_RETRIEVERS = ["tfidf", "bm25"]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(rows: list[dict[str, Any]]) -> dict[str, float]:
    precision = _mean([row["precision_at_k"] for row in rows])
    random_precision = _mean([row["random_precision"] for row in rows])

    return {
        "queries": len(rows),
        "precision_at_k": precision,
        "hit_rate_at_k": _mean([1.0 if row["hit_at_k"] else 0.0 for row in rows]),
        "mrr": _mean([row["reciprocal_rank"] for row in rows]),
        "random_precision": random_precision,
        "lift_over_random": precision / random_precision if random_precision else 0.0,
    }


def paired_comparison(
    baseline_rows: list[dict[str, Any]],
    other_rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Per-query win/loss count of `other` against `baseline` on precision@k.

    The p-value is a two-sided exact sign test: how likely a split this lopsided
    (or more) is if the two retrievers were equally good. Ties are ignored.
    """
    baseline_by_id = {row["id"]: row["precision_at_k"] for row in baseline_rows}

    wins = losses = ties = 0

    for row in other_rows:
        base = baseline_by_id[row["id"]]

        if row["precision_at_k"] > base:
            wins += 1
        elif row["precision_at_k"] < base:
            losses += 1
        else:
            ties += 1

    decided = wins + losses

    if decided == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(decided, i) for i in range(0, min(wins, losses) + 1))
        p_value = min(1.0, 2 * tail / 2**decided)

    return {"wins": wins, "losses": losses, "ties": ties, "sign_test_p": p_value}


def evaluate(
    retriever: Retriever,
    queries: list[dict[str, str]],
    docstore: pd.DataFrame,
    k: int = TOP_K,
    depth: int = MRR_DEPTH,
) -> dict[str, Any]:
    id_to_group, group_sizes = build_group_lookup(docstore)
    total_docs = len(docstore)

    rows: list[dict[str, Any]] = []

    for query in queries:
        target = (query["product"], query["issue"])
        group_size = group_sizes.get(target, 0)

        if group_size == 0:
            raise ValueError(
                f"Query {query['id']}: no documents in the index for "
                f"{target[0]} / {target[1]}. Check the label spelling."
            )

        ranked_ids = retriever(query["question"], depth)[:depth]
        flags = [id_to_group.get(complaint_id) == target for complaint_id in ranked_ids]

        # Precision divides by k, not by the number returned, so a retriever
        # that returns fewer than k results is penalised for the gap.
        precision = sum(flags[:k]) / k

        reciprocal_rank = 0.0
        for rank, is_relevant in enumerate(flags, start=1):
            if is_relevant:
                reciprocal_rank = 1.0 / rank
                break

        rows.append(
            {
                "id": query["id"],
                "question": query["question"],
                "strength": query["strength"],
                "group_size": group_size,
                "returned": len(ranked_ids),
                "precision_at_k": precision,
                "hit_at_k": any(flags[:k]),
                "reciprocal_rank": reciprocal_rank,
                # Expected precision if documents were picked at random.
                "random_precision": group_size / total_docs,
                "top_issues": [
                    (id_to_group.get(complaint_id) or ("?", "?"))[1]
                    for complaint_id in ranked_ids[:k]
                ],
            }
        )

    by_strength = {
        strength: summarise([row for row in rows if row["strength"] == strength])
        for strength in VALID_STRENGTHS
        if any(row["strength"] == strength for row in rows)
    }

    return {
        "k": k,
        "mrr_depth": depth,
        "summary": summarise(rows),
        "by_strength": by_strength,
        "per_query": rows,
    }


def run_evaluation(retriever_names: list[str]) -> dict[str, Any]:
    _, docstore, metadata = load_rag_artifacts()
    queries = load_golden_queries()

    retrievers = {
        name: evaluate(RETRIEVERS[name], queries, docstore)
        for name in retriever_names
    }

    baseline_name = retriever_names[0]
    comparisons = {
        name: paired_comparison(
            retrievers[baseline_name]["per_query"],
            data["per_query"],
        )
        for name, data in retrievers.items()
        if name != baseline_name
    }

    return {
        "index_version": metadata.get("index_version"),
        "doc_count": int(len(docstore)),
        "query_count": len(queries),
        "reference": baseline_name,
        "retrievers": retrievers,
        "comparisons": comparisons,
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def render_report(results: dict[str, Any]) -> str:
    k = TOP_K
    lines = [
        "# Retrieval evaluation",
        "",
        "Generated by `python -m src.rag.eval_retrieval --write-report`. "
        "Do not edit by hand.",
        "",
        f"- Index version: `{results['index_version']}`",
        f"- Documents indexed: {results['doc_count']}",
        f"- Golden queries: {results['query_count']} (`data/eval/golden_queries.jsonl`)",
        "",
        "## How relevance is judged",
        "",
        "A retrieved complaint is relevant to a query if its CFPB product and issue "
        "category equals the category the query was written for. Queries were written "
        "after reading about 8 complaints from each category.",
        "",
        "Each query is tagged by how consistent its category is: **strong** (most "
        "sampled complaints match the question), **mixed** (about half), **weak** "
        "(few), **broad** (a very large category, where almost any retriever scores "
        "well).",
        "",
        "## Limits of this evaluation",
        "",
        "- Categories are chosen by the person filing the complaint, so they are noisy. "
        "A retrieved complaint can be a good answer and still be scored wrong.",
        "- Queries were written by the project author, not collected from real users.",
        "- One query per category (two where a category splits in half) gives a small "
        "sample. Treat differences of a few points as noise.",
        "- `random P@k` is the precision you would get by picking complaints at "
        "random, so `lift` shows how much better than chance the retriever is.",
        "",
        "## Summary",
        "",
        f"| Retriever | P@{k} | Hit@{k} | MRR@{MRR_DEPTH} | Random P@{k} | Lift |",
        "|---|---|---|---|---|---|",
    ]

    for name, data in results["retrievers"].items():
        s = data["summary"]
        lines.append(
            f"| {name} | {_fmt(s['precision_at_k'])} | {_fmt(s['hit_rate_at_k'])} | "
            f"{_fmt(s['mrr'])} | {_fmt(s['random_precision'])} | "
            f"{s['lift_over_random']:.1f}x |"
        )

    comparisons = results.get("comparisons", {})

    if comparisons:
        baseline = results.get("reference", "baseline")
        lines += [
            "",
            f"## Paired comparison against {baseline}",
            "",
            f"For each query, does the retriever get a higher or lower P@{k} than "
            f"{baseline}? The sign-test p-value says how likely a split this lopsided "
            "is if the two were equally good. Below 0.05 suggests a real difference; "
            "with this few queries only large differences reach it.",
            "",
            f"| Retriever | Wins | Losses | Ties | Sign-test p |",
            "|---|---|---|---|---|",
        ]

        for name, c in comparisons.items():
            lines.append(
                f"| {name} | {c['wins']} | {c['losses']} | {c['ties']} | "
                f"{c['sign_test_p']:.3f} |"
            )

    for name, data in results["retrievers"].items():
        lines += [
            "",
            f"## {name}: by query strength",
            "",
            f"| Strength | Queries | P@{k} | Hit@{k} | MRR@{MRR_DEPTH} | Lift |",
            "|---|---|---|---|---|---|",
        ]

        for strength, s in data["by_strength"].items():
            lines.append(
                f"| {strength} | {s['queries']} | {_fmt(s['precision_at_k'])} | "
                f"{_fmt(s['hit_rate_at_k'])} | {_fmt(s['mrr'])} | "
                f"{s['lift_over_random']:.1f}x |"
            )

        lines += [
            "",
            f"## {name}: per query",
            "",
            f"| ID | Strength | Group size | P@{k} | Hit | RR | Question |",
            "|---|---|---|---|---|---|---|",
        ]

        for row in data["per_query"]:
            lines.append(
                f"| {row['id']} | {row['strength']} | {row['group_size']} | "
                f"{_fmt(row['precision_at_k'])} | {'yes' if row['hit_at_k'] else 'no'} | "
                f"{_fmt(row['reciprocal_rank'])} | {row['question']} |"
            )

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate complaint retrieval.")
    parser.add_argument(
        "--retriever",
        nargs="+",
        default=DEFAULT_RETRIEVERS,
        choices=sorted(RETRIEVERS),
        help="Retrievers to run. The first is the baseline the others are compared to.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write docs/rag_eval.md and data/processed/rag_retrieval_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_evaluation(args.retriever)

    print(f"Index: {results['index_version']}  docs: {results['doc_count']}  "
          f"queries: {results['query_count']}")

    for name, data in results["retrievers"].items():
        s = data["summary"]
        print(
            f"{name:>8}  P@{data['k']}={_fmt(s['precision_at_k'])}  "
            f"Hit@{data['k']}={_fmt(s['hit_rate_at_k'])}  "
            f"MRR={_fmt(s['mrr'])}  random={_fmt(s['random_precision'])}  "
            f"lift={s['lift_over_random']:.1f}x"
        )

    for name, c in results["comparisons"].items():
        print(
            f"{name} vs {results['reference']}: wins={c['wins']} losses={c['losses']} "
            f"ties={c['ties']} sign-test p={c['sign_test_p']:.3f}"
        )

    if args.write_report:
        RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(render_report(results), encoding="utf-8")

        print(f"Wrote {REPORT_PATH}")
        print(f"Wrote {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()