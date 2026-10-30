"""Evaluate RAG retrieval quality against a ground-truth eval set.

Metrics computed
----------------
Recall@k   — fraction of questions where the ground-truth source_file appears
             in the top-k retrieved results.
MRR        — Mean Reciprocal Rank: average of 1/rank of first correct result.
             Higher = correct answer appears closer to the top.
NDCG@k     — Normalised Discounted Cumulative Gain: position-weighted recall.
             A hit at rank 1 scores more than a hit at rank 5.

All three metrics are in [0, 1]; higher is better.

The script runs two passes — without HyDE and with HyDE — and prints a
side-by-side comparison so you can see whether query rewriting helped.

Usage
-----
    # First build the eval set (one-time):
    python scripts/build_eval_set.py --n-chunks 60

    # Then run evaluation:
    python scripts/run_eval.py
    python scripts/run_eval.py --eval-path data/eval/eval_set.jsonl --top-k 5
    python scripts/run_eval.py --no-hyde   # skip HyDE pass (faster)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _is_hit(results: list[dict], ground_truth_file: str) -> tuple[bool, int]:
    """Return (hit, rank) where rank is 1-indexed position of first correct result."""
    gt = Path(ground_truth_file).name  # compare filenames only, not full paths
    for rank, r in enumerate(results, 1):
        candidate = Path(str(r.get("source_file") or "")).name
        if candidate == gt:
            return True, rank
    return False, 0


def _ndcg(rank: int, k: int) -> float:
    """NDCG@k for a single binary relevance item at given rank (0 = not found)."""
    if rank == 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def evaluate(
    eval_path: Path,
    top_k: int = 5,
    use_hyde: bool = False,
) -> dict:
    """Run retrieval on every eval item, return aggregated metrics."""
    from src.indexer.retriever import retrieve

    config_path = PROJECT_ROOT / "configs" / "sources.yaml"
    items = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    hits_at_k = 0
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []
    details: list[dict] = []

    for i, item in enumerate(items, 1):
        question = item["question"]
        source_file = item["source_file"]
        label = f"[{i}/{len(items)}]"

        try:
            results = retrieve(
                question,
                config_path=config_path,
                top_k=top_k,
                use_reranker=False,
                use_hyde=use_hyde,
            )
        except Exception as exc:
            LOGGER.warning("%s retrieve failed: %s", label, exc)
            results = []

        hit, rank = _is_hit(results, source_file)
        hits_at_k += int(hit)
        rr = (1.0 / rank) if rank > 0 else 0.0
        nd = _ndcg(rank, top_k)
        reciprocal_ranks.append(rr)
        ndcg_scores.append(nd)
        details.append({
            "question": question[:60],
            "hit": hit,
            "rank": rank or "-",
            "source_file": Path(source_file).name,
        })

    n = len(items)
    return {
        "n": n,
        "top_k": top_k,
        "use_hyde": use_hyde,
        "recall_at_k": hits_at_k / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
        "ndcg_at_k": sum(ndcg_scores) / n if n else 0.0,
        "details": details,
    }


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def print_report(baseline: dict, hyde: dict | None = None) -> None:
    k = baseline["top_k"]
    print("\n" + "=" * 72)
    print(f"  RAG Retrieval Evaluation  (n={baseline['n']}, top_k={k})")
    print("=" * 72)

    header = f"  {'Metric':<20} {'Baseline (no HyDE)':>20}"
    if hyde:
        header += f"  {'With HyDE':>12}  {'Δ':>8}"
    print(header)
    print("  " + "-" * 68)

    metrics = [
        (f"Recall@{k}", "recall_at_k"),
        ("MRR", "mrr"),
        (f"NDCG@{k}", "ndcg_at_k"),
    ]
    for label, key in metrics:
        base_val = baseline[key]
        row = f"  {label:<20} {_fmt(base_val):>20}"
        if hyde:
            hyde_val = hyde[key]
            delta = hyde_val - base_val
            sign = "+" if delta >= 0 else ""
            row += f"  {_fmt(hyde_val):>12}  {sign}{_fmt(delta):>7}"
        print(row)

    print("=" * 72)

    if hyde:
        print("\n  Metric explanations:")
        print(f"  Recall@{k}  — fraction of questions answered within top-{k}")
        print("  MRR        — avg 1/rank of first correct hit (1.0 = always #1)")
        print(f"  NDCG@{k}    — position-weighted recall (hit at rank 1 > rank 5)")

    # Per-item detail: show misses only
    misses = [d for d in baseline["details"] if not d["hit"]]
    if misses:
        print(f"\n  Missed questions ({len(misses)}/{baseline['n']}):")
        for d in misses[:10]:
            print(f"    - {d['question']}")
            print(f"      ground truth: {d['source_file']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--eval-path", default="data/eval/eval_set.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-hyde", action="store_true", help="Skip the HyDE comparison pass")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    eval_path = PROJECT_ROOT / args.eval_path

    if not eval_path.exists():
        print(f"Eval set not found: {eval_path}")
        print("Run first:  python scripts/build_eval_set.py")
        sys.exit(1)

    print(f"Running baseline evaluation (top_k={args.top_k}) ...")
    baseline = evaluate(eval_path, top_k=args.top_k, use_hyde=False)

    hyde_result = None
    if not args.no_hyde:
        print("Running HyDE evaluation ...")
        hyde_result = evaluate(eval_path, top_k=args.top_k, use_hyde=True)

    print_report(baseline, hyde_result)


if __name__ == "__main__":
    main()

