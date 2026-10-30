"""Evaluate retrieval quality with lightweight smoke tests.

This is not a full benchmark. It catches obvious regressions after changing
crawlers, parsers, embeddings, or retrieval rules.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.tools import rag_search, search_company_docs, search_policy_docs


TEST_CASES = [
    {
        "name": "company_annual_report",
        "fn": "company",
        "stock_code": "300308",
        "theme": "AI 光模块",
        "expected_any": ["年度报告", "中际旭创"],
    },
    {
        "name": "company_optical_module",
        "fn": "company",
        "stock_code": "300308",
        "theme": "AI 光模块 800G 1.6T",
        "expected_any": ["光模块", "800G", "1.6T"],
    },
    {
        "name": "policy_ic_software",
        "fn": "policy",
        "theme": "人工智能 集成电路 软件 税收优惠",
        "question": "产业政策是否支持？",
        "expected_any": ["集成电路", "软件", "税收优惠"],
    },
    {
        "name": "generic_rag_query",
        "fn": "rag",
        "query": "中际旭创 年度报告 光模块 核心竞争力",
        "expected_any": ["中际旭创", "年度报告", "光模块"],
    },
]


def contains_any(items: list[dict], expected_terms: list[str]) -> bool:
    """Return True if retrieved text contains any expected term."""
    haystack = "\n".join(
        " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("content") or ""),
                str(item.get("url") or ""),
                str(item.get("source") or ""),
            ]
        )
        for item in items
    )
    return any(term in haystack for term in expected_terms)


def run_case(case: dict, top_k: int) -> dict:
    """Run one retrieval test case."""
    if case["fn"] == "company":
        results = search_company_docs(case["stock_code"], case["theme"], top_k=top_k)
    elif case["fn"] == "policy":
        results = search_policy_docs(case["theme"], case["question"], top_k=top_k)
    else:
        results = rag_search(case["query"], top_k=top_k)

    passed = bool(results) and contains_any(results, case["expected_any"])
    return {
        "name": case["name"],
        "passed": passed,
        "n_results": len(results),
        "top_titles": [item.get("title") for item in results[:3]],
        "top_sources": [item.get("source") for item in results[:3]],
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    results = [run_case(case, args.top_k) for case in TEST_CASES]
    passed_count = sum(1 for item in results if item["passed"])

    print("=" * 80)
    print("Retrieval Smoke Test")
    print("=" * 80)
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{status}] {item['name']}  n_results={item['n_results']}")
        print(f"       titles={item['top_titles']}")
        print(f"       sources={item['top_sources']}")

    print("=" * 80)
    print(f"Passed {passed_count}/{len(results)}")
    if passed_count != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

