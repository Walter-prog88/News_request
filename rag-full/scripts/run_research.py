"""Run deterministic RAG + market-data research workflow."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.research_workflow import collect_research_payload, save_report
from src.agent.trace import save_trace


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-code", required=True, help="Stock code, e.g. 300308 or 300308.SZ")
    parser.add_argument("--theme", required=True, help="Theme keyword, e.g. AI 光模块")
    parser.add_argument("--question", required=True, help="Research question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None, help="Optional output Markdown path")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM analysis and use rule-based synthesis")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    payload = collect_research_payload(
        args.stock_code,
        args.theme,
        args.question,
        top_k=args.top_k,
        include_llm_analysis=not args.no_llm,
    )
    report = payload["report"]

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    else:
        path = save_report(report, args.stock_code, args.theme)

    trace_path = save_trace(payload["trace"])
    print(report)
    print(f"\nSaved report: {path}")
    print(f"Saved trace: {trace_path}")


if __name__ == "__main__":
    main()

