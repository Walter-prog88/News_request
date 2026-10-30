"""End-to-end RAG comparison: cloud LLM vs local Ollama.

For each query in the eval set, runs the full RAG pipeline twice:
  1. Cloud  — provider configured in configs/sources.yaml (e.g. SiliconFlow/DeepSeek-V3)
  2. Local  — Ollama model specified via --local-model (default: qwen2.5:7b)

The retrieval layer (FAISS + BGE) is identical in both runs.
Only the LLM synthesis step differs.

Metrics reported
----------------
  Latency       — wall-clock seconds for the LLM synthesis call
  Answer length — proxy for response completeness
  Quality score — LLM-as-judge 1-5 (cloud model rates both answers)

Output saved to data/eval/cloud_vs_local.json and printed as a Markdown table.

Usage
-----
    python scripts/rag_cloud_vs_local.py
    python scripts/rag_cloud_vs_local.py --local-model qwen2.5:14b --n-queries 10
    python scripts/rag_cloud_vs_local.py --no-judge   # skip quality scoring
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OLLAMA_BASE = "http://localhost:11434"

JUDGE_PROMPT = """你是一位专业的投研回答质量评估专家。
请对以下投资研究问题和回答打分（1-5分）：
- 5分：准确、专业、有具体数据支撑
- 3分：基本正确但笼统
- 1分：错误或无意义

严格按格式输出：分数: X/5，然后一句理由。"""


def _load_cloud_config() -> dict:
    cfg_path = PROJECT_ROOT / "configs" / "sources.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        full = yaml.safe_load(f)
    return full.get("llm", {})


def _make_ollama_synthesizer(model: str):
    """Return a function that calls Ollama for LLM synthesis."""
    from src.agent.llm_synthesizer import _SYSTEM_PROMPT

    def _call(question: str, context: str) -> tuple[str, float]:
        prompt = f"{_SYSTEM_PROMPT}\n\n研究问题：{question}\n\n{context[:2000]}\n\n请基于以上材料给出专业分析。"
        t0 = time.monotonic()
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=180,
        )
        elapsed = time.monotonic() - t0
        resp.raise_for_status()
        answer = resp.json().get("response", "")
        return answer, elapsed

    return _call


def _make_cloud_synthesizer():
    """Return a function that calls the configured cloud LLM."""
    from src.agent.llm_synthesizer import _make_openai_client, get_llm_config, _SYSTEM_PROMPT

    cfg = get_llm_config()
    client = _make_openai_client(cfg)

    def _call(question: str, context: str) -> tuple[str, float]:
        t0 = time.monotonic()
        resp = client.chat.completions.create(
            model=cfg["model"],
            max_tokens=cfg["max_tokens"],
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"研究问题：{question}\n\n{context[:2000]}"},
            ],
        )
        elapsed = time.monotonic() - t0
        answer = resp.choices[0].message.content or ""
        return answer, elapsed

    return _call


def _judge(question: str, answer: str) -> float:
    """Rate answer quality 1-5 using cloud LLM as judge."""
    try:
        from src.agent.llm_synthesizer import _make_openai_client, get_llm_config
        cfg = get_llm_config()
        client = _make_openai_client(cfg)
        resp = client.chat.completions.create(
            model=cfg["model"],
            max_tokens=60,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"问题：{question}\n\n回答：{answer[:600]}"},
            ],
        )
        import re
        m = re.search(r"分数[:：]\s*([0-9.]+)", resp.choices[0].message.content or "")
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def _build_context_for_query(question: str) -> str:
    """Run RAG retrieval and return a context string for the LLM."""
    try:
        from src.agent.tools import rag_search
        docs = rag_search(question, top_k=5)
        parts = []
        for i, d in enumerate(docs, 1):
            title = d.get("title") or "未命名"
            content = (d.get("content") or "").replace("\n", " ")[:400]
            parts.append(f"[{i}] {title}\n{content}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"(RAG retrieval failed: {e})"


def run_comparison(
    queries: list[dict],
    local_model: str,
    use_judge: bool,
) -> list[dict]:
    cloud_fn = _make_cloud_synthesizer()
    local_fn = _make_ollama_synthesizer(local_model)

    results = []
    for i, item in enumerate(queries, 1):
        question = item["question"]
        print(f"\n[{i}/{len(queries)}] {question[:55]}...")

        context = _build_context_for_query(question)

        # Cloud
        print("  cloud ...", end=" ", flush=True)
        try:
            cloud_answer, cloud_latency = cloud_fn(question, context)
            print(f"{cloud_latency:.1f}s  {len(cloud_answer)} chars")
        except Exception as e:
            print(f"FAILED: {e}")
            cloud_answer, cloud_latency = "", 0.0

        # Local
        print(f"  local ({local_model}) ...", end=" ", flush=True)
        try:
            local_answer, local_latency = local_fn(question, context)
            print(f"{local_latency:.1f}s  {len(local_answer)} chars")
        except Exception as e:
            print(f"FAILED: {e}")
            local_answer, local_latency = "", 0.0

        cloud_quality = local_quality = 0.0
        if use_judge and cloud_answer and local_answer:
            print("  judging ...", end=" ", flush=True)
            cloud_quality = _judge(question, cloud_answer)
            local_quality = _judge(question, local_answer)
            print(f"cloud={cloud_quality:.1f}  local={local_quality:.1f}")

        results.append({
            "question": question,
            "cloud": {
                "latency_s": round(cloud_latency, 2),
                "answer_len": len(cloud_answer),
                "quality": cloud_quality,
                "answer": cloud_answer[:300],
            },
            "local": {
                "model": local_model,
                "latency_s": round(local_latency, 2),
                "answer_len": len(local_answer),
                "quality": local_quality,
                "answer": local_answer[:300],
            },
        })

    return results


def print_summary(results: list[dict], cloud_name: str, local_model: str) -> None:
    ok = [r for r in results if r["cloud"]["latency_s"] > 0 and r["local"]["latency_s"] > 0]
    if not ok:
        print("No complete results.")
        return

    def avg(key, side):
        vals = [r[side][key] for r in ok if r[side][key] > 0]
        return sum(vals) / len(vals) if vals else 0.0

    has_quality = any(r["cloud"]["quality"] > 0 for r in ok)

    print(f"\n\n## 云端 vs 本地 RAG 对比结果\n")
    print(f"| 指标 | 云端 ({cloud_name}) | 本地 ({local_model}) | 差值 |")
    print("|------|---------------------|----------------------|------|")

    cl = avg("latency_s", "cloud")
    lo = avg("latency_s", "local")
    print(f"| 平均延迟 (s) | {cl:.1f} | {lo:.1f} | {lo-cl:+.1f} |")

    cl = avg("answer_len", "cloud")
    lo = avg("answer_len", "local")
    print(f"| 平均回答长度 (chars) | {cl:.0f} | {lo:.0f} | {lo-cl:+.0f} |")

    if has_quality:
        cl = avg("quality", "cloud")
        lo = avg("quality", "local")
        print(f"| 质量得分 (1-5) | {cl:.2f} | {lo:.2f} | {lo-cl:+.2f} |")

    print(f"\n样本数：{len(ok)} 条查询")
    print("\n> 延迟：LLM synthesis 层耗时（检索层相同）；质量：LLM-as-judge 1-5分")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", default="data/eval/eval_set.jsonl")
    parser.add_argument("--local-model", default="qwen2.5:7b")
    parser.add_argument("--n-queries", type=int, default=10)
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    # Verify Ollama has the requested model
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        available = {m["name"] for m in r.json().get("models", [])}
        if args.local_model not in available:
            print(f"Model '{args.local_model}' not found. Available: {sorted(available)}")
            print(f"Run: ollama pull {args.local_model}")
            sys.exit(1)
    except Exception as e:
        print(f"Ollama not reachable: {e}")
        sys.exit(1)

    eval_path = PROJECT_ROOT / args.eval_path
    if not eval_path.exists():
        print(f"Eval set not found: {eval_path}")
        print("Run first: python scripts/build_eval_set.py")
        sys.exit(1)

    queries = [
        json.loads(line)
        for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.n_queries]

    cloud_cfg = _load_cloud_config()
    cloud_name = cloud_cfg.get("model", "cloud")

    print(f"Cloud model : {cloud_name}")
    print(f"Local model : {args.local_model}")
    print(f"Queries     : {len(queries)}")
    print(f"Judge       : {'enabled' if not args.no_judge else 'disabled'}")

    results = run_comparison(queries, args.local_model, use_judge=not args.no_judge)
    print_summary(results, cloud_name, args.local_model)

    out_path = PROJECT_ROOT / "data" / "eval" / "cloud_vs_local.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()

