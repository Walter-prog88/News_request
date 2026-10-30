"""Ollama local model benchmark: quantization and model size comparison.

Measures three dimensions for each model:
  - Inference speed   (tokens/s) — from Ollama's eval_duration / eval_count
  - Memory footprint  (GB RAM)   — resident set size delta during generation
  - Answer quality    (score)    — LLM-as-judge: GPT/DeepSeek rates 1-5

Outputs a Markdown table and saves raw results to data/eval/benchmark_results.json.

Usage
-----
    # Run full benchmark (all available models)
    python scripts/ollama_benchmark.py

    # Quick test with one model
    python scripts/ollama_benchmark.py --models qwen2.5:7b --n-prompts 5

    # Skip quality scoring (faster, no API cost)
    python scripts/ollama_benchmark.py --no-judge
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psutil
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OLLAMA_BASE = "http://localhost:11434"

# Fixed prompts covering Chinese financial Q&A scenarios
BENCHMARK_PROMPTS = [
    "请简要分析宁德时代在储能领域的核心竞争优势，100字以内。",
    "什么是量化投资中的IC（信息系数）？它如何衡量因子质量？",
    "A股上市公司发布业绩预告后，股价通常会有哪些反应规律？",
    "请解释RAG（检索增强生成）相比纯大模型的核心优势。",
    "半导体行业的护城河主要体现在哪些方面？",
    "什么是分析师一致预期？目标价上调通常意味着什么？",
    "请列举三个常用的股票技术分析指标并简述其含义。",
    "企业年报中的归母净利润和净利润有什么区别？",
]

JUDGE_SYSTEM = """你是一位专业的AI回答质量评估专家。
请对以下问题和回答打分，评估维度：准确性、专业性、简洁性。
输出格式严格为：分数: X/5（X为1-5的整数），然后一句简短理由。"""


def ollama_generate(model: str, prompt: str, timeout: int = 120) -> dict:
    """Call Ollama /api/generate and return parsed response."""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def measure_speed(response: dict) -> float:
    """Extract tokens/s from Ollama response metadata."""
    eval_count = response.get("eval_count", 0)
    eval_duration_ns = response.get("eval_duration", 1)
    if eval_count == 0:
        return 0.0
    return eval_count / (eval_duration_ns / 1e9)


def measure_memory_gb() -> float:
    """Return current process + children RAM usage in GB (RSS)."""
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            mem += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return mem / (1024 ** 3)


def get_ollama_process_memory() -> float:
    """Sum RSS of all Ollama-related processes (server + model runner)."""
    total = 0
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            if "ollama" in proc.info["name"].lower():
                total += proc.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / (1024 ** 3)


def judge_quality(question: str, answer: str) -> float:
    """Use configured LLM to rate answer quality 1-5. Returns 0.0 on failure."""
    try:
        from src.agent.llm_synthesizer import _make_openai_client, get_llm_config
        cfg = get_llm_config()
        client = _make_openai_client(cfg)
        content = f"问题：{question}\n\n回答：{answer}"
        resp = client.chat.completions.create(
            model=cfg["model"],
            max_tokens=60,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": content},
            ],
        )
        text = resp.choices[0].message.content or ""
        # Parse "分数: X/5"
        import re
        m = re.search(r"分数[:：]\s*([0-9.]+)", text)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def benchmark_model(model: str, prompts: list[str], use_judge: bool) -> dict:
    """Run all prompts against one model, return aggregated stats."""
    print(f"\n{'='*60}")
    print(f"  Model: {model}")
    print(f"{'='*60}")

    # Warm-up: one short call to load the model into memory
    print("  Warming up...", end=" ", flush=True)
    try:
        ollama_generate(model, "你好", timeout=60)
        print("done")
    except Exception as e:
        print(f"FAILED: {e}")
        return {"model": model, "error": str(e)}

    mem_loaded = get_ollama_process_memory()
    speeds: list[float] = []
    qualities: list[float] = []
    latencies: list[float] = []

    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}] {prompt[:40]}...", end=" ", flush=True)
        t0 = time.monotonic()
        try:
            resp = ollama_generate(model, prompt, timeout=180)
            elapsed = time.monotonic() - t0
            answer = resp.get("response", "")
            speed = measure_speed(resp)
            speeds.append(speed)
            latencies.append(elapsed)

            quality = 0.0
            if use_judge and answer:
                quality = judge_quality(prompt, answer)
                qualities.append(quality)

            print(f"{speed:.1f} tok/s  {elapsed:.1f}s  quality={quality:.1f}")
        except Exception as e:
            print(f"ERROR: {e}")

    if not speeds:
        return {"model": model, "error": "all prompts failed"}

    result = {
        "model": model,
        "avg_speed_toks": round(sum(speeds) / len(speeds), 2),
        "min_speed_toks": round(min(speeds), 2),
        "avg_latency_s": round(sum(latencies) / len(latencies), 1),
        "memory_gb": round(mem_loaded, 2),
        "n_prompts": len(speeds),
    }
    if qualities:
        result["avg_quality"] = round(sum(qualities) / len(qualities), 2)
    return result


def print_table(results: list[dict]) -> None:
    """Print results as a Markdown table."""
    ok = [r for r in results if "error" not in r]
    if not ok:
        print("\nNo successful results.")
        return

    has_quality = any("avg_quality" in r for r in ok)

    print("\n\n## 量化与参数规模对比结果\n")
    header = "| 模型 | 速度 (tok/s) | 平均延迟 (s) | 内存占用 (GB) |"
    sep    = "|------|-------------|-------------|--------------|"
    if has_quality:
        header += " 质量得分 (1-5) |"
        sep    += "----------------|"
    print(header)
    print(sep)
    for r in ok:
        row = (
            f"| {r['model']:<35} "
            f"| {r['avg_speed_toks']:>11.1f} "
            f"| {r['avg_latency_s']:>11.1f} "
            f"| {r['memory_gb']:>12.2f} |"
        )
        if has_quality:
            q = r.get("avg_quality", "-")
            row += f" {str(q):>14} |"
        print(row)

    print("\n> 速度：CPU 推理 tokens/s；内存：Ollama 进程 RSS；质量：LLM-as-judge 1-5分")

    for r in results:
        if "error" in r:
            print(f"\n⚠️  {r['model']} 失败：{r['error']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"],
        help="Ollama model tags to benchmark",
    )
    parser.add_argument("--n-prompts", type=int, default=len(BENCHMARK_PROMPTS))
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM quality scoring")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    prompts = BENCHMARK_PROMPTS[: args.n_prompts]
    use_judge = not args.no_judge

    # Verify Ollama is reachable
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        available = {m["name"] for m in r.json().get("models", [])}
    except Exception as e:
        print(f"Cannot reach Ollama at {OLLAMA_BASE}: {e}")
        sys.exit(1)

    # Filter to models that are actually downloaded
    to_run = []
    for model in args.models:
        if model in available:
            to_run.append(model)
        else:
            print(f"⚠️  {model} not found locally — run: ollama pull {model}")
    if not to_run:
        print("No models available. Pull at least one model first.")
        sys.exit(1)

    print(f"Benchmarking {len(to_run)} model(s), {len(prompts)} prompts each")
    print(f"Quality scoring: {'enabled (LLM-as-judge)' if use_judge else 'disabled'}")

    results = [benchmark_model(m, prompts, use_judge) for m in to_run]

    print_table(results)

    out_path = PROJECT_ROOT / "data" / "eval" / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()

