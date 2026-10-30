"""Build a synthetic evaluation set for RAG recall measurement.

How it works
------------
1. Sample N chunks randomly from the processed Markdown files.
2. For each chunk, call the configured LLM to generate one question whose
   answer is contained in that chunk.
3. Save results to data/eval/eval_set.jsonl.

Each line in eval_set.jsonl:
    {
        "question": "宁德时代储能业务2024年收入增速是多少？",
        "source_file": "data/processed/markdown/rc_300750_xxx.md",
        "doc_type": "research_report",
        "ts_code": "300750.SZ",
        "chunk_snippet": "储能业务收入同比增长78%..."  (first 200 chars)
    }

The source_file is the ground truth: a retrieval is a "hit" when the returned
result's source_file matches.  This is document-level matching — any chunk
from the same file counts — which handles the case where multiple chunks from
the same document are equally valid answers.

Limitations (acknowledged)
--------------------------
- Questions generated FROM a chunk are biased toward that chunk's vocabulary,
  making this eval set easier than real user queries.
- Use it as a fixed ruler for measuring relative change, not absolute quality.
- For stricter evaluation, replace generated questions with real user queries
  and manually verify the ground-truth source files.

Usage
-----
    python scripts/build_eval_set.py --n-chunks 60 --out data/eval/eval_set.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", flags=re.S)


def _parse_front_matter(text: str) -> tuple[dict, str]:
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip().strip("'\"")
        meta[k.strip()] = v if v != "null" else None
    return meta, m.group(2)


def _chunk_text(body: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Simple character-based chunking matching the indexer's default."""
    chunks: list[str] = []
    start = 0
    while start < len(body):
        end = min(start + chunk_size, len(body))
        chunks.append(body[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 60]


def _generate_question(chunk_text: str) -> str:
    """Ask the LLM to produce one question answerable from this chunk."""
    from src.agent.llm_synthesizer import _make_openai_client, get_llm_config

    cfg = get_llm_config()
    prompt = (
        "以下是一段金融文档内容：\n\n"
        f"{chunk_text[:600]}\n\n"
        "请根据这段内容，生成一个用户可能提问的问题（问题应该能从这段文字中找到答案）。"
        "只输出问题本身，不要解释，不要加序号。"
    )
    try:
        client = _make_openai_client(cfg)
        resp = client.chat.completions.create(
            model=cfg["model"],
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        LOGGER.warning("LLM question generation failed: %s", exc)
        return ""


def build_eval_set(
    markdown_dir: Path,
    out_path: Path,
    n_chunks: int = 60,
    seed: int = 42,
) -> int:
    """Sample chunks, generate questions, write eval_set.jsonl. Returns item count."""
    random.seed(seed)
    md_files = sorted(markdown_dir.glob("*.md"))
    if not md_files:
        raise RuntimeError(f"No markdown files found in {markdown_dir}")

    # Collect all (file, chunk) candidates first, then sample
    candidates: list[tuple[Path, dict, str]] = []
    for path in md_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        meta, body = _parse_front_matter(text)
        for chunk in _chunk_text(body):
            candidates.append((path, meta, chunk))

    LOGGER.info("Total candidate chunks: %d", len(candidates))
    sampled = random.sample(candidates, min(n_chunks, len(candidates)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, (path, meta, chunk) in enumerate(sampled, 1):
            LOGGER.info("[%d/%d] Generating question for %s ...", i, len(sampled), path.name)
            question = _generate_question(chunk)
            if not question:
                LOGGER.warning("  Skipping — no question generated")
                continue
            record = {
                "question": question,
                "source_file": str(path.relative_to(PROJECT_ROOT)),
                "doc_type": meta.get("source") or "unknown",
                "ts_code": meta.get("ts_code") or meta.get("stock_code") or "",
                "chunk_snippet": chunk[:200],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    LOGGER.info("Wrote %d eval items to %s", written, out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic RAG evaluation set")
    parser.add_argument("--n-chunks", type=int, default=60,
                        help="Number of chunks to sample (default 60)")
    parser.add_argument("--out", default="data/eval/eval_set.jsonl",
                        help="Output path (default data/eval/eval_set.jsonl)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    markdown_dir = PROJECT_ROOT / "data" / "processed" / "markdown"
    out_path = PROJECT_ROOT / args.out
    build_eval_set(markdown_dir, out_path, n_chunks=args.n_chunks, seed=args.seed)


if __name__ == "__main__":
    main()

