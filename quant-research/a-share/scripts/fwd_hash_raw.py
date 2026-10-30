"""前向检验原始数据记哈希(预注册 FORWARD_TEST_PREREG §4.8 第 1 条,前缀 fwd)。

在计算任何前向收益之前,对 cache/forward/raw/ 下全部文件与 cache/forward/m5_long.parquet 逐个记 sha256,
落盘 research/ledger/fwd_raw_manifest.json(含逐文件 sha256、大小、mtime 与整体摘要)。之后前向数据不得替换:
`python3 scripts/fwd_hash_raw.py --verify` 重算并与清单比对,不一致返回 1。

只读文件字节,不解析任何数值。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from afactor.config import CACHE, LEDGER, ROOT

FWD = CACHE / "forward"
OUT = LEDGER / "fwd_raw_manifest.json"


def sha256(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def collect() -> dict:
    files = sorted([p for p in (FWD / "raw").rglob("*") if p.is_file()]
                   + [FWD / "m5_long.parquet", FWD / "m5_long.meta.json"])
    out = {}
    for p in files:
        st = p.stat()
        out[str(p.relative_to(ROOT))] = dict(sha256=sha256(p), size=st.st_size,
                                             mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)))
    return out


def digest(entries: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(entries):
        h.update(f"{k}\t{entries[k]['sha256']}\t{entries[k]['size']}\n".encode())
    return h.hexdigest()


def main():
    verify = "--verify" in sys.argv[1:]
    t0 = time.time()
    ent = collect()
    dg = digest(ent)
    if verify:
        man = json.load(open(OUT))
        bad = [k for k in set(man["files"]) | set(ent)
               if man["files"].get(k, {}).get("sha256") != ent.get(k, {}).get("sha256")]
        print(f"前向原始数据校验:{len(ent)} 个文件,不一致 {len(bad)} 个;摘要 {dg} vs 清单 {man['files_digest']}")
        for k in sorted(bad)[:20]:
            print("  不一致:", k)
        sys.exit(1 if bad else 0)
    if OUT.exists():
        raise SystemExit(f"{OUT} 已存在;前向原始数据清单只写一次(用 --verify 校验)")
    man = dict(created_at=time.strftime("%Y-%m-%d %H:%M:%S %z"), n_files=len(ent), files_digest=dg,
               total_bytes=int(sum(v["size"] for v in ent.values())),
               note="前向检验开工时对 cache/forward/raw 与 m5_long 逐文件记 sha256(预注册 §4.8.1),之后数据不得替换",
               files=ent)
    json.dump(man, open(OUT, "w"), ensure_ascii=False, indent=0)
    print(f"写 {OUT.name}:{len(ent)} 个文件,{man['total_bytes'] / 1e6:.0f} MB,摘要 {dg},{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
