"""INC-T7 预注册冻结(前缀 inc_t7):只算文件哈希,不读任何数值、不算任何收益。

用法:python3 scripts/inc_t7_freeze.py            # 写 research/ledger/inc_t7_prereg_manifest.json(已存在则核对)
      python3 scripts/inc_t7_freeze.py verify     # 只核对
"""
from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "ledger" / "inc_t7_prereg_manifest.json"

FILES = [
    "research/INC_T7_PREREG.md",
    "src/afactor/strategy.py",
    "src/afactor/impact.py",
    "src/afactor/backtest.py",
    "src/afactor/lh_engine.py",
    "src/afactor/lh_panel.py",
    "src/afactor/panel.py",
    "src/afactor/config.py",
    "cache/_tmp/comp_v5_score_open10.parquet",
    "cache/_tmp/lh_step1_inputs/meta.json",
    "cache/_tmp/lh_step1_inputs/score_S1.npy",
    "cache/_tmp/lh_step1_inputs/pool_q92.npy",
    "cache/_tmp/lh_step1_inputs/V_d0.npy",
    "research/ledger/capacity_grid.csv",
    "research/ledger/lh_step1_grid.csv",
]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build() -> dict:
    rows = []
    for rel in FILES:
        p = ROOT / rel
        if not p.exists():
            rows.append(dict(path=rel, missing=True))
            continue
        st = p.stat()
        rows.append(dict(path=rel, sha256=sha(p), size=st.st_size,
                         mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))))
    return dict(task="INC-T7 调仓频率 × 缓冲带 成本-超额前沿(预注册冻结)",
                written_before_any_return_calc=True,
                frozen_at=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                files=rows)


def main():
    verify = len(sys.argv) > 1 and sys.argv[1] == "verify"
    cur = build()
    if OUT.exists():
        old = json.loads(OUT.read_text())
        a = {r["path"]: r.get("sha256") for r in old["files"]}
        b = {r["path"]: r.get("sha256") for r in cur["files"]}
        diff = [k for k in a if a[k] != b.get(k)]
        if diff:
            print("[inc_t7_freeze] 与冻结清单不一致:", diff)
            sys.exit(1)
        print(f"[inc_t7_freeze] 核对通过:{len(a)} 个文件与 {old['frozen_at']} 的冻结一致")
        return
    if verify:
        print("[inc_t7_freeze] 冻结清单不存在"); sys.exit(1)
    OUT.write_text(json.dumps(cur, ensure_ascii=False, indent=1))
    print(f"[inc_t7_freeze] 已冻结 {len(cur['files'])} 个文件 -> {OUT}")
    for r in cur["files"]:
        print("  ", r["path"], r.get("sha256", "MISSING")[:16], r.get("size"))


if __name__ == "__main__":
    main()
