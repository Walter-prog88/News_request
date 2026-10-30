"""INC-T4 步骤 2:流式构造弱因子等权组合 D(§3.3,与 run_midband_composite._accumulate 逐字同法)。

  z_n = ops.standardize(FA.REGISTRY[n][1](p), p["universe"]) × sign_n
  D   = Σ z_n / count,覆盖门槛 count >= len(set)//3(原脚本 n_min),否则 NaN

六个面板:D_{A,main,B} 与剔除 pead family 的 D′(*_nopead)。并集 636 个因子各构造一次,
同时喂给所有包含它的集合;≤3 个进程,每个进程按名字连续分块(相邻因子共用原料,省内存)。
进程内原料缓存超过 MEM_CAP 字节就清空(只影响速度,不影响数值)。

产出:cache/_tmp/inc_t4_D_{set}.parquet(float32,1455×5565),sha256 写入日志与 inc_t4_checks.csv。
用法:python3 scripts/inc_t4_build_scores.py [--procs 3]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t4_common as C  # noqa: E402
from afactor.config import TMP  # noqa: E402

warnings.filterwarnings("ignore")
MEM_CAP = 2.5e9
_G: dict = {}


def _init():
    from afactor.panel import Panel
    from afactor import factors as FA
    _G["p"] = Panel()
    FA.build_all()
    _G["u"] = _G["p"]["universe"]


def _cache_bytes(p) -> int:
    tot = 0
    for d in (p._cache, p._derived):
        for v in d.values():
            try:
                tot += int(v.memory_usage(index=False, deep=False).sum())
            except Exception:
                pass
    return tot


def _accumulate(job):
    from afactor import factors as FA, ops
    wid, items, set_names = job
    p, u = _G["p"], _G["u"]
    shape = (len(p.dates), len(p.codes))
    acc = {k: [np.zeros(shape, np.float32), np.zeros(shape, np.int16)] for k in set_names}
    fails, t0 = [], time.time()
    for i, (n, sg, sets) in enumerate(items):
        try:
            f = FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
            z = ops.standardize(f, u).to_numpy(np.float32) * sg
        except Exception as e:                       # 原脚本:构造失败的成员跳过(计数不加)
            fails.append((n, repr(e)[:200]))
            continue
        ok = np.isfinite(z)
        for k in sets:
            acc[k][0][ok] += z[ok]
            acc[k][1][ok] += 1
        del f, z, ok
        if _cache_bytes(p) > MEM_CAP:
            keep = {"universe": p._derived.get("universe")}
            p._cache.clear(); p._derived.clear()
            if keep["universe"] is not None:
                p._derived["universe"] = keep["universe"]
        if (i + 1) % 20 == 0:
            print(f"  [w{wid}] {i+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
    return wid, acc, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    C.check_prereg()
    mem = C.members()
    sets = mem["sets"]
    assert set(sets) == set(C.SETS)
    need: dict = {}
    for s in C.SETS:
        for n, sg in sets[s].items():
            need.setdefault((n, float(sg)), []).append(s)
    items = sorted([(n, sg, ss) for (n, sg), ss in need.items()], key=lambda x: x[0])
    print(f"(因子, 方向) {len(items)} 个(预注册并集 636)", flush=True)
    assert len(items) == 636
    W = a.procs
    blk = int(np.ceil(len(items) / W))
    jobs = [(i, items[i * blk:(i + 1) * blk], C.SETS) for i in range(W)]
    t0 = time.time()
    parts, fails = [], []
    tot = {k: [None, None] for k in C.SETS}
    with ProcessPoolExecutor(max_workers=W, initializer=_init) as ex:
        for fut in as_completed([ex.submit(_accumulate, j) for j in jobs]):
            wid, acc, fl = fut.result()
            parts.append(wid); fails += fl
            for k in C.SETS:                         # 逐块累加,不同时驻留多个分块
                s_, c_ = acc[k]
                if tot[k][0] is None:
                    tot[k] = [s_.astype(np.float64), c_.astype(np.int32)]
                else:
                    tot[k][0] += s_; tot[k][1] += c_
            del acc
            print(f"  worker {wid} 完成 {time.time()-t0:.0f}s,失败 {len(fl)}", flush=True)
    if fails:
        print("构造失败的因子:", fails, flush=True)
    from afactor.panel import Panel
    p = Panel()
    rows = []
    for k in C.SETS:
        s_, c_ = tot[k]
        nm = len(sets[k])
        n_min = max(1, nm // 3)
        with np.errstate(invalid="ignore", divide="ignore"):
            F = np.where(c_ >= n_min, s_ / np.maximum(c_, 1), np.nan).astype(np.float32)
        D = pd.DataFrame(F, index=p.dates, columns=p.codes)
        D.to_parquet(C.d_path(k))
        cov = float(np.isfinite(F).sum(1).mean())
        h = C.sha256_file(C.d_path(k))
        rows.append(dict(set=k, n_members=nm, n_min=n_min, mean_cover_per_day=cov, sha256=h))
        print(f"D_{k}: 成员 {nm} n_min {n_min} 日均覆盖 {cov:.0f} sha256 {h[:16]}…", flush=True)
    pd.DataFrame(rows).to_csv(TMP / f"{C.PFX}_D_manifest.csv", index=False)
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
