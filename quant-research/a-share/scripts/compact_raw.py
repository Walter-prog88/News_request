"""压缩 new_data 原始分钟数据(逐文件转换 -> 逐字节校验 -> 删除原件)。

策略(实测于 2026-08-20):
  open_adj / close_adj -> parquet + zstd + float32   约 7.0x(价格重复性高)
  amt                  -> pkl.gz + float32           约 2.4x(高熵,parquet 反而更差)

float32 的相对误差 ~2e-8,分钟收益误差 ~1e-8,相对 IC 门槛 0.025 完全可忽略。
安全性:每个文件先写新文件、读回逐元素比对(NaN 位置一致、数值等于 float32 往返值),
        校验通过才删除原件;峰值额外占用仅一个文件。中断可重入。

    python scripts/compact_raw.py --dry-run      # 只报告预计收益
    python scripts/compact_raw.py                # 实际转换
    python scripts/compact_raw.py --field amt    # 只转某个字段
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "new_data")
PLAN = {"close_adj": "parquet", "open_adj": "parquet", "amt": "gz"}


def target_path(src: str, kind: str) -> str:
    return src[:-4] + (".parquet" if kind == "parquet" else ".pkl.gz")


def convert_one(src: str, kind: str) -> tuple[int, int]:
    df = pd.read_pickle(src)
    orig = os.path.getsize(src)
    d32 = df.astype(np.float32)
    dst = target_path(src, kind)
    tmp = dst + ".tmp"
    if kind == "parquet":
        d32.to_parquet(tmp, compression="zstd", compression_level=9)
    else:
        # 必须显式指定 compression:pandas 按扩展名推断,而 .tmp 后缀会让它写成裸 pickle
        d32.to_pickle(tmp, compression="gzip")
    back = (pd.read_parquet(tmp) if kind == "parquet"
            else pd.read_pickle(tmp, compression="gzip"))
    # 校验:索引/列一致 + NaN 位置一致 + 数值等于 float32 往返值(逐元素)
    assert back.index.equals(df.index) and list(back.columns) == list(df.columns), f"{src} 索引/列不一致"
    a, b = d32.to_numpy(), back.to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{src} NaN 位置不一致"
    m = ~np.isnan(a)
    assert np.array_equal(a[m], b[m]), f"{src} 数值不一致"
    os.replace(tmp, dst)
    os.remove(src)
    return orig, os.path.getsize(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--field", default=None)
    a = ap.parse_args()
    fields = [a.field] if a.field else list(PLAN)
    tot_o = tot_n = 0
    for fld in fields:
        kind = PLAN[fld]
        files = sorted(glob.glob(os.path.join(RAW, fld, "*.pkl")))
        if not files:
            print(f"{fld}: 无待转换 .pkl(可能已完成)");  continue
        print(f"\n=== {fld} -> {kind}  ({len(files)} 个文件) ===", flush=True)
        if a.dry_run:
            sz = sum(os.path.getsize(f) for f in files)
            ratio = 7.0 if kind == "parquet" else 2.45
            print(f"  当前 {sz/1e9:.1f} GB -> 预计 {sz/1e9/ratio:.1f} GB(约 {ratio:.1f}x)")
            tot_o += sz; tot_n += sz / ratio
            continue
        fo = fn = 0
        t0 = time.time()
        for i, f in enumerate(files, 1):
            o, n = convert_one(f, kind)
            fo += o; fn += n
            if i % 10 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {fo/1e9:.1f} -> {fn/1e9:.1f} GB, {time.time()-t0:.0f}s", flush=True)
        print(f"  {fld}: {fo/1e9:.2f} GB -> {fn/1e9:.2f} GB (省 {(fo-fn)/1e9:.2f} GB, {fo/max(fn,1):.2f}x)")
        tot_o += fo; tot_n += fn
    print(f"\n合计 {tot_o/1e9:.1f} GB -> {tot_n/1e9:.1f} GB,节省 {(tot_o-tot_n)/1e9:.1f} GB")


if __name__ == "__main__":
    main()
