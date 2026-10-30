"""合并月度 npz -> 按特征的宽表 parquet,并在 2026-01-01 处物理切分研究集/封存集。"""
from __future__ import annotations
import glob, os, re
import numpy as np
import pandas as pd
from afactor.config import TMP, DAILY, SEALED, SEAL_START

BATCH = 25


def main():
    files = sorted(glob.glob(str(TMP / "*.npz")))
    months = [re.search(r"(\d{6})", os.path.basename(f)).group(1) for f in files]
    print(f"{len(files)} 个月度文件")

    all_dates, all_codes, feat_names = set(), set(), None
    meta = {}
    for f, m in zip(files, months):
        z = np.load(f, allow_pickle=True)
        d = [str(x) for x in z["dates"]]
        c = [str(x) for x in z["codes"]]
        meta[m] = (d, c)
        all_dates.update(d); all_codes.update(c)
        if feat_names is None:
            feat_names = sorted(k for k in z.files if k not in ("dates", "codes"))
        z.close()

    dates = pd.DatetimeIndex(sorted(all_dates))
    codes = pd.Index(sorted(all_codes), name="ts_code")
    di = {d: i for i, d in enumerate(dates.strftime("%Y-%m-%d"))}
    ci = {c: i for i, c in enumerate(codes)}
    print(f"面板 {len(dates)} 日 x {len(codes)} 股 x {len(feat_names)} 特征")
    print(f"日期范围 {dates[0].date()} ~ {dates[-1].date()}")

    seal_mask = dates >= pd.Timestamp(SEAL_START)
    print(f"研究集 {int((~seal_mask).sum())} 日 | 封存集 {int(seal_mask.sum())} 日")

    for b0 in range(0, len(feat_names), BATCH):
        batch = feat_names[b0:b0 + BATCH]
        buf = {k: np.full((len(dates), len(codes)), np.nan, np.float32) for k in batch}
        for f, m in zip(files, months):
            z = np.load(f, allow_pickle=True)
            d, c = meta[m]
            ri = np.array([di[x] for x in d]); cj = np.array([ci[x] for x in c])
            for k in batch:
                buf[k][np.ix_(ri, cj)] = z[k]
            z.close()
        for k in batch:
            df = pd.DataFrame(buf[k], index=dates, columns=codes)
            df.index.name = "date"
            df.loc[~seal_mask].to_parquet(DAILY / f"{k}.parquet")
            df.loc[seal_mask].to_parquet(SEALED / f"{k}.parquet")
        print(f"  已写 {b0 + len(batch)}/{len(feat_names)}", flush=True)

    pd.Series(feat_names).to_csv(DAILY / "_features.csv", index=False, header=["feature"])
    print("done")


if __name__ == "__main__":
    main()
