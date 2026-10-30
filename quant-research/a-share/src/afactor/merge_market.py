"""合并市场互动特征月度 npz -> cache/daily(研究期)与 cache/sealed(封存期)。"""
from __future__ import annotations
import glob, os, re
import numpy as np
import pandas as pd
from afactor.config import DAILY, SEALED, SEAL_START
from afactor.build_market import TMP_MKT

BATCH = 16


def main():
    files = sorted(glob.glob(str(TMP_MKT / "*.npz")))
    months = [re.search(r"(\d{6})", os.path.basename(f)).group(1) for f in files]
    print(f"{len(files)} 个月度文件")
    all_dates, all_codes, feat_names, meta = set(), set(), None, {}
    for f, m in zip(files, months):
        z = np.load(f, allow_pickle=True)
        d = [str(x) for x in z["dates"]]; c = [str(x) for x in z["codes"]]
        meta[m] = (d, c); all_dates.update(d); all_codes.update(c)
        if feat_names is None:
            feat_names = sorted(k for k in z.files if k not in ("dates", "codes"))
        z.close()
    dates = pd.DatetimeIndex(sorted(all_dates))
    codes = pd.Index(sorted(all_codes), name="ts_code")
    di = {d: i for i, d in enumerate(dates.strftime("%Y-%m-%d"))}
    ci = {c: i for i, c in enumerate(codes)}
    seal = dates >= pd.Timestamp(SEAL_START)
    print(f"面板 {len(dates)}日 x {len(codes)}股 x {len(feat_names)}特征;研究 {int((~seal).sum())} 封存 {int(seal.sum())}")
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
            df = pd.DataFrame(buf[k], index=dates, columns=codes); df.index.name = "date"
            df.loc[~seal].to_parquet(DAILY / f"{k}.parquet")
            df.loc[seal].to_parquet(SEALED / f"{k}.parquet")
        print(f"  {b0+len(batch)}/{len(feat_names)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
