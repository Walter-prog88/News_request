"""Round B input: daily amount + signed-flow imbalance for ALL 2005 symbols.

The existing flow store covers only the 1178 domestic-equity symbols; bond,
gold, HK and overseas ETFs have neither a liquidity gate nor flow factors.
This script aggregates the 6.8GB minute pivots into two daily panels:

  daily_amount_all.parquet   sum of minute amount per session
  flow_ratio_all.parquet     tick-rule signed amount / total amount per session

Memory plan for a 16GB machine: load one pickle at a time, dump to float32
memmaps (~3.4GB each on disk), free, then stream column blocks. Peak ~10GB,
so run this ALONE, never alongside the sweep workflow.

Free cross-check: for the 1178 covered symbols the result must match the
frozen npz store's daily_amount; max relative error is written to the report.
"""

import gc
import json
import os

import numpy as np
import pandas as pd
import os as _os
# Default to a persistent root: the old fallback pointed at the research
# session's scratch dir, which the OS purges, taking every cached panel
# with it. rebuild_caches.sh regenerates this tree bit-identically.
_SCRATCH = _os.environ.get(
    "INDEX_CACHE_ROOT", _os.path.expanduser("~/index-cache"))

SRC = "/path/to/user/Desktop/ETF-Research/full-etf project2 "
CACHE = os.environ.get(
    "ZOO_CACHE",
    _SCRATCH + "/zoo_cache",
)
FLOW_NPZ = (
    "/path/to/user/Desktop/ETF-Research/index-project-active-etf-upper-bound/research/results/"
    "active_etf_upper_bound/core_flow_main.npz"
)
BLOCK = 200


def dump_memmap(name: str) -> tuple:
    """Pickle -> float32 .npy on disk; returns (index, columns)."""
    path = f"{CACHE}/{name}.f32.npy"
    df = pd.read_pickle(f"{SRC}/etf_pivot_{name}_qfq.pkl")
    idx, cols = df.index, list(df.columns)
    if not os.path.exists(path):
        np.save(path, df.values.astype(np.float32))
    del df
    gc.collect()
    return idx, cols


def main() -> None:
    os.makedirs(CACHE, exist_ok=True)
    print("loading amount pickle ...", flush=True)
    idx, cols = dump_memmap("amount")
    print("loading close pickle ...", flush=True)
    idx2, cols2 = dump_memmap("close")
    # the two pivots may order columns differently; align close to amount order
    if list(cols) != list(cols2):
        assert set(cols) == set(cols2) and len(idx) == len(idx2)
        order = [cols2.index(c) for c in cols]
        C = np.load(f"{CACHE}/close.f32.npy", mmap_mode="r")
        np.save(f"{CACHE}/close_aligned.f32.npy",
                np.asarray(C[:, order], dtype=np.float32))
        os.replace(f"{CACHE}/close_aligned.f32.npy", f"{CACHE}/close.f32.npy")
        cols2 = cols
    assert len(idx) == len(idx2)

    days = idx.normalize()
    uday = days.unique().sort_values()
    codes = uday.get_indexer(days)
    nday, nsym = len(uday), len(cols)

    A = np.load(f"{CACHE}/amount.f32.npy", mmap_mode="r")
    C = np.load(f"{CACHE}/close.f32.npy", mmap_mode="r")

    amt_daily = np.zeros((nday, nsym), dtype=np.float64)
    signed_daily = np.zeros((nday, nsym), dtype=np.float64)

    # day starts, to zero the overnight diff inside each session
    day_start = np.zeros(len(idx), dtype=bool)
    day_start[0] = True
    day_start[1:] = codes[1:] != codes[:-1]

    for lo in range(0, nsym, BLOCK):
        hi = min(nsym, lo + BLOCK)
        a = np.asarray(A[:, lo:hi], dtype=np.float64)
        c = np.asarray(C[:, lo:hi], dtype=np.float64)
        a = np.nan_to_num(a)
        dc = np.zeros_like(c)
        dc[1:] = c[1:] - c[:-1]
        dc[day_start] = 0.0
        dc = np.nan_to_num(dc)
        s = np.sign(dc) * a
        amt_daily[:, lo:hi] = np.vstack(
            [np.bincount(codes, weights=a[:, j], minlength=nday)
             for j in range(hi - lo)]).T
        signed_daily[:, lo:hi] = np.vstack(
            [np.bincount(codes, weights=s[:, j], minlength=nday)
             for j in range(hi - lo)]).T
        print(f"  block {lo}-{hi} done", flush=True)

    amt = pd.DataFrame(amt_daily, index=uday, columns=cols)
    flow = pd.DataFrame(
        np.where(amt_daily > 0, signed_daily / amt_daily, np.nan),
        index=uday, columns=cols)
    amt.to_parquet(f"{CACHE}/daily_amount_all.parquet")
    flow.to_parquet(f"{CACHE}/flow_ratio_all.parquet")

    # cross-check against the frozen npz for the covered 1178
    z = np.load(FLOW_NPZ, allow_pickle=True)
    ref = pd.DataFrame(z["daily_amount"], index=pd.DatetimeIndex(z["index"]),
                       columns=z["cols"])
    common = [s for s in ref.columns if s in amt.columns]
    mine = amt.reindex(index=ref.index)[common]
    both = ref[common].stack().to_frame("ref").join(
        mine.stack().rename("mine")).dropna()
    both = both[both["ref"] > 0]
    relerr = (both["mine"] - both["ref"]).abs() / both["ref"]
    report = {"rows": len(both), "max_rel_err": float(relerr.max()),
              "p999_rel_err": float(relerr.quantile(0.999))}
    with open(f"{CACHE}/minute_agg_report.json", "w") as fh:
        json.dump(report, fh)
    print("cross-check:", report)
    for name in ("amount", "close"):
        os.remove(f"{CACHE}/{name}.f32.npy")


if __name__ == "__main__":
    main()
