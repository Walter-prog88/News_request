"""Training matrix for the showcase model stack.

Features : 312 factors (legacy 210 daily price-volume + 102 intraday flow),
           cross-sectionally percentile-ranked per day, NaN preserved.
Labels   : next-open to next-next-open return (signal at close t, trade at
           open t+1, marked at open t+2), residualized against the gated
           cross-section mean; plus its within-day percentile rank for the
           rank-objective model.
Universe : domestic-equity ETF pool (flow-store columns), liquidity gate v2.

Outputs land in SHOWCASE cache as flat arrays: X, y_res, y_rank, row day/sym
indices, and a manifest.
"""

import json
import os

import numpy as np
import os as _os
# Default to a persistent root: the old fallback pointed at the research
# session's scratch dir, which the OS purges, taking every cached panel
# with it. rebuild_caches.sh regenerates this tree bit-identically.
_SCRATCH = _os.environ.get(
    "INDEX_CACHE_ROOT", _os.path.expanduser("~/index-cache"))

SCRATCH = _SCRATCH
ZOO = f"{SCRATCH}/zoo_cache"
OUT = f"{SCRATCH}/showcase_cache"
UB = "/path/to/user/Desktop/ETF-Research/index-project-active-etf-upper-bound"
LEGACY = f"{UB}/research/active_system_v1/factors_v2.npz"
FLOW = (f"{UB}/research/results/active_etf_upper_bound/core_flow_main.npz")

RET_CLIP = 0.25


def rank01(m: np.ndarray) -> np.ndarray:
    """Row-wise percentile rank in [0,1], NaN preserved. m: (F, N)."""
    out = np.full(m.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(m)
    cnt = finite.sum(axis=1)
    order = np.argsort(np.where(finite, m, np.inf), axis=1, kind="stable")
    ranks = np.empty(m.shape, dtype=np.int32)
    rng = np.arange(m.shape[1], dtype=np.int32)
    np.put_along_axis(ranks, order, rng[None, :], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = ranks / np.maximum(cnt - 1, 1)[:, None]
    out[finite] = vals[finite].astype(np.float32)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(f"{ZOO}/meta.json"))
    zdates = np.array(meta["dates"], dtype="datetime64[D]")
    zsyms = list(meta["symbols"])

    lg = np.load(LEGACY, allow_pickle=True)
    fl = np.load(FLOW, allow_pickle=True)
    ldates = lg["index"].astype("datetime64[D]")
    fdates = fl["index"].astype("datetime64[D]")
    common = np.intersect1d(np.intersect1d(ldates, fdates), zdates)
    print("common sessions:", len(common), common[0], common[-1])

    syms = [s for s in fl["cols"] if s in set(lg["cols"]) and s in set(zsyms)]
    print("universe:", len(syms))
    li = {s: i for i, s in enumerate(lg["cols"])}
    fi = {s: i for i, s in enumerate(fl["cols"])}
    zi = {s: i for i, s in enumerate(zsyms)}
    lsym = np.array([li[s] for s in syms])
    fsym = np.array([fi[s] for s in syms])
    zsym = np.array([zi[s] for s in syms])
    lt = {d: i for i, d in enumerate(ldates)}
    ft = {d: i for i, d in enumerate(fdates)}
    zt = {d: i for i, d in enumerate(zdates)}

    n_z = len(zdates)
    op = np.load(f"{ZOO}/continuous_open.npy")[:n_z][:, zsym]
    valid = np.load(f"{ZOO}/daily_ohlc_valid.npy")[:n_z][:, zsym].astype(bool)
    gate = np.load(f"{ZOO}/liq_gate_v2.npy")[:n_z][:, zsym].astype(bool)
    op = np.where(valid & (op > 0), op, np.nan)

    names = [str(x) for x in lg["names"]] + [str(x) for x in fl["names"]]
    A_l, A_f = lg["A"], fl["A"]

    Xs, ys, yr, drow, srow = [], [], [], [], []
    for t_i, d in enumerate(common[:-2]):
        z0, z1, z2 = zt[d], zt[common[t_i + 1]], zt[common[t_i + 2]]
        o1, o2 = op[z1], op[z2]
        with np.errstate(invalid="ignore"):
            lab = o2 / o1 - 1.0
        ok = (gate[z0] & valid[z0] & np.isfinite(lab)
              & (np.abs(lab) < RET_CLIP))
        if ok.sum() < 30:
            continue
        idx = np.where(ok)[0]
        F = np.concatenate([A_l[:, lt[d], lsym[idx]],
                            A_f[:, ft[d], fsym[idx]]], axis=0)
        R = rank01(F).T                       # (n, 312)
        y = lab[idx].astype(np.float32)
        y_res = y - y.mean()
        y_rank = rank01(y[None, :])[0]
        Xs.append(R)
        ys.append(y_res)
        yr.append(y_rank)
        drow.append(np.full(len(idx), t_i, dtype=np.int32))
        srow.append(idx.astype(np.int32))
        if t_i % 250 == 0:
            print(f"  {d} rows={len(idx)}", flush=True)

    X = np.concatenate(Xs)
    np.save(f"{OUT}/X.npy", X)
    np.save(f"{OUT}/y_res.npy", np.concatenate(ys))
    np.save(f"{OUT}/y_rank.npy", np.concatenate(yr))
    np.save(f"{OUT}/row_day.npy", np.concatenate(drow))
    np.save(f"{OUT}/row_sym.npy", np.concatenate(srow))
    with open(f"{OUT}/manifest.json", "w") as fh:
        json.dump({"n_rows": int(X.shape[0]), "n_features": int(X.shape[1]),
                   "feature_names": names,
                   "dates": [str(d) for d in common],
                   "symbols": syms}, fh)
    print("rows:", X.shape, "nan share:",
          round(float(np.isnan(X).mean()), 4))


if __name__ == "__main__":
    main()
