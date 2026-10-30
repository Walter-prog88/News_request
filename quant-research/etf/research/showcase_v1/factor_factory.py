"""Genetic-programming factor factory for the showcase stack.

The 312-factor matrix is a human-curated library: someone decided in advance
which shapes were worth measuring. This module removes that decision. It
searches formula space directly — random expression trees over 19 raw daily
panels and a causal operator set — and keeps whatever survives a rank-IC
screen and a redundancy screen.

Two disciplines make the chapter defensible rather than decorative:

  * Fitness is scored on days < DEV_END only. Every survivor also carries its
    IC on the untouched tail. Selection never saw those days, so the held-out
    column is the honest read on whether the search found structure or noise.
  * A complexity penalty is charged per node, so a 5-node formula beats a
    13-node formula of equal IC. Simple survivors are the ones that can be
    read, argued with, and falsified.

The headline retrain then follows the house convention (full-sample fit,
signal t, fill t+1 open, 0.5bp/side, Top-4) so the delta against the
312-only baseline is apples-to-apples.
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sweep_dial import CACHE, ZOO, load_env, run_topk, stats

OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
TAG = os.environ.get("FACTORY_TAG", "")

DEV_END = int(os.environ.get("FACTORY_DEV_END", 1400))
                        # fitness horizon; days >= this are never selected on
POP = 400
GENS = 12
ISLANDS = 4             # independent restarts; a single deme collapses onto
                        # one strong operand and starves the dedup pool
ELITE_FRAC = 0.10
IMMIGRANT_FRAC = 0.12   # fresh random blood each generation, same reason
TOURNEY = 5
MAX_DEPTH = 3
COMPLEXITY = 0.0002
P_CROSSOVER = 0.60
P_TERMINAL = 0.30       # early-stop probability when growing a subtree
IC_MIN = 0.020          # primary screen on |dev IC|
IC_FLOOR = 0.012        # relaxed screen if the primary leaves < MIN_KEEP
MIN_KEEP, MAX_KEEP = 250, 350
DEDUP_RHO = 0.70
DEDUP_ROWS = 120_000
MIN_NAMES = 20          # days thinner than this carry no usable IC
MIN_COVER = 0.30        # a formula must be finite on this share of dev rows
PANEL_CACHE = 100       # subtree panels held in RAM (~6.7MB each)

TS_W = (3, 5, 10, 21, 63)
BIN_W = (5, 10, 21, 63)
UNARY_SIMPLE = ("neg", "abs", "sign", "log1p_abs", "rank_cs", "zscore_cs")
UNARY_TS = ("ts_mean", "ts_std", "ts_max", "ts_min", "ts_delta", "ts_sum",
            "delay", "ts_rank")
BIN_SIMPLE = ("add", "sub", "mul", "div_safe")
BIN_TS = ("ts_corr", "ts_cov")

ZOO_FIELDS = ("ret_cc", "overnight_ret_continuous", "ret_oc_continuous",
              "open30_ret_continuous", "last30_ret", "rv_closepath_continuous",
              "up_rv_continuous", "down_rv_continuous",
              "high_time_ordinal_continuous", "zr_share_continuous",
              "n_price_moves", "path_abs_return_continuous", "flat_bar_ratio")


# ----------------------------------------------------------------- operands

def build_operands(man, dates):
    """(T,N) raw panels on the showcase grid, invalid sessions blanked."""
    meta = json.load(open(f"{ZOO}/meta.json"))
    zt = {np.datetime64(d, "D"): i for i, d in enumerate(meta["dates"])}
    zi = {s: i for i, s in enumerate(meta["symbols"])}
    zsym = np.array([zi[s] for s in man["symbols"]])
    zrow = np.array([zt[d] for d in dates])
    n_z = len(meta["dates"])

    def grab(field):
        a = np.load(f"{ZOO}/{field}.npy", mmap_mode="r")[:n_z]
        return np.asarray(a[:, zsym][zrow], dtype=np.float32)

    valid = grab("daily_ohlc_valid").astype(bool)
    ops, names = [], []

    def add(nm, arr):
        a = np.asarray(arr, dtype=np.float32)
        a = np.where(valid & np.isfinite(a), a, np.nan).astype(np.float32)
        ops.append(a)
        names.append(nm)

    for f in ZOO_FIELDS:
        add(f, grab(f))

    close, high, low = grab("close"), grab("high_continuous"), grab(
        "low_continuous")
    with np.errstate(invalid="ignore", divide="ignore"):
        add("hl_range", (high - low) / close)
        add("close_pos", (close - low) / np.where(high > low, high - low,
                                                  np.nan))
        add("hi_gap", high / close - 1.0)
        add("lo_gap", 1.0 - low / close)

    idx = pd.DatetimeIndex(dates)
    amt = pd.read_parquet(f"{ZOO}/daily_amount_all.parquet").reindex(
        index=idx, columns=man["symbols"]).values.astype(np.float32)
    flow = pd.read_parquet(f"{ZOO}/flow_ratio_all.parquet").reindex(
        index=idx, columns=man["symbols"]).values.astype(np.float32)
    # amount is unadjusted money traded; log tames the scale without changing
    # any cross-sectional ordering the search can act on
    add("log_amount", np.log1p(np.where(amt > 0, amt, np.nan)))
    add("flow_ratio", flow)
    return ops, names


# ---------------------------------------------------------------- operators

def _clean(a):
    a = np.asarray(a, dtype=np.float32)
    return np.where(np.isfinite(a), a, np.nan).astype(np.float32)


def _mp(w):
    return max(2, int(round(w * 0.6)))


def rank_cs(x):
    """Row-wise percentile rank in [0,1], NaN preserved."""
    out = np.full(x.shape, np.nan, np.float32)
    fin = np.isfinite(x)
    cnt = fin.sum(axis=1)
    order = np.argsort(np.where(fin, x, np.inf), axis=1, kind="stable")
    ranks = np.empty(x.shape, np.int32)
    rng = np.arange(x.shape[1], dtype=np.int32)
    np.put_along_axis(ranks, order, rng[None, :], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = ranks / np.maximum(cnt - 1, 1)[:, None]
    out[fin] = vals[fin].astype(np.float32)
    return out


def zscore_cs(x):
    mu = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=1, keepdims=True)
    sd = np.nanstd(np.where(np.isfinite(x), x, np.nan), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _clean((x - mu) / np.where(sd > 0, sd, np.nan))


def delay(x, w):
    out = np.full(x.shape, np.nan, np.float32)
    out[w:] = x[:-w]
    return out


def apply_unary(name, x, w):
    if name == "neg":
        return -x
    if name == "abs":
        return np.abs(x)
    if name == "sign":
        return _clean(np.sign(x))
    if name == "log1p_abs":
        return _clean(np.sign(x) * np.log1p(np.abs(x)))
    if name == "rank_cs":
        return rank_cs(x)
    if name == "zscore_cs":
        return zscore_cs(x)
    if name == "delay":
        return delay(x, w)
    if name == "ts_delta":
        return _clean(x - delay(x, w))
    d = pd.DataFrame(x, copy=False).rolling(w, min_periods=_mp(w))
    if name == "ts_mean":
        return _clean(d.mean().values)
    if name == "ts_std":
        return _clean(d.std().values)
    if name == "ts_max":
        return _clean(d.max().values)
    if name == "ts_min":
        return _clean(d.min().values)
    if name == "ts_sum":
        return _clean(d.sum().values)
    if name == "ts_rank":
        return _clean(d.rank(pct=True).values)
    raise KeyError(name)


def apply_binary(name, a, b, w):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if name == "add":
            return _clean(a + b)
        if name == "sub":
            return _clean(a - b)
        if name == "mul":
            return _clean(a * b)
        if name == "div_safe":
            return _clean(a / np.where(np.abs(b) > 1e-9, b, np.nan))
    da = pd.DataFrame(a, copy=False)
    db = pd.DataFrame(b, copy=False)
    r = da.rolling(w, min_periods=_mp(w))
    if name == "ts_corr":
        return _clean(r.corr(db).values)
    if name == "ts_cov":
        return _clean(r.cov(db).values)
    raise KeyError(name)


# ------------------------------------------------------------------- trees
# node := ("v", i) | ("u", op, w, child) | ("b", op, w, left, right)

def node_str(n, opnames):
    if n[0] == "v":
        return opnames[n[1]]
    if n[0] == "u":
        head = n[1] if n[2] is None else f"{n[1]}_{n[2]}"
        return f"{head}({node_str(n[3], opnames)})"
    head = n[1] if n[2] is None else f"{n[1]}_{n[2]}"
    return (f"{head}({node_str(n[3], opnames)}, "
            f"{node_str(n[4], opnames)})")


def n_nodes(n):
    if n[0] == "v":
        return 1
    if n[0] == "u":
        return 1 + n_nodes(n[3])
    return 1 + n_nodes(n[3]) + n_nodes(n[4])


def depth(n):
    if n[0] == "v":
        return 0
    if n[0] == "u":
        return 1 + depth(n[3])
    return 1 + max(depth(n[3]), depth(n[4]))


def rand_unary(rng):
    if rng.random() < 0.45:
        return ("u", UNARY_SIMPLE[rng.integers(len(UNARY_SIMPLE))], None)
    return ("u", UNARY_TS[rng.integers(len(UNARY_TS))],
            int(TS_W[rng.integers(len(TS_W))]))


def rand_binary(rng):
    if rng.random() < 0.65:
        return ("b", BIN_SIMPLE[rng.integers(len(BIN_SIMPLE))], None)
    return ("b", BIN_TS[rng.integers(len(BIN_TS))],
            int(BIN_W[rng.integers(len(BIN_W))]))


def rand_tree(rng, n_vars, dleft, full=False):
    if dleft == 0 or (not full and rng.random() < P_TERMINAL):
        return ("v", int(rng.integers(n_vars)))
    if rng.random() < 0.42:
        k, op, w = rand_binary(rng)
        return (k, op, w, rand_tree(rng, n_vars, dleft - 1, full),
                rand_tree(rng, n_vars, dleft - 1, full))
    k, op, w = rand_unary(rng)
    return (k, op, w, rand_tree(rng, n_vars, dleft - 1, full))


def collect(n, acc=None, path=()):
    acc = [] if acc is None else acc
    acc.append(path)
    if n[0] == "u":
        collect(n[3], acc, path + (3,))
    elif n[0] == "b":
        collect(n[3], acc, path + (3,))
        collect(n[4], acc, path + (4,))
    return acc


def get_at(n, path):
    for p in path:
        n = n[p]
    return n


def set_at(n, path, sub):
    if not path:
        return sub
    p = path[0]
    child = set_at(n[p], path[1:], sub)
    lst = list(n)
    lst[p] = child
    return tuple(lst)


def crossover(rng, a, b):
    pa_all, pb_all = collect(a), collect(b)
    for _ in range(8):
        pa = pa_all[rng.integers(len(pa_all))]
        pb = pb_all[rng.integers(len(pb_all))]
        cand = set_at(a, pa, get_at(b, pb))
        if depth(cand) <= MAX_DEPTH:
            return cand
    return a


def mutate(rng, t, n_vars):
    paths = collect(t)
    p = paths[rng.integers(len(paths))]
    node = get_at(t, p)
    roll = rng.random()
    if roll < 0.40:                                   # swap the operator
        if node[0] == "u":
            k, op, w = rand_unary(rng)
            node = (k, op, w, node[3])
        elif node[0] == "b":
            k, op, w = rand_binary(rng)
            node = (k, op, w, node[3], node[4])
        else:
            node = ("v", int(rng.integers(n_vars)))
    elif roll < 0.75:                                 # regrow a subtree
        room = MAX_DEPTH - len(p)
        node = rand_tree(rng, n_vars, max(0, room))
    else:                                             # hoist a child up
        node = node[3] if node[0] != "v" else ("v", int(rng.integers(n_vars)))
    cand = set_at(t, p, node)
    return cand if depth(cand) <= MAX_DEPTH else t


# --------------------------------------------------------------- evaluation

class Evaluator:
    """Panel evaluation with a small LRU of subtree results."""

    def __init__(self, operands, opnames, day, sym, y_rank, n_days):
        self.ops = operands
        self.names = opnames
        self.cache = {}
        self.order = []
        self.day = day
        self.sym = sym
        self.y = y_rank.astype(np.float64)
        self.n_days = n_days
        self.starts = np.searchsorted(day, np.arange(n_days + 1))
        self.dev = day < DEV_END
        self.held = ~self.dev

    def panel(self, n):
        if n[0] == "v":
            return self.ops[n[1]]
        key = node_str(n, self.names)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if n[0] == "u":
            val = apply_unary(n[1], self.panel(n[3]), n[2])
        else:
            val = apply_binary(n[1], self.panel(n[3]), self.panel(n[4]), n[2])
        self.cache[key] = val
        self.order.append(key)
        while len(self.order) > PANEL_CACHE:
            self.cache.pop(self.order.pop(0), None)
        return val

    def row_ranks(self, vals):
        """Within-day percentile rank of `vals` over the training rows."""
        fin = np.isfinite(vals)
        o = np.lexsort((vals, self.day))          # NaN sorts last per day
        ds = self.day[o]
        pos = np.arange(len(vals)) - self.starts[ds]
        cnt = np.bincount(self.day[fin], minlength=self.n_days).astype(
            np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            r_sorted = pos / np.maximum(cnt[ds] - 1.0, 1.0)
        xr = np.full(len(vals), np.nan)
        xr[o] = r_sorted
        xr[~fin] = np.nan
        return xr, fin

    def _ic_series(self, xr, fin, mask):
        m = fin & mask
        d = self.day[m]
        x = xr[m]
        y = self.y[m]
        c = np.bincount(d, minlength=self.n_days).astype(np.float64)
        sx = np.bincount(d, weights=x, minlength=self.n_days)
        sy = np.bincount(d, weights=y, minlength=self.n_days)
        sxx = np.bincount(d, weights=x * x, minlength=self.n_days)
        syy = np.bincount(d, weights=y * y, minlength=self.n_days)
        sxy = np.bincount(d, weights=x * y, minlength=self.n_days)
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = sxy / c - (sx / c) * (sy / c)
            vx = sxx / c - (sx / c) ** 2
            vy = syy / c - (sy / c) ** 2
            ic = cov / np.sqrt(np.where(vx > 0, vx, np.nan)
                               * np.where(vy > 0, vy, np.nan))
        ok = (c >= MIN_NAMES) & np.isfinite(ic)
        return ic[ok]

    def score(self, tree):
        vals = self.panel(tree)[self.day, self.sym].astype(np.float64)
        cover = np.isfinite(vals[self.dev]).mean()
        nn = n_nodes(tree)
        if cover < MIN_COVER:
            return {"ic_dev": 0.0, "ic_held": 0.0, "ic_ir": 0.0,
                    "n_nodes": nn, "fit": -9.0, "cover": float(cover)}
        xr, fin = self.row_ranks(vals)
        ic_d = self._ic_series(xr, fin, self.dev)
        ic_h = self._ic_series(xr, fin, self.held)
        if len(ic_d) < 200:
            return {"ic_dev": 0.0, "ic_held": 0.0, "ic_ir": 0.0,
                    "n_nodes": nn, "fit": -9.0, "cover": float(cover)}
        md = float(ic_d.mean())
        sd = float(ic_d.std())
        return {"ic_dev": md,
                "ic_held": float(ic_h.mean()) if len(ic_h) > 50 else 0.0,
                "ic_ir": float(md / sd * np.sqrt(252)) if sd > 0 else 0.0,
                "n_nodes": nn, "cover": float(cover),
                "fit": abs(md) - COMPLEXITY * nn}


# --------------------------------------------------------------------- main

def win_stats(r):
    return {"2019-2021": stats(r.loc["2019":"2021"]),
            "2022-2023": stats(r.loc["2022":"2023"]),
            "2024-2026": stats(r.loc["2024":]),
            "full": stats(r)}


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    t_start = time.time()
    os.makedirs(OUT, exist_ok=True)
    man, dates, G, O2O = load_env()
    T, N = len(dates), len(man["symbols"])
    day = np.load(f"{CACHE}/row_day.npy").astype(np.int64)
    sym = np.load(f"{CACHE}/row_sym.npy").astype(np.int64)
    y_rank = np.load(f"{CACHE}/y_rank.npy")
    n_rows = len(day)

    operands, opnames = build_operands(man, dates)
    n_vars = len(operands)
    print(f"operands={n_vars} T={T} N={N} rows={n_rows} "
          f"dev_days<{DEV_END} ({str(dates[DEV_END])})", flush=True)

    ev = Evaluator(operands, opnames, day, sym, y_rank, T)

    memo = {}
    generated = 0

    def evaluate(tree):
        key = node_str(tree, opnames)
        hit = memo.get(key)
        if hit is None:
            hit = ev.score(tree)
            hit["tree"] = tree
            memo[key] = hit
        return hit

    n_elite = int(POP * ELITE_FRAC)
    n_imm = int(POP * IMMIGRANT_FRAC)
    for isl in range(ISLANDS):
        rng = np.random.default_rng(20260816 + 101 * isl)
        pop = []                                  # ramped half-and-half
        for i in range(POP):
            d = 1 + (i % MAX_DEPTH)
            pop.append(rand_tree(rng, n_vars, d, full=(i % 2 == 0)))
        generated += POP

        for gen in range(GENS + 1):
            scored = [(evaluate(t)["fit"], t) for t in pop]
            scored.sort(key=lambda z: -z[0])
            best = scored[0]
            print(f"isl {isl} gen {gen:2d}  best_fit={best[0]:.4f}  "
                  f"ic_dev={memo[node_str(best[1], opnames)]['ic_dev']:+.4f}"
                  f"  uniq={len(memo)}  {time.time()-t_start:.0f}s  "
                  f"{node_str(best[1], opnames)[:80]}", flush=True)
            if gen == GENS:
                break
            nxt = [t for _, t in scored[:n_elite]]
            nxt += [rand_tree(rng, n_vars, 1 + int(rng.integers(MAX_DEPTH)))
                    for _ in range(n_imm)]
            generated += n_imm

            def tourney(sc=scored):
                cand = [sc[int(i)]
                        for i in rng.integers(len(sc), size=TOURNEY)]
                return max(cand, key=lambda z: z[0])[1]

            while len(nxt) < POP:
                if rng.random() < P_CROSSOVER:
                    child = crossover(rng, tourney(), tourney())
                else:
                    child = mutate(rng, tourney(), n_vars)
                nxt.append(child)
                generated += 1
            pop = nxt

    # ---- screen, dedup, persist -------------------------------------------
    rows = [dict(formula=k, node_count=v["n_nodes"], ic_dev=v["ic_dev"],
                 ic_held=v["ic_held"], ic_ir=v["ic_ir"], cover=v["cover"],
                 fit=v["fit"]) for k, v in memo.items() if v["fit"] > -1.0]
    rows.sort(key=lambda r: -r["fit"])
    passed = sum(1 for r in rows if abs(r["ic_dev"]) >= IC_MIN)
    print(f"evaluated={len(memo)} scorable={len(rows)} passed_ic={passed}",
          flush=True)

    sub = np.sort(np.random.default_rng(7).choice(
        n_rows, min(DEDUP_ROWS, n_rows), replace=False))
    panel_path = f"{CACHE}/factory_panel_values{TAG}.npy"
    store = np.lib.format.open_memmap(panel_path, mode="w+", dtype=np.float32,
                                      shape=(MAX_KEEP, T, N))
    X2 = np.lib.format.open_memmap(f"{CACHE}/X2{TAG}.npy", mode="w+",
                                   dtype=np.float32,
                                   shape=(n_rows, 312 + MAX_KEEP))
    Xb = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    step = 50_000
    for i0 in range(0, n_rows, step):
        X2[i0:i0 + step, :312] = Xb[i0:i0 + step]
    del Xb

    kept_names, kept_meta = [], []
    keep_sig = np.zeros((MAX_KEEP, len(sub)), np.float32)   # z-scored, |.|=1
    ev.cache.clear()
    ev.order.clear()
    floors, fi = [IC_MIN, IC_FLOOR, 0.006], 0
    for r in rows:                       # rows are sorted by fit, descending
        k = len(kept_names)
        if k >= MAX_KEEP:
            break
        # the primary screen holds; it steps down only if dedup has thinned
        # the quota below the target band
        while (r["fit"] < floors[fi] and k < MIN_KEEP
               and fi + 1 < len(floors)):
            fi += 1
        if r["fit"] < floors[fi]:
            break
        P = ev.panel(memo[r["formula"]]["tree"])
        vals = P[day, sym].astype(np.float64)
        xr, _fin = ev.row_ranks(vals)
        s = np.nan_to_num(xr[sub], nan=0.5).astype(np.float32)
        s -= s.mean()
        nrm = float(np.linalg.norm(s))
        if nrm <= 1e-6:
            continue
        s /= nrm
        if k and float(np.abs(keep_sig[:k] @ s).max()) > DEDUP_RHO:
            continue
        keep_sig[k] = s
        store[k] = np.where(np.isfinite(P), P, np.nan).astype(np.float32)
        X2[:, 312 + k] = xr.astype(np.float32)
        kept_names.append(r["formula"])
        kept_meta.append(r)
        if len(kept_names) % 50 == 0:
            print(f"  kept {len(kept_names)}  ({time.time()-t_start:.0f}s)",
                  flush=True)
    k = len(kept_names)
    print(f"after_dedup={k}  ({time.time()-t_start:.0f}s)", flush=True)

    store.flush()
    np.savez(f"{CACHE}/factory_panel{TAG}.npz", panel=store[:k],
             names=np.array(kept_names))
    del store
    os.remove(panel_path)

    keep_set = set(kept_names)
    df = pd.DataFrame(rows)
    df["kept"] = df["formula"].isin(keep_set)
    df = df[["formula", "node_count", "ic_dev", "ic_held", "ic_ir", "cover",
             "kept"]]
    df.to_csv(f"{OUT}/factory_factors{TAG}.csv", index=False)
    funnel = {"generated": int(generated), "evaluated": int(len(memo)),
              "passed_ic": int(passed), "after_dedup": int(k),
              "ic_min": IC_MIN, "ic_floor": IC_FLOOR,
              "dedup_rho": DEDUP_RHO, "dev_end_day": DEV_END,
              "dev_end_date": str(dates[DEV_END]),
              "pop": POP, "gens": GENS, "islands": ISLANDS,
              "immigrant_frac": IMMIGRANT_FRAC, "max_depth": MAX_DEPTH}
    with open(f"{OUT}/factory_funnel{TAG}.json", "w") as fh:
        json.dump(funnel, fh, indent=1)

    # ---- retrain on 312 + factory -----------------------------------------
    ev.cache.clear()
    ev.order.clear()
    del operands, ev.ops, keep_sig
    Xfull = X2[:, :312 + k]
    yres = np.load(f"{CACHE}/y_res.npy")
    preds = np.zeros(n_rows)
    for s in (0, 1):
        m = HistGradientBoostingRegressor(
            learning_rate=0.05, max_leaf_nodes=31, max_iter=10,
            min_samples_leaf=3000, l2_regularization=1.0, random_state=s)
        m.fit(Xfull, yres)
        preds += m.predict(Xfull)
        print(f"  fit seed {s} done ({time.time()-t_start:.0f}s)", flush=True)
    P = np.full((T, N), np.nan, np.float32)
    P[day, sym] = preds / 2
    np.save(f"{CACHE}/factory_score{TAG}.npy", P)
    r_new = run_topk(P, G, O2O, dates)
    r_base = run_topk(np.load(f"{CACHE}/dial_iter10.npy"), G, O2O, dates)

    curve = {"dial": {"max_iter": 10, "min_samples_leaf": 3000},
             "n_features": {"baseline": 312, "factory": 312 + k},
             "baseline_312": win_stats(r_base),
             "factory_312_plus": win_stats(r_new)}
    pd.DataFrame({"baseline_312": (1 + r_base).cumprod(),
                  "factory_312_plus": (1 + r_new).cumprod()}
                 ).to_parquet(f"{OUT}/factory_nav{TAG}.parquet")
    with open(f"{OUT}/factory_curve{TAG}.json", "w") as fh:
        json.dump(curve, fh, indent=1)

    top = sorted(kept_meta, key=lambda z: -abs(z["ic_dev"]))[:12]
    summary = {"funnel": funnel,
               "top12": [{"formula": t["formula"], "nodes": t["node_count"],
                          "ic_dev": round(t["ic_dev"], 4),
                          "ic_held": round(t["ic_held"], 4),
                          "ic_ir": round(t["ic_ir"], 2)} for t in top],
               "curve": curve,
               "runtime_min": round((time.time() - t_start) / 60, 1)}
    print("SUMMARY_JSON " + json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
