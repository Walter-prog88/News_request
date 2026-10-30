"""Two explainer artefacts for the showcase report.

Chart 1  strategy vs 方案内等权 vs 全池等权 — NAV (log) plus drawdown panel.
         方案内等权 = equal weight buy-and-hold of every ETF the strategy ever
         held. If the strategy only beat the market by picking a good corner
         of the universe, this line catches it.

Chart 2  per-ETF wealth contribution, money-weighted and additive:
             contrib_i = Σ_t  w_{i,t} · r_{i,t+1} · NAV_t
         which reconciles to final NAV − initial capital by construction; the
         reconciliation residual is asserted, matching the old ledger's
         accounting discipline.

Writes results/explainer_{nav,contrib}.csv|json and the SVG fragments the
report inlines.
"""

import json

import numpy as np
import pandas as pd

from sweep_dial import CACHE, COST_SIDE, TOPK, load_env, stats

OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
import sys as _sys
_sys.path.insert(0, "/path/to/user/Desktop/ETF-Research/index project/research/zoo_v1")
from harness import STORE as _STORE
MASTER = (f"{_STORE}/universe/"
          "classification_versions/classification-cb3a08d444b73aaa85ec727a/"
          "symbol_master.parquet")
DIAL = {"max_iter": 10, "min_samples_leaf": 3000}
TOP_N_CHART = 18


def build_stack(X, yres, yrank, day, sym, dates, nsym):
    from sklearn.ensemble import HistGradientBoostingRegressor

    def fit(y):
        p = np.zeros(len(y))
        for s in (0, 1):
            m = HistGradientBoostingRegressor(
                learning_rate=0.05, max_leaf_nodes=31, random_state=s,
                l2_regularization=1.0, **DIAL)
            m.fit(X, y)
            p += m.predict(X)
        P = np.full((len(dates), nsym), np.nan, np.float32)
        P[day, sym] = p / 2
        return P

    P_gbdt = np.load(f"{CACHE}/dial_iter10.npy")
    P_rank = fit(yrank)
    P_mlp = np.load(f"{CACHE}/insample/score_mlp.npy")

    def z(P):
        mu = np.nanmean(P, axis=1, keepdims=True)
        sd = np.nanstd(P, axis=1, keepdims=True)
        return (P - mu) / np.where(sd > 0, sd, np.nan)
    return np.nanmean(np.stack([z(P_gbdt), z(P_rank), z(P_mlp)]), axis=0)


def run_with_ledger(P, G, O2O, dates, nsym):
    """Top-4 run that also records per-symbol money-weighted contribution."""
    w_prev = np.zeros(nsym)
    nav = 1.0
    rets, contrib = [], np.zeros(nsym)
    cost_total = 0.0
    held = np.zeros(nsym, dtype=bool)
    for t in range(len(dates) - 2):
        s = np.where(G[t], P[t], np.nan)
        k = np.where(np.isfinite(s))[0]
        if len(k) >= TOPK:
            top = k[np.argsort(s[k])][-TOPK:]
            w = np.zeros(nsym)
            w[top] = 1.0 / TOPK
        else:
            w = w_prev
        held |= w > 0
        r_t = np.nan_to_num(O2O[t + 1])
        pnl = w * r_t
        contrib += pnl * nav                 # money-weighted, additive
        turn = np.abs(w - w_prev).sum()
        cost = turn * COST_SIDE
        cost_total += cost * nav
        day_ret = float(pnl.sum()) - cost
        rets.append(day_ret)
        nav *= (1 + day_ret)
        w_prev = w
    r = pd.Series(rets, index=pd.DatetimeIndex(dates[2:]))
    return r, contrib, cost_total, held, nav


def eq_weight(mask, G, O2O, dates, rebal=21):
    """Equal-weight book over `mask` names, monthly rebalanced, same cost."""
    nsym = len(mask)
    w_prev = np.zeros(nsym)
    rets = []
    for t in range(len(dates) - 2):
        if t % rebal == 0 or w_prev.sum() == 0:
            elig = np.where(mask & G[t] & np.isfinite(O2O[t + 1]))[0]
            w = np.zeros(nsym)
            if len(elig):
                w[elig] = 1.0 / len(elig)
        else:
            w = w_prev
        turn = np.abs(w - w_prev).sum()
        rets.append(float((w * np.nan_to_num(O2O[t + 1])).sum())
                    - turn * COST_SIDE)
        w_prev = w
    return pd.Series(rets, index=pd.DatetimeIndex(dates[2:]))


def svg_nav(nav: pd.DataFrame, dd: pd.DataFrame) -> str:
    W, H1, H2 = 880, 330, 150
    cols = {"strategy": ("融合主线策略", "var(--up)", 2.6),
            "selected_eq": ("方案内等权(策略持有过的ETF)", "var(--gold)", 1.9),
            "pool_eq": ("全池等权", "var(--dim)", 1.6)}
    step = max(1, len(nav) // 520)
    sub, sdd = nav.iloc[::step], dd.iloc[::step]
    n = len(sub)
    logs = {c: np.log10(sub[c].values) for c in cols}
    y0 = min(v.min() for v in logs.values()) - .05
    y1 = max(v.max() for v in logs.values()) + .1

    def px(i):
        return i / (n - 1) * (W - 130) + 62

    def py(v):
        return H1 - 28 - (v - y0) / (y1 - y0) * (H1 - 46)
    body = []
    for g in range(int(np.floor(y0)), int(np.ceil(y1)) + 1):
        gy = py(g)
        if 16 < gy < H1 - 16:
            body.append(f'<line x1="62" x2="{W-68}" y1="{gy:.0f}" y2="{gy:.0f}"'
                        f' stroke="var(--hair)"/>'
                        f'<text x="54" y="{gy+4:.0f}" text-anchor="end" '
                        f'fill="var(--dim)" font-size="11">{10**g:g}x</text>')
    for c, (lab, col, sw) in cols.items():
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}"
                       for i, v in enumerate(logs[c]))
        body.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
                    f'stroke-width="{sw}"/>')
        body.append(f'<text x="{W-62}" y="{py(logs[c][-1]):.0f}" '
                    f'fill="{col}" font-size="12">{sub[c].iloc[-1]:.1f}x</text>')
    leg = "".join(f'<tspan fill="{v[1]}">■</tspan>'
                  f'<tspan fill="var(--muted)"> {v[0]}   </tspan>'
                  for v in cols.values())
    body.append(f'<text x="62" y="15" font-size="12">{leg}</text>')

    d0 = float(sdd.min().min())

    def dy(v):
        return 24 + (v / d0) * (H2 - 46) if d0 else 24
    body2 = [f'<text x="62" y="14" font-size="11" fill="var(--muted)">'
             f'回撤</text>']
    for c, (lab, col, sw) in cols.items():
        pts = " ".join(f"{px(i):.1f},{dy(v):.1f}"
                       for i, v in enumerate(sdd[c].values))
        body2.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
                     f'stroke-width="1.4" opacity=".85"/>')
    for frac in (0.5, 1.0):
        gy = dy(d0 * frac)
        body2.append(f'<line x1="62" x2="{W-68}" y1="{gy:.0f}" y2="{gy:.0f}" '
                     f'stroke="var(--hair)"/><text x="54" y="{gy+4:.0f}" '
                     f'text-anchor="end" fill="var(--dim)" font-size="11">'
                     f'{d0*frac*100:.0f}%</text>')
    years = pd.DatetimeIndex(sub.index).year
    for yr in sorted(set(years))[1:]:
        i = int(np.argmax(years == yr))
        body2.append(f'<text x="{px(i):.0f}" y="{H2-6}" text-anchor="middle" '
                     f'fill="var(--dim)" font-size="11">{yr}</text>')
    return (f'<svg viewBox="0 0 {W} {H1}" role="img">{"".join(body)}</svg>'
            f'<svg viewBox="0 0 {W} {H2}" role="img">{"".join(body2)}</svg>')


def svg_contrib(rows) -> str:
    W, rh = 880, 26
    H = len(rows) * rh + 26
    vmax = max(abs(r["pct"]) for r in rows)
    zero = 430
    span = W - zero - 90
    out = []
    for i, r in enumerate(rows):
        y = 12 + i * rh
        bw = abs(r["pct"]) / vmax * span
        pos = r["pct"] >= 0
        x = zero if pos else zero - bw
        col = "var(--up)" if pos else "var(--down)"
        out.append(
            f'<text x="{zero-12}" y="{y+16}" text-anchor="end" font-size="12" '
            f'fill="var(--ink)">{r["label"]}</text>'
            f'<rect x="{x:.0f}" y="{y+4}" width="{max(bw,1):.0f}" height="15" '
            f'fill="{col}" rx="2"/>'
            f'<text x="{x+bw+8 if pos else x-8:.0f}" y="{y+16}" font-size="11" '
            f'text-anchor="{"start" if pos else "end"}" fill="var(--muted)">'
            f'{r["pct"]:+.0f}pp</text>')
    out.append(f'<line x1="{zero}" x2="{zero}" y1="6" y2="{H-8}" '
               f'stroke="var(--hair)"/>')
    return f'<svg viewBox="0 0 {W} {H}" role="img">{"".join(out)}</svg>'


def main():
    man, dates, G, O2O = load_env()
    syms = man["symbols"]
    nsym = len(syms)
    X = np.asarray(np.load(f"{CACHE}/X.npy", mmap_mode="r"))
    yres = np.load(f"{CACHE}/y_res.npy")
    yrank = np.load(f"{CACHE}/y_rank.npy")
    day = np.load(f"{CACHE}/row_day.npy")
    sym = np.load(f"{CACHE}/row_sym.npy")

    P = build_stack(X, yres, yrank, day, sym, dates, nsym)
    r, contrib, cost_total, held, nav_end = run_with_ledger(
        P, G, O2O, dates, nsym)

    resid = (contrib.sum() - cost_total) - (nav_end - 1.0)
    print(f"reconciliation residual: {resid:.3e}")
    assert abs(resid) < 1e-6, "contribution ledger does not reconcile"

    pool = np.ones(nsym, dtype=bool)
    curves = pd.DataFrame({
        "strategy": r,
        "selected_eq": eq_weight(held, G, O2O, dates),
        "pool_eq": eq_weight(pool, G, O2O, dates)})
    nav = (1 + curves.fillna(0)).cumprod()
    dd = nav / nav.cummax() - 1
    nav.to_parquet(f"{OUT}/explainer_nav.parquet")

    m = pd.read_parquet(MASTER).set_index("symbol")
    tot_pos = contrib[contrib > 0].sum()
    rows = []
    for i in np.argsort(-np.abs(contrib)):
        if contrib[i] == 0:
            continue
        s = syms[i]
        nm = (m.at[s, "tracked_index_name"] if s in m.index else None) or (
            m.at[s, "name"] if s in m.index else s)
        rows.append({"symbol": s, "label": f"{nm}  {s.split('.')[0]}",
                     "contrib_rmb_per_1m": float(contrib[i] * 1e6),
                     "pct": float(contrib[i] * 100),
                     "share_of_positive_%": float(
                         100 * contrib[i] / tot_pos) if tot_pos else 0.0})
    pd.DataFrame(rows).to_csv(f"{OUT}/explainer_contrib.csv", index=False)

    summary = {
        "windows": {k: stats(curves[k].dropna()) for k in curves},
        "n_names_ever_held": int(held.sum()),
        "cost_total_pct_of_initial": round(float(cost_total * 100), 2),
        "reconciliation_residual": float(resid),
        "top10_share_of_positive_%": round(
            float(sum(r["share_of_positive_%"] for r in rows[:10])), 1),
        "final_nav": {k: float(nav[k].iloc[-1]) for k in nav},
    }
    with open(f"{OUT}/explainer_summary.json", "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    with open(f"{OUT}/explainer_charts.svg", "w") as fh:
        fh.write(svg_nav(nav, dd) + "\n<!--SPLIT-->\n"
                 + svg_contrib(rows[:TOP_N_CHART]))
    print(json.dumps(summary, ensure_ascii=False, indent=1)[:900])


if __name__ == "__main__":
    main()
