"""Causal walk-forward selection over every module in module_returns/.

The selector is the strategy. Meta-parameters are the four frozen variants
from README.md; the primary is (trail=252, thr=0.3), declared before any
composite number was seen. Losers stay in the pool on purpose: if the
selector picks noise, the bleed shows up in the curve, which is the honest
price of exploration.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

import harness as H

VARIANTS = [(126, 0.0), (126, 0.3), (252, 0.0), (252, 0.3)]
PRIMARY = (252, 0.3)
REB = 5
CORR_CAP = 0.85
W_CAP = 0.25
META_COST_BP = 10.0
MIN_OBS = 400


def load_modules() -> pd.DataFrame:
    frames = {}
    for f in glob.glob(f"{H.MODDIR}/*.parquet"):
        mid = os.path.basename(f)[:-8]
        s = pd.read_parquet(f)["net"]
        if s.notna().sum() >= MIN_OBS:
            frames[mid] = s
    return pd.DataFrame(frames)


def compose(mods: pd.DataFrame, trail: int, thr: float):
    dates = mods.index
    w_hist = pd.DataFrame(0.0, index=dates, columns=mods.columns)
    current = pd.Series(0.0, index=mods.columns)
    for i in range(trail, len(dates), REB):
        past = mods.iloc[max(0, i - trail): i]
        enough = past.notna().sum() >= min(126, trail)
        mu, sd = past.mean(), past.std()
        ts = (mu / sd * np.sqrt(252)).where(enough & (sd > 0))
        elig = ts[ts > thr].sort_values(ascending=False)
        chosen = []
        if len(elig):
            corr = past[elig.index].corr()
            for m in elig.index:
                if all(abs(corr.loc[m, c]) < CORR_CAP for c in chosen):
                    chosen.append(m)
        current = pd.Series(0.0, index=mods.columns)
        if chosen:
            iv = 1.0 / sd[chosen]
            w = (iv / iv.sum()).clip(upper=W_CAP)
            current[chosen] = (w / w.sum()).values
        w_hist.iloc[i: i + REB] = current.values
    gross = (w_hist * mods.fillna(0.0)).sum(axis=1)
    turn = w_hist.diff().abs().sum(axis=1).fillna(0.0)
    return gross - turn * META_COST_BP / 1e4, w_hist


def compose_v2(mods: pd.DataFrame, trail: int = 252, thr_in: float = 0.5,
               thr_out: float = 0.0, top_k: int = 12, smooth: float = 0.5,
               reb: int = 21):
    """v2 selector (README amendment 2026-08-15, after the v1 diagnosis).

    v1 failed for mechanical reasons measured in the ledger: 44% of the book
    replaced every 5 sessions (1.9%/yr meta cost on a 2.8%-vol book) and the
    good modules diluted to 3% weights across ~33 holdings. v2 fixes the
    mechanism: monthly rebalance, entry/exit hysteresis, a top-K cap, and
    weight smoothing. Parameters are declared once, not tuned.
    """
    dates = mods.index
    w_hist = pd.DataFrame(0.0, index=dates, columns=mods.columns)
    current = pd.Series(0.0, index=mods.columns)
    holding: list = []
    for i in range(trail, len(dates), reb):
        past = mods.iloc[max(0, i - trail): i]
        enough = past.notna().sum() >= min(126, trail)
        mu, sd = past.mean(), past.std()
        ts = (mu / sd * np.sqrt(252)).where(enough & (sd > 0))
        # keep incumbents above the exit bar, then fill with new entries
        keep = [m for m in holding if (ts.get(m) or -9) > thr_out]
        cand = ts[ts > thr_in].sort_values(ascending=False)
        corr = past.corr() if len(cand) else None
        for m in cand.index:
            if len(keep) >= top_k:
                break
            if m in keep:
                continue
            if all(abs(corr.loc[m, c]) < CORR_CAP for c in keep if c in corr):
                keep.append(m)
        holding = keep[:top_k]
        target = pd.Series(0.0, index=mods.columns)
        if holding:
            iv = 1.0 / sd[holding]
            w = (iv / iv.sum()).clip(upper=W_CAP)
            target[holding] = (w / w.sum()).values
        current = smooth * current + (1 - smooth) * target
        w_hist.iloc[i: i + reb] = current.values
    gross = (w_hist * mods.fillna(0.0)).sum(axis=1)
    turn = w_hist.diff().abs().sum(axis=1).fillna(0.0)
    return gross - turn * META_COST_BP / 1e4, w_hist


V2_VARIANTS = {
    "v2_primary": {},                       # declared default
    "v2_k8": {"top_k": 8},
    "v2_k16": {"top_k": 16},
    "v2_in03": {"thr_in": 0.3},
}


# ---------------------------------------------------------------- v3: family

def family_key(mid: str) -> str:
    """Mechanical family id from the ledger spec — no performance fields.

    Collapses window/holding/liq-gate variants of one construction into one
    family, so the selector picks information sources, not parameter twins.
    """
    import re
    with open(f"{H.ROOT}/trial_ledger.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            if row["id"] == mid:
                spec, axis = row["spec"], row["axis"]
                break
        else:
            return mid
    if axis.startswith("sleeve"):
        return f"{axis}.{spec['kind']}"
    if spec["kind"] == "flow":
        base = re.match(r"((?:prev|same)_[a-z0-9]+?)_", spec["name"] + "_")
        fam = base.group(1) if base else spec["name"]
        return f"{axis}.{fam}.q{spec['q']}"
    col = spec.get("col") or spec.get("kind")
    return (f"{axis}.{spec['kind']}.{col}.s{int(spec.get('sign', 1))}"
            f".{spec.get('domain', '')}")


def build_families(mods: pd.DataFrame) -> pd.DataFrame:
    groups: dict = {}
    for mid in mods.columns:
        groups.setdefault(family_key(mid), []).append(mid)
    fam = pd.DataFrame({k: mods[v].mean(axis=1) for k, v in groups.items()})
    return fam


def compose_v3(fams: pd.DataFrame, trail: int = 252, thr_in: float = 0.5,
               exit_trail: int = 126, top_k: int = 8, reb: int = 21):
    """v3 (README amendment): family-level selection, fast exit, no smoothing.

    Entry: trailing-252 Sharpe > 0.5, capped at top_k, corr gate 0.85.
    Exit: trailing-126 Sharpe < 0 — a family that stops paying is cut within
    months, which is what v2's slow exit failed to do in 2022.
    """
    dates = fams.index
    w_hist = pd.DataFrame(0.0, index=dates, columns=fams.columns)
    holding: list = []
    for i in range(trail, len(dates), reb):
        past = fams.iloc[max(0, i - trail): i]
        recent = fams.iloc[max(0, i - exit_trail): i]
        enough = past.notna().sum() >= min(126, trail)
        ts = (past.mean() / past.std() * np.sqrt(252)).where(
            enough & (past.std() > 0))
        ts_fast = recent.mean() / recent.std().replace(0, np.nan) * np.sqrt(252)
        holding = [m for m in holding if (ts_fast.get(m) or -9) > 0]
        cand = ts[ts > thr_in].sort_values(ascending=False)
        corr = past.corr() if len(cand) else None
        for m in cand.index:
            if len(holding) >= top_k:
                break
            if m in holding:
                continue
            if all(abs(corr.loc[m, c]) < CORR_CAP
                   for c in holding if c in corr):
                holding.append(m)
        target = pd.Series(0.0, index=fams.columns)
        if holding:
            iv = 1.0 / past.std()[holding]
            w = (iv / iv.sum()).clip(upper=W_CAP)
            target[holding] = (w / w.sum()).values
        w_hist.iloc[i: i + reb] = target.values
    gross = (w_hist * fams.fillna(0.0)).sum(axis=1)
    turn = w_hist.diff().abs().sum(axis=1).fillna(0.0)
    return gross - turn * META_COST_BP / 1e4, w_hist


V3_VARIANTS = {
    "v3_primary": {},
    "v3_k12": {"top_k": 12},
    "v3_in03": {"thr_in": 0.3},
    "v3_exit63": {"exit_trail": 63},
}


# ----------------------------------------------------- structural composite

# Membership rule (mechanical, README amendment): a family enters iff it
# contains >= 1 module that PASSED verification (extra-lag decay graceful,
# permutation p < 0.1 where applicable). Where the dom/all twins of one
# construction both pass, the dom one is kept. Fixed thereafter — the three
# selector generations (v1/v2/v3) measured dynamic selection skill at ~zero,
# so the composite holds structure, not trailing performance.
VERIFIED_PASS = [
    "flow_q10_9940c331ed", "flow_q10_004fb4e98b", "flow_q10_27ed035e4b",
    "flow_q10_d98d55d6da", "mom_53a47a6d64", "shape_381f643493",
    "onid_0a19f3a1ad",
    # gated (v2) sleeve modules replace the ungated ones: the v1 sleeve
    # numbers carried ~25% untradeable-name inflation, closed in round B
    "sleeve_v2_fa9abd7fff", "sleeve_v2_6facec6627", "sleeve_v2_5a01f0df62",
    "sleeve_v2_b3c880c202",
]
DROP_TWINS = {"mom.mom.mom.s1.all"}


def compose_structural(fams: pd.DataFrame, members: list,
                       vol_win: int = 63, reb: int = 21):
    """Fixed membership, causal risk parity, monthly rebalance."""
    sub = fams[members]
    vol = sub.rolling(vol_win, min_periods=30).std()
    w = pd.DataFrame(0.0, index=sub.index, columns=members)
    for i in range(vol_win, len(sub), reb):
        v = vol.iloc[i - 1]
        iv = (1.0 / v).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if iv.sum() > 0:
            w.iloc[i: i + reb] = (iv / iv.sum()).values
    gross = (w * sub.fillna(0.0)).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    return gross - turn * META_COST_BP / 1e4, w


def structural_members(fams: pd.DataFrame) -> list:
    keys = {family_key(mid) for mid in VERIFIED_PASS}
    keys -= DROP_TWINS
    return sorted(k for k in keys if k in fams.columns)


def main():
    mods = load_modules()
    print(f"modules in pool: {mods.shape[1]}")
    outdir = f"{H.ROOT}/composite"
    os.makedirs(outdir, exist_ok=True)
    report = {}
    for trail, thr in VARIANTS:
        name = f"trail{trail}_thr{thr}"
        net, w_hist = compose(mods, trail, thr)
        st = H.stats(net)
        report[name] = {"primary": False, "stats": st}
        net.rename("net").to_frame().to_parquet(f"{outdir}/{name}.parquet")
        share = w_hist.mean().sort_values(ascending=False)
        report[name]["top_weights"] = {
            k: round(float(v), 4) for k, v in share.head(15).items() if v > 0}
    for name, kw in V2_VARIANTS.items():
        net, w_hist = compose_v2(mods, **kw)
        st = H.stats(net)
        report[name] = {"primary": False, "stats": st}
        net.rename("net").to_frame().to_parquet(f"{outdir}/{name}.parquet")
        share = w_hist.mean().sort_values(ascending=False)
        report[name]["top_weights"] = {
            k: round(float(v), 4) for k, v in share.head(15).items() if v > 0}
    fams = build_families(mods)
    print(f"families: {fams.shape[1]}")
    fams.to_parquet(f"{outdir}/families.parquet")
    for name, kw in V3_VARIANTS.items():
        net, w_hist = compose_v3(fams, **kw)
        st = H.stats(net)
        report[name] = {"primary": False, "stats": st}
        net.rename("net").to_frame().to_parquet(f"{outdir}/{name}.parquet")
        share = w_hist.mean().sort_values(ascending=False)
        report[name]["top_weights"] = {
            k: round(float(v), 4) for k, v in share.head(15).items() if v > 0}

    # structural composite: fixed verified membership, causal risk parity
    members = structural_members(fams)
    print("structural members:", members)
    net, w_hist = compose_structural(fams, members)
    st = H.stats(net)
    report["v_struct"] = {"primary": True, "stats": st, "members": members,
                          "avg_weights": {k: round(float(v), 4) for k, v in
                                          w_hist.mean().items() if v > 0}}
    net.rename("net").to_frame().to_parquet(f"{outdir}/v_struct.parquet")
    # leave-one-out robustness instead of selection
    loo = {}
    for drop in members:
        net_d, _ = compose_structural(fams, [m for m in members if m != drop])
        loo[drop] = H.stats(net_d)["worst_clean"]
    report["v_struct"]["leave_one_out_worst_clean"] = loo

    # registry composite: causal risk parity DIRECTLY over the verified
    # modules. Caveat (disclosed in the report): verification targets were
    # drawn from the top of the clean-window ranking, so membership embeds
    # one layer of performance selection; the sealed smoke window and the
    # forward paper ledger are the only unbiased judges of this curve.
    reg = [m for m in VERIFIED_PASS if m in mods.columns]
    net, w_hist = compose_structural(mods, reg)
    st = H.stats(net)
    report["v_registry"] = {
        "primary": False, "stats": st, "members": reg,
        "avg_weights": {k: round(float(v), 4) for k, v in
                        w_hist.mean().items() if v > 0}}
    net.rename("net").to_frame().to_parquet(f"{outdir}/v_registry.parquet")
    loo = {}
    for drop in reg:
        net_d, _ = compose_structural(mods, [m for m in reg if m != drop])
        loo[drop] = H.stats(net_d)["worst_clean"]
    report["v_registry"]["leave_one_out_worst_clean"] = loo
    corr = mods[reg].corr().round(3)
    corr.to_csv(f"{outdir}/registry_corr.csv")
    report["v_registry"]["max_abs_offdiag_corr"] = float(
        corr.where(~np.eye(len(reg), dtype=bool)).abs().max().max())

    # v_final: the P3 rule as pre-registered — ONE weight per information
    # source (verified modules averaged within family first), risk parity
    # across legs. This deduplicates the flow twins the registry carried.
    groups: dict = {}
    for mid in reg:
        groups.setdefault(family_key(mid), []).append(mid)
    legs = pd.DataFrame({k: mods[v].mean(axis=1) for k, v in groups.items()})
    net, w_hist = compose_structural(legs, list(legs.columns))
    st = H.stats(net)
    lcorr = legs.corr().round(3)
    lcorr.to_csv(f"{outdir}/final_leg_corr.csv")
    report["v_final"] = {
        "primary": False, "stats": st,
        "legs": {k: v for k, v in groups.items()},
        "avg_weights": {k: round(float(v), 4) for k, v in
                        w_hist.mean().items() if v > 0},
        "max_abs_offdiag_corr": float(
            lcorr.where(~np.eye(len(legs.columns), dtype=bool))
            .abs().max().max())}
    net.rename("net").to_frame().to_parquet(f"{outdir}/v_final.parquet")
    loo = {}
    for drop in legs.columns:
        net_d, _ = compose_structural(
            legs, [m for m in legs.columns if m != drop])
        loo[drop] = H.stats(net_d)["worst_clean"]
    report["v_final"]["leave_one_out_worst_clean"] = loo
    with open(f"{outdir}/summary.json", "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    for name, r in report.items():
        tag = " *PRIMARY*" if r["primary"] else ""
        st = r["stats"]
        row = {k: (st[k] or {}).get("sharpe") for k in
               ("y19_21", "y22_23", "y24_26", "pre24", "full")}
        print(f"{name}{tag}: {row}  worst_clean={st['worst_clean']}")


if __name__ == "__main__":
    main()
