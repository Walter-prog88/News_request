"""THE one-shot smoke test on the sealed window (2026-05-22 .. 2026-07-24).

Runs with ZOO_UNSEAL=1. Only the six legs whose signals rebuild from live
daily data are smoked; the three flow legs need the upstream minute factor
pipeline and are excluded (disclosed operational gap). 42 sessions decide
nothing statistically — this is a did-it-blow-up check, nothing more.
"""

import json
import os

import numpy as np
import pandas as pd

os.environ["ZOO_UNSEAL"] = "1"

import harness as H
import zoo_axes as Z

SMOKE_LO, SMOKE_HI = "2026-05-22", "2026-07-24"
LOCK = f"{H.ROOT}/composite/smoke_used.lock"

LEGS_CS = {
    "mom.252skip21.dom.h21": {"kind": "mom", "w": 252, "skip": 21,
                              "domain": "dom", "hold": 21, "q": 5},
    "shape.hightime21.all.h21": {"kind": "rollmean",
                                 "col": "high_time_ordinal_continuous",
                                 "w": 21, "domain": "all", "hold": 21, "q": 5},
    "onid.overnight63.all.h21": {"kind": "rollsum",
                                 "col": "overnight_ret_continuous",
                                 "w": 63, "sign": -1, "domain": "all",
                                 "hold": 21, "q": 5},
}
SLEEVE_KINDS = {"tsmom_vol_63": {"kind": "tsmom_vol", "w": 63},
                "tsmom_vol_252": {"kind": "tsmom_vol", "w": 252},
                "risk_parity": {"kind": "risk_parity_all"},
                "month_turn": {"kind": "month_turn"}}


def main():
    if os.path.exists(LOCK):
        raise SystemExit("smoke window already spent — refusing to run again")
    P = H.Panels()
    assert not P.sealed
    legs = {}
    for name, spec in LEGS_CS.items():
        sig = Z.make_signal(P, spec)
        legs[name] = H.backtest_cs(P, sig, spec["hold"], spec["q"],
                                   spec["domain"], gate=P.liq2())
    sl = P.sleeves_v2()
    for name, want in SLEEVE_KINDS.items():
        for spec, wt in Z.sleeve_specs(P, sl):
            if spec == want:
                legs[name] = H.backtest_sleeve(P, wt)
    df = pd.DataFrame(legs)

    # causal risk parity, same construction as compose_structural
    vol = df.rolling(63, min_periods=30).std()
    w = pd.DataFrame(0.0, index=df.index, columns=df.columns)
    for i in range(63, len(df), 21):
        iv = (1.0 / vol.iloc[i - 1]).replace([np.inf, -np.inf],
                                             np.nan).fillna(0.0)
        if iv.sum() > 0:
            w.iloc[i: i + 21] = (iv / iv.sum()).values
    comp = ((w * df.fillna(0.0)).sum(axis=1)
            - w.diff().abs().sum(axis=1).fillna(0.0) * 10 / 1e4)

    out = {}
    for name in list(df.columns) + ["composite_ex_flow"]:
        r = (comp if name == "composite_ex_flow" else df[name])
        r = r.loc[SMOKE_LO:SMOKE_HI].dropna()
        out[name] = {
            "n": len(r),
            "cum_ret_%": round(float((1 + r).prod() - 1) * 100, 2),
            "sharpe_ann": (round(float(r.mean() / r.std() * np.sqrt(252)), 2)
                           if len(r) > 5 and r.std() > 0 else None),
            "worst_day_%": round(float(r.min()) * 100, 2) if len(r) else None,
        }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open(f"{H.ROOT}/composite/smoke_result.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    with open(LOCK, "w") as fh:
        fh.write("spent\n")


if __name__ == "__main__":
    main()
