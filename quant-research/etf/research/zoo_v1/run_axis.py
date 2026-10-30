"""CLI: run one frozen axis of the zoo, or verify one module.

  run_axis.py <axis>                      axis in AXES | sleeve | flow_h1 |
                                          flow_h3 | flow_q10
  run_axis.py verify --id <module_id>     extra-lag + permutation audit

All numbers flow through harness.evaluate conventions; results land in
results/<axis>.json, the ledger, and module_returns/.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

import harness as H
import zoo_axes as Z

RESULTS = f"{H.ROOT}/results"


def run_cs_axis(P, axis, specs):
    rows = []
    for spec in specs:
        sig = Z.make_signal(P, spec)
        gate = P.liq2() if spec.get("gate") == "v2" else None
        net = H.backtest_cs(P, sig, spec["hold"], spec["q"], spec["domain"],
                            gate=gate)
        st = H.stats(net)
        mid = H.record(axis, spec, st, net)
        rows.append({"id": mid, "spec": spec,
                     "worst_clean": st["worst_clean"],
                     "full": (st.get("full") or {}).get("sharpe")})
    return rows


def run_sleeve_axis(P, axis="sleeve", sl=None):
    rows = []
    for spec, wt in Z.sleeve_specs(P, sl):
        net = H.backtest_sleeve(P, wt, sl)
        st = H.stats(net)
        mid = H.record(axis, spec, st, net)
        rows.append({"id": mid, "spec": spec,
                     "worst_clean": st["worst_clean"],
                     "full": (st.get("full") or {}).get("sharpe")})
    return rows


def verify(module_id: str, perms: int = 50):
    """Extra-lag decay + within-date permutation placebo for one module."""
    spec = None
    with open(H.LEDGER) as fh:
        for line in fh:
            row = json.loads(line)
            if row["id"] == module_id:
                spec = row["spec"]
                axis = row["axis"]
    if spec is None:
        raise SystemExit(f"id not in ledger: {module_id}")
    P = H.Panels()
    if axis.startswith("sleeve"):
        sl = {"sleeve_v2": P.sleeves_v2,
              "sleeve_ix": P.sleeves_ix}.get(axis, lambda: None)()
        for s, wt in Z.sleeve_specs(P, sl):
            if s == spec:
                base = H.stats(H.backtest_sleeve(P, wt, sl))
                lag = H.stats(H.backtest_sleeve(P, wt.shift(1).fillna(0.0), sl))
                out = {"id": module_id, "spec": spec,
                       "full": (base["full"] or {}).get("sharpe"),
                       "lag1_full": (lag["full"] or {}).get("sharpe"),
                       "perm_p": None}
                print(json.dumps(out))
                return
        raise SystemExit("sleeve spec not regenerated")

    sig = Z.make_signal(P, spec)
    base_net = H.backtest_cs(P, sig, spec["hold"], spec["q"], spec["domain"])
    base = H.stats(base_net)
    lag_sig = np.vstack([np.full((1, sig.shape[1]), np.nan), sig[:-1]])
    lag = H.stats(H.backtest_cs(P, lag_sig, spec["hold"], spec["q"],
                                spec["domain"]))

    real = (base["full"] or {}).get("sharpe") or 0.0
    rng = np.random.default_rng(
        int.from_bytes(module_id.encode()[-8:], "little") % (2**32))
    hits = 0
    for _ in range(perms):
        perm = sig.copy()
        for t in range(perm.shape[0]):
            ok = np.isfinite(perm[t])
            if ok.sum() > 1:
                vals = perm[t, ok]
                rng.shuffle(vals)
                perm[t, ok] = vals
        pst = H.stats(H.backtest_cs(P, perm, spec["hold"], spec["q"],
                                    spec["domain"]))
        if ((pst["full"] or {}).get("sharpe") or -9) >= real:
            hits += 1
    out = {"id": module_id, "spec": spec, "full": real,
           "lag1_full": (lag["full"] or {}).get("sharpe"),
           "perm_p": round((hits + 1) / (perms + 1), 4)}
    print(json.dumps(out))


def main():
    if sys.argv[1] == "verify":
        vid = sys.argv[sys.argv.index("--id") + 1]
        perms = (int(sys.argv[sys.argv.index("--perms") + 1])
                 if "--perms" in sys.argv else 50)
        verify(vid, perms)
        return
    axis = sys.argv[1]
    os.makedirs(RESULTS, exist_ok=True)
    P = H.Panels()
    if axis == "sleeve":
        rows = run_sleeve_axis(P)
    elif axis == "sleeve_v2":
        rows = run_sleeve_axis(P, "sleeve_v2", P.sleeves_v2())
    elif axis == "sleeve_ix":
        rows = run_sleeve_axis(P, "sleeve_ix", P.sleeves_ix())
    elif axis == "flow_xa":
        rows = run_cs_axis(P, axis, Z.flow_xa_axis())
    elif axis in ("flow_h1", "flow_h3"):
        rows = run_cs_axis(P, axis, Z.flow_axis(hold=int(axis[-1])))
    elif axis == "flow_q10":
        rows = run_cs_axis(P, axis, Z.flow_axis(hold=5, quantile=10))
    else:
        rows = run_cs_axis(P, axis, Z.AXES[axis])
    with open(f"{RESULTS}/{axis}.json", "w") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    top = sorted([r for r in rows if r["worst_clean"] is not None],
                 key=lambda r: -r["worst_clean"])[:10]
    print(f"axis={axis} runs={len(rows)}")
    for r in top:
        print(f"  {r['id']}  worst={r['worst_clean']:6.2f}  "
              f"full={r['full']}  {json.dumps(r['spec'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
