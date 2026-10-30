"""Deflated Sharpe: does anything survive the search's own selection pressure?

Bailey & Lopez de Prado: after N trials on T observations, the expected
maximum Sharpe of pure noise is

  E[max SR_daily] ~ (1-g)*z(1-1/N) + g*z(1-1/(N*e)),  g = Euler-Mascheroni

Report each headline curve against two denominators:
  N_full   every ledger trial + composite variant (fully conservative)
  N_family one per mechanical family (trials inside a family are twins)
"""

import glob
import json
import math

import numpy as np
import pandas as pd
from scipy import stats as sps

import harness as H
import walkforward as W

GAMMA = 0.5772156649


def expected_max_sr(n: int, t: int) -> float:
    z1 = sps.norm.ppf(1 - 1.0 / n)
    z2 = sps.norm.ppf(1 - 1.0 / (n * math.e))
    return ((1 - GAMMA) * z1 + GAMMA * z2) / math.sqrt(t)


def psr(sr_daily: float, hurdle_daily: float, t: int) -> float:
    se = math.sqrt((1 + 0.5 * sr_daily**2) / t)
    return float(sps.norm.cdf((sr_daily - hurdle_daily) / se))


def main():
    n_trials = sum(1 for _ in open(H.LEDGER))
    n_composites = len(glob.glob(f"{H.ROOT}/composite/*.parquet"))
    mods = W.load_modules()
    n_family = len({W.family_key(m) for m in mods.columns})
    n_full = n_trials + n_composites

    targets = {
        "v_final": f"{H.ROOT}/composite/v_final.parquet",
        "v_registry": f"{H.ROOT}/composite/v_registry.parquet",
        "核心单族 sleeve_v2.tsmom_vol(w252)":
            f"{H.MODDIR}/sleeve_v2_6facec6627.parquet",
    }
    out = {}
    for name, path in targets.items():
        r = pd.read_parquet(path)["net"].dropna()
        t = len(r)
        sr_d = float(r.mean() / r.std())
        row = {"sharpe_ann": round(sr_d * math.sqrt(252), 3), "T": t}
        for label, n in [("N_full", n_full), ("N_family", n_family)]:
            h = expected_max_sr(n, t)
            row[label] = {
                "N": n,
                "噪音最大值期望(年化)": round(h * math.sqrt(252), 3),
                "PSR超过噪音最大值": round(psr(sr_d, h, t), 3),
            }
        out[name] = row
    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open(f"{H.ROOT}/composite/deflation.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
