"""INC-T4 步骤 1:按 run_midband_composite.py::_select 原规则复现弱因子集合(§3.2),计数门槛不过即停。

产出:cache/_tmp/inc_t4_members.json(三集 + 三个 D′,含方向)、research/ledger/inc_t4_members.csv。
本步骤只读既有 IC 账本(research/ic_series),不计算任何收益 / IC。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t4_common as C  # noqa: E402
from afactor.config import LEDGER, TMP  # noqa: E402
from afactor.fast_eval import IC_DIR  # noqa: E402
from afactor import factors as FA  # noqa: E402


def _select(ics, lo, hi, yrs):
    """与 scripts/run_midband_composite.py::_select 逐字相同。"""
    a, b = yrs
    out = {}
    for n, s in ics.items():
        w = s[(s.index.year >= a) & (s.index.year <= b)].dropna()
        if len(w) < C.MIN_DAYS:
            continue
        m, sd = w.mean(), w.std(ddof=1)
        if sd <= 0:
            continue
        t = m / sd * np.sqrt(len(w))
        by_year = w.groupby(w.index.year).mean()
        if len(by_year) < 2 or not (np.sign(by_year) == np.sign(m)).all():
            continue
        if lo <= abs(m) < hi:
            out[n] = (float(m), float(t), float(np.sign(m)))
    return out


def main():
    C.check_prereg()
    FA.build_all()
    reg = set(FA.REGISTRY)
    names = [f.stem.replace("__open", "") for f in IC_DIR.glob("*__open.parquet")]
    names = [n for n in names if n in reg and FA.REGISTRY[n][0] not in ("control", "composite")]
    ics = {}
    for n in sorted(names):
        s = pd.read_parquet(IC_DIR / f"{n}__open.parquet")
        ics[n] = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    print(f"载入 IC 序列 {len(ics)} 条(预注册:1870)", flush=True)
    assert len(ics) == 1870, len(ics)
    comp = json.load(open(LEDGER / "comp_v5_members.json"))
    comp_m = set(n for n in comp["members"] if n in FA.REGISTRY)
    assert len(comp_m) == 24, len(comp_m)

    out, rows = {}, []
    for s, yrs in C.SPLITS.items():
        D = _select(ics, C.MID_LO, C.MID_HI, yrs)
        med = float(np.median([abs(v[0]) for v in D.values()]))
        print(f"D_{s}: 选择窗 {yrs} 成员 {len(D)}(应 {C.EXPECT_N[s]}),|IC| 中位 {med:.4f}(应 {C.EXPECT_MED[s]})",
              flush=True)
        if len(D) != C.EXPECT_N[s] or round(med, 4) != C.EXPECT_MED[s]:
            raise SystemExit(f"停止规则 §11.1:D_{s} 计数/中位数不符")
        nopead = {n: v for n, v in D.items() if FA.REGISTRY[n][0] != "pead"}
        if len(nopead) != C.EXPECT_NOPEAD[s]:
            raise SystemExit(f"D_{s}′ 计数 {len(nopead)} ≠ {C.EXPECT_NOPEAD[s]}")
        out[s] = {n: v[2] for n, v in sorted(D.items())}
        out[f"{s}_nopead"] = {n: v[2] for n, v in sorted(nopead.items())}
        fam = pd.Series({n: FA.REGISTRY[n][0] for n in D}).value_counts()
        print("   family:", dict(fam), flush=True)
        print(f"   pead 成员:{sorted(n for n in D if FA.REGISTRY[n][0] == 'pead')};"
              f" 与 COMP_v5 重叠 {len(set(D) & comp_m)}:{sorted(set(D) & comp_m)}", flush=True)
        for n, (m, t, sg) in sorted(D.items()):
            rows.append(dict(set=f"D_{s}", name=n, family=FA.REGISTRY[n][0], ic_mean_sel=m, ic_t_sel=t, sign=int(sg),
                             in_comp_v5=n in comp_m, in_nopead=n in nopead,
                             sel_window=f"{yrs[0]}-{yrs[1]}"))
    # 方向冲突检查
    signs = {}
    for s in C.SPLITS:
        for n, sg in out[s].items():
            signs.setdefault(n, set()).add(sg)
    conflict = [n for n, v in signs.items() if len(v) > 1]
    print(f"并集 {len(signs)} 个(预注册 636),方向冲突 {len(conflict)}", flush=True)
    assert len(signs) == 636 and not conflict
    ov = {f"{a}&{b}": len(set(out[a]) & set(out[b])) for a, b in (("A", "main"), ("A", "B"), ("main", "B"))}
    print("两两重叠:", ov, flush=True)

    f = TMP / f"{C.PFX}_members.json"
    json.dump(dict(prereg_sha=C.PREREG_SHA, sets=out, overlap=ov, union=len(signs)), open(f, "w"),
              ensure_ascii=False, indent=1)
    pd.DataFrame(rows).to_csv(LEDGER / f"{C.PFX}_members.csv", index=False)
    print(f"members sha256 = {C.sha256_file(f)}", flush=True)
    print(f"members.csv sha256 = {C.sha256_file(LEDGER / f'{C.PFX}_members.csv')}", flush=True)


if __name__ == "__main__":
    main()
