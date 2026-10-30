"""INC-T3 诊断(描述性,读收益之后加,不进判定):主格剔除面板触及的池内股票日中,基线排雷实际持有的比例与权重,按年。
产出 research/ledger/inc_t3_overlay_touch_by_year.csv。用法 python3 scripts/inc_t3_diag_touch.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import inc_t3_common as C, inc_t3_overlay as OV
X = OV.Ctx()
W0 = X.weights(None)
win = OV.exclusion_windows(X.ev, 0.10, 10, 0, X.T, "main")
E = OV.panel_from_windows(win, X.T, X.N)
UE = X.U & E & X.m[:, None]
held = UE & (W0 > 0)
yrs = X.p.dates.year
rows = []
for y in range(2020, 2026):
    k = yrs == y
    rows.append(dict(year=y, excl_stockdays=int(UE[k].sum()), held_by_base=int(held[k].sum()),
                     share_held=float(held[k].sum() / max(UE[k].sum(), 1)),
                     excl_weight_share=float((W0 * UE)[k].sum(axis=1).mean())))
D = pd.DataFrame(rows)
print(D)
D.to_csv(C.LEDGER / "inc_t3_overlay_touch_by_year.csv", index=False)
