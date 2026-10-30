"""验证"曾创年内新高 + 回调 + 反抽"规则(用户假设,2026-08-28)。

与之前 R1-R7 的关键区别:不用"最近涨多少"(120日涨幅),而用**高点的时间位置**
—— 一年内创过 252 日新高、且高点是近期的。二者在数据上几乎正交
(如英维克 120日涨幅 -41.7% 全池 7.2 分位,但 83 天前刚创年内新高)。

主证据窗口 2024-09-24~2025-12-31(规则来自 2026 买入的取证 -> 2026 只作参考)。
口径:T 日收盘信号 -> T+1 开盘买 -> T+1+h 卖;相对同日全池等权。
"""
import os, sys, glob, warnings; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.config import DAILY, SEALED
pd.set_option("display.width", 300)

def load_recent(col):
    rows = []
    for f in glob.glob("cache/recent_daily/parts/*.csv"):
        if os.path.getsize(f) < 50: continue
        d = pd.read_csv(f)
        if len(d) == 0 or col not in d.columns: continue
        d = d[d.tradestatus == 1]; rows.append(d[["date", "code", col]])
    r = pd.concat(rows, ignore_index=True)
    r["ts"] = r["code"].str.upper().str.replace(r"^(SH|SZ)\.(\d+)$", r"\2.\1", regex=True)
    r["date"] = pd.to_datetime(r["date"]); r[col] = pd.to_numeric(r[col], errors="coerce")
    return r.pivot(index="date", columns="ts", values=col)
def cat(name, col):
    a = pd.concat([pd.read_parquet(DAILY / f"{name}.parquet"), pd.read_parquet(SEALED / f"{name}.parquet")])
    rec = load_recent(col); both = a.columns.union(rec.columns)
    return pd.concat([a.reindex(columns=both), rec.reindex(columns=both)]).sort_index()

C = cat("close_px", "close"); A = cat("amount", "amount")
TW = pd.concat([pd.read_parquet(DAILY / "twap_open10.parquet"), pd.read_parquet(SEALED / "twap_open10.parquet")])
TW = TW.reindex(index=C.index, columns=C.columns).fillna(C)
adv = A.rolling(20, min_periods=10).mean(); pool = ((adv >= 2e7) & C.notna()).fillna(False)
rk = lambda X: X.where(pool).rank(axis=1, pct=True)

hi252 = C.rolling(252, min_periods=200).max()
is_newhigh = (C >= hi252 - 1e-9) & pool                       # 当日创 252 日新高
dist_hi = C / hi252 - 1.0                                     # 距 252 日高点(<=0)
# 高点距今日数:最近一次创新高距今多少个交易日
age = pd.DataFrame(np.nan, index=C.index, columns=C.columns)
last = np.full(C.shape[1], np.nan)
NH = is_newhigh.to_numpy(); 
for i in range(len(C)):
    last = np.where(NH[i], i, last)
    age.iloc[i] = i - last
r5 = rk(C / C.shift(5) - 1); r20 = rk(C / C.shift(20) - 1); r120 = rk(C / C.shift(120) - 1)
liq = rk(adv); yr = C / C.shift(252) - 1

FRESH = {k: (age <= k) & (age >= 1) for k in (40, 60, 90, 120)}
RULES = {
 "R0 全池":                                   pool,
 "N1 一年内创过新高(高点<=60日)":               FRESH[60],
 "N2 N1 + 回落 15%~55%":                      FRESH[60] & (dist_hi <= -0.15) & (dist_hi >= -0.55),
 "N3 N2 + 5日反抽(>0.70)":                    FRESH[60] & (dist_hi <= -0.15) & (dist_hi >= -0.55) & (r5 > 0.70),
 "N4 N3 + 成交额前10% [完整规则]":              FRESH[60] & (dist_hi <= -0.15) & (dist_hi >= -0.55) & (r5 > 0.70) & (liq > 0.90),
 "N5 N4 但高点<=40日(更新鲜)":                 FRESH[40] & (dist_hi <= -0.15) & (dist_hi >= -0.55) & (r5 > 0.70) & (liq > 0.90),
 "N6 N4 但高点<=120日(更宽)":                  FRESH[120] & (dist_hi <= -0.15) & (dist_hi >= -0.55) & (r5 > 0.70) & (liq > 0.90),
 "N7 N4 + 年涨幅>50%(大牛股)":                 FRESH[60] & (dist_hi <= -0.15) & (dist_hi >= -0.55) & (r5 > 0.70) & (liq > 0.90) & (yr > 0.5),
 "对照 X1 高点很旧(>120日)+同样回落反抽":        (age > 120) & (dist_hi <= -0.15) & (dist_hi >= -0.55) & (r5 > 0.70) & (liq > 0.90),
 "对照 X2 旧规则 R4(120日涨幅口径)":            (r120 > 0.7) & (liq > 0.90) & (r20 < 0.45) & (r5 > 0.70),
 "对照 X3 N4 去掉5日反抽":                     FRESH[60] & (dist_hi <= -0.15) & (dist_hi >= -0.55) & (liq > 0.90),
 "对照 X4 当日就在创新高+高流动":               is_newhigh & (liq > 0.90),
}
HS = [1, 2, 3, 5, 10, 20]
FWD = {h: TW.shift(-(1 + h)) / TW.shift(-1) - 1 for h in HS}
SEGS = {"主证据 2024-09-24~2025-12-31": (C.index >= "2024-09-24") & (C.index <= "2025-12-31"),
        "旁证 2020-01~2024-09-23": (C.index >= "2020-01-01") & (C.index <= "2024-09-23"),
        "⚠已污染 2026": (C.index >= "2026-01-01")}
for sname, m in SEGS.items():
    print("=" * 200); print(f"{sname}  ({int(m.sum())} 天)"); print("=" * 200)
    print(f'{"规则":36s}{"日均只数":>8s}' + "".join(f"{'h'+str(h):>14s}" for h in HS) + f'{"胜率h3":>8s}')
    for rn, cond in RULES.items():
        sel = (cond & pool & m[:, None]).fillna(False)
        n = sel.sum(axis=1)[m]; n = n[n > 0]
        if len(n) < 20: print(f"{rn:36s}{'样本不足':>8s}"); continue
        line = f"{rn:36s}{n.median():8.0f}"
        for h in HS:
            F = FWD[h].where(pool); base = F[m].mean(axis=1)
            ex = (F.where(sel)[m].mean(axis=1) - base).dropna()
            t = ex.mean() / ex.std() * np.sqrt(len(ex)) if len(ex) > 10 else np.nan
            line += f" {ex.mean()*100:+7.2f}%({t:+4.1f})"
        F3 = FWD[3].where(pool); b3 = F3[m].mean(axis=1)
        wr = ((F3.where(sel)[m].mean(axis=1) - b3) > 0).mean()
        print(line + f"{wr*100:7.0f}%")
    print()
