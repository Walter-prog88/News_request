"""验证从交易者实际买入提炼的规则(2026-08-28)。

规则来源:27 笔实际买入的全池百分位中位数 ——
  额亿 97.5 / 5日涨 82.4 / 120日涨 84.7 / 区间位置 81.9 / 20日涨 29.5 / 距20日高 38.6
提炼为:【中长期强 + 中期回落 + 近周反抽 + 高流动性】

⚠ 纪律:规则从 2026-06~08 的买入提炼 -> 2026 已污染,只能参考。
   **主证据窗口 = 2024-09-24 ~ 2025-12-31**(用户指定 924 之后,且规则未从这段提炼)。
   另报 2020-2024-09-23 作为"更早期是否也成立"的旁证。
口径:T 日收盘信号 -> T+1 开盘买 -> T+1+h 开盘卖(twap_open10);相对同日全池等权。
"""
import os, sys, glob, warnings; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.config import DAILY, SEALED
pd.set_option("display.width", 260)

def load_recent(name, col):
    rows = []
    for f in glob.glob("cache/recent_daily/parts/*.csv"):
        if os.path.getsize(f) < 50: continue
        d = pd.read_csv(f)
        if len(d) == 0 or col not in d.columns: continue
        d = d[d.tradestatus == 1]
        rows.append(d[["date", "code", col]])
    r = pd.concat(rows, ignore_index=True)
    r["ts"] = r["code"].str.upper().str.replace(r"^(SH|SZ)\.(\d+)$", r"\2.\1", regex=True)
    r["date"] = pd.to_datetime(r["date"]); r[col] = pd.to_numeric(r[col], errors="coerce")
    return r.pivot(index="date", columns="ts", values=col)

def cat(name, col):
    a = pd.concat([pd.read_parquet(DAILY / f"{name}.parquet"), pd.read_parquet(SEALED / f"{name}.parquet")])
    rec = load_recent(name, col); both = a.columns.union(rec.columns)
    return pd.concat([a.reindex(columns=both), rec.reindex(columns=both)]).sort_index()

C = cat("close_px", "close"); A = cat("amount", "amount")
TW = pd.concat([pd.read_parquet(DAILY / "twap_open10.parquet"), pd.read_parquet(SEALED / "twap_open10.parquet")])
TW = TW.reindex(index=C.index, columns=C.columns)          # 8/18 后无分钟数据 -> 用收盘价代
TW = TW.fillna(C)
print(f"面板 {C.index.min().date()} ~ {C.index.max().date()}  ({C.shape[1]} 只);8/18 起用收盘价替代开盘 TWAP\n")

adv20 = A.rolling(20, min_periods=10).mean()
pool = ((adv20 >= 2e7) & C.notna()).fillna(False)
rk = lambda X: X.where(pool).rank(axis=1, pct=True)
r5   = rk(C / C.shift(5) - 1)
r20  = rk(C / C.shift(20) - 1)
r120 = rk(C / C.shift(120) - 1)
d20h = rk(C / C.rolling(20, min_periods=10).max() - 1)
posq = rk((C - C.rolling(252, min_periods=120).min()) / (C.rolling(252, min_periods=120).max() - C.rolling(252, min_periods=120).min()))
liq  = rk(adv20)

RULES = {
 "R0 全池(基准)":                 pool,
 "R1 只要 120日强(>0.7)":          r120 > 0.7,
 "R2 R1+高流动(额>0.90)":          (r120 > 0.7) & (liq > 0.90),
 "R3 R2+20日弱(<0.45)":           (r120 > 0.7) & (liq > 0.90) & (r20 < 0.45),
 "R4 R3+5日反抽(>0.70) [完整规则]": (r120 > 0.7) & (liq > 0.90) & (r20 < 0.45) & (r5 > 0.70),
 "R5 R4 去掉流动性条件":            (r120 > 0.7) & (r20 < 0.45) & (r5 > 0.70),
 "R6 R4 换成 20日强(对照)":         (r120 > 0.7) & (liq > 0.90) & (r20 > 0.55) & (r5 > 0.70),
 "R7 只要 5日反抽+高流动(对照)":      (liq > 0.90) & (r5 > 0.70),
}
HS = [1, 2, 3, 5, 10, 20]
FWD = {h: TW.shift(-(1 + h)) / TW.shift(-1) - 1 for h in HS}
SEGS = {"主证据 2024-09-24~2025-12-31": (C.index >= "2024-09-24") & (C.index <= "2025-12-31"),
        "旁证 2020-01~2024-09-23":     (C.index >= "2020-01-01") & (C.index <= "2024-09-23"),
        "⚠已污染 2026 全年":            (C.index >= "2026-01-01")}

for sname, m in SEGS.items():
    print("=" * 210); print(f"{sname}   ({int(m.sum())} 个交易日)"); print("=" * 210)
    print(f'{"规则":30s}{"日均只数":>9s}' + "".join(f"{'超额 h'+str(h):>13s}" for h in HS) + f'{"胜率h3":>9s}')
    for rname, cond in RULES.items():
        sel = (cond & pool & m[:, None]).fillna(False)
        n = sel.sum(axis=1)[m]; n = n[n > 0]
        if len(n) < 20: print(f"{rname:30s}{'样本不足':>9s}"); continue
        line = f"{rname:30s}{n.median():9.0f}"
        for h in HS:
            F = FWD[h].where(pool)
            base = F[m].mean(axis=1)
            g = F.where(sel)[m].mean(axis=1)
            ex = (g - base).dropna()
            t = ex.mean() / ex.std() * np.sqrt(len(ex)) if len(ex) > 10 else np.nan
            line += f"  {ex.mean()*100:+7.2f}%({t:+4.1f})"
        F3 = FWD[3].where(pool); b3 = F3[m].mean(axis=1)
        wr = ((F3.where(sel)[m].mean(axis=1) - b3) > 0).mean()
        print(line + f"{wr*100:8.0f}%")
    print()
