"""交易者买入取证(完整版):合并 cache/sealed + cache/recent_daily,覆盖到 2026-08-27。
三问:(1) 买点当天形态 (2) 买入后的表现 (3) 持有期。
⚠ 2026 已用于本次取证 -> 由此提炼的规则只能在 2024-09-24~2025-12-31 验证。"""
import os, sys, csv, glob, warnings; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.config import DAILY, SEALED
pd.set_option("display.width", 280)

def load_recent():
    rows = []
    for f in glob.glob("cache/recent_daily/parts/*.csv"):
        if os.path.getsize(f) < 50: continue
        d = pd.read_csv(f)
        if len(d) == 0: continue
        rows.append(d[["date", "code", "close", "high", "amount", "tradestatus"]])
    r = pd.concat(rows, ignore_index=True)
    r = r[r.tradestatus == 1]
    r["ts"] = r["code"].str.upper().str.replace(r"^(SH|SZ)\.(\d+)$", r"\2.\1", regex=True)
    r["date"] = pd.to_datetime(r["date"])
    for c in ("close", "high", "amount"): r[c] = pd.to_numeric(r[c], errors="coerce")
    return (r.pivot(index="date", columns="ts", values="close"),
            r.pivot(index="date", columns="ts", values="high"),
            r.pivot(index="date", columns="ts", values="amount"))

def cat(name, rec):
    a = pd.concat([pd.read_parquet(DAILY / f"{name}.parquet"), pd.read_parquet(SEALED / f"{name}.parquet")])
    both = a.columns.union(rec.columns)
    return pd.concat([a.reindex(columns=both), rec.reindex(columns=both)]).sort_index()

rc, rh, ra = load_recent()
C, H, A = cat("close_px", rc), cat("high_px", rh), cat("amount", ra)
print(f"合并后面板 {C.index.min().date()} ~ {C.index.max().date()},{C.shape[1]} 只\n")

D = "/path/to/user/Documents/ChatGPT/Stock model"
ev = list(csv.DictReader(open(D + "/data/normalized/events.csv", encoding="utf-8")))
buys = [e for e in ev if e["event_type"] in ("buy", "buy_to_target")
        and e["security_code"].isdigit() and len(e["security_code"]) == 6]
to_ts = lambda c: c + (".SH" if c[0] in "65" else ".SZ")
adv20 = A.rolling(20, min_periods=10).mean(); poolm = (adv20 >= 2e7) & C.notna()

def feats(code, d):
    ts = to_ts(code)
    if ts not in C.columns: return None
    idx = C.index[C.index <= pd.Timestamp(d)]
    if len(idx) < 130: return None
    t = idx[-1]
    c = C[ts].loc[:t].dropna(); a = A[ts].loc[:t].dropna()
    if len(c) < 130 or (pd.Timestamp(d) - t).days > 4: return None
    px = c.iloc[-1]
    f = dict(code=code, buy=str(pd.Timestamp(d).date()), asof=str(t.date()))
    f["距252高%"] = px / c.iloc[-252:].max() * 100 - 100
    f["距20高%"] = px / c.iloc[-20:].max() * 100 - 100
    f["区间位置"] = (px - c.iloc[-252:].min()) / (c.iloc[-252:].max() - c.iloc[-252:].min())
    for w in (5, 20, 60, 120): f[f"{w}日涨%"] = (px / c.iloc[-w-1] - 1) * 100
    r_ = c.pct_change()
    f["波动收敛"] = r_.iloc[-20:].std() / r_.iloc[-60:].std()
    f["量比5/60"] = a.iloc[-5:].mean() / a.iloc[-60:].mean()
    f["额亿"] = a.iloc[-20:].mean() / 1e8
    # 买入后表现
    fut = C[ts].loc[t:].dropna()
    for h in (1, 3, 5, 10):
        f[f"后{h}日%"] = (fut.iloc[h] / px - 1) * 100 if len(fut) > h else np.nan
    return f

rows = [x for e in buys if (x := feats(e["security_code"], e["event_date_start"]))]
df = pd.DataFrame(rows)
fc = ["距252高%", "距20高%", "区间位置", "5日涨%", "20日涨%", "60日涨%", "120日涨%", "波动收敛", "量比5/60", "额亿"]
pc = [f"后{h}日%" for h in (1, 3, 5, 10)]
print("=" * 200); print(f"A. 买点当天形态 + 买入后表现({len(df)} 笔,已覆盖到 2026-08-27)"); print("=" * 200)
print(df[["code", "buy"] + fc + pc].to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
print("\n中位数:"); print(df[fc + pc].median().round(2).to_string())

# 百分位
def pool_pct(d, f):
    t = C.index[C.index <= pd.Timestamp(d)][-1]
    cs = poolm.loc[t][poolm.loc[t]].index
    sub = C.loc[:t, cs]; sa = A.loc[:t, cs]; px = sub.iloc[-1]; rr = sub.pct_change()
    dd = {"距252高%": px / sub.iloc[-252:].max() * 100 - 100, "距20高%": px / sub.iloc[-20:].max() * 100 - 100,
          "区间位置": (px - sub.iloc[-252:].min()) / (sub.iloc[-252:].max() - sub.iloc[-252:].min()),
          "5日涨%": (px / sub.iloc[-6] - 1) * 100, "20日涨%": (px / sub.iloc[-21] - 1) * 100,
          "60日涨%": (px / sub.iloc[-61] - 1) * 100, "120日涨%": (px / sub.iloc[-121] - 1) * 100,
          "波动收敛": rr.iloc[-20:].std() / rr.iloc[-60:].std(), "量比5/60": sa.iloc[-5:].mean() / sa.iloc[-60:].mean(),
          "额亿": sa.iloc[-20:].mean() / 1e8}
    return {k: (v.dropna() < f[k]).mean() * 100 for k, v in dd.items() if pd.notna(f.get(k))}
P = pd.DataFrame([dict(code=r["code"], buy=r["buy"], **pool_pct(r["buy"], r)) for _, r in df.iterrows()])
print("\n" + "=" * 200); print("B. 买点在当日全池中的百分位(50=中位)"); print("=" * 200)
print(P.to_string(index=False, float_format=lambda x: f"{x:6.1f}"))
print("\n百分位中位数:"); print(P[fc].median().round(1).to_string())

# 持有期
ps = pd.read_csv(D + "/data/normalized/position_state.csv")
ps["trading_date"] = pd.to_datetime(ps["trading_date"])
hold = []
for code, g in ps.groupby("security_code"):
    ds = sorted(g["trading_date"].unique())
    hold.append(dict(code=str(code), 名称=g["security_name"].iloc[0], 首见=str(pd.Timestamp(ds[0]).date()),
                     末见=str(pd.Timestamp(ds[-1]).date()), 出现截面数=len(ds)))
Hd = pd.DataFrame(hold).sort_values("首见")
print("\n" + "=" * 200); print(f"C. 持仓存续(共 {len(Hd)} 个标的,13 个持仓截面)"); print("=" * 200)
print(Hd.to_string(index=False))
df.to_csv("research/trader_forensics_full.csv", index=False); P.to_csv("research/trader_forensics_pct.csv", index=False)
