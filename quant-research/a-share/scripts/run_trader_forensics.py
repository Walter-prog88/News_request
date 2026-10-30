"""交易者实际买入的取证分析(2026-08-27)。
数据:/path/to/user/Documents/ChatGPT/Stock model/data/normalized/events.csv(带代码/价格/金额的买入)
行情:cache/sealed 覆盖 2026-01-05~2026-08-17。**仅描述性取证**;由此得到的任何规则必须回到 2020-2025 验证。
"""
import os, sys, csv, warnings; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.config import DAILY, SEALED
pd.set_option("display.width", 260)

D = "/path/to/user/Documents/ChatGPT/Stock model"
ev = list(csv.DictReader(open(D + "/data/normalized/events.csv", encoding="utf-8")))
buys = [e for e in ev if e["event_type"] in ("buy", "buy_to_target")
        and e["security_code"].isdigit() and len(e["security_code"]) == 6]
def to_ts(c): return c + (".SH" if c[0] in "65" else ".SZ")

C = pd.concat([pd.read_parquet(DAILY / "close_px.parquet"), pd.read_parquet(SEALED / "close_px.parquet")])
A = pd.concat([pd.read_parquet(DAILY / "amount.parquet"), pd.read_parquet(SEALED / "amount.parquet")])
H = pd.concat([pd.read_parquet(DAILY / "high_px.parquet"), pd.read_parquet(SEALED / "high_px.parquet")]) if (SEALED/"high_px.parquet").exists() else None
LAST = C.index.max()
adv20 = A.rolling(20, min_periods=10).mean()
pool = (adv20 >= 2e7) & C.notna()

def feats(code, d):
    """在日期 d(或 d 之前最后一个有数据的日)描述该股形态。全部只用 d 及之前。"""
    ts = to_ts(code)
    if ts not in C.columns: return None
    idx = C.index[C.index <= pd.Timestamp(d)]
    if len(idx) < 130: return None
    t = idx[-1]; asof_gap = (pd.Timestamp(d) - t).days
    c = C[ts].loc[:t].dropna()
    a = A[ts].loc[:t].dropna()
    if len(c) < 130: return None
    px = c.iloc[-1]
    f = dict(code=code, ts=ts, buy_date=d, asof=str(t.date()), 数据滞后天=asof_gap)
    f["距252日高%"] = px / c.iloc[-252:].max() * 100 - 100
    f["距60日高%"] = px / c.iloc[-60:].max() * 100 - 100
    f["距20日高%"] = px / c.iloc[-20:].max() * 100 - 100
    f["252日区间位置"] = (px - c.iloc[-252:].min()) / (c.iloc[-252:].max() - c.iloc[-252:].min())
    for w in (5, 20, 60, 120):
        f[f"{w}日涨%"] = (px / c.iloc[-w-1] - 1) * 100
    r = c.pct_change()
    f["20日波动%"] = r.iloc[-20:].std() * 100
    f["波动收敛"] = r.iloc[-20:].std() / r.iloc[-60:].std()
    box = c.iloc[-25:]
    f["25日箱体宽%"] = (box.max() - box.min()) / box.mean() * 100
    f["量比5/60"] = a.iloc[-5:].mean() / a.iloc[-60:].mean()
    f["量比20/120"] = a.iloc[-20:].mean() / a.iloc[-120:].mean()
    f["MA20上方"] = px > c.iloc[-20:].mean()
    f["MA60上方"] = px > c.iloc[-60:].mean()
    f["日均额亿"] = a.iloc[-20:].mean() / 1e8
    return f

rows = [x for e in buys if (x := feats(e["security_code"], e["event_date_start"]))]
df = pd.DataFrame(rows)
covered = df[df["数据滞后天"] <= 3]; stale = df[df["数据滞后天"] > 3]
cols = ["code", "buy_date", "asof", "距252日高%", "距60日高%", "距20日高%", "252日区间位置",
        "5日涨%", "20日涨%", "60日涨%", "120日涨%", "20日波动%", "波动收敛", "25日箱体宽%",
        "量比5/60", "量比20/120", "日均额亿"]
print("=" * 250)
print(f"A. 买入当日形态(行情覆盖内,{len(covered)} 笔)")
print("=" * 250)
print(covered[cols].to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
print(f"\n中位数:"); print(covered[cols[3:]].median().round(2).to_string())

print("\n" + "=" * 250)
print(f"B. 数据外的买入 —— 截至 {LAST.date()} 的形态({len(stale)} 笔,含 8/27 三只)")
print("=" * 250)
print(stale[cols].to_string(index=False, float_format=lambda x: f"{x:7.2f}"))

# C. 对照:同期全池的分布
print("\n" + "=" * 250)
print("C. 对照 —— 买入日全池(ADV>=2000万)的同口径分位,看他的买点落在哪")
print("=" * 250)
def pool_dist(d, keys):
    idx = C.index[C.index <= pd.Timestamp(d)]
    t = idx[-1]
    ok = pool.loc[t]; cs = ok[ok].index
    out = {}
    sub = C.loc[:t, cs]; sa = A.loc[:t, cs]
    px = sub.iloc[-1]
    out["距252日高%"] = (px / sub.iloc[-252:].max() * 100 - 100)
    out["距20日高%"] = (px / sub.iloc[-20:].max() * 100 - 100)
    out["252日区间位置"] = (px - sub.iloc[-252:].min()) / (sub.iloc[-252:].max() - sub.iloc[-252:].min())
    out["20日涨%"] = (px / sub.iloc[-21] - 1) * 100
    out["60日涨%"] = (px / sub.iloc[-61] - 1) * 100
    out["120日涨%"] = (px / sub.iloc[-121] - 1) * 100
    rr = sub.pct_change()
    out["波动收敛"] = rr.iloc[-20:].std() / rr.iloc[-60:].std()
    out["量比5/60"] = sa.iloc[-5:].mean() / sa.iloc[-60:].mean()
    out["日均额亿"] = sa.iloc[-20:].mean() / 1e8
    return out
pcts = []
for _, r_ in df.iterrows():
    pdst = pool_dist(r_["buy_date"], None)
    row = dict(code=r_["code"], buy_date=r_["buy_date"])
    for k, s in pdst.items():
        v = r_.get(k)
        row[k] = (s.dropna() < v).mean() * 100 if pd.notna(v) else np.nan
    pcts.append(row)
P = pd.DataFrame(pcts)
print("(数值 = 该股在当日全池中的百分位,50=中位,越大越高)")
print(P.to_string(index=False, float_format=lambda x: f"{x:6.1f}"))
print("\n所有买入的百分位中位数:")
print(P.drop(columns=["code", "buy_date"]).median().round(1).to_string())
df.to_csv("research/trader_buy_forensics.csv", index=False)
P.to_csv("research/trader_buy_percentiles.csv", index=False)
