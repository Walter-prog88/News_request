"""实验 b2:强势篮子内,**哪一类信号**有用?
上一轮结论:我们的因子(反转/低波/GBT)在篮子内调权 ≈ 等权,而 ETF(市值加权)赢等权。
本轮把"加权方式"和"信号方向"拆开:
  - 市值加权(vw):复现 ETF 的做法本身
  - 各类信号的**名次倾斜**(tilt):动量(20/60/120)、市值、换手、波动,以及各自的反向
判据:能否跑赢篮子等权,并接近/超过市值加权。
"""
import os
from pathlib import Path, sys, warnings, glob; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import strategy as ST, newdata as ND
from afactor.config import LEDGER
pd.set_option("display.width", 260)
MODE, VER = "open10", "V100"
p = Panel(); d = p.dates; years = np.asarray(d.year)
cons = pd.read_csv("data_8:21/index_cons/csindex_cons_snapshot_20260821.csv", dtype=str)
to_code = lambda x: x.zfill(6) + (".SH" if x.zfill(6)[0] in "56" else ".SZ")
basket = sorted(set(to_code(c) for c in cons["成分券代码"]) & set(p.codes))
B = pd.DataFrame(False, index=d, columns=p.codes); B[basket] = True
u = ST.universe(p, VER) & B
c = p.raw("close_px"); amt = p.raw("amount")
mv = ND.total_mv(d, p.codes)
sig = {
    "市值(大→重)": np.log(mv),
    "动量 20 日": c / c.shift(20) - 1,
    "动量 60 日": c / c.shift(60) - 1,
    "动量 120 日": c / c.shift(120) - 1,
    "成交额 20 日均": np.log(amt.rolling(20, min_periods=10).mean()),
    "波动 20 日": c.pct_change().rolling(20, min_periods=10).std(),
    "换手趋势(20/60)": amt.rolling(20, min_periods=10).mean() / amt.rolling(60, min_periods=30).mean(),
}
sig = {k: v.reindex(index=d, columns=p.codes).shift(1) for k, v in sig.items()}   # T-1 收盘可得
segs = {"2023": years == 2023, "2024 924前": (d >= "2024-01-01") & (d <= "2024-09-23"),
        "924后A": (d >= "2024-09-24") & (d <= "2025-06-30"), "2025H2": (d >= "2025-07-01") & (d <= "2025-12-31"),
        "924后全段": (d >= "2024-09-24") & (d <= "2025-12-31"), "2023-25 全段": years >= 2023}
# 外部 ETF 数据仓路径:用 ETF_STORE 环境变量覆盖
S = os.environ.get("ETF_STORE",
     str(Path(__file__).resolve().parents[2] / "ETF-Research" /
         "full-etf project2 " / "full-etf-universe-store"))
ETFS = ["512930.SH", "512480.SH", "515880.SH", "159852.SZ", "159516.SZ"]
fs = sorted(glob.glob(f"{S}/daily/versions/daily-base-v2-*/batch=daily_base/yr=*/bkt=*/part.parquet"))
etf = pd.concat([pd.read_parquet(f, columns=["symbol", "session_date", "close"]) for f in fs])
etf = etf[etf.symbol.isin(ETFS)].pivot(index="session_date", columns="symbol", values="close").sort_index()
etf.index = pd.to_datetime(etf.index); etf_ew = etf.reindex(d).ffill().pct_change().mean(axis=1)
zero = pd.DataFrame(0.0, index=d, columns=p.codes)
rows = []
def run(name, score, scheme, n, m):
    r = ST.run_strategy(score, p, VER, MODE, scheme=scheme, n=n, rebalance=5, date_mask=m, universe_override=u)
    pa = r.get("phase_ann") or [r["ann_return"]]
    rows.append(dict(段=sname, 策略=name, 年化=r["ann_return"], 夏普=r["sharpe"], 回撤=r["max_dd"],
                     换手=r.get("turnover"), 相位极差=max(pa) - min(pa)))
for sname, m in segs.items():
    m = np.asarray(m)
    run("篮子等权(基准)", zero, "top", 100000, m)
    run("篮子市值加权(≈ETF做法)", mv, "vw", None, m)
    for k, v in sig.items():
        run(f"倾斜↑ {k}", v, "tilt", 1.0, m)
        run(f"倾斜↓ {k}", -v, "tilt", 1.0, m)
    r = etf_ew[m].dropna(); nv = (1 + r).cumprod()
    rows.append(dict(段=sname, 策略="ETF 等权(5只,参照)", 年化=float(nv.iloc[-1] ** (242 / len(r)) - 1),
                     夏普=float(r.mean() / r.std() * np.sqrt(242)), 回撤=float((nv / nv.cummax() - 1).min()), 换手=np.nan, 相位极差=np.nan))
df = pd.DataFrame(rows); df.to_csv(LEDGER / "basket_signals.csv", index=False)
for sname in segs:
    sub = df[df.段 == sname].drop(columns="段").sort_values("年化", ascending=False)
    print(f"\n--- {sname} ---"); print(sub.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
