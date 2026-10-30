"""实验 b:强势篮子(AI/半导体/通信/软件指数成分)内,用因子/模型调权 vs 篮子等权 vs ETF vs 2×ETF。
用户目标:验证"在强势股里做灵活交易能否放大收益、强过买 ETF 甚至杠杆 ETF"。
第一版用 2026-08-21 当前成分(含事后偏差,篮子 vs ETF 那一比会高估篮子;篮子内对比两边同篮子,基本抵消)。
打分 = B×GBT 滚动 OOS(全市场训练),以及两个单因子作稳健性对照。"""
import os
from pathlib import Path, sys, warnings, glob; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import strategy as ST, factors as FA
from afactor.config import LEDGER
pd.set_option("display.width", 260)
MODE = os.environ.get("BT_MODE", "open10"); VER = "V100"
p = Panel(); d = p.dates; years = np.asarray(d.year)
# ---- 篮子 ----
cons = pd.read_csv("data_8:21/index_cons/csindex_cons_snapshot_20260821.csv", dtype=str)
def to_code(x):
    x = x.zfill(6); return x + (".SH" if x[0] in "56" else ".SZ")
basket = sorted(set(to_code(c) for c in cons["成分券代码"]) & set(p.codes))
B = pd.DataFrame(False, index=d, columns=p.codes); B[basket] = True
u = ST.universe(p, VER) & B
print(f"篮子 {len(basket)} 只(面板内),日均可交易 {int(u.sum(axis=1).mean())} 只 | 执行 {MODE}")
# ---- 打分 ----
sc = pd.DataFrame(np.nan, index=d, columns=p.codes, dtype=np.float32)
for y in [2023, 2024, 2025]:
    s = pd.read_parquet(LEDGER / f"scores/B_gbt_open_h5_{y}.parquet"); sc.loc[s.index, s.columns] = s.to_numpy()
FA.build_all()
ivol = -FA.REGISTRY["ivol_20"][1](p).reindex(index=d, columns=p.codes)      # 低波为好
rev = -FA.REGISTRY["rev_20"][1](p).reindex(index=d, columns=p.codes)        # 反转:近期跌为好
zero = pd.DataFrame(0.0, index=d, columns=p.codes)
# ---- ETF 净值(store)----
# 外部 ETF 数据仓路径:用 ETF_STORE 环境变量覆盖
S = os.environ.get("ETF_STORE",
     str(Path(__file__).resolve().parents[2] / "ETF-Research" /
         "full-etf project2 " / "full-etf-universe-store"))
ETFS = {"512930.SH": "人工智能ETF", "512480.SH": "半导体ETF", "515880.SH": "通信ETF", "159852.SZ": "软件ETF", "159516.SZ": "科创芯片ETF"}
fs = sorted(glob.glob(f"{S}/daily/versions/daily-base-v2-*/batch=daily_base/yr=*/bkt=*/part.parquet"))
etf = pd.concat([pd.read_parquet(f, columns=["symbol", "session_date", "close"]) for f in fs])
etf = etf[etf.symbol.isin(ETFS)].pivot(index="session_date", columns="symbol", values="close").sort_index()
etf.index = pd.to_datetime(etf.index); etf = etf.reindex(d).ffill()
etf_ret = etf.pct_change().fillna(0.0)
etf_ew = etf_ret.mean(axis=1)
MARGIN = 0.06 / 242
etf_2x = 2 * etf_ew - MARGIN                                   # 日内再平衡 2 倍(含 6% 融资利率)
print("ETF 覆盖:", {k: f"{etf[k].first_valid_index().date()}" for k in etf.columns})
# ---- 分段 ----
segs = {"2023": years == 2023, "2024 924前": (d >= "2024-01-01") & (d <= "2024-09-23"),
        "924后A": (d >= "2024-09-24") & (d <= "2025-06-30"), "2025H2": (d >= "2025-07-01") & (d <= "2025-12-31"),
        "924后全段": (d >= "2024-09-24") & (d <= "2025-12-31"), "2023-25 全段": years >= 2023}
def stat(r):
    r = r.dropna(); nv = (1 + r).cumprod()
    return dict(年化=float(nv.iloc[-1] ** (242 / len(r)) - 1), 夏普=float(r.mean() / r.std() * np.sqrt(242)) if r.std() > 0 else np.nan,
                回撤=float((nv / nv.cummax() - 1).min()))
rows = []
strategies = {
    "篮子等权(基准)": (zero, "top", 100000),
    "篮子内排雷 剔最差30%": (sc, "exclude", None),
    "篮子内线性倾斜(GBT)": (sc, "tilt", 1.0),
    "篮子内强倾斜(GBT, ^2)": (sc, "tilt", 2.0),
    "篮子内 Top50%(GBT)": (sc, "top", None),
    "篮子内 Top20(GBT)": (sc, "top", 20),
    "篮子内倾斜(低波单因子)": (ivol, "tilt", 1.0),
    "篮子内倾斜(反转单因子)": (rev, "tilt", 1.0),
}
for sname, m in segs.items():
    m = np.asarray(m)
    for name, (score, scheme, n) in strategies.items():
        kw = dict(scheme=scheme, rebalance=5, date_mask=m)
        if scheme == "exclude": kw.update(buffer=(0.20, 0.35))
        elif scheme == "top" and n is None: kw.update(n=max(5, int(u[m].sum(axis=1).median() * 0.5)))
        else: kw.update(n=n)
        r = ST.run_strategy(score, p, VER, MODE, universe_override=u, **kw)
        pa = r.get("phase_ann") or [r["ann_return"]]
        rows.append(dict(段=sname, 策略=name, 年化=r["ann_return"], 夏普=r["sharpe"], 回撤=r["max_dd"], 仓位=r.get("exposure"), 换手=r.get("turnover"), 相位极差=max(pa) - min(pa)))
    for name, r in [("ETF 等权(5只)", etf_ew[m]), ("2×ETF(日再平衡,含6%利息)", etf_2x[m])] + [(f"ETF {ETFS[k]}", etf_ret[k][m]) for k in etf.columns]:
        rows.append(dict(段=sname, 策略=name, **stat(r), 仓位=np.nan, 换手=np.nan, 相位极差=np.nan))
df = pd.DataFrame(rows); df.to_csv(LEDGER / f"basket_tilt_{MODE}.csv", index=False)
for sname in segs:
    print(f"\n--- {sname} ---"); print(df[df.段 == sname].drop(columns="段").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
