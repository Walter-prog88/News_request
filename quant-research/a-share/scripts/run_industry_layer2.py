"""行业层补充实验:(a) 5 槽位在 V100 上(参与率不卡);(b) 状态层在 2020-2025 全期的行业级检验。"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
from afactor.config import LEDGER
from afactor.panel import Panel
from afactor import strategy as ST
sys.path.insert(0, os.path.dirname(__file__))
from run_industry_layer import industry_labels, regime_state, MIN_MEMBERS, MOM_WIN, K_IND
pd.set_option("display.width", 250)

p = Panel(); dates, codes = p.dates, p.codes; years = np.asarray(dates.year)
close = p.raw("close_px")
# ===== (b) 行业级:状态层 2020-2025 全期(不依赖个股打分)=====
u = ST.universe(p, "V1000"); r = close.pct_change().where(u)
tw = p.raw("twap_open10"); r_exec = (tw.shift(-2) / tw.shift(-1) - 1).where(u)   # T 信号可交易口径
lab = industry_labels(codes).dropna()
cnt = lab[lab.index.isin(u.columns[u.mean() > 0.2])].value_counts(); lab = lab[lab.map(cnt) >= MIN_MEMBERS]
ind_ret = r[lab.index].T.groupby(lab).mean().T
S = regime_state(ind_ret, dates)
ind_mom = ind_ret.add(1).rolling(MOM_WIN, min_periods=60).apply(np.prod, raw=True) - 1
top = ind_mom.rank(axis=1, ascending=False) <= K_IND
bot = ind_mom.rank(axis=1, ascending=True) <= K_IND
nxt = r_exec[lab.index].T.groupby(lab).mean().T            # T 信号 -> twap[T+1]->twap[T+2],与主回测同口径(审查 F1)
eq = nxt.mean(axis=1); trend = nxt.where(top).mean(axis=1); anti = nxt.where(bot).mean(axis=1)
S_b = S.reindex(dates)
switch = pd.Series(np.where(S_b > 0, trend, eq), index=dates)
print("=== (b) 行业级(主回测口径 twap[T+1]->[T+2],不含成本):动量前5行业 vs 全行业等权 vs 状态切换 ===")
print(f"{'年':6s}{'趋势市占比':>10s}{'前5行业超额':>12s}{'后5行业超额':>12s}{'状态切换超额':>12s}   S均值")
for y in range(2020, 2026):
    m = years == y
    f = lambda s: ((1 + s[m]).prod() - 1) * 100
    tf = (S_b[m] > 0).mean() * 100
    print(f"{y:<6d}{tf:>9.0f}%{f(trend)-f(eq):>+11.1f}pp{f(anti)-f(eq):>+11.1f}pp{f(switch)-f(eq):>+11.1f}pp   {S_b[m].mean():+.3f}")
print("(前5行业超额 = 恒定追行业动量 的增量;状态切换 = 只在 S>0 时追,否则等权)")

# ===== (a) 5 槽位在 V100 =====
OOS = [2023, 2024, 2025]
score = pd.DataFrame(np.nan, index=dates, columns=codes, dtype=np.float32)
for y in OOS:
    s = pd.read_parquet(LEDGER / f"scores/B_gbt_open_h5_{y}.parquet"); score.loc[s.index, s.columns] = s.to_numpy()
u100 = ST.universe(p, "V100"); score = score.where(u100)
within = pd.DataFrame(np.nan, index=dates, columns=lab.index, dtype=np.float32)
for ind, mem in lab.groupby(lab).groups.items():
    within[list(mem)] = score[list(mem)].rank(axis=1, ascending=False, method="first").to_numpy()
within = within.reindex(columns=codes)
ind_rank = ind_mom.rank(axis=1, ascending=False)
stock_ind_rank = pd.DataFrame({c: ind_rank[lab[c]] for c in lab.index}).reindex(columns=codes)
in_top = (stock_ind_rank <= K_IND).fillna(False)
print("\n=== (a) V100(100万):5 槽位 ===")
rows = []
for name, sc, n in [("纯Top5", score, 5), ("纯Top20", score, 20),
                    ("恒定趋势 5行业×1只", -within.where(in_top & (within <= 1)), 5),
                    ("恒定趋势 5行业×4只", -within.where(in_top & (within <= 4)), 20)]:
    for y in OOS:
        dm = years == y
        bm = ST.benchmark(p, "V100", "open10", date_mask=dm)
        r_ = ST.run_strategy(sc, p, "V100", "open10", scheme="top", n=n, rebalance=5, date_mask=dm)
        pa = r_.get("phase_ann") or [r_["ann_return"]]
        rows.append(dict(变体=name, 年=y, 超额=r_["ann_return"] - bm["ann_return"], 夏普=r_["sharpe"],
                         回撤=r_["max_dd"], 仓位=r_.get("exposure"), 持仓=r_["holdings"], 相位极差=max(pa) - min(pa)))
df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n三年平均:")
print(df.groupby("变体", sort=False).agg(超额=("超额", "mean"), 夏普=("夏普", "mean"), 回撤=("回撤", "mean"),
                                       最差年=("超额", "min"), 仓位=("仓位", "mean"), 相位极差=("相位极差", "mean")).round(4).to_string())
df.to_csv(LEDGER / "industry_layer2.csv", index=False)
