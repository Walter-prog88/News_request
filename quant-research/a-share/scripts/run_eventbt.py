"""小碎步法事件驱动复现:默认参数(未调),分段报告 + 两条基线。
分段:2020-21(趋势段,考场)/ 2022-24(轮动段,风险披露)/ 2025H1(轮动)/ 2025H2(趋势起点,withhold)。
基线 A:守门集合等权(用户硬门槛);基线 B:不守门的同一套入场/退出规则。"""
import os, sys, warnings, json; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import eventbt as EB, strategy as ST
from afactor.config import LEDGER
pd.set_option("display.width", 250)
WITHHOLD = os.environ.get("EB_SHOW_H2", "0") != "1"
p = Panel(); d = p.dates
raw = pd.read_csv('data_8:21/source_raw/baostock_stock_industry_raw.csv'); raw['code'] = raw['code'].str.upper().str.replace(r'^(SH|SZ)\.(\d+)$', r'\2.\1', regex=True)
lab = pd.Series(dict(zip(raw.code, raw.industry))).reindex(p.codes).replace('', np.nan)
segs = {"2020-21 趋势段": (d >= "2020-01-01") & (d <= "2021-12-31"),
        "2022-24 轮动段": (d >= "2022-01-01") & (d <= "2024-12-31"),
        "2025H1 轮动": (d >= "2025-01-01") & (d <= "2025-06-30"),
        "2025H2 趋势起点": (d >= "2025-07-01") & (d <= "2025-12-31")}
version = os.environ.get("EB_VERSION", "V100")
u = ST.universe(p, version)
variants = {"小碎步(守门)": EB.Rules(),
            "同规则不守门(基线B)": EB.Rules(use_sector_gate=False, use_stock_gate=False)}
rows = []
for name, rules in variants.items():
    sig = EB.build_signals(p, rules, u, lab)
    for sname, m in segs.items():
        if WITHHOLD and "H2" in sname: continue
        r = EB.run_event_bt(p, rules, version, date_mask=np.asarray(m), sector_labels=lab, universe=u, signals=sig)
        rows.append(dict(变体=name, 段=sname, 年化=r["ann_return"], 夏普=r["sharpe"], 回撤=r["max_dd"], 仓位=r["exposure"],
                         交易数=r["n_trades"], 胜率=r["win_rate"], 单笔均值=r["avg_ret"], 持有天=r["avg_hold"],
                         退出=json.dumps(r["exit_mix"], ensure_ascii=False)))
    if name.startswith("小碎步"):
        # 基线 A:同一守门集合(strong ∩ universe)的等权,周频,尾盘口径
        gate = (sig["strong"] & u)
        ew = pd.DataFrame(0.0, index=d, columns=p.codes).where(gate)
        for sname, m in segs.items():
            if WITHHOLD and "H2" in sname: continue
            rA = ST.run_strategy(ew, p, version, "tail1450", scheme="top", n=100000, rebalance=5, date_mask=np.asarray(m))
            pool = ST.benchmark(p, version, "tail1450", date_mask=np.asarray(m))
            rows.append(dict(变体="基线A:守门集合等权", 段=sname, 年化=rA["ann_return"], 夏普=rA["sharpe"], 回撤=rA["max_dd"], 仓位=rA.get("exposure"),
                             交易数=np.nan, 胜率=np.nan, 单笔均值=np.nan, 持有天=np.nan, 退出=f"守门集合中位 {int(gate[m].sum(axis=1).median())} 只"))
            rows.append(dict(变体="参照:全池等权", 段=sname, 年化=pool["ann_return"], 夏普=pool.get("sharpe"), 回撤=pool["max_dd"], 仓位=1.0,
                             交易数=np.nan, 胜率=np.nan, 单笔均值=np.nan, 持有天=np.nan, 退出=""))
df = pd.DataFrame(rows); df.to_csv(LEDGER / f"eventbt_{version}.csv", index=False)
print(f"=== 小碎步法事件驱动复现 | {version} | 尾盘口径 | 默认参数未调 | 2025H2 {'已隐藏' if WITHHOLD else '显示'} ===")
for sname in segs:
    if WITHHOLD and "H2" in sname: continue
    print(f"\n--- {sname} ---")
    print(df[df.段 == sname].drop(columns="段").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
