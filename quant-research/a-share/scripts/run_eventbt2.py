"""小碎步法复现 v2:审查修复后的变体网格。每个变体只改一件事。
段:2020H2-21(趋势段;rp 需 120 天,故从 2020-07 起,基线同窗)/ 2022-24 / 2025H1;2025H2 withhold。
对照:基线A 守门集合等权;基线B 同规则不守门;随机对照 = 同守门集合内随机入场(10 次)。"""
import os, sys, warnings, json; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import eventbt as EB, strategy as ST
from afactor.config import LEDGER
pd.set_option("display.width", 260)
WITHHOLD = os.environ.get("EB_SHOW_H2", "0") != "1"
version = os.environ.get("EB_VERSION", "V100")
p = Panel(); d = p.dates
raw = pd.read_csv('data_8:21/source_raw/baostock_stock_industry_raw.csv'); raw['code'] = raw['code'].str.upper().str.replace(r'^(SH|SZ)\.(\d+)$', r'\2.\1', regex=True)
lab = pd.Series(dict(zip(raw.code, raw.industry))).reindex(p.codes).replace('', np.nan)
segs = {"2020H2-21 趋势段": (d >= "2020-07-01") & (d <= "2021-12-31"),
        "2022-24 轮动段": (d >= "2022-01-01") & (d <= "2024-12-31"),
        "2025H1 轮动": (d >= "2025-01-01") & (d <= "2025-06-30"),
        "2025H2 趋势起点": (d >= "2025-07-01") & (d <= "2025-12-31")}
u = ST.universe(p, version)
R = EB.Rules
variants = {
    "V0 审查前同款(仅低开收红,动量排序)": R(path_breakout=False, rank_by="momentum"),
    "V1 两路径,板块排序":                R(rank_by="sector"),
    "V2 仅突破前高回踩,板块排序":         R(path_lowopen=False, rank_by="sector"),
    "V3 两路径,龙头(市值)排序":          R(rank_by="cap"),
    "V4 V1+早盘退出(低开3%出/早盘+3%落袋)": R(rank_by="sector", am_exit_gap=0.03, am_tp=0.03),
    "V5 V4+整槽轮换":                    R(rank_by="sector", am_exit_gap=0.03, am_tp=0.03, swap_when_full=True),
    "V6 V1 只守板块门(不守个股门)":       R(rank_by="sector", use_stock_gate=False),
    "V7 V1 不止损(只 MA20/时间/止盈)":    R(rank_by="sector", sl=9.0),
}
rows = []
def add(name, seg, r, extra=""):
    rows.append(dict(变体=name, 段=seg, 年化=r.get("ann_return"), 夏普=r.get("sharpe"), 回撤=r.get("max_dd"), 仓位=r.get("exposure"),
                     交易=r.get("n_trades", np.nan), 胜率=r.get("win_rate", np.nan), 持有天=r.get("avg_hold", np.nan),
                     退出=json.dumps(r.get("exit_mix", {}), ensure_ascii=False) if r.get("exit_mix") else extra))
sig_cache = {}
for name, rules in variants.items():
    sig = EB.build_signals(p, rules, u, lab); sig_cache[name] = sig
    for sname, m in segs.items():
        if WITHHOLD and "H2" in sname and "2025" in sname: continue
        r = EB.run_event_bt(p, rules, version, date_mask=np.asarray(m), sector_labels=lab, universe=u, signals=sig)
        add(name, sname, r)
# 基线 A / B / 随机对照(用 V1 的守门集合)
sig1 = sig_cache["V1 两路径,板块排序"]; gate = (sig1["strong"] & u)
rng = np.random.default_rng(0)
for sname, m in segs.items():
    if WITHHOLD and "H2" in sname and "2025" in sname: continue
    m = np.asarray(m)
    ew = pd.DataFrame(0.0, index=d, columns=p.codes).where(gate)
    rA = ST.run_strategy(ew, p, version, "tail1450", scheme="top", n=100000, rebalance=5, date_mask=m)
    add("基线A 守门集合等权(周频)", sname, rA, f"守门集合中位 {int(gate[m].sum(axis=1).median())} 只")
    pool = ST.benchmark(p, version, "tail1450", date_mask=m); add("参照 全池等权", sname, pool)
    rB = EB.run_event_bt(p, R(rank_by="sector", use_sector_gate=False, use_stock_gate=False), version, date_mask=m, sector_labels=lab, universe=u)
    add("基线B 同规则不守门", sname, rB)
    # 随机对照:守门集合内每日随机挑与 V1 相同数量的候选(保持入场节奏),随机排序
    anns = []
    for k in range(10):
        e = sig1["entry"]; n_day = e.sum(axis=1)
        rnd = pd.DataFrame(rng.random(gate.shape), index=d, columns=p.codes).where(gate)
        thr = rnd.apply(lambda row: np.nan, axis=1)  # placeholder
        # 每日从守门集合里随机取 n_day 只作为 entry
        rank = rnd.rank(axis=1, method="first")
        e_rnd = (rank.le(n_day, axis=0)) & gate
        s2 = dict(sig1); s2["entry"] = e_rnd; s2["score"] = rnd.where(e_rnd); s2["score_all"] = rnd
        rr = EB.run_event_bt(p, R(rank_by="none"), version, date_mask=m, sector_labels=lab, universe=u, signals=s2)
        anns.append(rr["ann_return"])
    a = np.array(anns); v1 = [x for x in rows if x["变体"].startswith("V1 ") and x["段"] == sname][0]["年化"]
    add("随机对照 守门内随机入场(10次)", sname, dict(ann_return=float(a.mean()), sharpe=np.nan, max_dd=np.nan, exposure=np.nan),
        f"分布 [{a.min()*100:+.1f}%, {a.max()*100:+.1f}%];V1 落在第 {int((a < v1).mean()*100)} 百分位")
df = pd.DataFrame(rows); df.to_csv(LEDGER / f"eventbt2_{version}.csv", index=False)
print(f"=== 小碎步法复现 v2 | {version} | 尾盘口径 | 2025H2 {'隐藏' if WITHHOLD else '显示'} ===")
for sname in segs:
    if WITHHOLD and "H2" in sname and "2025" in sname: continue
    print(f"\n--- {sname} ---")
    print(df[df.段 == sname].drop(columns="段").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
