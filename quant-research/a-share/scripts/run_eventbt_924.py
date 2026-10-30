"""小碎步法复现 —— 以 2024-09-24 之后为考场(用户指定)。
三段:924 前(对照)/ 924 后 A(2024-09-24~2025-06-30)/ 924 后 B(2025H2,首次揭开,此后不再调参)。
变体与 v2 相同(规则在揭开 B 段之前已固定)。"""
import os, sys, warnings, json; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import eventbt as EB, strategy as ST
from afactor.config import LEDGER
pd.set_option("display.width", 260)
version = os.environ.get("EB_VERSION", "V100")
p = Panel(); d = p.dates
raw = pd.read_csv('data_8:21/source_raw/baostock_stock_industry_raw.csv'); raw['code'] = raw['code'].str.upper().str.replace(r'^(SH|SZ)\.(\d+)$', r'\2.\1', regex=True)
lab = pd.Series(dict(zip(raw.code, raw.industry))).reindex(p.codes).replace('', np.nan)
segs = {"924前 2024-01~09-23": (d >= "2024-01-01") & (d <= "2024-09-23"),
        "924后A 2024-09-24~2025-06": (d >= "2024-09-24") & (d <= "2025-06-30"),
        "924后B 2025H2(首次揭开)": (d >= "2025-07-01") & (d <= "2025-12-31"),
        "924后全段 2024-09-24~2025-12": (d >= "2024-09-24") & (d <= "2025-12-31")}
u = ST.universe(p, version); R = EB.Rules
variants = {"V1 两路径,板块排序": R(rank_by="sector"),
            "V2 仅突破前高回踩": R(path_lowopen=False, rank_by="sector"),
            "V6 V1 只守板块门": R(rank_by="sector", use_stock_gate=False),
            "V7 V1 不止损": R(rank_by="sector", sl=9.0)}
rows = []
def add(name, seg, r, extra=""):
    rows.append(dict(变体=name, 段=seg, 年化=r.get("ann_return"), 夏普=r.get("sharpe"), 回撤=r.get("max_dd"), 仓位=r.get("exposure"),
                     交易=r.get("n_trades", np.nan), 胜率=r.get("win_rate", np.nan), 持有天=r.get("avg_hold", np.nan),
                     退出=json.dumps(r.get("exit_mix", {}), ensure_ascii=False) if r.get("exit_mix") else extra))
sigs = {n: EB.build_signals(p, r_, u, lab) for n, r_ in variants.items()}
for name, rules in variants.items():
    for sname, m in segs.items():
        add(name, sname, EB.run_event_bt(p, rules, version, date_mask=np.asarray(m), sector_labels=lab, universe=u, signals=sigs[name]))
sig1 = sigs["V1 两路径,板块排序"]; gate = (sig1["strong"] & u); rng = np.random.default_rng(0)
for sname, m in segs.items():
    m = np.asarray(m)
    ew = pd.DataFrame(0.0, index=d, columns=p.codes).where(gate)
    add("基线A 守门集合等权", sname, ST.run_strategy(ew, p, version, "tail1450", scheme="top", n=100000, rebalance=5, date_mask=m), f"守门中位 {int(gate[m].sum(axis=1).median())} 只")
    add("参照 全池等权", sname, ST.benchmark(p, version, "tail1450", date_mask=m))
    add("基线B 同规则不守门", sname, EB.run_event_bt(p, R(rank_by="sector", use_sector_gate=False, use_stock_gate=False), version, date_mask=m, sector_labels=lab, universe=u))
    anns = []
    for k in range(10):
        e = sig1["entry"]; n_day = e.sum(axis=1)
        rnd = pd.DataFrame(rng.random(gate.shape), index=d, columns=p.codes).where(gate)
        e_rnd = (rnd.rank(axis=1, method="first").le(n_day, axis=0)) & gate
        s2 = dict(sig1); s2["entry"] = e_rnd; s2["score"] = rnd.where(e_rnd); s2["score_all"] = rnd
        anns.append(EB.run_event_bt(p, R(rank_by="none"), version, date_mask=m, sector_labels=lab, universe=u, signals=s2)["ann_return"])
    a = np.array(anns); v1 = [x for x in rows if x["变体"].startswith("V1 ") and x["段"] == sname][0]["年化"]; v2 = [x for x in rows if x["变体"].startswith("V2 ") and x["段"] == sname][0]["年化"]
    add("随机对照 守门内随机入场×10", sname, dict(ann_return=float(a.mean())), f"[{a.min()*100:+.1f}%, {a.max()*100:+.1f}%]; V1 第{int((a<v1).mean()*100)}百分位, V2 第{int((a<v2).mean()*100)}百分位")
df = pd.DataFrame(rows); df.to_csv(LEDGER / f"eventbt_924_{version}.csv", index=False)
for sname in segs:
    print(f"\n--- {sname} ---"); print(df[df.段 == sname].drop(columns="段").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
