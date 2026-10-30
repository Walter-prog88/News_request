"""XSB2-E1:深回调低吸主变体 + 全基线,分段表。协议 research/XSB2_PROTOCOL.md §6。

段:2021 / 2022 / 2023 / 2024-01~09-23 / 924~2025-06 / 2025H2 / 全段 2021-01~2025-12。
基线:A 守门集合等权 / B 同触发不守门 / 守门内随机入场×20 / 全池等权。
"""
import os, sys, warnings, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import eventbt as EB, strategy as ST
from afactor.eventbt_xsb2 import XRules, build_xsb2_signals
from afactor.config import LEDGER

pd.set_option("display.width", 300)
version = os.environ.get("XSB2_VERSION", "V100")
p = Panel(); d = p.dates
sw = pd.read_csv("stock_sw_industry.csv", encoding="utf-8-sig")
lab = pd.Series(dict(zip(sw.ts_code, sw.sw_industry))).reindex(p.codes).replace("未知", np.nan)
u = ST.universe(p, version)

segs = {"2021": (d >= "2021-01-01") & (d <= "2021-12-31"),
        "2022": (d >= "2022-01-01") & (d <= "2022-12-31"),
        "2023": (d >= "2023-01-01") & (d <= "2023-12-31"),
        "2024至924前": (d >= "2024-01-01") & (d <= "2024-09-23"),
        "924后A": (d >= "2024-09-24") & (d <= "2025-06-30"),
        "2025H2": (d >= "2025-07-01") & (d <= "2025-12-31"),
        "全段2021-2025": (d >= "2021-01-01") & (d <= "2025-12-31")}

# 引擎规则(退出层):tp 数值关,sl20,MA60,60 日,8×7.5%
ER = EB.Rules(tp=9.0, sl=0.20, exit_ma=60, max_hold=60, n_slots=8, slot_w=0.075,
              cooldown=5, am_exit_gap=0.03, rank_by="sector")

variants = {
    "主 T1∪T3": XRules(),
    "仅T1 放量反转": XRules(t3=False),
    "仅T3 缩量左侧": XRules(t1=False),
    "T1∪T2∪T3": XRules(t2=True),
    "主 不带extra退出": XRules(use_tp_hi=False, use_decay=False),
}
rows = []
def add(name, seg, r, extra=""):
    rows.append(dict(变体=name, 段=seg, 年化=r.get("ann_return"), 夏普=r.get("sharpe"),
                     回撤=r.get("max_dd"), 仓位=r.get("exposure"), 交易=r.get("n_trades", np.nan),
                     胜率=r.get("win_rate", np.nan), 单笔=r.get("avg_ret", np.nan),
                     持有天=r.get("avg_hold", np.nan),
                     退出=json.dumps(r.get("exit_mix", {}), ensure_ascii=False) if r.get("exit_mix") else extra))

sigs = {n: build_xsb2_signals(p, xr, u, lab) for n, xr in variants.items()}
main_sig = sigs["主 T1∪T3"]
print("主变体入场信号日均候选:", float(main_sig["entry"].sum(axis=1).mean()),
      " 守门集合日均:", float(main_sig["gate"].sum(axis=1).mean()), flush=True)

for name in variants:
    er = ER if name != "主 不带extra退出" else ER
    for sname, m in segs.items():
        add(name, sname, EB.run_event_bt(p, er, version, date_mask=np.asarray(m),
                                         sector_labels=lab, universe=u, signals=sigs[name]))
    print("done", name, flush=True)

# 基线 B:同触发不守门
sig_ng = build_xsb2_signals(p, XRules(gates_on=False), u, lab)
gate = main_sig["gate"]
rng = np.random.default_rng(0)
for sname, m in segs.items():
    m = np.asarray(m)
    ew = pd.DataFrame(0.0, index=d, columns=p.codes).where(gate)
    add("基线A 守门集合等权", sname,
        ST.run_strategy(ew, p, version, "tail1450", scheme="top", n=100000, rebalance=5, date_mask=m),
        f"守门中位 {int(gate[m].sum(axis=1).median())} 只")
    add("参照 全池等权", sname, ST.benchmark(p, version, "tail1450", date_mask=m))
    add("基线B 同触发不守门", sname, EB.run_event_bt(p, ER, version, date_mask=m,
                                              sector_labels=lab, universe=u, signals=sig_ng))
    anns = []
    e = main_sig["entry"]; n_day = e.sum(axis=1)
    for k in range(20):
        rnd = pd.DataFrame(rng.random(gate.shape), index=d, columns=p.codes).where(gate)
        e_rnd = (rnd.rank(axis=1, method="first").le(n_day, axis=0)) & gate
        s2 = dict(main_sig); s2["entry"] = e_rnd; s2["score"] = rnd.where(e_rnd); s2["score_all"] = rnd
        anns.append(EB.run_event_bt(p, ER, version, date_mask=m, sector_labels=lab,
                                    universe=u, signals=s2)["ann_return"])
    a = np.array([x for x in anns if np.isfinite(x)])
    v_main = [x for x in rows if x["变体"] == "主 T1∪T3" and x["段"] == sname][0]["年化"]
    add("随机对照 守门内随机×20", sname, dict(ann_return=float(a.mean())),
        f"[{a.min()*100:+.1f}%, {a.max()*100:+.1f}%]; 主变体第{int((a < v_main).mean()*100)}百分位")
    print("done 基线", sname, flush=True)

df = pd.DataFrame(rows)
df.to_csv(LEDGER / f"xsb2_e1_{version}.csv", index=False)
for sname in segs:
    print(f"\n--- {sname} ---")
    print(df[df.段 == sname].drop(columns="段").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
