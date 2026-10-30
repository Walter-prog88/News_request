"""XSB2-E4:幸存家族验证 —— 滞后一天 + 种子扰动 + 全段/924后 双窗。

E3 幸存家族(5/96,聚在同一格族):T3 缩量左侧 × pb_min=0.30 × exit_ma 关 × sl{12,20,关} × rally{0.5,1.0}。
验证:(a) 入场推迟 1 日重跑,衰减须平缓(慢变量形态信号应不敏感);
     (b) 排序种子扰动 ×5(score 加微噪声),报极差(槽位争用路径噪声);
     (c) am_gap 关闭消融(退出层依赖检查)。
"""
import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import eventbt as EB, strategy as ST
from afactor.eventbt_xsb2 import XRules, build_xsb2_signals
from afactor.config import LEDGER

version = os.environ.get("XSB2_VERSION", "V100")
p = Panel(); d = p.dates
sw = pd.read_csv("stock_sw_industry.csv", encoding="utf-8-sig")
lab = pd.Series(dict(zip(sw.ts_code, sw.sw_industry))).reindex(p.codes).replace("未知", np.nan)
u = ST.universe(p, version)
segs = {"全段2021-2025": (d >= "2021-01-01") & (d <= "2025-12-31"),
        "924后A": (d >= "2024-09-24") & (d <= "2025-06-30"),
        "2025H2": (d >= "2025-07-01") & (d <= "2025-12-31")}

WINNER = dict(t1=False, t3=True, pb_min=0.30, exit_ma=None)
rows = []
def add(tag, seg, r):
    rows.append(dict(检验=tag, 段=seg, 年化=r.get("ann_return"), 夏普=r.get("sharpe"),
                     回撤=r.get("max_dd"), 仓位=r.get("exposure"), 交易=r.get("n_trades", np.nan),
                     胜率=r.get("win_rate", np.nan)))

for rally in [0.5, 1.0]:
    for sl_tag, sl in [("sl12", 0.12), ("sl20", 0.20)]:
        xr = XRules(rally_min=rally, **WINNER)
        sig = build_xsb2_signals(p, xr, u, lab)
        er = EB.Rules(tp=9.0, sl=sl, exit_ma=60, max_hold=60, n_slots=8, slot_w=0.075,
                      cooldown=5, am_exit_gap=0.03)
        base_tag = f"r{rally}_{sl_tag}"
        for sname, m in segs.items():
            m = np.asarray(m)
            add(f"{base_tag} 原始", sname, EB.run_event_bt(p, er, version, date_mask=m,
                                                        sector_labels=lab, universe=u, signals=sig))
            # (a) 滞后一天
            lag = dict(sig)
            lag["entry"] = sig["entry"].shift(1).fillna(False)
            lag["score"] = sig["score_all"].where(lag["entry"])
            add(f"{base_tag} 滞后1日", sname, EB.run_event_bt(p, er, version, date_mask=m,
                                                          sector_labels=lab, universe=u, signals=lag))
            # (c) am_gap 关
            er2 = EB.Rules(tp=9.0, sl=sl, exit_ma=60, max_hold=60, n_slots=8, slot_w=0.075,
                           cooldown=5, am_exit_gap=None)
            add(f"{base_tag} 无am_gap", sname, EB.run_event_bt(p, er2, version, date_mask=m,
                                                           sector_labels=lab, universe=u, signals=sig))
            # (b) 种子扰动 ×5
            anns = []
            rng = np.random.default_rng(7)
            for k in range(5):
                s2 = dict(sig)
                noise = pd.DataFrame(rng.random(sig["score_all"].shape) * 1e-6,
                                     index=d, columns=p.codes)
                s2["score"] = (sig["score_all"] + noise).where(sig["entry"])
                s2["score_all"] = sig["score_all"] + noise
                anns.append(EB.run_event_bt(p, er, version, date_mask=m, sector_labels=lab,
                                            universe=u, signals=s2)["ann_return"])
            a = np.array(anns)
            rows.append(dict(检验=f"{base_tag} 种子×5", 段=sname, 年化=float(a.mean()),
                             夏普=np.nan, 回撤=np.nan, 仓位=np.nan,
                             交易=np.nan, 胜率=float(a.max() - a.min())))  # 胜率列借放极差
        print("done", base_tag, flush=True)

df = pd.DataFrame(rows)
df.to_csv(LEDGER / f"xsb2_e4_{version}.csv", index=False)
for sname in segs:
    print(f"\n--- {sname} ---")
    print(df[df.段 == sname].drop(columns="段").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n注:种子×5 行的'胜率'列 = 年化极差(max-min)")
