"""XSB2-E3:冻结网格扫描(协议 §1)——裁决步。

网格:rally_min{0.5,1.0} × pb_min{0.20,0.30} × 触发{T1,T3,T1∪T3,T1∪T2∪T3} × sl{0.12,0.20,关}
     × exit_ma{关,250}(E2 修订臂)= 96 变体。
每个唯一门(rally,pb)配:守门集合等权 + 守门内随机×5(轻量;主变体的×20 在 E2)。
判据:同时 > 守门等权 且 > 随机均值 且 > 全池,才算候选;预期按 E1/E2 为全灭,如实记录。
"""
import os, sys, warnings, itertools
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
        "924后A": (d >= "2024-09-24") & (d <= "2025-06-30")}
TRIG = {"T1": dict(t1=True, t3=False), "T3": dict(t1=False, t3=True),
        "T1uT3": dict(), "T1uT2uT3": dict(t2=True)}
rows = []
gate_cache = {}
rng = np.random.default_rng(0)

for rally, pb, (tname, tkw), ma in itertools.product(
        [0.5, 1.0], [0.20, 0.30], TRIG.items(), [None, 250]):
    xr = XRules(rally_min=rally, pb_min=pb, exit_ma=ma, **tkw)
    sig = build_xsb2_signals(p, xr, u, lab)
    gkey = (rally, pb)
    if gkey not in gate_cache:
        gate = sig["gate"]
        ew = pd.DataFrame(0.0, index=d, columns=p.codes).where(gate)
        ref = {}
        for sname, m in segs.items():
            m = np.asarray(m)
            ref[sname] = dict(
                gate_ew=ST.run_strategy(ew, p, version, "tail1450", scheme="top",
                                        n=100000, rebalance=5, date_mask=m)["ann_return"],
                pool=ST.benchmark(p, version, "tail1450", date_mask=m)["ann_return"])
        gate_cache[gkey] = (gate, ref)
        print("gate", gkey, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in ref.items()}, flush=True)
    gate, ref = gate_cache[gkey]
    for sl_tag, sl in [("sl12", 0.12), ("sl20", 0.20), ("slOFF", 9.0)]:
        er = EB.Rules(tp=9.0, sl=sl, exit_ma=60, max_hold=60, n_slots=8, slot_w=0.075,
                      cooldown=5, am_exit_gap=0.03)
        for sname, m in segs.items():
            m = np.asarray(m)
            r = EB.run_event_bt(p, er, version, date_mask=m, sector_labels=lab,
                                universe=u, signals=sig)
            # 随机×5(同门同退出臂同 sl,配对)
            anns = []
            n_day = sig["entry"].sum(axis=1)
            for k in range(5):
                rnd = pd.DataFrame(rng.random(gate.shape), index=d, columns=p.codes).where(gate)
                e_rnd = (rnd.rank(axis=1, method="first").le(n_day, axis=0)) & gate
                s2 = dict(sig); s2["entry"] = e_rnd; s2["score"] = rnd.where(e_rnd); s2["score_all"] = rnd
                anns.append(EB.run_event_bt(p, er, version, date_mask=m, sector_labels=lab,
                                            universe=u, signals=s2)["ann_return"])
            a = np.array([x for x in anns if np.isfinite(x)])
            rows.append(dict(rally=rally, pb=pb, trig=tname, exit_ma=ma or 0, sl=sl_tag, 段=sname,
                             年化=r["ann_return"], 夏普=r["sharpe"], 回撤=r["max_dd"],
                             交易=r["n_trades"], 胜率=r["win_rate"],
                             随机均=float(a.mean()) if len(a) else np.nan,
                             守门等权=ref[sname]["gate_ew"], 全池=ref[sname]["pool"],
                             过三关=bool(r["ann_return"] > max(float(a.mean()) if len(a) else -9,
                                                          ref[sname]["gate_ew"], ref[sname]["pool"]))))
    print("done", rally, pb, tname, ma, flush=True)

df = pd.DataFrame(rows)
df.to_csv(LEDGER / f"xsb2_e3_{version}.csv", index=False)
full = df[df.段 == "全段2021-2025"]
print(f"\n=== 全段:{len(full)} 变体,过三关 {int(full.过三关.sum())} 个 ===")
print(full.sort_values("年化", ascending=False).head(12).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
post = df[df.段 == "924后A"]
print(f"\n=== 924后A:过三关 {int(post.过三关.sum())} 个 ===")
print(post.sort_values("年化", ascending=False).head(8).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
