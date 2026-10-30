"""XSB2-E2:退出臂修正(网格既有 exit_ma{60,关} + 修订新增 MA250 臂)。

E1 失败机理(先写机理再改参数,walkforward 代际纪律):
深回调入场天然在 MA60 下方(取证中位 -15%),exit_ma=60 令 80% 交易开仓即被踢出,
平均持有 3.5 天,策略退化为换手机器。E1 的负收益首先是退出层与入场层自相矛盾的工程产物,
不能据以裁决入场信号。修订:exit_ma ∈ {None, 250}(250 与 G2 年线门自洽);其余参数不动。
网格修订已入账:去偏分母按账本行数累计。
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

ER = EB.Rules(tp=9.0, sl=0.20, exit_ma=60, max_hold=60, n_slots=8, slot_w=0.075,
              cooldown=5, am_exit_gap=0.03, rank_by="sector")

variants = {}
for ma_tag, ma in [("noMA", None), ("MA250", 250)]:
    variants[f"主 T1∪T3 {ma_tag}"] = XRules(exit_ma=ma)
    variants[f"仅T1 {ma_tag}"] = XRules(t3=False, exit_ma=ma)
    variants[f"仅T3 {ma_tag}"] = XRules(t1=False, exit_ma=ma)
    variants[f"T1∪T2∪T3 {ma_tag}"] = XRules(t2=True, exit_ma=ma)

rows = []
def add(name, seg, r, extra=""):
    rows.append(dict(变体=name, 段=seg, 年化=r.get("ann_return"), 夏普=r.get("sharpe"),
                     回撤=r.get("max_dd"), 仓位=r.get("exposure"), 交易=r.get("n_trades", np.nan),
                     胜率=r.get("win_rate", np.nan), 单笔=r.get("avg_ret", np.nan),
                     持有天=r.get("avg_hold", np.nan),
                     退出=json.dumps(r.get("exit_mix", {}), ensure_ascii=False) if r.get("exit_mix") else extra))

sigs = {n: build_xsb2_signals(p, xr, u, lab) for n, xr in variants.items()}
for name in variants:
    for sname, m in segs.items():
        add(name, sname, EB.run_event_bt(p, ER, version, date_mask=np.asarray(m),
                                         sector_labels=lab, universe=u, signals=sigs[name]))
    print("done", name, flush=True)

# 基线(与 E1 相同的 A/池;随机对照按两个退出臂各配一组)
main_noma = sigs["主 T1∪T3 noMA"]; gate = main_noma["gate"]
rng = np.random.default_rng(0)
for sname, m in segs.items():
    m = np.asarray(m)
    ew = pd.DataFrame(0.0, index=d, columns=p.codes).where(gate)
    add("基线A 守门集合等权", sname,
        ST.run_strategy(ew, p, version, "tail1450", scheme="top", n=100000, rebalance=5, date_mask=m),
        f"守门中位 {int(gate[m].sum(axis=1).median())} 只")
    add("参照 全池等权", sname, ST.benchmark(p, version, "tail1450", date_mask=m))
    for ma_tag in ["noMA", "MA250"]:
        base = sigs[f"主 T1∪T3 {ma_tag}"]
        anns = []
        e = base["entry"]; n_day = e.sum(axis=1)
        for k in range(20):
            rnd = pd.DataFrame(rng.random(gate.shape), index=d, columns=p.codes).where(gate)
            e_rnd = (rnd.rank(axis=1, method="first").le(n_day, axis=0)) & gate
            s2 = dict(base); s2["entry"] = e_rnd; s2["score"] = rnd.where(e_rnd); s2["score_all"] = rnd
            anns.append(EB.run_event_bt(p, ER, version, date_mask=m, sector_labels=lab,
                                        universe=u, signals=s2)["ann_return"])
        a = np.array([x for x in anns if np.isfinite(x)])
        v_main = [x for x in rows if x["变体"] == f"主 T1∪T3 {ma_tag}" and x["段"] == sname][0]["年化"]
        add(f"随机×20 {ma_tag}", sname, dict(ann_return=float(a.mean())),
            f"[{a.min()*100:+.1f}%, {a.max()*100:+.1f}%]; 主{ma_tag}第{int((a < v_main).mean()*100)}百分位")
    print("done 基线", sname, flush=True)

df = pd.DataFrame(rows)
df.to_csv(LEDGER / f"xsb2_e2_{version}.csv", index=False)
for sname in segs:
    print(f"\n--- {sname} ---")
    print(df[df.段 == sname].drop(columns="段").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
