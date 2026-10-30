"""E2 补充:守门内随机入场 x30(配对同种子)在 默认 / 仅换MA60 / 最好1 三套退出规则下的分布。
判别"改退出"是让信号更有效(策略相对随机分布的分位上升),还是只让随机入场也变好(纯 β/持有期暴露)。"""
import os, sys, warnings, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import eventbt as EB, strategy as ST
from afactor.config import LEDGER
version = "V100"; T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
p = Panel(); d = p.dates
raw = pd.read_csv('data_8:21/source_raw/baostock_stock_industry_raw.csv'); raw['code'] = raw['code'].str.upper().str.replace(r'^(SH|SZ)\.(\d+)$', r'\2.\1', regex=True)
lab = pd.Series(dict(zip(raw.code, raw.industry))).reindex(p.codes).replace('', np.nan)
m = np.asarray((d >= "2024-09-24") & (d <= "2025-12-31"))
u = ST.universe(p, version); R = EB.Rules
sigs = {ma: EB.build_signals(p, R(rank_by="sector", exit_ma=ma), u, lab) for ma in (20, 60)}
gate = (sigs[20]["strong"] & u)
def monthly(rets):
    r = rets.copy(); r.index = pd.DatetimeIndex(r.index); return (1 + r).groupby(r.index.to_period("M")).prod() - 1
keys = {"默认 tp8/sl5/ma20/h15": (0.08, 0.05, 20, 15), "仅换MA60 tp8/sl5/ma60/h15": (0.08, 0.05, 60, 15),
        "最好1 tp25/sl关/ma60/h40": (0.25, 9.0, 60, 40), "最好2 tp25/sl8/ma60/h40": (0.25, 0.08, 60, 40)}
N_REP = 30
# 预生成 30 套随机入场(所有规则共用 -> 严格配对)
rng = np.random.default_rng(0); n_day = sigs[20]["entry"].sum(axis=1); draws = []
for k in range(N_REP):
    rnd = pd.DataFrame(rng.random(gate.shape), index=d, columns=p.codes).where(gate)
    e_rnd = (rnd.rank(axis=1, method="first").le(n_day, axis=0)) & gate
    draws.append((e_rnd, rnd))
log("draws ready")
out = {}; monthly_rc = {}
for label, (tp, sl, ma, mh) in keys.items():
    rules = R(rank_by="sector", tp=tp, sl=sl, exit_ma=ma, max_hold=mh)
    strat = EB.run_event_bt(p, rules, version, date_mask=m, sector_labels=lab, universe=u, signals=sigs[ma])
    anns, shs, dds, mons = [], [], [], []
    for e_rnd, rnd in draws:
        s2 = dict(sigs[ma]); s2["entry"] = e_rnd; s2["score"] = rnd.where(e_rnd); s2["score_all"] = rnd
        r = EB.run_event_bt(p, R(rank_by="none", tp=tp, sl=sl, exit_ma=ma, max_hold=mh), version, date_mask=m, sector_labels=lab, universe=u, signals=s2)
        anns.append(r["ann_return"]); shs.append(r["sharpe"]); dds.append(r["max_dd"]); mons.append(monthly(r["rets"]))
    a = np.array(anns); sh = np.array(shs); dd = np.array(dds)
    out[label] = dict(strat_ann=strat["ann_return"], strat_sharpe=strat["sharpe"], strat_dd=strat["max_dd"], rc_ann=a, rc_sharpe=sh, rc_dd=dd)
    monthly_rc[label] = pd.concat(mons, axis=1)
    log(f"{label}: 策略 年化={strat['ann_return']:+.3f} 夏普={strat['sharpe']:.2f} 回撤={strat['max_dd']:.3f} | 随机x{N_REP} 年化 均值={a.mean():+.3f} 中位={np.median(a):+.3f} "
        f"P10={np.percentile(a,10):+.3f} P90={np.percentile(a,90):+.3f} [{a.min():+.3f},{a.max():+.3f}] 夏普均值={sh.mean():.2f} 回撤均值={dd.mean():.3f} | 策略在随机分布第{int((a<strat['ann_return']).mean()*100)}百分位; 超随机均值 {strat['ann_return']-a.mean():+.3f}")
labs = list(keys)
print("\n===== 配对差(同一随机入场序列,换退出规则)=====")
base = out[labs[0]]
for lb in labs[1:]:
    diff = out[lb]["rc_ann"] - base["rc_ann"]; sd = out[lb]["strat_ann"] - base["strat_ann"]
    print(f"{lb} − 默认:随机对照差 均值={diff.mean():+.3f} 中位={np.median(diff):+.3f} [{diff.min():+.3f},{diff.max():+.3f}] 正比例={(diff>0).mean():.2f} | 策略差={sd:+.3f} | 策略差−随机差均值={sd-diff.mean():+.3f}")
print("\n===== 随机对照逐月收益(30 次的中位 / P10 / P90,%)=====")
tab = {}
for lb in labs:
    Mr = monthly_rc[lb]; tab[f"{lb} 中位"] = Mr.median(axis=1); tab[f"{lb} P10"] = Mr.quantile(0.1, axis=1); tab[f"{lb} P90"] = Mr.quantile(0.9, axis=1)
T = pd.DataFrame(tab); T.to_csv(LEDGER / "eventbt_e2_rc_monthly.csv")
print((T[[c for c in T.columns if "中位" in c]] * 100).to_string(float_format=lambda x: f"{x:+.1f}"))
pd.DataFrame({lb: out[lb]["rc_ann"] for lb in labs}).to_csv(LEDGER / "eventbt_e2_rc_ann.csv", index=False)
log("done")
