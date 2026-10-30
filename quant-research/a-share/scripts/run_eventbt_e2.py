"""E2:固定止盈/止损砍掉了多少 —— 在默认守门(Rules 默认, rank_by='sector')下扫退出规则。
tp {0.08,0.15,0.25,9.0} x sl {0.05,0.08,9.0} x exit_ma {20,60} x max_hold {15,40} = 48 组。
考场 2024-09-24 ~ 2025-12-31(已全部揭开,本次为样本内诊断)。
输出:48 组绩效表;最好 3 组 + 默认 的逐月收益表(vs 全池等权 vs 守门集合等权);
最好一组 vs 守门集合等权逐月差;守门内随机入场 x10 在默认/最好退出规则下的分布(配对同种子)。"""
import os, sys, warnings, json, itertools, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import eventbt as EB, strategy as ST
from afactor.config import LEDGER
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50)
version = os.environ.get("EB_VERSION", "V100")
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

p = Panel(); d = p.dates
raw = pd.read_csv('data_8:21/source_raw/baostock_stock_industry_raw.csv'); raw['code'] = raw['code'].str.upper().str.replace(r'^(SH|SZ)\.(\d+)$', r'\2.\1', regex=True)
lab = pd.Series(dict(zip(raw.code, raw.industry))).reindex(p.codes).replace('', np.nan)
m = np.asarray((d >= "2024-09-24") & (d <= "2025-12-31"))
u = ST.universe(p, version); R = EB.Rules

# 信号只依赖 exit_ma(ma_exit),其余两套共用
sigs = {ma: EB.build_signals(p, R(rank_by="sector", exit_ma=ma), u, lab) for ma in (20, 60)}
log("signals built")

def monthly(rets: pd.Series) -> pd.Series:
    r = rets.copy(); r.index = pd.DatetimeIndex(r.index)
    return (1 + r).groupby(r.index.to_period("M")).prod() - 1

grid = list(itertools.product([0.08, 0.15, 0.25, 9.0], [0.05, 0.08, 9.0], [20, 60], [15, 40]))
rows, results = [], {}
for tp, sl, ma, mh in grid:
    rules = R(rank_by="sector", tp=tp, sl=sl, exit_ma=ma, max_hold=mh)
    r = EB.run_event_bt(p, rules, version, date_mask=m, sector_labels=lab, universe=u, signals=sigs[ma])
    key = (tp, sl, ma, mh); results[key] = r
    rows.append(dict(tp=tp, sl=sl, exit_ma=ma, max_hold=mh, 年化=r["ann_return"], 夏普=r["sharpe"], 回撤=r["max_dd"], 仓位=r["exposure"],
                     交易=r["n_trades"], 胜率=r["win_rate"], 平均单笔=r["avg_ret"], 持有天=r["avg_hold"], 期末净值=float(r["nav"].iloc[-1]),
                     退出=json.dumps(r["exit_mix"], ensure_ascii=False)))
    log(f"tp={tp} sl={sl} ma={ma} mh={mh}: ann={r['ann_return']:+.3f} sharpe={r['sharpe']:.2f} dd={r['max_dd']:.3f} n={r['n_trades']} hold={r['avg_hold']:.1f} wr={r['win_rate']:.2f}")
df = pd.DataFrame(rows).sort_values("年化", ascending=False).reset_index(drop=True)
df.to_csv(LEDGER / f"eventbt_e2_grid_{version}.csv", index=False)
print("\n===== E2 48 组退出规则扫描(2024-09-24~2025-12-31,样本内;按年化降序)=====")
print(df.to_string(index=True, float_format=lambda x: f"{x:.3f}"))

# 边际效应:按每个维度取均值
print("\n===== 各维度边际(年化均值 / 夏普均值 / 回撤均值)=====")
for col in ["tp", "sl", "exit_ma", "max_hold"]:
    g = df.groupby(col).agg(年化均值=("年化", "mean"), 夏普均值=("夏普", "mean"), 回撤均值=("回撤", "mean"), 交易均值=("交易", "mean"), 持有天均值=("持有天", "mean"))
    print(f"-- {col} --"); print(g.to_string(float_format=lambda x: f"{x:.3f}"))

# ---- 基线:守门集合等权 A、全池等权 ----
sig1 = sigs[20]; gate = (sig1["strong"] & u)
ew = pd.DataFrame(0.0, index=d, columns=p.codes).where(gate)
gate_ew = ST.run_strategy(ew, p, version, "tail1450", scheme="top", n=100000, rebalance=5, date_mask=m)
pool_ew = ST.benchmark(p, version, "tail1450", date_mask=m)
log(f"gateEW ann={gate_ew['ann_return']:+.3f} poolEW ann={pool_ew['ann_return']:+.3f}")

# ---- 逐月收益表 ----
default_key = (0.08, 0.05, 20, 15)
best_keys = [tuple(x) for x in df.head(3)[["tp", "sl", "exit_ma", "max_hold"]].to_numpy()]
best_keys = [(float(a), float(b), int(c), int(e)) for a, b, c, e in best_keys]
def kname(k): return f"tp{k[0]:g}/sl{k[1]:g}/ma{k[2]}/h{k[3]}"
cols = {f"默认 {kname(default_key)}": monthly(results[default_key]["rets"])}
for i, k in enumerate(best_keys, 1):
    cols[f"最好{i} {kname(k)}"] = monthly(results[k]["rets"])
cols["守门集合等权"] = monthly(gate_ew["rets"]); cols["全池等权"] = monthly(pool_ew["rets"])
M = pd.DataFrame(cols)
M[f"最好1−守门等权"] = M.iloc[:, 1] - M["守门集合等权"]
M.to_csv(LEDGER / f"eventbt_e2_monthly_{version}.csv")
print("\n===== 逐月收益表(自然月,%)=====")
print((M * 100).to_string(float_format=lambda x: f"{x:+.1f}"))
print("\n累计(期末净值):", {c: round(float((1 + M[c]).prod()), 3) for c in M.columns if "−" not in c})
print("月收益统计(均值 / 中位 / 正月数 / ≥5% 月数 / 总月数):")
for c in M.columns:
    s = M[c]; print(f"  {c:28s} mean={s.mean()*100:+.2f}% med={s.median()*100:+.2f}% pos={int((s>0).sum())} ge5={int((s>=0.05).sum())} n={len(s)}")

# ---- 随机对照:守门内随机入场 x10,配对同种子,默认 vs 最好 退出规则 ----
def random_control(key, seed=0, n_rep=10):
    rng = np.random.default_rng(seed); anns = []; mdds = []
    tp, sl, ma, mh = key; base = sigs[ma]
    e = base["entry"]; n_day = e.sum(axis=1)
    for k in range(n_rep):
        rnd = pd.DataFrame(rng.random(gate.shape), index=d, columns=p.codes).where(gate)
        e_rnd = (rnd.rank(axis=1, method="first").le(n_day, axis=0)) & gate
        s2 = dict(base); s2["entry"] = e_rnd; s2["score"] = rnd.where(e_rnd); s2["score_all"] = rnd
        r = EB.run_event_bt(p, R(rank_by="none", tp=tp, sl=sl, exit_ma=ma, max_hold=mh), version, date_mask=m, sector_labels=lab, universe=u, signals=s2)
        anns.append(r["ann_return"]); mdds.append(r["max_dd"])
    return np.array(anns), np.array(mdds)
print("\n===== 随机对照:守门内随机入场 x10(同种子配对)=====")
rc = {}
for label, key in [("默认", default_key), ("最好1", best_keys[0])]:
    a, dd = random_control(key); strat = results[key]["ann_return"]; rc[label] = a
    log(f"{label} {kname(key)}: 策略年化={strat:+.3f}; 随机均值={a.mean():+.3f} 中位={np.median(a):+.3f} [{a.min():+.3f},{a.max():+.3f}] 随机回撤均值={dd.mean():.3f}; 策略在随机分布第{int((a<strat).mean()*100)}百分位")
    print("   随机 10 次年化:", np.round(a, 3).tolist())
diff = rc["最好1"] - rc["默认"]
print(f"配对差(最好1 − 默认)随机对照:均值={diff.mean():+.3f} 中位={np.median(diff):+.3f} [{diff.min():+.3f},{diff.max():+.3f}];"
      f" 策略差={results[best_keys[0]]['ann_return'] - results[default_key]['ann_return']:+.3f}")
log("done")
