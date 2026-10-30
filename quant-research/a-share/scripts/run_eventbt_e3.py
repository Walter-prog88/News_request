"""E3 仓位与槽位 —— 小碎步法事件引擎,考场 2024-09-24 ~ 2025-12-31(样本内诊断,非验证)。

扫描 (n_slots, slot_w):(5,0.11) 基准 / (5,0.15) 插值点 / (5,0.18) / (8,0.11) / (3,0.30) / (5,0.20) 满仓上限
规则两版:Rules(rank_by='sector') 与 Rules(rank_by='sector', sl=9.0)(不止损)
资金两版:V100(100 万)与 V1000(1000 万,接近交易者 2026 年规模)
输出:年化/夏普/回撤/平均仓位/满槽天数占比/参与率卡住入场次数/部分成交次数/成交率;
      逐月收益表((5,0.18) 与 (3,0.30) vs 全池等权 vs 守门集合等权)。
现金 = 1 − Σ槽位,引擎不用额外参数;(5,0.18) 自带 10% 现金底。
"""
import os, sys, warnings, json, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import eventbt as EB, strategy as ST
from afactor.config import LEDGER
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

p = Panel(); d = p.dates
assert d.max() <= pd.Timestamp("2025-12-31"), "不得触及封存期"
raw = pd.read_csv('data_8:21/source_raw/baostock_stock_industry_raw.csv'); raw['code'] = raw['code'].str.upper().str.replace(r'^(SH|SZ)\.(\d+)$', r'\2.\1', regex=True)
lab = pd.Series(dict(zip(raw.code, raw.industry))).reindex(p.codes).replace('', np.nan)
mask = np.asarray((d >= "2024-09-24") & (d <= "2025-12-31"))
R = EB.Rules
bases = {"止损5%": dict(rank_by="sector"), "不止损": dict(rank_by="sector", sl=9.0)}
grid = [(5, 0.11), (5, 0.15), (5, 0.18), (8, 0.11), (3, 0.30), (5, 0.20)]
versions = ["V100", "V1000"]


def pos_count_series(trades: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    """由成交记录重建逐日持仓只数(完整退出才释放槽)。"""
    n = pd.Series(0, index=idx, dtype=int)
    if len(trades) == 0: return n
    delta = pd.Series(0, index=idx, dtype=int)
    for _, r in trades.iterrows():
        if r.side == "buy": delta[r.date] += 1
        elif r.side == "sell" and r.frac >= 0.999: delta[r.date] -= 1
    return delta.cumsum()


def monthly(rets: pd.Series, m) -> pd.Series:
    r = rets[(rets.index >= "2024-09-24") & (rets.index <= "2025-12-31")]
    return (1 + r).groupby(r.index.to_period("M")).prod() - 1


rows, monthly_cols, nav_store = [], {}, {}
for version in versions:
    u = ST.universe(p, version)
    log(f"== {version}: 池中位 {int(u[mask].sum(axis=1).median())} 只 ==")
    sig = EB.build_signals(p, R(rank_by="sector"), u, lab)   # 信号不依赖 sl / 槽位,两版规则共用
    log("signals built")
    for bname, bkw in bases.items():
        for (ns, sw) in grid:
            rules = R(n_slots=ns, slot_w=sw, **bkw)
            r = EB.run_event_bt(p, rules, version, date_mask=mask, sector_labels=lab, universe=u, signals=sig)
            tr = r["trades"]; buys = tr[tr.side == "buy"] if len(tr) else tr
            st = r["stats"]
            npos = pos_count_series(tr, r["nav"].index)
            full_frac = float((npos >= ns).mean())
            n_buy_attempt = len(buys) + st["cap_blocked_entries"]
            row = dict(版本=version, 规则=bname, 槽位=ns, 槽重=sw, 最大仓位=round(ns * sw, 2),
                       年化=r["ann_return"], 夏普=r["sharpe"], 回撤=r["max_dd"], 平均仓位=r["exposure"],
                       满槽天占比=full_frac, 平均持仓只数=float(npos.mean()),
                       交易=r["n_trades"], 胜率=r["win_rate"], 单笔均收=r["avg_ret"], 持有天=r["avg_hold"],
                       入场尝试=n_buy_attempt, 卡住入场=st["cap_blocked_entries"], 部分成交入场=st["partial_entries"],
                       部分成交退出=st["partial_exits"], 入场成交率=len(buys) / n_buy_attempt if n_buy_attempt else np.nan,
                       买入均成交比=float(buys.frac.mean()) if len(buys) else np.nan,
                       候选日=st["cand_days"], 候选总数=st["cand_total"],
                       退出=json.dumps(r["exit_mix"], ensure_ascii=False))
            rows.append(row)
            key = f"{version}|{bname}|({ns},{sw})"
            nav_store[key] = r["nav"]
            if (ns, sw) in [(5, 0.18), (3, 0.30), (5, 0.11)]:
                monthly_cols[key] = monthly(r["rets"], mask)
            log(f"{key}: ann {r['ann_return']*100:+.1f}% sh {r['sharpe']:.2f} dd {r['max_dd']*100:.1f}% expo {r['exposure']:.3f} full {full_frac:.2f} blocked {st['cap_blocked_entries']} partial {st['partial_entries']}")
    # 基线:守门集合等权(基线 A)与全池等权
    gate = (sig["strong"] & u)
    ew = pd.DataFrame(0.0, index=d, columns=p.codes).where(gate)
    bA = ST.run_strategy(ew, p, version, "tail1450", scheme="top", n=100000, rebalance=5, date_mask=mask)
    bU = ST.benchmark(p, version, "tail1450", date_mask=mask)
    rows.append(dict(版本=version, 规则="基线A 守门集合等权", 槽位=np.nan, 槽重=np.nan, 最大仓位=1.0, 年化=bA["ann_return"], 夏普=bA["sharpe"], 回撤=bA["max_dd"], 平均仓位=bA["exposure"]))
    rows.append(dict(版本=version, 规则="参照 全池等权", 槽位=np.nan, 槽重=np.nan, 最大仓位=1.0, 年化=bU["ann_return"], 夏普=bU["sharpe"], 回撤=bU["max_dd"], 平均仓位=1.0))
    monthly_cols[f"{version}|守门集合等权"] = monthly(bA["rets"], mask)
    monthly_cols[f"{version}|全池等权"] = monthly(bU["rets"], mask)
    nav_store[f"{version}|守门集合等权"] = bA["nav"]; nav_store[f"{version}|全池等权"] = bU["nav"]
    log(f"{version} 基线A 守门等权 ann {bA['ann_return']*100:+.1f}%  全池等权 ann {bU['ann_return']*100:+.1f}%")

df = pd.DataFrame(rows)
df.to_csv(LEDGER / "eventbt_e3.csv", index=False)
mdf = pd.DataFrame(monthly_cols)
mdf.to_csv(LEDGER / "eventbt_e3_monthly.csv")
pd.DataFrame(nav_store).to_csv(LEDGER / "eventbt_e3_nav.csv")

print("\n" + "=" * 100 + "\n扫描结果(考场 2024-09-24~2025-12-31,样本内)\n" + "=" * 100)
show = ["版本", "规则", "槽位", "槽重", "最大仓位", "年化", "夏普", "回撤", "平均仓位", "满槽天占比", "平均持仓只数", "交易", "胜率", "持有天",
        "入场尝试", "卡住入场", "部分成交入场", "部分成交退出", "入场成交率", "买入均成交比"]
for v in versions:
    print(f"\n--- {v} ---")
    print(df[df.版本 == v][show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n退出结构:")
print(df[["版本", "规则", "槽位", "槽重", "退出"]].dropna(subset=["槽位"]).to_string(index=False))

print("\n" + "=" * 100 + "\n逐月收益表(自然月;2024-09 为 09-24~09-30 残月)\n" + "=" * 100)
for v in versions:
    cols = [c for c in mdf.columns if c.startswith(v + "|")]
    sub = mdf[cols].copy(); sub.columns = [c.split("|", 1)[1] for c in cols]
    print(f"\n--- {v} ---")
    print((sub * 100).to_string(float_format=lambda x: f"{x:+.1f}"))
    print("月收益统计(%): 中位 / 均值 / ≥+5% 月数 / <0 月数 / 共 N 月")
    for c in sub.columns:
        s = sub[c].dropna() * 100
        print(f"  {c:28s} {s.median():+5.1f} / {s.mean():+5.1f} / {int((s >= 5).sum()):2d} / {int((s < 0).sum()):2d} / {len(s)}")

print("\n" + "=" * 100 + "\n仓位 55% -> 90% 的倍数(同规则同版本,(5,0.18)/(5,0.11) 与 (3,0.30)/(5,0.11))\n" + "=" * 100)
for v in versions:
    for b in bases:
        base = df[(df.版本 == v) & (df.规则 == b) & (df.槽位 == 5) & (df.槽重 == 0.11)].iloc[0]
        for (ns, sw) in grid:
            if (ns, sw) == (5, 0.11): continue
            x = df[(df.版本 == v) & (df.规则 == b) & (df.槽位 == ns) & (df.槽重 == sw)].iloc[0]
            k = x.年化 / base.年化 if base.年化 else np.nan
            ke = x.平均仓位 / base.平均仓位
            print(f"{v} {b} ({ns},{sw}): 年化 {x.年化*100:+.1f}% vs {base.年化*100:+.1f}% -> ×{k:.2f}; 平均仓位 {x.平均仓位:.3f} vs {base.平均仓位:.3f} -> ×{ke:.2f}; 回撤 {x.回撤*100:.1f}% vs {base.回撤*100:.1f}% (×{x.回撤/base.回撤:.2f}); 夏普 {x.夏普:.2f} vs {base.夏普:.2f}")
log("done")
