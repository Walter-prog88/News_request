"""E4 补充:把 引擎 V1 与 交易者 的年化差距按 仓位 / 守门 / 入场筛选 / 退出规则 分解(样本内 2024-09-24~2025-12-31)。"""
import os, sys, warnings; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import eventbt as EB, strategy as ST
p = Panel(); d = p.dates; codes = p.codes; T = len(d)
raw = pd.read_csv('data_8:21/source_raw/baostock_stock_industry_raw.csv'); raw['code'] = raw['code'].str.upper().str.replace(r'^(SH|SZ)\.(\d+)$', r'\2.\1', regex=True)
lab = pd.Series(dict(zip(raw.code, raw.industry))).reindex(codes).replace('', np.nan)
u = ST.universe(p, "V100"); R = EB.Rules; m = np.asarray((d >= "2024-09-24") & (d <= "2025-12-31"))
sig = EB.build_signals(p, R(rank_by="sector"), u, lab); gate = (sig["strong"] & u).fillna(False)
px = p.raw("twap_tail8").to_numpy(np.float64); close = p.raw("close_px").to_numpy(np.float64); tradable = p["twap_tail8_f"].notna().to_numpy(bool)
H = 5; reb = np.flatnonzero(m)[::H]; cost_rt = 7e-4
def ann(s): nav = (1 + s).cumprod(); return float(nav.iloc[-1] ** (242 / len(s)) - 1)
def ew_line(mask, expo, seed=None, n=None):
    rng = np.random.default_rng(seed); nav = 1.0; rets = {}
    for t in reb:
        te = min(t + H, T - 1)
        ok = mask[t] & tradable[t] & np.isfinite(px[te]) & np.isfinite(px[t]); c = np.flatnonzero(ok)
        if len(c) == 0:
            for k in range(t, te + 1): rets[k] = 0.0
            continue
        if n is not None and len(c) > n: c = rng.choice(c, n, replace=False)
        seg = pd.DataFrame(close[t:te + 1, c] / px[t, c]).ffill().fillna(1.0).to_numpy()
        X = px[te, c] / px[t, c]; seg[-1] = np.where(np.isfinite(X), X, seg[-1])
        P = (1 - expo) + expo * seg.mean(axis=1); prev = 1.0
        for k, day in enumerate(range(t, te + 1)):
            r = P[k] / prev - 1.0 - (expo * cost_rt if k == te - t else 0.0); rets[day] = r; prev = P[k]
    s = pd.Series(rets).sort_index(); s.index = d[s.index]; return s
G = gate.to_numpy(bool); U = u.to_numpy(bool)
print("仓位 0.55 / 0.77 / 1.00 下的年化(%):")
for nm, mask in (("全池等权", U), ("守门等权", G)):
    print(f"  {nm}: " + "  ".join(f"e={e:.2f} {ann(ew_line(mask, e))*100:+.1f}%" for e in (0.55, 0.77, 1.0)))
for e in (0.55, 0.77):
    a = np.array([ann(ew_line(G, e, seed=k, n=5)) for k in range(30)]); print(f"  守门随机5只 e={e:.2f}: 均值 {a.mean()*100:+.1f}% 中位 {np.median(a)*100:+.1f}% [{a.min()*100:+.1f}%, {a.max()*100:+.1f}%]")
# 引擎:默认槽位 0.11(55%)vs 0.154(77%)vs 0.184(92%)
for sw in (0.11, 0.154, 0.184):
    r = EB.run_event_bt(p, R(rank_by="sector", slot_w=sw), "V100", date_mask=m, sector_labels=lab, universe=u, signals=sig)
    print(f"  引擎V1 slot_w={sw:.3f}(仓位≈{5*sw:.2f}): 年化 {r['ann_return']*100:+.1f}%  回撤 {r['max_dd']*100:.1f}%  实际仓位 {r['exposure']:.2f}  交易 {r['n_trades']} 胜率 {r['win_rate']:.3f}")
# 入场集合(每日 entry 触发)等权、固定 5 日持有,每日新开一批(5 相位平均)——隔离"入场筛选"的价值
E = sig["entry"].to_numpy(bool) & tradable
fwd = np.where(np.isfinite(px), 1, np.nan); 
rows = []
for t in np.flatnonzero(m):
    te = min(t + H, T - 1)
    for nm, mask in (("入场触发集合", E[t]), ("守门集合", G[t] & tradable[t]), ("全池", U[t] & tradable[t])):
        ok = mask & np.isfinite(px[te]) & np.isfinite(px[t]); c = np.flatnonzero(ok)
        rows.append(dict(date=d[t], set=nm, n=len(c), f5=float(np.mean(px[te, c] / px[t, c] - 1)) if len(c) else np.nan))
D = pd.DataFrame(rows).pivot(index="date", columns="set", values="f5"); Nn = pd.DataFrame(rows).pivot(index="date", columns="set", values="n")
print("\n每日 5 日前瞻收益(tail->tail,所有交易日,等权):")
print("  均值(%):", (D.mean() * 100).round(2).to_dict(), " 日均只数:", Nn.mean().round(0).to_dict())
dd = D.dropna(); print(f"  入场触发集合 − 守门集合:{(dd['入场触发集合']-dd['守门集合']).mean()*100:+.2f}pp/5日(t={((dd['入场触发集合']-dd['守门集合']).mean()/(dd['入场触发集合']-dd['守门集合']).std()*np.sqrt(len(dd))):.2f}, n={len(dd)})")
print(f"  守门集合 − 全池:{(dd['守门集合']-dd['全池']).mean()*100:+.2f}pp/5日(t={((dd['守门集合']-dd['全池']).mean()/(dd['守门集合']-dd['全池']).std()*np.sqrt(len(dd))):.2f})")
