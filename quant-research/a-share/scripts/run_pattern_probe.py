"""形态 vs 因子框架:为什么截面 IC 挖不到形态(2026-08-23)。

用户观察:交易者做的可能是"中长期趋势 -> 回调 -> 横盘震荡 -> 重新走强"的形态,
而我们 2028 个因子没抓住。本脚本验证三个机制,不做策略:

E1 稀释:形态作为二值信号,全截面 Rank IC(会被 0.025 关卡刷掉)vs 命中集合的条件收益(可能很大)
E2 持有期:命中集合在 h=1/5/10/20/40/60 的超额,看兑现期
E3 张成:用现有 2028 个因子能否重建形态标签(逻辑回归 AUC)—— 若不能,说明因子库里没有这个信息
E4 顺序:同样 120 日涨幅、不同路径的股票,未来收益是否不同 —— 证明"压成一个数"丢掉了信息

全部只用 2020-2025(2026 封存已用完),信息时点 T 日收盘。
"""
import os, sys, warnings, json; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import strategy as ST
from afactor.config import LEDGER
from scipy.stats import rankdata
pd.set_option("display.width", 250)

p = Panel(); u = ST.universe(p, "V1000")
c = p.raw("close_px"); tw = p.raw("twap_open10"); amt = p.raw("amount")
years = np.asarray(p.dates.year)
研究 = (years >= 2020) & (years <= 2025)

# ---------- 形态定义(全部用 T 日收盘,无前视)----------
ret120 = c / c.shift(120) - 1.0                      # 中长期趋势
hi60 = c.rolling(60, min_periods=40).max()
dd = c / hi60 - 1.0                                  # 距 60 日高点(回调深度)
w = 25                                               # 横盘窗口
box_hi = c.rolling(w, min_periods=15).max(); box_lo = c.rolling(w, min_periods=15).min()
box_rng = (box_hi - box_lo) / c.rolling(w, min_periods=15).mean()      # 箱体宽度
vol20 = c.pct_change().rolling(20, min_periods=10).std()
vol60 = c.pct_change().rolling(60, min_periods=30).std()
squeeze = vol20 / vol60                                                # 波动收敛
amt_r = amt.rolling(5, min_periods=3).mean() / amt.rolling(60, min_periods=30).mean()
brk = c >= box_hi.shift(1)                                             # 突破箱体上沿

PAT = {
 "① 只要趋势(涨120日>20%)": (ret120 > 0.20),
 "②+回调过(距60日高-25%~-8%)": (ret120 > 0.20) & (dd <= -0.08) & (dd >= -0.25),
 "③+横盘(箱体<18% 且波动收敛<0.9)": (ret120 > 0.20) & (dd <= -0.08) & (dd >= -0.25) & (box_rng < 0.18) & (squeeze < 0.9),
 "④+突破箱体(完整形态)": (ret120 > 0.20) & (dd <= -0.08) & (dd >= -0.25) & (box_rng < 0.18) & (squeeze < 0.9) & brk,
 "④+放量突破": (ret120 > 0.20) & (dd <= -0.08) & (dd >= -0.25) & (box_rng < 0.18) & (squeeze < 0.9) & brk & (amt_r > 1.3),
}
HS = [1, 5, 10, 20, 40, 60]
FWD = {h: (tw.shift(-(1 + h)) / tw.shift(-1) - 1).where(u) for h in HS}

print("=" * 130)
print("E1/E2 稀释与持有期:形态作为二值信号的全截面 Rank IC,vs 命中集合的条件超额")
print("=" * 130)
print(f'{"形态":36s}{"命中率":>8s}{"日均只数":>9s}' + "".join(f"{'IC h'+str(h):>10s}" for h in HS) + "   |" + "".join(f"{'超额 h'+str(h):>11s}" for h in HS))
rows = []
for name, m in PAT.items():
    M = (m & u & 研究[:, None]).fillna(False)
    hit = M.to_numpy(bool)
    rate = hit.sum() / (u.to_numpy(bool) & 研究[:, None]).sum()
    line = f"{name:36s}{rate*100:7.2f}%{hit.sum()/研究.sum():9.0f}"
    ics, exs = [], []
    for h in HS:
        F = FWD[h].to_numpy(np.float32); U = (u.to_numpy(bool) & 研究[:, None])
        ic_list = []
        for i in np.flatnonzero(研究)[::3]:
            ok = U[i] & np.isfinite(F[i])
            if ok.sum() < 200 or hit[i].sum() == 0: continue
            ic_list.append(np.corrcoef(rankdata(hit[i, ok].astype(float)), rankdata(F[i, ok]))[0, 1])
        ic = np.nanmean(ic_list) if ic_list else np.nan
        hv = F[hit & U]; bv = F[(~hit) & U]
        ex = np.nanmean(hv) - np.nanmean(bv)
        ics.append(ic); exs.append(ex)
    line += "".join(f"{x:+10.4f}" for x in ics) + "   |" + "".join(f"{x*100:+10.2f}%" for x in exs)
    print(line)
    rows.append(dict(形态=name, 命中率=rate, **{f"IC_h{h}": ics[i] for i, h in enumerate(HS)},
                     **{f"超额_h{h}": exs[i] for i, h in enumerate(HS)}))
pd.DataFrame(rows).to_csv(LEDGER / "pattern_probe.csv", index=False)
print(f"\n关卡是 |IC| >= 0.025。上表任何一行 IC 低于 0.025 但超额显著为正 -> 该形态会被因子层刷掉。")

# ---------- E4 顺序:同样涨幅、不同路径 ----------
print("\n" + "=" * 130)
print("E4 顺序信息:同样的 120 日涨幅,不同路径,未来 20 日收益是否不同")
print("=" * 130)
path_eff = (c / c.shift(120) - 1).abs() / c.pct_change().abs().rolling(120, min_periods=60).sum()   # 路径效率
q_ret = ret120.where(u).rank(axis=1, pct=True)
q_eff = path_eff.where(u).rank(axis=1, pct=True)
F20 = FWD[20].to_numpy(np.float32); U = (u.to_numpy(bool) & 研究[:, None])
QR, QE = q_ret.to_numpy(np.float32), q_eff.to_numpy(np.float32)
print(f'{"120日涨幅分位":>14s}' + "".join(f"{'路径效率 Q'+str(k):>13s}" for k in range(1, 6)) + f"{'最高-最低':>12s}")
for lo, hi, lab in [(0.8, 1.0, "最高 20%"), (0.6, 0.8, "60-80%"), (0.4, 0.6, "40-60%")]:
    sel = U & (QR > lo) & (QR <= hi)
    vals = []
    for k in range(5):
        s2 = sel & (QE > k * 0.2) & (QE <= (k + 1) * 0.2)
        vals.append(np.nanmean(F20[s2]) * 100)
    print(f'{lab:>14s}' + "".join(f"{v:+12.2f}%" for v in vals) + f"{vals[-1]-vals[0]:+11.2f}pp")
print("(路径效率 = |总涨幅| / 逐日绝对涨幅之和;Q5 = 走得最'直';同一列内涨幅相同,只有路径不同)")
