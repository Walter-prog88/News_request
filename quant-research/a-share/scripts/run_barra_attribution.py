"""Barra 式超额归因 —— 把风格补全到行业标准的十个，行业补到申万一级全部。

上一版归因只放了 7 个日频风格 + 8 个行业,漏掉整个 价值/盈利/成长/杠杆 一族,
而 COMP_v5 里恰好有三个基本面交互因子(fx_fu_roe__P__ivol_20 等),
它们的收益完全可能被 EP/BP 解释掉。本脚本补全后重测,回答一个问题:

    上一版剩下的 1.75pp(主口径)/2.45pp(尾盘) 残差 alpha,
    在更严的尺子下还剩多少?

## 与上一版的两个方法差异

1. **纯因子收益(pure factor returns)**。上一版用"rank 标准化 + 美元中性多空"造风格收益,
   风格之间相关性极高(市值↔流动性 0.72、换手↔波动 0.94),VIF 高达 13~28,
   单个 beta 不可解读。本版改用 Barra 的做法:**每日截面 WLS 回归**

       r_i = f_country + sum_k X_ik * f_k + sum_j Ind_ij * f_j + u_i

   权重 sqrt(市值),行业收益加市值加权和为零的约束做识别。
   回归系数 f 就是"纯因子收益"——持有该因子单位暴露、其余因子零暴露的组合收益。
   这样造出来的因子收益近似正交,贡献可以逐项解读。

2. **风格补全到 Barra CNE 的十个**:
   Size / Beta / Momentum / Residual Volatility / Non-linear Size /
   Book-to-Price / Liquidity / Earnings Yield / Growth / Leverage
   行业:申万一级全部(而非前 8 个)。

只读研究期面板,不触碰封存样本。
产出:research/ledger/barra_attribution_{summary,factor_returns}.csv
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel, TailPanel
from afactor import factors as FA, combine, strategy as ST, ops, newdata as ND
from afactor.strategy import TRADING_DAYS

pd.set_option("display.width", 240)

MODES = ["open10", "tail1450"]
VERSION = "V1000"
PLAN = dict(scheme="exclude", buffer=(0.20, 0.35), rebalance=5)


# ---------------------------------------------------------------- 工具 / 暴露度 / 纯因子收益
# 2026-10 起这些函数搬进库 src/afactor/barra.py(逻辑不变,核对见 research/ledger/risk_move_check.csv)。
# 这里 re-export 同名对象,`from run_barra_attribution import nw_t, assert_aligned, build_exposures`
# 等旧写法照常可用(run_barra_attribution_top20/topfrac.py、run_comp_split.py 依赖它)。
from afactor.barra import (WINSOR, nw_t, zscore, fwd_returns, assert_aligned,   # noqa: E402,F401
                           build_exposures, _reg, pure_factor_returns)


# ---------------------------------------------------------------- 主流程
def main():
    FA.build_all()
    man = json.load(open(LEDGER / "comp_v5_members.json"))
    members, signs = man["members"], man["signs"]

    rows, fr_store = [], {}
    for mode in MODES:
        print(f"\n{'='*100}\n{mode}\n{'='*100}")
        p = TailPanel() if ST.EXEC_MODES[mode]["tail"] else Panel()
        R = fwd_returns(p, mode)
        u = p["universe"].reindex_like(R).fillna(False)

        print("构建暴露度…")
        X = build_exposures(p, u)
        print(f"  风格 {len(X)} 个: {list(X)}")

        print("逐日截面 WLS 求纯因子收益…")
        F = pure_factor_returns(R, X, u, p)
        print(f"  得到 {F.shape[0]} 天 x {F.shape[1]} 个因子")

        fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
               .replace([np.inf, -np.inf], np.nan) for n in members if n in FA.REGISTRY}
        score = combine.combine_equal(combine.prepare(fac, signs, p["universe"]))
        strat = ST.run_strategy(score, p, VERSION, mode, **PLAN)["rets"]
        bench = ST.benchmark(p, VERSION, mode)["rets"]
        assert_aligned(strat, bench, "策略 vs 池等权")

        idx = strat.index.intersection(bench.index).intersection(F.dropna(how="any").index)
        y = (strat.reindex(idx) - bench.reindex(idx)).astype(float)
        Fi = F.reindex(idx).astype(float)
        styles = [c for c in Fi.columns if not c.startswith("行业:") and c != "Country"]
        indus = [c for c in Fi.columns if c.startswith("行业:")]

        A = np.column_stack([np.ones(len(idx)), Fi[styles + indus].to_numpy()])
        b, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
        resid = y.to_numpy() - A @ b
        r2 = 1.0 - resid.var() / y.to_numpy().var()

        terms = styles + indus
        contrib = {t: b[i + 1] * Fi[t].mean() * TRADING_DAYS for i, t in enumerate(terms)}
        sty_sum = sum(contrib[t] for t in styles)
        ind_sum = sum(contrib[t] for t in indus)
        alpha = b[0] * TRADING_DAYS
        tot = y.mean() * TRADING_DAYS

        for i, t in enumerate(terms):
            rows.append(dict(mode=mode, term=t, kind="风格" if t in styles else "行业",
                             beta=b[i + 1], factor_ann=Fi[t].mean() * TRADING_DAYS,
                             contrib_ann=contrib[t]))
        rows.append(dict(mode=mode, term="__alpha", kind="残差", beta=np.nan,
                         factor_ann=np.nan, contrib_ann=alpha))
        rows.append(dict(mode=mode, term="__总超额", kind="合计", beta=np.nan,
                         factor_ann=np.nan, contrib_ann=tot))
        rows.append(dict(mode=mode, term="__风格合计", kind="合计", beta=np.nan,
                         factor_ann=np.nan, contrib_ann=sty_sum))
        rows.append(dict(mode=mode, term="__行业合计", kind="合计", beta=np.nan,
                         factor_ann=np.nan, contrib_ann=ind_sum))
        rows.append(dict(mode=mode, term="__R2", kind="合计", beta=np.nan,
                         factor_ann=np.nan, contrib_ann=r2))
        for q in (0, 5, 10, 20):
            rows.append(dict(mode=mode, term=f"__alpha_t_q{q}", kind="合计", beta=np.nan,
                             factor_ann=np.nan, contrib_ann=nw_t(resid + b[0], q)))

        fr_store[mode] = Fi
        print(f"\n  总超额 {tot:+.4f} | 风格 {sty_sum:+.4f} | 行业 {ind_sum:+.4f} "
              f"| alpha {alpha:+.4f} | R2 {r2:.4f}")
        print(f"  alpha t: q0={nw_t(resid + b[0], 0):.2f} q5={nw_t(resid + b[0], 5):.2f} "
              f"q10={nw_t(resid + b[0], 10):.2f} q20={nw_t(resid + b[0], 20):.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "barra_attribution_summary.csv", index=False)
    pd.concat({k: v for k, v in fr_store.items()}, axis=1).to_csv(
        LEDGER / "barra_factor_returns.csv")

    print("\n" + "=" * 110)
    for mode in MODES:
        d = df[df["mode"] == mode]
        print(f"\n--- {mode} 风格 ---")
        print(d[d.kind == "风格"][["term", "beta", "factor_ann", "contrib_ann"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        print(d[d.kind == "合计"][["term", "contrib_ann"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"\n已写 {LEDGER/'barra_attribution_summary.csv'}")


if __name__ == "__main__":
    main()
