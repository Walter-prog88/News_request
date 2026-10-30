"""PEAD 第 2 步:事件引擎门槛(预注册第 6 节"引擎门槛",读任何臂的结果之前执行并落盘)。

被测对象 = afactor.pead_engine.run_account(callable 目标 + 逐日重试卖出 + 最低佣金/整手钩子)。
三条回归断言:
  ① 每日 exposure <= 1;
  ② "合格池全体等权、每 5 日调仓"对合格池等权基准(每日再平衡、不含成本)的超额 ≈ −成本
     (残差 = 超额 + 成本,来自 5 日内权重漂移,应为小量);
  ③ corr(策略日收益, 基准日收益.shift(k)) 峰值在 k = 0。
另加一条等价性核对:同一组权重走 pead_engine(rebalance=1 + 自判调仓日)与走
strategy.run_weights(rebalance=5, normalize=False, 单相位)应基本一致,差异只来自逐日重试卖出。
门槛用"无整手、无最低佣金"的口径(整手在全池 4000 只等权下没有意义),资金 1000 万。
产出:research/ledger/pead_engine_gate.csv,日志 logs/pead_engine_gate.log。
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from afactor.config import LEDGER
from afactor.panel import Panel
from afactor import strategy as ST, earnings_events as EE, pead_engine as PE

CAP = 1e7


def pool_benchmark(p, Q: pd.DataFrame) -> pd.Series:
    """合格池等权(不含成本):信号行 t 的池,吃 twap_open10[t+2]/twap_open10[t+1]-1,记在 t。"""
    px, _ = ST.exec_prices(p, "open10")
    g = px.shift(-1) / px - 1.0                                  # 索引在成交日 te
    m = g.where(Q.shift(1).fillna(False).astype(bool)).mean(axis=1)
    return m.shift(-1).iloc[:len(p.dates) - 2].fillna(0.0)


def main():
    t0 = time.time()
    p = Panel()
    dates, codes = p.dates, p.codes
    Q = EE.qualified_pool(p)
    Qn = Q.to_numpy(bool)
    bench = pool_benchmark(p, Q)
    star = PE.star_mask(codes)
    dummy = np.ones((len(dates), len(codes)))
    rows = []
    rets_all = []
    for ph in range(5):
        batches = {}
        for te in range(1, len(dates)):
            if te % 5 != ph:
                continue
            idx = np.flatnonzero(Qn[te - 1])
            if len(idx):
                batches[te] = (idx, np.full(len(idx), 1.0 / len(idx)))
        out = PE.run_account(p, batches, ph, CAP, dummy, dummy, star, lots=False, min_commission=0.0)
        d = out["daily"]
        r = d["ret"]; c = d["cost_total"]
        rets_all.append(r.rename(ph))
        # 等价性:同一权重走 strategy.run_weights(rebalance=5)。simulate 的相位锚在信号行:
        # 信号行 t 调仓 <=> (t - phase_sim) % 5 == 0;本模块 te = t+1,te % 5 == ph -> phase_sim = (ph-1) % 5
        W = np.zeros((len(dates), len(codes)))
        for te, (idx, w) in batches.items():
            W[te - 1, idx] = w
        ref = ST.run_weights(W, p, "open10", CAP, rebalance=5, avg_phases=True, phases=[(ph - 1) % 5],
                             normalize=False, participation_on_nav=True)
        rr = ref["rets"]
        a, b = r.align(rr, join="inner")
        b_al = bench.reindex(r.index)
        cs = {k: float(r.corr(b_al.shift(k))) for k in (-2, -1, 0, 1, 2)}
        peak = max(cs, key=cs.get)
        for y, idx_y in r.groupby(r.index.year).groups.items():
            ex = (r.loc[idx_y] - b_al.loc[idx_y]).mean() * ST.TRADING_DAYS
            cst = c.loc[idx_y].mean() * ST.TRADING_DAYS
            rows.append(dict(check="residual_by_year", phase=ph, year=int(y), excess_pp=ex * 100,
                             cost_pp=cst * 100, residual_pp=(ex + cst) * 100))
        rows.append(dict(check="exposure_max", phase=ph, value=float(d["exposure"].max()),
                         passed=bool(d["exposure"].max() <= 1 + 1e-9)))
        rows.append(dict(check="corr_peak_k", phase=ph, value=peak, passed=peak == 0,
                         detail=" ".join(f"{k:+d}:{v:+.4f}" for k, v in cs.items())))
        rows.append(dict(check="equiv_vs_run_weights", phase=ph,
                         value=float((a - b).abs().max()),
                         detail=f"ann diff {((a-b).mean()*ST.TRADING_DAYS)*100:+.4f}pp; "
                                f"pending_stock_days {out['stats']['pending_stock_days']}, "
                                f"pending_days {out['stats']['pending_days']}"))
        print(f"phase {ph}: exposure max {d['exposure'].max():.6f}; corr peak k={peak} {cs}; "
              f"equiv max|diff| {(a-b).abs().max():.2e}; pending {out['stats']} [{time.time()-t0:.0f}s]", flush=True)
    R = pd.concat(rets_all, axis=1).mean(axis=1)
    b_al = bench.reindex(R.index)
    for y, idx_y in R.groupby(R.index.year).groups.items():
        rows.append(dict(check="residual_by_year_phaseavg", phase=-1, year=int(y),
                         excess_pp=(R.loc[idx_y] - b_al.loc[idx_y]).mean() * ST.TRADING_DAYS * 100))
    df = pd.DataFrame(rows)
    # 相位平均的成本与残差
    rb = df[df.check == "residual_by_year"].groupby("year")[["excess_pp", "cost_pp", "residual_pp"]].mean()
    for y, r_ in rb.iterrows():
        df.loc[(df.check == "residual_by_year_phaseavg") & (df.year == y), "cost_pp"] = r_["cost_pp"]
        df.loc[(df.check == "residual_by_year_phaseavg") & (df.year == y), "residual_pp"] = \
            df.loc[(df.check == "residual_by_year_phaseavg") & (df.year == y), "excess_pp"] + r_["cost_pp"]
    df.to_csv(LEDGER / "pead_engine_gate.csv", index=False)
    print(df[df.check.isin(["residual_by_year_phaseavg"])].to_string())
    ok = df[df.check.isin(["exposure_max", "corr_peak_k"])]["passed"].all()
    print("GATE", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
