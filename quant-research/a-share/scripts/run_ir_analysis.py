"""IR / 跟踪误差 / 分年度超额 —— 补 baselines.csv 缺的那一块。

背景:`run_baselines.py` 把 run_strategy 返回的 `rets`(逐日收益 Series)直接塞进
DataFrame 再 to_csv,pandas 会把 Series 转成**截断的 repr 字符串**(1453 天只落 10 行),
基准的逐日序列则根本没落盘。结果是账本里有年化、夏普、回撤,唯独没有
**超额的波动**,因而算不出 IR,也做不了 t 检验。

本脚本不改任何研究口径,只做三件事:
  1. 取 run_strategy 与 benchmark 的逐日收益(二者按设计同索引,见 strategy.benchmark 文档),
     完整落盘到 parquet;
  2. 计算全期与分年度的 超额 / 跟踪误差 / IR / t 值(含 Newey-West 修正);
  3. 额外给出相对**可投资宽基指数**(中证500 / 中证1000)的超额,因为池等权基准
     既买不到、又含幸存者偏差,只用它算超额会被质疑。

只读研究期面板(Panel / TailPanel),**不触碰封存样本**。
产出:research/ledger/ir_analysis_{summary,yearly}.csv、ir_daily_series.parquet
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel, TailPanel
from afactor import factors as FA, combine, strategy as ST, newdata as ND
from afactor.strategy import TRADING_DAYS

pd.set_option("display.width", 220)

# 要分析的策略:主口径与尾盘口径下的排雷缓冲 20/35,外加一个 TopN 基线作对照
PLANS = [
    ("排雷缓冲20/35", dict(scheme="exclude", buffer=(0.20, 0.35), rebalance=5)),
    ("基线 Top50 周频", dict(scheme="top", n=50, rebalance=5)),
]
MODES = ["open10", "tail1450"]
VERSION = "V1000"
INDEXES = {"000905": "中证500", "000852": "中证1000"}


def build_score(p, members, signs):
    fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
           .replace([np.inf, -np.inf], np.nan) for n in members if n in FA.REGISTRY}
    return combine.combine_equal(combine.prepare(fac, signs, p["universe"]))


def nw_t(x: pd.Series, lag: int) -> float:
    """Newey-West(1987) 修正的均值 t 值,Bartlett 核,带宽 = lag。

    ⚠ 这里的自相关方向与直觉相反:实测 NW 后 t **比朴素 t 更大**,说明逐日超额
    是负自相关的(主口径下主因是开-开与收-收的隔夜错位,构成 MA(1))。
    负自相关下带宽越大 t 越大,所以"带宽取大一点更保守"在这里不成立——
    必须报 q 敏感性,见 nw_t_curve()。
    自协方差按 n 归一(标准写法);先前按 n-L 归一会把 |gamma_L| 放大 n/(n-L) 倍,
    在 gamma_L 为负时使 t 偏大,方向反保守。
    """
    x = pd.Series(x).dropna().to_numpy(float)
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    s = float((e ** 2).sum() / n)
    for L in range(1, lag + 1):
        c = float((e[L:] * e[:-L]).sum() / n)
        s += 2.0 * (1.0 - L / (lag + 1.0)) * c
    if s <= 0:
        return np.nan
    return float(x.mean() / np.sqrt(s / n))


def nw_t_curve(x: pd.Series, qs=(0, 5, 7, 10, 20)) -> dict:
    """NW t 对带宽 q 的敏感性。q=0 即朴素 t。"""
    return {f"t_nw_q{q}": (nw_t(x, q) if q > 0 else
                           float(pd.Series(x).mean() / pd.Series(x).std() * np.sqrt(len(x))))
            for q in qs}


def ann_geom(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float((1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0)


def stats_pair(rs: pd.Series, rb: pd.Series, label: str, bench_label: str) -> dict:
    """给定策略与基准的逐日收益(同索引),算超额/TE/IR/t。"""
    idx = rs.index.intersection(rb.index)
    rs, rb = rs.reindex(idx).fillna(0.0), rb.reindex(idx).fillna(0.0)
    d = rs - rb                                        # 逐日超额(算术差)
    yrs = len(idx) / TRADING_DAYS
    te = float(d.std() * np.sqrt(TRADING_DAYS))        # 跟踪误差(年化)
    exc_arith = float(d.mean() * TRADING_DAYS)         # 算术超额(与 TE 同口径,用于算 IR)
    exc_geom = ann_geom(rs) - ann_geom(rb)             # 几何超额(与账本 excess 口径一致)
    ir = exc_arith / te if te > 0 else np.nan
    return dict(
        strategy=label, benchmark=bench_label, n_days=len(idx), years=round(yrs, 3),
        ann_strategy=ann_geom(rs), ann_benchmark=ann_geom(rb),
        excess_geom=exc_geom, excess_arith=exc_arith,
        tracking_error=te, IR=ir,   # IR 分子是 excess_arith,不是 excess_geom

        t_naive=float(d.mean() / d.std() * np.sqrt(len(d))) if d.std() > 0 else np.nan,
        t_newey_west=nw_t(d, lag=10),                  # 5 日调仓 -> 带宽 max(5, 2h)=10
        corr_with_bench=float(rs.corr(rb)),
        **nw_t_curve(d),
    )


def index_rets_on(p, mode, idx_on_signal_days: pd.Index, code: str) -> pd.Series:
    """把宽基指数的日收益搬到与策略同一套【信号日】索引上。

    ⚠ 这里踩过一次错位一天的坑,记下推导:
      主口径 open10:信号日 t -> 次日开盘买 -> 再次日开盘卖,持有区间 [open(t+1), open(t+2)],
                    实质覆盖**交易日 t+1**;
      尾盘口径 tail1450:信号日 t -> 当日尾盘买 -> 次日尾盘卖,区间 [close(t), close(t+1)],
                    同样覆盖**交易日 t+1**。
    两个口径覆盖的都是 t+1 这一天,所以指数一律取 close(t+1)/close(t)-1 打在信号日 t,
    **不再按 lag 额外 shift**。先前按 benchmark() 的写法多 shift 了一次 lag,主口径下
    corr(策略, 指数) 从 0.70 掉到 0.23、峰值跑到 k=+1,尾盘口径因 lag=0 侥幸未受影响。
    指数只有收盘价,主口径(开-开)与之仍有执行时点差异,属量级参照,不是精确可交易对照。
    """
    px = ND.index_daily()[code].reindex(p.dates).ffill()
    r = (px.shift(-1) / px - 1.0)                      # r[t] = 交易日 t+1 的指数收益
    return r.reindex(idx_on_signal_days).fillna(0.0)


def assert_aligned(rs: pd.Series, rb: pd.Series, tag: str) -> str:
    """回归断言:corr(策略_t, 基准_{t+k}) 的峰值必须在 k=0,否则就是错开了一天。"""
    ks = (-2, -1, 0, 1, 2)
    cs = {k: float(rs.corr(rb.shift(k))) for k in ks}
    peak = max(ks, key=lambda k: cs[k])
    line = "  ".join(f"k={k:+d}:{cs[k]:+.4f}" for k in ks)
    flag = "OK" if peak == 0 else "MISALIGNED"
    print(f"  [{flag}] {tag}: {line}  峰值 k={peak:+d}")
    if peak != 0:
        raise AssertionError(f"{tag} 与基准错位 {peak} 天,拒绝出数")
    return flag


def main():
    FA.build_all()
    man = json.load(open(LEDGER / "comp_v5_members.json"))
    members, signs = man["members"], man["signs"]

    summary, yearly, series = [], [], {}

    for mode in MODES:
        p = TailPanel() if ST.EXEC_MODES[mode]["tail"] else Panel()
        score = build_score(p, members, signs)
        bm = ST.benchmark(p, VERSION, mode)
        rb_pool = bm["rets"]
        series[f"{mode}|基准 池等权"] = rb_pool

        for code, cname in INDEXES.items():
            series[f"{mode}|基准 {cname}"] = index_rets_on(p, mode, rb_pool.index, code)

        for label, kw in PLANS:
            r = ST.run_strategy(score, p, VERSION, mode, **kw)
            rs = r["rets"]
            series[f"{mode}|{label}"] = rs

            # --- 出数之前先过对齐断言 ---
            print(f"\n[对齐检查] {mode} × {label}")
            assert_aligned(rs, rb_pool, "vs 池等权")
            for cn in INDEXES.values():
                assert_aligned(rs, series[f"{mode}|基准 {cn}"], f"vs {cn}")

            # --- 全期:相对三个基准 ---
            for bl, rb in [("池等权", rb_pool)] + [
                    (cn, series[f"{mode}|基准 {cn}"]) for cn in INDEXES.values()]:
                row = stats_pair(rs, rb, label, bl)
                row.update(mode=mode, version=VERSION,
                           turnover=r["turnover"], cost_bps=r["cost_bps"],
                           holdings=r["holdings"], max_dd=r["max_dd"],
                           sharpe=r["sharpe"], vol=r["vol"])
                summary.append(row)

            # --- 分年度:只对池等权与中证500 ---
            for bl, rb in ([("池等权", rb_pool)] +
                           [(cn, series[f"{mode}|基准 {cn}"]) for cn in INDEXES.values()]):
                idx = rs.index.intersection(rb.index)
                for y, sub in pd.Series(idx, index=idx).groupby(idx.year):
                    if len(sub) < 60:
                        continue
                    row = stats_pair(rs.reindex(sub.index), rb.reindex(sub.index), label, bl)
                    row.update(mode=mode, year=int(y), version=VERSION)
                    yearly.append(row)

    sm = pd.DataFrame(summary)
    yr = pd.DataFrame(yearly)
    sm.to_csv(LEDGER / "ir_analysis_summary.csv", index=False)
    yr.to_csv(LEDGER / "ir_analysis_yearly.csv", index=False)
    ser = pd.DataFrame(series).sort_index()
    ser.to_parquet(LEDGER / "ir_daily_series.parquet")

    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\n" + "=" * 130)
    print("全期:超额 / 跟踪误差 / IR")
    print("=" * 130)
    cols = ["mode", "strategy", "benchmark", "years", "ann_strategy", "ann_benchmark",
            "excess_geom", "excess_arith", "tracking_error", "IR", "t_naive",
            "t_nw_q7", "t_newey_west", "t_nw_q20", "corr_with_bench"]
    print(sm[cols].to_string(index=False))

    print("\n" + "=" * 130)
    print("分年度超额(vs 池等权 与 vs 中证500)")
    print("=" * 130)
    ycols = ["mode", "strategy", "benchmark", "year", "ann_strategy", "ann_benchmark",
             "excess_geom", "excess_arith", "tracking_error", "IR"]
    print(yr[ycols].to_string(index=False))

    print(f"\n已写 {LEDGER/'ir_analysis_summary.csv'}")
    print(f"已写 {LEDGER/'ir_analysis_yearly.csv'}")
    print(f"已写 {LEDGER/'ir_daily_series.parquet'}  形状 {ser.shape}")


if __name__ == "__main__":
    main()
