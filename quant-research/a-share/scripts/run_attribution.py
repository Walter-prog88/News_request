"""超额归因 —— 把排雷策略的超额拆成 风格暴露 / 行业暴露 / 残差 alpha。

这是评审提示词任务一第 3 条:"拆成 ①选股alpha ②风格/行业暴露 ③交易性收益,
哪块占主导?若②占大头,直接指出这不是 alpha。"

## 方法(收益归因,returns-based)

1. 构造风格因子的**日收益序列**。对每个连续型风格 S:
   当日在决策池内做截面 rank 标准化 -> z;构造美元中性多空权重 w = z / sum|z|;
   风格因子日收益 = sum_i w_i * r_i,其中 r 与策略同口径(信号日 t 吃 te->te+1 的收益)。
2. 行业:用申万一级,每个行业等权收益减全池等权收益,取前 K 大行业作为解释变量,
   其余并入"其他行业"。
3. 把策略的日超额对这些风格/行业因子做 OLS:
      excess_t = alpha + sum_k beta_k * f_{k,t} + eps_t
   报 alpha(年化)、alpha 的 Newey-West t(带宽 q in {0,5,10})、R^2、
   每个风格的 beta / t / 年化收益贡献(= beta * 该因子年化收益)。

## 已知的先验(来自 README §11,决定了这次要重点看什么)
   因子对市值中性化保留率 1.012(无暴露)、对行业 0.900(损失 10%),
   但市值+行业+流动性三者一起正交后只剩 0.635 —— **流动性是唯一实质暴露**。
   所以本次归因的核心问题是:超额里流动性风格占多大比重。

只读研究期面板,不触碰封存样本。
产出:research/ledger/attribution_{summary,daily}.csv
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
PLAN = ("排雷缓冲20/35", dict(scheme="exclude", buffer=(0.20, 0.35), rebalance=5))
TOP_INDUSTRIES = 8          # 单列出的行业数,其余并入"其他行业"


# ---------------------------------------------------------------- 工具
def nw_t(x: np.ndarray, q: int) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    s = float((e ** 2).sum() / n)
    for L in range(1, q + 1):
        s += 2.0 * (1.0 - L / (q + 1.0)) * float((e[L:] * e[:-L]).sum() / n)
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def nw_se_ols(X: np.ndarray, resid: np.ndarray, q: int) -> np.ndarray:
    """OLS 系数的 Newey-West 标准误(Bartlett 核)。"""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    u = X * resid[:, None]
    S = u.T @ u / n
    for L in range(1, q + 1):
        w = 1.0 - L / (q + 1.0)
        G = u[L:].T @ u[:-L] / n
        S += w * (G + G.T)
    V = n * XtX_inv @ S @ XtX_inv
    return np.sqrt(np.maximum(np.diag(V), 0.0))


def ann(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1) if len(r) else np.nan


def assert_aligned(rs: pd.Series, rb: pd.Series, tag: str):
    ks = (-2, -1, 0, 1, 2)
    cs = {k: float(rs.corr(rb.shift(k))) for k in ks}
    peak = max(ks, key=lambda k: cs[k])
    print("  [%s] %-22s %s  峰值 k=%+d" % ("OK" if peak == 0 else "MISALIGNED", tag,
          "  ".join(f"k={k:+d}:{cs[k]:+.3f}" for k in ks), peak))
    if peak != 0:
        raise AssertionError(f"{tag} 错位 {peak} 天,拒绝出数")


# ---------------------------------------------------------------- 风格因子收益
def fwd_returns(p, mode) -> pd.DataFrame:
    """与 strategy.benchmark 同口径的个股前瞻收益:索引在信号日 t,吃 te->te+1。"""
    px, _ = ST.exec_prices(p, mode)
    lag = ST.EXEC_MODES[mode]["lag"]
    g = px.shift(-1) / px - 1.0                       # 索引在成交日 te
    return g.shift(-lag).iloc[:len(p.dates) - 1 - lag]


def style_return(z: pd.DataFrame, R: pd.DataFrame, u: pd.DataFrame) -> pd.Series:
    """美元中性多空组合的日收益。z 为截面标准化后的风格分。"""
    z = z.where(u).reindex_like(R)
    z = z.sub(z.mean(axis=1), axis=0)                 # 截面去均值
    w = z.div(z.abs().sum(axis=1), axis=0)            # 归一到 sum|w| = 1
    return (w * R).sum(axis=1, min_count=1)


def cs_rank_z(x: pd.DataFrame, u: pd.DataFrame) -> pd.DataFrame:
    """截面 rank 后映射到均值 0、标准差约 1。"""
    r = x.where(u).rank(axis=1, pct=True)
    return (r - 0.5) * np.sqrt(12.0)


def build_styles(p, u, R) -> dict[str, pd.Series]:
    mv = FA._real(p, "mv")                            # 真实总市值(未复权价 x 股本)
    turn = FA._real(p, "turn")                        # 真实换手率
    close = p["close_px"]                             # 前复权收盘(收益率无前视,仅用于算动量/波动)
    mom21 = close / close.shift(21) - 1.0
    mom63 = close / close.shift(63) - 1.0
    rev5 = -(close / close.shift(5) - 1.0)            # 取负:高分 = 近期跌得多
    vol20 = (close / close.shift(1) - 1.0).rolling(20).std()

    raw = {
        "市值(log)": ops.signed_log(mv),
        "流动性(log成交额20日)": ops.signed_log(p["adv20"]),
        "换手率(20日均)": turn.rolling(20).mean(),
        "动量(21日)": mom21,
        "动量(63日)": mom63,
        "反转(5日)": rev5,
        "波动率(20日)": vol20,
    }
    return {k: style_return(cs_rank_z(v, u), R, u) for k, v in raw.items()}


def build_industry(p, u, R) -> tuple[dict[str, pd.Series], pd.Series]:
    """行业相对收益:每个申万一级行业等权收益 − 全池等权收益。"""
    L = ND.sw_industry_labels(p.dates, p.codes).reindex_like(R)
    names = ND.sw_industry_names()
    pool = R.where(u).mean(axis=1)
    counts = L.where(u).stack().value_counts()
    keep = [c for c in counts.index[:TOP_INDUSTRIES]]
    out = {}
    for c in keep:
        m = (L == c) & u
        out[f"行业:{names.get(c, str(c))}"] = R.where(m).mean(axis=1) - pool
    return out, pool


# ---------------------------------------------------------------- 主流程
def main():
    FA.build_all()
    man = json.load(open(LEDGER / "comp_v5_members.json"))
    members, signs = man["members"], man["signs"]

    rows, daily = [], {}
    for mode in MODES:
        p = TailPanel() if ST.EXEC_MODES[mode]["tail"] else Panel()
        u = p["universe"]
        R = fwd_returns(p, mode)
        u = u.reindex_like(R).fillna(False)

        fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
               .replace([np.inf, -np.inf], np.nan) for n in members if n in FA.REGISTRY}
        score = combine.combine_equal(combine.prepare(fac, signs, p["universe"]))
        label, kw = PLAN
        strat = ST.run_strategy(score, p, VERSION, mode, **kw)["rets"]
        bench = ST.benchmark(p, VERSION, mode)["rets"]

        print(f"\n[对齐检查] {mode}")
        assert_aligned(strat, bench, "策略 vs 池等权")

        styles = build_styles(p, u, R)
        inds, _ = build_industry(p, u, R)
        allf = {**styles, **inds}

        idx = strat.index.intersection(bench.index)
        for f in allf.values():
            idx = idx.intersection(f.dropna().index)
        idx = idx.sort_values()

        y = (strat.reindex(idx) - bench.reindex(idx)).astype(float)
        F = pd.DataFrame({k: v.reindex(idx).astype(float) for k, v in allf.items()})
        F = F.fillna(0.0)

        daily[f"{mode}|超额"] = y
        for k, v in F.items():
            daily[f"{mode}|{k}"] = v

        X = np.column_stack([np.ones(len(idx)), F.to_numpy()])
        b, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
        resid = y.to_numpy() - X @ b
        se = nw_se_ols(X, resid, q=10)
        tstat = b / np.where(se > 0, se, np.nan)
        r2 = 1.0 - resid.var() / y.to_numpy().var()

        alpha_d = b[0]
        rows.append(dict(mode=mode, term="alpha(残差)", beta=np.nan,
                         t_q10=tstat[0], factor_ann=np.nan,
                         contrib_ann=alpha_d * TRADING_DAYS,
                         note="截距年化"))
        for j, name in enumerate(F.columns, start=1):
            f_ann = F[name].mean() * TRADING_DAYS
            rows.append(dict(mode=mode, term=name, beta=b[j], t_q10=tstat[j],
                             factor_ann=f_ann, contrib_ann=b[j] * f_ann, note=""))
        rows.append(dict(mode=mode, term="__总超额(算术年化)", beta=np.nan, t_q10=np.nan,
                         factor_ann=np.nan, contrib_ann=y.mean() * TRADING_DAYS,
                         note=f"R2={r2:.4f}"))
        rows.append(dict(mode=mode, term="__风格+行业合计", beta=np.nan, t_q10=np.nan,
                         factor_ann=np.nan,
                         contrib_ann=sum(b[j] * F.iloc[:, j - 1].mean() * TRADING_DAYS
                                         for j in range(1, len(b))),
                         note=""))
        # alpha 的 NW t 敏感性
        for q in (0, 5, 10, 20):
            se_q = nw_se_ols(X, resid, q=q)
            rows.append(dict(mode=mode, term=f"__alpha_t_q{q}", beta=np.nan,
                             t_q10=alpha_d / se_q[0] if se_q[0] > 0 else np.nan,
                             factor_ann=np.nan, contrib_ann=np.nan, note="alpha t 带宽敏感性"))

        print(f"\n{mode}: 总超额(算术年化) {y.mean()*TRADING_DAYS:+.4f}  "
              f"alpha {alpha_d*TRADING_DAYS:+.4f}  R2 {r2:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "attribution_summary.csv", index=False)
    pd.DataFrame(daily).to_csv(LEDGER / "attribution_daily.csv")

    print("\n" + "=" * 120)
    print("超额归因")
    print("=" * 120)
    for mode in MODES:
        d = df[df["mode"] == mode]
        print(f"\n--- {mode} ---")
        main_rows = d[~d.term.str.startswith("__")]
        print(main_rows[["term", "beta", "t_q10", "factor_ann", "contrib_ann"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        print(d[d.term.str.startswith("__")][["term", "t_q10", "contrib_ann", "note"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print(f"\n已写 {LEDGER/'attribution_summary.csv'}")
    print(f"已写 {LEDGER/'attribution_daily.csv'}")


if __name__ == "__main__":
    main()
