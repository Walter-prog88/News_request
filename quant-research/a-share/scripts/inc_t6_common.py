"""INC-T6(基本面定义场地、量价做场地内增量)公共口径与工具。

预注册:research/INC_T6_PREREG.md(sha256 76b006f2…a6e45,冻结于 2026-10-11 04:04 JST)。
本模块只放口径常量、场地构建、逐臂回测与统计工具;参数全部取自预注册,不得按结果改动。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from afactor.config import LEDGER, TMP, ROOT, REPORTS  # noqa: E402

PFX = "inc_t6"
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
ARMS_DIR = TMP / f"{PFX}_arms"
ARMS_DIR.mkdir(parents=True, exist_ok=True)

PREREG = ROOT / "research" / "INC_T6_PREREG.md"
PREREG_SHA = "76b006f28f25b50e4e3c401eff9717c4cdd798a401700b91f7393be9ba7e6a45"

TD = 242
VERSION = "V1000"
MODE = "open10"
AUM = 1e7
REBALANCE = 5
NW_BW = 10                                     # §9.1 Bartlett 带宽 L = max(5, 2h) = 10
Z95 = 1.96
EXCL_BUF = (0.20, 0.35)                        # 排雷主口径
TOP_BUF = (0.10, 0.15)                         # 选优主口径
TOP_BUF20 = (0.20, 0.25)                       # 选优敏感性
EXCL_BUF_D = (0.40, 0.55)                      # A11-d 亏损场地的差异化缓冲
MIN_SCORED = 300                               # §3.4 最小场地规模
MIN_SCORED_FRAC = 0.95

WINDOWS = {
    "dev": ("2021-01-04", "2023-12-31"),
    "test": ("2024-01-02", "2025-12-31"),
    "full": ("2021-01-04", "2025-12-31"),
    "y2021": ("2021-01-01", "2021-12-31"), "y2022": ("2022-01-01", "2022-12-31"),
    "y2023": ("2023-01-01", "2023-12-31"), "y2024": ("2024-01-01", "2024-12-31"),
    "y2025": ("2025-01-01", "2025-12-31"),
    "stress": ("2024-01-02", "2024-02-07"),
}
TEST_WINDOWS = {"test", "y2024", "y2025", "stress", "full"}   # 含检验窗日子的窗口(阶段 A 落盘前不得计算)

# capacity_grid.csv  COMP_v5_exclude | open10 | Y1.0 | pov | tick | const | 1e7(§10 前置:A10 复现)
CAP_REF = {"full": 0.053175, "dev": 0.043395, "test": 0.069878}
CAP_REF_RIDE = {"full": 0.052923, "dev": 0.043212, "test": 0.069506}

# 13 个主场地(§4)。编码:D1 0=亏损 1=盈利;D2/D3/D4 1=低 2=中 3=高(-1 = NaN,不进任何场地)
CELLS = ["D1_loss", "D1_profit", "D2_low", "D2_mid", "D2_high", "D3_low", "D3_mid", "D3_high",
         "D4_low", "D4_mid", "D4_high", "Q", "Qm"]
CELL_DESC = {"D1_loss": "TTM 归母净利<0(亏损)", "D1_profit": "TTM 归母净利>=0(盈利)",
             "D2_low": "营业利润同比 低1/3(下滑)", "D2_mid": "营业利润同比 中1/3", "D2_high": "营业利润同比 高1/3(增长)",
             "D3_low": "ROE 低1/3", "D3_mid": "ROE 中1/3", "D3_high": "ROE 高1/3",
             "D4_low": "EP 低1/3(贵,含亏损)", "D4_mid": "EP 中1/3", "D4_high": "EP 高1/3(便宜)",
             "Q": "质量成长 = ROE高 ∩ 增长", "Qm": "差场地 Q⁻ = ROE低 ∩ 下滑"}
CONFIRM_CELL = "Q"
# 阶段 A 候选:Q 以外全部场地(预注册写"11 个",按"Q 以外全部"执行 = 12 个,见 inc_t6_deviations.csv)
STAGEA_CELLS = [c for c in CELLS if c != CONFIRM_CELL]
H1_PAIRS = [("D1_loss", "D1_profit", "H1(验证性)"), ("D2_low", "D2_high", "H1′ 下滑−增长"),
            ("D3_low", "D3_high", "H1′ 低ROE−高ROE"), ("Qm", "Q", "H1′ Q⁻−Q"),
            ("D4_low", "D4_high", "H1′ 贵−便宜")]

STAGEA_FILE = LEDGER / f"{PFX}_stageA_selection.csv"
CELLS_FILE = TMP / f"{PFX}_cells.parquet"


# ====================================================================== 杂项
def check_prereg() -> str:
    h = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
    if h != PREREG_SHA:
        raise RuntimeError(f"预注册文件 sha256 变了:{h}(应为 {PREREG_SHA})—— 正文不得改写")
    return h


class Logger:
    def __init__(self, stage: str):
        self.f = open(LOGS / f"{PFX}_{stage}.log", "a")
        self.t0 = time.time()
        self(f"===== stage {stage} start {time.strftime('%Y-%m-%d %H:%M:%S')} =====")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        line = f"[{time.time() - self.t0:7.1f}s] {s}"
        print(line, flush=True)
        self.f.write(line + "\n"); self.f.flush()


def stageA_done() -> bool:
    return STAGEA_FILE.exists()


def require_stageA(what: str):
    """§9.3 / §13:阶段 A 名单落盘前不得计算、打印或写入任何检验窗统计。"""
    if not stageA_done():
        raise RuntimeError(f"{what}:阶段 A 入围名单 {STAGEA_FILE.name} 尚未落盘,不得计算检验窗统计")


# ====================================================================== 统计
def nw_se_mean(x: np.ndarray, bw: int = NW_BW) -> float:
    """均值的 Newey-West 标准误(Bartlett 核,带宽 bw)。"""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return np.nan
    u = x - x.mean()
    lrv = float(np.sum(u * u)) / n                     # 不用 @(numpy 2 + Accelerate 的 matmul 假警告,见 strategy.py)
    for L in range(1, bw + 1):
        lrv += 2.0 * (1.0 - L / (bw + 1)) * float(np.sum(u[L:] * u[:-L])) / n
    return float(np.sqrt(lrv / n)) if lrv > 0 else np.nan


def geo_ann(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float((1.0 + r).prod() ** (TD / len(r)) - 1.0)


def wslice(s: pd.Series | pd.DataFrame, w: str):
    a, b = WINDOWS[w]
    if w in TEST_WINDOWS:
        require_stageA(f"窗口 {w}")
    return s.loc[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]


def quarter_t(d: pd.Series) -> tuple[float, int]:
    """以日历季度为独立单位的 t(季度内日超额之和)。"""
    q = d.groupby([d.index.year, d.index.quarter]).sum()
    q = q[d.groupby([d.index.year, d.index.quarter]).size() >= 20]
    n = len(q)
    if n < 3 or q.std() <= 0:
        return np.nan, n
    return float(q.mean() / q.std() * np.sqrt(n)), n


def diff_stats(d: pd.Series, w: str) -> dict:
    """日差序列(超额或配对差)在窗口 w 内的统计:算术年化、NW t、95% 区间、季度 t。"""
    x = wslice(d.dropna(), w)
    n = len(x)
    if n < 20:
        return dict(n_days=n)
    mu = float(x.mean())
    se = nw_se_mean(x.to_numpy())
    qt, nq = quarter_t(x)
    ann, se_ann = mu * TD, se * TD
    return dict(n_days=n, ann=ann, t_nw=mu / se if se and se > 0 else np.nan, se_ann=se_ann,
                lo95=ann - Z95 * se_ann, hi95=ann + Z95 * se_ann, t_quarter=qt, n_quarters=nq,
                te_ann=float(x.std() * np.sqrt(TD)))


def align_peak(rs: pd.Series, rb: pd.Series, ks=range(-3, 4)) -> tuple[int, dict]:
    a, b = rs.align(rb, join="inner")
    cs = {k: float(a.corr(b.shift(k))) for k in ks}
    return max(cs, key=lambda k: cs[k] if np.isfinite(cs[k]) else -9), cs


def max_dd(r: pd.Series) -> float:
    nav = (1 + r.fillna(0)).cumprod()
    return float((nav / nav.cummax() - 1).min())


# ====================================================================== 面板与输入
_P = None


def panel():
    global _P
    if _P is None:
        from afactor.panel import Panel
        _P = Panel()
    return _P


def pool(p) -> pd.DataFrame:
    from afactor import strategy as ST
    return ST.universe(p, VERSION)


def comp_score(p) -> pd.DataFrame:
    return pd.read_parquet(TMP / "comp_v5_score_open10.parquet").reindex(index=p.dates, columns=p.codes)


def vp_score_path():
    return TMP / f"{PFX}_score_vp.parquet"


def clone_scores(p) -> dict:
    """C1 = −20 日 log(真实换手率) 均值(换手率 <= 0 记缺失);C2 = −log(真实总市值)。"""
    from afactor import factors as FA, ops
    turn = FA._real(p, "turn").astype(np.float64)
    lt = np.log(turn.where(turn > 0))
    c1 = -ops.ts_mean(lt, 20)
    mv = FA._real(p, "mv").astype(np.float64)
    c2 = -np.log(mv.where(mv > 0))
    return {"C1": c1.astype(np.float32), "C2": c2.astype(np.float32)}


def fund_vars(p, lag_rows: int = 0) -> dict:
    """§4 场地变量(PIT,可见日 <= t)。lag_rows=1:基本面生效日再推迟 1 个交易日(§11 敏感性)。
    MV 是当日收盘后可知的市场数据,不随生效日敏感性移动。"""
    from afactor import factors as FA
    ni = FA._fu(p, "n_income_attr_p_ttm").astype(np.float64)
    eq = FA._fu(p, "total_hldr_eqy_exc_min_int_avg").astype(np.float64)
    opy = FA._fu(p, "operate_profit_yoy").astype(np.float64)
    niq = FA._fu(p, "n_income_attr_p_q").astype(np.float64)
    if lag_rows:
        ni, eq, opy, niq = (x.shift(lag_rows) for x in (ni, eq, opy, niq))
    mv = FA._real(p, "mv").astype(np.float64)
    roe = FA._safe_ratio(ni, eq)                          # = REGISTRY["fu_roe"] 同式
    ep = FA._safe_ratio(ni, mv)                           # 用真实总市值
    return dict(NI=ni, EQ=eq, OPY=opy, MV=mv, ROE=roe, EP=ep, NIQ=niq)


def tercile_code(x: pd.DataFrame, U: pd.DataFrame, n: int = 3) -> np.ndarray:
    """当日决策池内按秩分档(rank(pct=True));1..n,NaN / 池外 = -1。三分位边界:<=1/3, (1/3,2/3], >2/3。"""
    xv = x.where(U)
    q = xv.rank(axis=1, pct=True).to_numpy(np.float64)
    out = np.full(q.shape, -1, np.int8)
    fin = np.isfinite(q)
    k = np.ceil(q * n - 1e-12)                            # q<=1/n -> 1, ...
    k = np.clip(k, 1, n)
    out[fin] = k[fin].astype(np.int8)
    return out


def build_cell_codes(p, U: pd.DataFrame, fv: dict) -> dict:
    """返回每维的 T x N int8 编码。"""
    Uv = U.to_numpy(bool)
    ni = fv["NI"].to_numpy(np.float64)
    d1 = np.full(ni.shape, -1, np.int8)
    fin = Uv & np.isfinite(ni)
    d1[fin & (ni < 0)] = 0
    d1[fin & (ni >= 0)] = 1
    niq = fv["NIQ"].to_numpy(np.float64)
    d1q = np.full(niq.shape, -1, np.int8)
    finq = Uv & np.isfinite(niq)
    d1q[finq & (niq < 0)] = 0
    d1q[finq & (niq >= 0)] = 1
    return dict(D1=d1, D1q=d1q,
                D2=tercile_code(fv["OPY"], U), D3=tercile_code(fv["ROE"], U), D4=tercile_code(fv["EP"], U),
                D2q5=tercile_code(fv["OPY"], U, 5), D3q5=tercile_code(fv["ROE"], U, 5),
                D4q5=tercile_code(fv["EP"], U, 5))


def cell_masks_from_codes(codes: dict, which: str = "main") -> dict:
    """编码 -> 场地布尔掩码(T x N ndarray)。"""
    d1, d2, d3, d4 = codes["D1"], codes["D2"], codes["D3"], codes["D4"]
    if which == "main":
        return {"D1_loss": d1 == 0, "D1_profit": d1 == 1,
                "D2_low": d2 == 1, "D2_mid": d2 == 2, "D2_high": d2 == 3,
                "D3_low": d3 == 1, "D3_mid": d3 == 2, "D3_high": d3 == 3,
                "D4_low": d4 == 1, "D4_mid": d4 == 2, "D4_high": d4 == 3,
                "Q": (d3 == 3) & (d2 == 3), "Qm": (d3 == 1) & (d2 == 1)}
    if which == "q5":
        out = {}
        for d in ("D2", "D3", "D4"):
            for k in range(1, 6):
                out[f"{d}q5_{k}"] = codes[f"{d}q5"] == k
        return out
    if which == "d1q":
        return {"D1q_loss": codes["D1q"] == 0, "D1q_profit": codes["D1q"] == 1}
    if which == "d1fc":
        return {"D1fc_loss": codes["D1fc"] == 0, "D1fc_profit": codes["D1fc"] == 1}
    raise ValueError(which)


def forecast_loss_code(p, U: pd.DataFrame) -> np.ndarray:
    """D1″(探索性):最近一次业绩预告类型 ∈ {首亏, 续亏, 增亏} -> 0,其余类型 -> 1,无预告 -> -1。
    公告日后一交易日生效(side="right",与 newdata.surprise_panels 同口径),状态 ffill 到下一次预告。"""
    from afactor import newdata as ND
    fc = ND._read_dated_csv("forecast.csv", "ann_date")
    fc["ann_date"] = ND._ymd(fc["ann_date"]); fc["end_date"] = ND._ymd(fc["end_date"])
    fc = fc[fc["ts_code"].isin(p.codes) & fc["ann_date"].notna()]
    fc = fc.sort_values(["ts_code", "end_date", "ann_date"]).drop_duplicates(["ts_code", "end_date"], keep="last")
    fc["is_loss"] = fc["type"].isin(["首亏", "续亏", "增亏"]).astype(float)
    st = ND._event_panel(fc, "is_loss", "ann_date", p.dates, p.codes, lag_days=1, ffill=True)
    v = st.to_numpy(np.float64)
    Uv = U.to_numpy(bool)
    out = np.full(v.shape, -1, np.int8)
    out[Uv & (v == 1)] = 0
    out[Uv & (v == 0)] = 1
    return out


def save_codes(p, codes: dict, path=CELLS_FILE):
    """场地编码落盘(长表:池内股票日;date, code, 各维 int8)。"""
    T, N = codes["D1"].shape
    any_in = np.zeros((T, N), bool)
    for v in codes.values():
        any_in |= v >= 0
    ti, ni = np.nonzero(any_in)
    df = pd.DataFrame({"date": p.dates[ti], "code": p.codes[ni]})
    for k, v in codes.items():
        df[k] = v[ti, ni]
    df.to_parquet(path, compression="zstd")


def load_codes(p, path=CELLS_FILE) -> dict:
    df = pd.read_parquet(path)
    ti = p.dates.get_indexer(pd.DatetimeIndex(df["date"]))
    ni = p.codes.get_indexer(pd.Index(df["code"]))
    out = {}
    for k in df.columns:
        if k in ("date", "code"):
            continue
        a = np.full((len(p.dates), len(p.codes)), -1, np.int8)
        a[ti, ni] = df[k].to_numpy(np.int8)
        out[k] = a
    return out


def to_frame(p, m: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(m, index=p.dates, columns=p.codes)


# ====================================================================== 权重
def w_const(p, U: np.ndarray) -> np.ndarray:
    """U0 / A00 / A01:场地全体等权(= run_strategy(常数, scheme='exclude', exclude_frac=0))。"""
    from afactor.backtest import target_weights
    F = np.ones(U.shape, np.float32)
    return target_weights(F, U, top_n=None, sign=1, scheme="exclude", top_frac=0.0)


def w_exclude(S: np.ndarray, U: np.ndarray, buffer=EXCL_BUF, frac: float = 0.3) -> np.ndarray:
    """U1 / A10 / A11:与 strategy.build_weights(scheme='exclude', buffer=...) 同法。"""
    from afactor.backtest import weights_exclude_buffer, target_weights
    if buffer is None:
        return target_weights(S, U, top_n=None, sign=1, scheme="exclude", top_frac=frac)
    return weights_exclude_buffer(S, U, 1, *buffer)


def w_top(S: np.ndarray, U: np.ndarray, buffer=TOP_BUF) -> np.ndarray:
    """U2 / C1 / C2:选优 + 缓冲带(scripts/run_barra_attribution_topfrac.weights_top_buffer)。"""
    from run_barra_attribution_topfrac import weights_top_buffer
    return weights_top_buffer(S, U, 1, *buffer)


def w_merge_groups(parts: list[np.ndarray]) -> np.ndarray:
    """多个分组各自排雷后的保留集合并,整体等权(A11-n / A11-d)。"""
    keep = np.zeros(parts[0].shape, bool)
    for W in parts:
        keep |= W > 0
    n = keep.sum(axis=1, keepdims=True)
    return np.divide(keep.astype(np.float32), n, out=np.zeros(keep.shape, np.float32), where=n > 0)


# ====================================================================== 成本与回测
_CM: dict = {}


def cost_model(p, kind: str):
    """Y1 主口径 = impact.CostModel(Y=1, 一跳);Y0.5 / Y1.5;legacy = None(佣金 1bp + 印花税)。"""
    from afactor import impact as IM
    if kind == "legacy":
        return None
    y = {"Y1": 1.0, "Y0.5": 0.5, "Y1.5": 1.5}[kind]
    if y not in _CM:
        _CM[y] = IM.CostModel(p, impact_y=y, spread_mult=1.0)
    return _CM[y]


def _atomic_parquet(df: pd.DataFrame, f):
    """先写临时文件再改名(两个进程并行时同名文件不会写坏)。"""
    tmp = f.with_name(f.name + f".{os.getpid()}.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, f)


REC_KEEP = ["ret", "gross_ret", "cost_total", "cost_commission", "cost_stamp", "cost_spread", "cost_impact",
            "turnover", "exposure", "n_hold", "part_bound_frac", "unfilled"]


def arm_path(tag: str):
    return ARMS_DIR / f"{PFX}_arm_{tag}.parquet"


def run_arm(p, tag: str, W: np.ndarray, cost: str = "Y1", capital: float = AUM, keep_holdings: bool = False,
            log=print, force: bool = False) -> pd.DataFrame:
    """单臂回测(5 相位平均、open10、从 2020-01-02 连续跑、不传 date_mask),逐日记录落盘。可断点续跑。"""
    from afactor import strategy as ST
    f = arm_path(tag)
    if f.exists() and not force:
        return pd.read_parquet(f)
    t0 = time.time()
    cm = cost_model(p, cost)
    res = ST.run_weights(W, p, MODE, capital, rebalance=REBALANCE, avg_phases=True, cost_model=cm, record=True)
    rec = res["record"]
    daily = rec["daily"][REC_KEEP].copy()
    assert np.allclose(daily["ret"].to_numpy(), res["rets"].reindex(daily.index).to_numpy(), atol=1e-12)
    by = rec["daily_by_phase"]
    ph = by.pivot_table(index=by.index, columns="phase", values="ret")
    for c in ph.columns:
        daily[f"ph{int(c)}"] = ph[c].reindex(daily.index)
    daily["exposure_max_phase"] = by.groupby(level=0)["exposure"].max().reindex(daily.index)
    _atomic_parquet(daily, f)
    if keep_holdings:
        H = rec["holdings"]
        H = H.loc[:, (H != 0).any(axis=0)]
        H.to_parquet(ARMS_DIR / f"{PFX}_hold_{tag}.parquet")
    log(f"  arm {tag:42s} turn {res['turnover']:.4f} cost {res['cost_bps']:6.2f}bp/d hold {res['holdings']:7.1f} "
        f"expmax {daily['exposure_max_phase'].max():.6f}  {time.time() - t0:5.1f}s")
    return daily


def bench_path(tag: str):
    return ARMS_DIR / f"{PFX}_bench_{tag}.parquet"


def run_bench(p, tag: str, U: np.ndarray, force: bool = False) -> pd.Series:
    """场地同频(5 日)等权基准:ST.benchmark(p, V1000, open10, rebalance=5, universe_override=U_c)。"""
    from afactor import strategy as ST
    f = bench_path(tag)
    if f.exists() and not force:
        return pd.read_parquet(f)["ret"]
    b = ST.benchmark(p, VERSION, MODE, rebalance=REBALANCE, universe_override=to_frame(p, U))["rets"]
    _atomic_parquet(b.rename("ret").to_frame(), f)
    return b


def load_arm(tag: str) -> pd.DataFrame:
    return pd.read_parquet(arm_path(tag))


def load_bench(tag: str) -> pd.Series:
    return pd.read_parquet(bench_path(tag))["ret"]


def bench_engine_like(P: np.ndarray, U: np.ndarray, notrade_te: np.ndarray | None, k: int = REBALANCE,
                      lag: int = 1, engine_like: bool = True) -> pd.Series:
    """与引擎同口径的无摩擦同频等权基准(审查 P1 修复,偏离表第 14 行;诊断用,不得作为基准)。

    与 ST._pool_ew_drift_returns 同一日历(t0 = 0,相位 ph 的调仓日 = {t: (t-ph) % k == 0})、同一漂移与满仓冷启动,
    差别只在调仓日对"成交日 te 不可交易"的股票:
      engine_like=False  或 notrade_te=None:每只都重置(池 ∩ te 有价)等权 = 官方基准(核对用,应逐位相同);
      engine_like=True:不可交易的持仓保留其漂移后的权重(simulate 的 stuck 规则:stuck = ~can & h>0,
                       budget = 1 - Σ stuck,目标在可交易股票上归一到 budget),不可交易的非持仓不买。
    notrade_te 为按成交日 te 索引的 (T, N) 布尔数组(True = te 不可成交);te 无价(px[te] 或 px[te+1] 非有限)
    的股票在 engine_like 下一律视为不可交易。P 为 (T, N) 成交价,U 为信号日决策池 (T, N)。
    返回按信号日 t 索引(p.dates[:T-1-lag])的 k 相位平均日收益。"""
    T, N = P.shape
    n_sig = T - 1 - lag
    with np.errstate(invalid="ignore", divide="ignore"):
        G = P[1 + lag:1 + lag + n_sig] / P[lag:lag + n_sig] - 1.0
    NT = notrade_te[lag:lag + n_sig] if notrade_te is not None else np.zeros((n_sig, N), bool)
    W = np.zeros((k, N))
    R = np.zeros((k, n_sig))
    for t in range(n_sig):
        g = G[t]
        fin = np.isfinite(g)
        gz = np.where(fin, g, 0.0)
        reb = ((t - np.arange(k)) % k) == 0
        todo = reb | (W.sum(axis=1) <= 0)
        if todo.any():
            if engine_like and notrade_te is not None:
                nt = NT[t] | ~fin
                m = U[t] & ~nt
                n = int(m.sum())
                for ph in np.flatnonzero(todo):
                    stuck = nt & (W[ph] > 1e-12)
                    budget = max(0.0, 1.0 - float(W[ph][stuck].sum()))
                    W[ph] = np.where(stuck, W[ph], 0.0) + (np.where(m, budget / n, 0.0) if n > 0 else 0.0)
            else:
                m = U[t] & fin
                n = int(m.sum())
                W[todo] = np.where(m, 1.0 / n, 0.0) if n > 0 else 0.0
        r = (W * gz).sum(axis=1)
        R[:, t] = r
        grow = 1.0 + r
        ok = grow > 1e-9
        W[ok] = W[ok] * (1.0 + gz) / grow[ok, None]
    return R.mean(axis=0)


def arm_window_stats(daily: pd.DataFrame, bench: pd.Series, w: str) -> dict:
    """单臂在窗口 w 内:对基准的净超额(算术年化 / NW t / 区间 / 季度 t)、几何年化差、毛超额、成本、换手等。"""
    r = daily["ret"]
    a, b = r.align(bench, join="inner")
    ex = (a - b)
    st = diff_stats(ex, w)
    out = {f"ex_{k}": v for k, v in st.items()}
    rr, bb = wslice(a, w), wslice(b, w)
    dd = wslice(daily.reindex(a.index), w)
    out.update(ann_geo=geo_ann(rr), bench_geo=geo_ann(bb), ex_geo=geo_ann(rr) - geo_ann(bb),
               gross_ex_ann=float((dd["gross_ret"] - bb).mean() * TD),
               cost_ann=float(dd["cost_total"].mean() * TD),
               cost_commission_bp=float(dd["cost_commission"].mean() * TD * 1e4),
               cost_stamp_bp=float(dd["cost_stamp"].mean() * TD * 1e4),
               cost_spread_bp=float(dd["cost_spread"].mean() * TD * 1e4),
               cost_impact_bp=float(dd["cost_impact"].mean() * TD * 1e4),
               turnover_daily=float(dd["turnover"].mean()), holdings=float(dd["n_hold"].mean()),
               exposure_max=float(dd["exposure_max_phase"].max()),
               part_bound_frac=float(dd["part_bound_frac"].mean(skipna=True)),
               max_dd=max_dd(rr), bench_max_dd=max_dd(bb))
    phc = [c for c in dd.columns if c.startswith("ph") and c[2:].isdigit()]
    if phc:
        pex = [float((dd[c] - bb).mean() * TD) for c in phc]
        out.update(phase_ex_min=min(pex), phase_ex_max=max(pex), phase_ex_range=max(pex) - min(pex))
    return out
