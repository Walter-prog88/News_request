"""前向检验适配器(前缀 fwd):**只放宽日期守卫、只换数据源,公式一律不改**(预注册 FORWARD_TEST_PREREG §4.4)。

冻结代码对 2026 年日期有多处硬守卫(build_daily_5m 的月份守卫、impact / barra 的 assert_not_sealed、
barra 的 RESEARCH_END 断言、statind 的标签文件只覆盖研究期、index_proxy 的 SEAL_END、earnings_events.CUTOFF)。
本模块的每个适配器都只做两件事之一:
  (1) 放宽日期守卫:在上下文内把守卫函数 / 常量换成放行版本,退出时恢复;
  (2) 换数据源:把"研究期原料"换成前向面板 / 前向原始数据,计算仍调用冻结函数本身。
每个适配器在前向使用之前,都要先在研究期数据上与冻结代码的原输出逐位比较(scripts/fwd_adapter_equiv.py,
结果 research/ledger/fwd_adapter_equiv.csv);不通过的不得使用。

**只有前缀 fwd 的前向检验代码可以 import 本模块。** 2026-01-01 ~ 2026-08-30 的数据只作特征回看 / 预热输入,
不得据此计算或报告任何策略收益、IC、超额。
"""
from __future__ import annotations

import contextlib
import time

import numpy as np
import pandas as pd

from afactor import minute5 as M5
from afactor.config import N_BARS, load_raw_minute

# 规范字段名 <- 候选近似(与 build_daily_5m.finalize 同一映射,常量取自 minute5,不得再改选)
CANON = {"twap_open10": f"twap_open10__{M5.TWAP5_METHOD}",
         "amt_open10": f"amt_open10__{M5.AMT_OPEN10_METHOD}",
         "vwap": f"vwap__{M5.VWAP5_METHOD}",
         "amt_openauc_share": f"amt_openauc_share__{M5.OPENAUC5_METHOD}"}
FIELDS = tuple(M5.M5_SIGNAL_FIELDS) + tuple(M5.M5_EXEC_FIELDS)        # 16 个 5 分钟口径原料


def _canonical(feats: dict) -> dict:
    """_day_features_5m 的输出(含全部候选)-> 16 个规范字段。"""
    out = {}
    for f in FIELDS:
        out[f] = feats[CANON[f]] if f in CANON else feats[f]
    return out


# =============================================================================================
# 适配器 A:1 分钟 -> 5 分钟口径日频原料(build_daily_5m.process_month 去掉月份守卫)
# =============================================================================================
def minute_month_features(month: str, drop_bj: bool = True, clean: bool = True) -> dict:
    """build_daily_5m.process_month 的前向版:同一套逐日计算(float32 读入 -> 逐日 float64 ->
    minute5.aggregate_day -> build_daily_5m._day_features_5m -> np.where(valid, ·, NaN) -> float32),
    只去掉两处守卫(month > 202512 拒绝、assert_not_sealed)。

    clean=True:先对三张宽表做 fwd_data.clean_minute_month(2026 文件里的 UTC 时间戳行与缺 09:30 行;
                研究期文件上它是恒等变换)。
    drop_bj=True:读入后去掉北交所列 —— 预注册 §4.1:semibeta_mp 的市场均值只用沪深;北交所没有前向数据源。
                研究期等价性检验(2025-12)用 drop_bj=False,与 cache/m5 同一输入。
    返回 dict(dates, codes, fields={规范字段: float32 (D, N)}, diag)。"""
    from afactor import build_daily_5m as BD5
    from afactor import fwd_data as FD
    t0 = time.time()
    C = load_raw_minute("close_adj", month)
    O = load_raw_minute("open_adj", month)
    V = load_raw_minute("amt", month)
    diag = {}
    if clean:
        C, dC = FD.clean_minute_month(C)
        O, dO = FD.clean_minute_month(O)
        V, dV = FD.clean_minute_month(V)
        diag = dict(utc_rows=dC["utc_rows"], conflict_cells=dC["conflict_cells"] + dO["conflict_cells"]
                    + dV["conflict_cells"], missing_slots=dC["missing_slots"])
    if drop_bj:
        keep = [c for c in C.columns if not str(c).endswith(".BJ")]
        C, O, V = C[keep], O[keep], V[keep]
    codes = C.columns
    assert list(O.columns) == list(codes) and list(V.columns) == list(codes), f"{month} 列不一致"
    assert O.index.equals(C.index) and V.index.equals(C.index), f"{month} 索引不一致"
    days = C.index.normalize().values
    Ca, Oa, Va = C.to_numpy(np.float32), O.to_numpy(np.float32), V.to_numpy(np.float32)
    del C, O, V
    uniq, starts, counts = np.unique(days, return_index=True, return_counts=True)
    order = np.argsort(starts)
    feats, bad, good = {}, [], []
    for d, s, n in zip(uniq[order], starts[order], counts[order]):
        if n != N_BARS:
            bad.append((str(pd.Timestamp(d).date()), int(n)))
            continue
        P_ = Ca[s:s + n].astype(np.float64); O_ = Oa[s:s + n].astype(np.float64)
        V_ = Va[s:s + n].astype(np.float64)
        valid = ~np.isnan(P_).all(axis=0)
        P5, O5, V5 = M5.aggregate_day(P_, O_, V_)
        f = _canonical(BD5._day_features_5m(P5, O5, V5))
        for k, val in f.items():
            feats.setdefault(k, []).append(np.where(valid, val, np.nan))
        good.append(pd.Timestamp(d))
    arrs = {k: np.asarray(v, dtype=np.float32) for k, v in feats.items()}
    diag.update(bad_days=bad, n_days=len(good), n_codes=len(codes), secs=round(time.time() - t0, 1))
    return dict(dates=pd.DatetimeIndex(good), codes=pd.Index(codes), fields=arrs, diag=diag)


# =============================================================================================
# 适配器 B:baostock 5 分钟 -> 同一套 5 分钟口径日频原料
# =============================================================================================
def bs5_day_features(P5: np.ndarray, O5: np.ndarray, V5: np.ndarray, close_d: np.ndarray,
                     open_d: np.ndarray, amount_d: np.ndarray, adj: np.ndarray, ok: np.ndarray) -> dict:
    """单日 baostock 5 分钟 (48, N)(**不复权**)-> 16 个规范字段 (N,)。

    数据源替换只做预注册 §4.1/§4.2 写定的三步,公式仍是 build_daily_5m._day_features_5m:
      1. 0 价当 NaN(无成交 bar);组内价格按 1 分钟数据的约定前值填充(1 分钟数据 amt=0 的分钟价格为前值填充):
         bar 收盘缺失 -> 上一根收盘(首根缺失 -> 日线开盘);bar 开盘缺失 -> 上一根收盘(首根由 reconcile 设为日线开盘);
      2. minute5.reconcile_with_daily:bar47 收盘 := 日线收盘、bar0 开盘 := 日线开盘、bar47 成交额 := 日成交额 − 前 47 根;
      3. 价格乘以复权系数 adj(以 2026-08-17 为基准的后复权,F_t = Π_{08-17 < e <= t} prev_close_e / preclose_e)。
    ok = 当日有真实成交(日线 tradestatus=1、成交额>0、48 根齐全);其余股票全部字段置 NaN(1 分钟口径"整日 NaN = 停牌")。"""
    from afactor import build_daily_5m as BD5
    P5 = np.where(P5 > 0, P5, np.nan).astype(np.float64)
    O5 = np.where(O5 > 0, O5, np.nan).astype(np.float64)
    V5 = np.where(np.isfinite(V5), V5, 0.0).astype(np.float64)
    # 1. 组内前值填充(只影响无成交 bar;QA:全窗口 149 行 0 价,其中 144 行是 3 个停牌股票日)
    seed = np.where(np.isfinite(open_d), open_d, np.nan)
    prev = seed.copy()
    for k in range(P5.shape[0]):
        P5[k] = np.where(np.isfinite(P5[k]), P5[k], prev)
        if k > 0:
            O5[k] = np.where(np.isfinite(O5[k]), O5[k], P5[k - 1])
        prev = P5[k]
    # 2. 日线修正(价格同为不复权)
    P5, O5, V5 = M5.reconcile_with_daily(P5, O5, V5, close_d, open_d, amount_d)
    # 3. 复权到 08-17 基准
    P5 = P5 * adj[None, :]
    O5 = O5 * adj[None, :]
    f = _canonical(BD5._day_features_5m(P5, O5, V5))
    return {k: np.where(ok, v, np.nan).astype(np.float32) for k, v in f.items()}


# =============================================================================================
# 适配器 C:放宽日期守卫的上下文(impact / barra);换数据源(未复权收盘价)
# =============================================================================================
@contextlib.contextmanager
def lifted_guards(raw_close: pd.DataFrame | None = None, research_end: str = "2026-10-09"):
    """在上下文内放宽冻结代码的日期守卫,退出时恢复:
      - afactor.impact.assert_not_sealed、afactor.barra.assert_not_sealed -> 放行;
      - afactor.barra.RESEARCH_END -> research_end(RiskModel.build 里 `p.dates.max() <= RESEARCH_END` 的断言);
      - raw_close 给定时:newdata.raw_close(dates, codes) -> 前向面板的未复权收盘价(impact.tick_floor 的唯一外部数据源;
        前向窗口 daily_basic 只到 08-21,08-18 起换成 baostock 日线未复权收盘)。
    计算函数本身一个字都不改。"""
    from afactor import impact as IM, barra as BR, newdata as ND
    saved = dict(im_ans=IM.assert_not_sealed, br_ans=BR.assert_not_sealed, br_end=BR.RESEARCH_END,
                 nd_rc=ND.raw_close)
    noop = lambda *a, **k: None
    IM.assert_not_sealed = noop
    BR.assert_not_sealed = noop
    BR.RESEARCH_END = research_end
    if raw_close is not None:
        rc = raw_close

        def _raw_close(dates, codes):
            return rc.reindex(index=pd.DatetimeIndex(dates), columns=codes).astype(np.float32)
        ND.raw_close = _raw_close
    try:
        yield
    finally:
        IM.assert_not_sealed = saved["im_ans"]
        BR.assert_not_sealed = saved["br_ans"]
        BR.RESEARCH_END = saved["br_end"]
        ND.raw_close = saved["nd_rc"]


# =============================================================================================
# 适配器 D:统计行业标签(statind.build_labels 的逐季计算,换数据源 + 只算指定季度,不写研究期标签文件)
# =============================================================================================
def statind_labels_for_quarters(close: pd.DataFrame, quarters: list[str], research_q0: str = "2020Q1") -> pd.DataFrame:
    """与 statind.build_labels 同一算法(逐季末 120 日收益、KMeans K=30、n_init=4、标签用于下一季),
    只做两处数据源替换:输入 close 换成前向面板;只计算 quarters 里列出的"标签所用季度"(例如 2025Q4 -> 用于 2026Q1)。
    random_state 沿用研究期的季度序号 qi(研究面板首季 2020Q1 = 0),与研究期同一季度的种子相同。
    不写 statind.LABEL_FILE(研究期标签文件)。返回 DataFrame[close.index × close.columns],-1 = 无标签。"""
    from sklearn.cluster import KMeans
    from afactor import statind as SI
    ret = (close / close.shift(1) - 1.0).clip(-0.25, 0.25)
    dates = ret.index
    per = dates.to_period("Q")
    q_ends = pd.Series(dates).groupby(per).max()
    L = pd.DataFrame(-1, index=dates, columns=ret.columns, dtype=np.int16)
    q0 = pd.Period(research_q0, "Q")
    qlist = list(q_ends.index)
    for q in quarters:
        q = pd.Period(q, "Q")
        if q not in q_ends.index:
            continue
        qe = q_ends[q]
        qi = (q - q0).n
        pos = dates.get_loc(qe)
        if pos < SI.LOOKBACK:
            continue
        win = ret.iloc[pos - SI.LOOKBACK + 1: pos + 1]
        valid = win.notna().sum() >= SI.MIN_VALID
        cols = valid[valid].index
        X = win[cols].to_numpy(np.float64)
        mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
        ok = sd > 1e-9
        X = SI._safe_std(X, mu, sd)
        X = np.where(np.isfinite(X), X, 0.0).T
        km = KMeans(n_clusters=SI.K, n_init=4, random_state=qi).fit(X[ok])
        lab = np.full(len(cols), -1, np.int16)
        lab[ok] = km.labels_
        start = pos + 1
        k_next = qlist.index(q) + 1
        end = dates.get_loc(q_ends[qlist[k_next]]) if k_next < len(qlist) else len(dates) - 1
        if start > end:
            continue
        L.iloc[start:end + 1, [L.columns.get_loc(c) for c in cols]] = lab[None, :]
    return L
