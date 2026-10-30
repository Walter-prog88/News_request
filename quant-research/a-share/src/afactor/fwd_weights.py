"""前向检验的指数基准(前缀 fwd):中证 1000 实际成分权重还原 + 真实 000852 对齐到信号日。

预注册 FORWARD_TEST_PREREG §4.5:
  - 权重来源:index_stock_cons_weight_csindex("000852") 快照(权重日 d_S = 2026-09-30,成分名单日 10-09;
    有效条件由 scripts/fwd_decisions.py 在计算任何收益之前核对,全部满足 -> 用实际成分);
  - 决策日 t 的收盘权重:w_i(t) ∝ w_i(d_S) × P_i(t) / P_i(d_S),P = 前向面板复权收盘(§4.2 拼接后),停牌按前值。
    d_S 晚于 t 时用到了 t 之后的价格,但它只是精确还原 t 日的指数权重(t 日权重本来就是公开信息),没有前视;
    股息带来约 0.1%/月量级的权重漂移;临时调整(调入 / 调出)不处理(快照没有纳入日期);
  - 成交时点计价:index_proxy.exec_time_bench_returns(p, "open10", W)(冻结函数,原样调用);
  - T3 优化器内部用的基准权重也取同一个 W[t](决策时点只用收盘权重)。
  - 真实 000852(新浪指数日线收盘,价格指数):r[t] = close(t+1)/close(t) − 1 记在信号日 t,
    与 index_proxy.index_returns_on_signal_days 同式,只换数据源(那边只读到 2025-12-31 的 ETF 项目快照);只作参考。
只有前缀 fwd 的前向检验代码可以 import 本模块。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afactor.config import CACHE

RAW = CACHE / "forward" / "raw"
CSI_W_FILE = RAW / "ak_csindex" / "000852_cons_weight_fetched20261010.csv"
CSI_C_FILE = RAW / "ak_csindex" / "000852_cons_fetched20261010.csv"
IDX_FILE = RAW / "ak_index" / "sh000852_fetched20261010.csv"


def csi1000_snapshot() -> tuple[pd.Series, pd.Timestamp]:
    """中证 1000 权重快照 -> (按项目代码索引、和为 1 的权重 Series, 权重日 d_S)。"""
    w = pd.read_csv(CSI_W_FILE, dtype={"成分券代码": str})
    dS = pd.Timestamp(w["日期"].iloc[0])
    ex = np.where(w["交易所"].str.contains("上海"), ".SH", np.where(w["交易所"].str.contains("深圳"), ".SZ", ".BJ"))
    code = w["成分券代码"].str.zfill(6) + ex
    s = pd.Series(w["权重"].to_numpy(np.float64), index=pd.Index(code, name="ts_code"))
    return s / s.sum(), dS


def backout_close_weights(snap: pd.Series, dS, anchor: pd.DataFrame, anchor_at_dS: pd.Series | None = None) -> pd.DataFrame:
    """由快照权重反推 anchor 各行(收盘时点)的权重:w_i(t) ∝ snap_i × anchor_i(t) / anchor_i(d_S),行和归一。
    anchor = 复权收盘价(停牌按前值);anchor_at_dS 默认取 anchor 在 d_S 那一行。缺价成分的比值记 1(保持快照权重)。"""
    codes = anchor.columns
    w0 = snap.reindex(codes).fillna(0.0).to_numpy(np.float64)
    a0 = (anchor.loc[pd.Timestamp(dS)] if anchor_at_dS is None else anchor_at_dS).reindex(codes).to_numpy(np.float64)
    A = anchor.to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = A / a0[None, :]
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0)
    W = w0[None, :] * ratio
    s = W.sum(axis=1, keepdims=True)
    W = np.where(s > 0, W / np.where(s > 0, s, 1.0), 0.0)
    return pd.DataFrame(W, index=anchor.index, columns=codes)


def csi1000_close_weights(p, start="2026-08-17") -> pd.DataFrame:
    """前向面板上的中证 1000 收盘权重 W[t](t >= start 的行有值,之前全 0;优化器与评价只用前向窗口的行)。
    anchor 与 index_proxy.exec_time_bench_returns 同一收盘价:close_px.where(valid).ffill()。"""
    snap, dS = csi1000_snapshot()
    anchor = p.raw("close_px").where(p["valid"]).ffill().reindex(columns=p.codes)
    W = backout_close_weights(snap, dS, anchor)
    W.loc[W.index < pd.Timestamp(start)] = 0.0
    return W.astype(np.float32)


def real_index_on_signal_days(dates: pd.DatetimeIndex, mode: str = "open10") -> pd.Series:
    """真实 000852(新浪快照)对齐到引擎信号日:r[t] = close(t+1)/close(t) − 1,索引 dates[:T−1−lag]
    (与 index_proxy.index_returns_on_signal_days 逐字同式,只换数据源)。"""
    from afactor import strategy as ST
    lag = ST.EXEC_MODES[mode]["lag"]
    d = pd.read_csv(IDX_FILE)
    px = pd.Series(d["close"].to_numpy(np.float64), index=pd.to_datetime(d["date"])).sort_index()
    px = px.reindex(dates).ffill()
    r = px.shift(-1) / px - 1.0
    idx = dates[:len(dates) - 1 - lag]
    return r.reindex(idx).fillna(0.0).rename("000852")
