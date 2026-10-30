"""阶段4:时序模型 —— 动态仓位(进攻/防守)。

用户口径:做"指数动量过强降仓、回调后加仓"式的动态仓位;**不做固定现金比例**
(跌出价值时要敢上仓位)。决策变量:总仓位 + 核心/卫星配比。

⚠ 这是全项目最易过拟合的一层:只有 1455 个交易日、状态切换一年才几次,
独立样本比截面模型少几个数量级。故:状态数 <=3、参数个位数、必须 walk-forward、
与"满仓"对照。报告中单独标注其可信度。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from afactor.panel import Panel


def market_features(p: Panel, index_code: str = "000905") -> pd.DataFrame:
    """指数与全市场状态特征(全部只用 T 日及之前的信息)。

    指数用 ETF 项目冻结的宽基日线(默认中证500,与我们全A等权池最接近);
    另叠加自建的全市场等权指数与市场广度(截面上涨占比)。
    """
    from afactor import newdata as ND
    idx = ND.index_daily()
    s = None
    if index_code in idx.columns:
        raw = idx[index_code]
        last = raw.dropna().index.max()
        stale = p.dates[p.dates > last]
        if len(stale):
            # 静默 ffill 会让 mom5/mom20 僵化为 0、dd60 冻结,规则仓位退化成常数。
            raise ValueError(
                f"指数快照 {index_code} 只到 {last.date()},但面板需要到 "
                f"{p.dates[-1].date()}(缺 {len(stale)} 天)。请先刷新指数快照,"
                f"或显式传 index_code=None 改用自建等权指数。")
        s = raw.reindex(p.dates).ffill()
    r = p["ret_c2c"].where(p["universe"])
    mkt = r.mean(axis=1)                      # 自建全A等权日收益
    eq = (1 + mkt.fillna(0)).cumprod()        # 自建等权指数
    px = s if s is not None else eq
    out = pd.DataFrame(index=p.dates)
    for w in (5, 20, 60, 120):
        out[f"mom{w}"] = px / px.shift(w) - 1.0
    out["dd"] = px / px.cummax() - 1.0                        # 距历史高点的回撤
    out["dd60"] = px / px.rolling(60, min_periods=20).max() - 1.0
    out["vol20"] = mkt.rolling(20, min_periods=10).std() * np.sqrt(242)
    out["vol60"] = mkt.rolling(60, min_periods=30).std() * np.sqrt(242)
    out["vol_ratio"] = out["vol20"] / out["vol60"].replace(0, np.nan)
    out["breadth"] = (r > 0).sum(axis=1) / r.notna().sum(axis=1).replace(0, np.nan)
    out["breadth20"] = out["breadth"].rolling(20, min_periods=10).mean()
    out["disp"] = r.std(axis=1)                                # 截面分化度
    out["disp20"] = out["disp"].rolling(20, min_periods=10).mean()
    out["eq_mom20"] = eq / eq.shift(20) - 1.0                  # 自建指数动量(与宽基互补)
    return out


def rule_position(feat: pd.DataFrame, mom_hi: float = 0.15, dd_lo: float = -0.12,
                  lo: float = 0.5, hi: float = 1.0, tail: bool = False) -> pd.Series:
    """规则版动态仓位(可解释基线,3 个阈值参数)。

    动量过强 -> 降仓(过热);回调够深 -> 加仓(跌出价值);其余持中。
    参数少、方向由先验决定,不做网格搜索择优。

    对齐(2026-08-22 审查修正):策略日收益按**信号日 t** 索引,主口径下 t 日
    收盘后决策,feat[t] 用的正是 t 日收盘信息 -> pos[t]=f(feat[t]),**不移位**。
    此前无条件 shift(1) 多滞后一天:2024 年化 -7.45%->-3.92%、2025 差 2.36pp,
    单是这个对齐动作的摆幅就大于择时本身的净价值。
    尾盘口径 14:50 决策时当日全日指数尚不可得,必须 shift(1)。
    """
    mom = feat["mom60"]
    dd = feat["dd60"]
    pos = pd.Series(0.75, index=feat.index)
    pos[mom > mom_hi] = lo                    # 60 日涨幅过大 -> 防守
    pos[dd < dd_lo] = hi                      # 回撤够深 -> 进攻
    pos[(mom > mom_hi) & (dd < dd_lo)] = 0.75  # 冲突时持中
    return (pos.shift(1) if tail else pos).fillna(0.75)


def forward_market(p: Panel, horizon: int = 20) -> pd.Series:
    """未来 horizon 日的全A等权累计收益(fit_position_model 的标签)。"""
    r = p["ret_c2c"].where(p["universe"]).mean(axis=1).fillna(0.0)
    eq = (1 + r).cumprod()
    return eq.shift(-horizon) / eq - 1.0


def fit_position_model(feat: pd.DataFrame, fwd: pd.Series, train_mask: np.ndarray,
                       horizon: int = 20, kind: str = "logit"):
    """小模型版:用状态特征预测未来 horizon 日市场方向,输出仓位。

    两处修正:标签跨 horizon 天,训练窗末尾必须 purge 掉 horizon 天,否则直接
    看到测试段开头;LogisticRegression 带 L2 惩罚,特征量纲差一个数量级
    (mom 是小数、vol 是年化)必须先标准化,否则惩罚项事实上只作用于小量纲特征。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = feat.to_numpy(np.float64)
    y = (fwd > 0).astype(int).to_numpy()
    ok = np.isfinite(X).all(axis=1) & np.isfinite(fwd.to_numpy())
    tr = train_mask & ok
    idx = np.flatnonzero(tr)
    if len(idx) > horizon:
        tr = tr.copy(); tr[idx[-horizon:]] = False      # purge
    if tr.sum() < 200:
        return None
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    m.fit(X[tr], y[tr])
    return m


def model_position(m, feat: pd.DataFrame, lo: float = 0.4, hi: float = 1.0,
                   tail: bool = False) -> pd.Series:
    if m is None:
        return pd.Series(1.0, index=feat.index)
    X = feat.to_numpy(np.float64)
    ok = np.isfinite(X).all(axis=1)
    pos = pd.Series(np.nan, index=feat.index)
    if ok.sum():
        prob = m.predict_proba(X[ok])[:, 1]
        pos.iloc[np.flatnonzero(ok)] = lo + (hi - lo) * np.clip(prob, 0, 1)
    pos = pos.ffill()                       # 特征缺失 -> 维持上一状态,不猜 0
    return (pos.shift(1) if tail else pos).fillna(0.75)


def apply_position(nav_ret: pd.Series, pos: pd.Series, cash_ret: float = 0.0,
                   switch_bps: float = 1.0) -> pd.Series:
    """把仓位应用到策略日收益上。

    cash_ret : 空仓部分的日利率(默认 0 = 保守,实际货基约 48bp/年)
    switch_bps: 仓位切换的单向成本(|Δp| * bps),此前完全未计
    """
    p = pos.reindex(nav_ret.index).fillna(1.0)
    switch = p.diff().abs().fillna(0.0) * switch_bps * 1e-4
    return nav_ret * p + (1 - p) * cash_ret - switch
