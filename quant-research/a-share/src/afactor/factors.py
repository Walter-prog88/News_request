"""因子库:经典逻辑族 + 机械算子网格。

每个因子是 (name, family, fn, note),fn(panel) -> DataFrame [date x code]。
约定:因子值高 <-> 预期未来收益高(方向已在定义处调整),但检验不依赖该约定。
禁止使用绝对价格水平(前复权含未来分红信息),只用价格比值。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from afactor import ops
from afactor.panel import Panel

REGISTRY: dict[str, tuple[str, callable, str]] = {}


def add(name: str, family: str, fn, note: str = ""):
    if name in REGISTRY:
        raise KeyError(f"因子重名: {name}")
    REGISTRY[name] = (family, fn, note)


# ================= 通用构件 =================
def _ret(p): return p["ret_c2c"]
def _on(p): return p["overnight"]
def _turn(p): return p["turnover_proxy"]


def _rolling_split_by(value: pd.DataFrame, sorter: pd.DataFrame, w: int, high_minus_low=True):
    """滚动窗口内按 sorter 排序,取高半边 value 之和减低半边之和(兴业"理想反转/理想振幅"口径)。"""
    V = value.to_numpy(np.float32); S = sorter.to_numpy(np.float32)
    T, N = V.shape
    out = np.full((T, N), np.nan, np.float32)
    half = w // 2
    for t in range(w - 1, T):
        v = V[t - w + 1:t + 1]; s = S[t - w + 1:t + 1]
        ok = np.isfinite(v) & np.isfinite(s)
        if ok.sum() == 0:
            continue
        s2 = np.where(ok, s, -np.inf)
        order = np.argsort(-s2, axis=0)
        vs = np.take_along_axis(np.where(ok, v, np.nan), order, axis=0)
        hi = np.nanmean(vs[:half], axis=0)
        lo = np.nanmean(vs[half:], axis=0)
        good = ok.sum(axis=0) >= w * 0.6
        res = (hi - lo) if high_minus_low else (lo - hi)
        out[t] = np.where(good, res, np.nan)
    return pd.DataFrame(out, index=value.index, columns=value.columns)


def _ivol(p, w):
    """相对等权市场的特质波动率(无风格因子数据时的最简版:单因子市场模型残差波动)。"""
    r = _ret(p)
    mkt = r.where(p["universe"]).mean(axis=1)
    mktF = pd.DataFrame(np.repeat(mkt.to_numpy(np.float32)[:, None], r.shape[1], 1),
                        index=r.index, columns=r.columns)
    var_m = mkt.rolling(w, min_periods=max(5, w // 2)).var()
    beta = ops.ts_cov(r, mktF, w).div(var_m.replace(0, np.nan), axis=0)
    resid = r - beta.mul(mkt, axis=0)
    return ops.ts_std(resid, w)


# ================= 1. 动量 / 反转 =================
def _build_momentum():
    W_REV = [1, 2, 3, 5, 10, 20, 40, 60, 120]
    for w in W_REV:
        add(f"rev_{w}", "momentum",
            lambda p, w=w: -ops.ts_pct(p.raw("close_px"), w),
            f"{w}日反转")
    for w, s in [(20, 1), (40, 5), (60, 5), (120, 20), (250, 20)]:
        add(f"mom_{w}s{s}", "momentum",
            lambda p, w=w, s=s: ops.ts_delay(p.raw("close_px"), s) /
            ops.ts_delay(p.raw("close_px"), w) - 1,
            f"{w}日动量跳过最近{s}日")
    for w in [5, 10, 20, 60]:
        add(f"mom_overnight_{w}", "momentum",
            lambda p, w=w: ops.ts_sum(_on(p), w), f"{w}日隔夜动量")
        add(f"mom_intraday_{w}", "momentum",
            lambda p, w=w: ops.ts_sum(p.raw("ret_intraday"), w), f"{w}日日内动量")
        add(f"on_minus_id_{w}", "momentum",
            lambda p, w=w: ops.ts_sum(_on(p), w) - ops.ts_sum(p.raw("ret_intraday"), w),
            f"{w}日隔夜减日内")
    for w in [20, 60, 120]:
        add(f"fip_{w}", "momentum",
            lambda p, w=w: np.sign(ops.ts_pct(p.raw("close_px"), w)) *
            (ops.ts_mean((_ret(p) < 0).astype(float), w) - ops.ts_mean((_ret(p) > 0).astype(float), w)),
            f"{w}日信息离散度FIP(路径依赖动量)")
        add(f"ret_skew_{w}", "momentum", lambda p, w=w: -ops.ts_skew(_ret(p), w), f"{w}日收益偏度(负向)")
        add(f"ret_kurt_{w}", "momentum", lambda p, w=w: -ops.ts_kurt(_ret(p), w), f"{w}日收益峰度(负向)")
    for w in [10, 20, 40]:
        add(f"decay_rev_{w}", "momentum",
            lambda p, w=w: -ops.ts_decay(_ret(p), w), f"{w}日时间加权反转")
    for w in [20, 40]:
        add(f"ideal_rev_{w}", "momentum",
            lambda p, w=w: -_rolling_split_by(_ret(p), p.raw("amount"), w),
            f"理想反转{w}:高成交额日收益减低成交额日收益(负向)")
        add(f"ideal_rev_turn_{w}", "momentum",
            lambda p, w=w: -_rolling_split_by(_ret(p), _turn(p), w),
            f"理想反转{w}(按换手代理切割)")
        add(f"ideal_rng_{w}", "momentum",
            lambda p, w=w: -_rolling_split_by(_ret(p), p.raw("rng"), w),
            f"理想振幅{w}:高振幅日收益减低振幅日收益(负向)")
    for w in [5, 20, 60]:
        add(f"max_ret_{w}", "momentum", lambda p, w=w: -ops.ts_max(_ret(p), w), f"{w}日最大单日收益(彩票偏好负向)")
        add(f"min_ret_{w}", "momentum", lambda p, w=w: ops.ts_min(_ret(p), w), f"{w}日最小单日收益")
        add(f"rev_voladj_{w}", "momentum",
            lambda p, w=w: -ops.safe_div(ops.ts_pct(p.raw("close_px"), w), ops.ts_std(_ret(p), w)),
            f"{w}日波动调整反转")
        add(f"absret_{w}", "momentum", lambda p, w=w: -ops.ts_mean(_ret(p).abs(), w), f"{w}日平均绝对收益(负向)")
    for w in [120, 250]:
        add(f"near_high_{w}", "momentum",
            lambda p, w=w: ops.safe_div(p.raw("close_px"), ops.ts_max(p.raw("close_px"), w)),
            f"距{w}日高点比值")
        add(f"near_low_{w}", "momentum",
            lambda p, w=w: -ops.safe_div(p.raw("close_px"), ops.ts_min(p.raw("close_px"), w)),
            f"距{w}日低点比值(负向)")


# ================= 2. 波动率 / 尾部 =================
def _build_volatility():
    for w in [5, 10, 20, 60]:
        add(f"rvol_{w}", "volatility", lambda p, w=w: -ops.ts_mean(p.raw("rvol"), w), f"{w}日已实现波动(负向)")
        add(f"rsj_{w}", "volatility", lambda p, w=w: -ops.ts_mean(p.raw("rsj"), w), f"{w}日已实现符号跳跃(负向)")
        add(f"rskew_{w}", "volatility", lambda p, w=w: -ops.ts_mean(p.raw("rskew"), w), f"{w}日已实现偏度(负向)")
    for w in [5, 20, 60, 120]:
        add(f"dvol_{w}", "volatility", lambda p, w=w: -ops.ts_std(_ret(p), w), f"{w}日日收益波动(负向)")
    for w in [20, 60, 120]:
        add(f"ivol_{w}", "volatility", lambda p, w=w: -_ivol(p, w), f"{w}日特质波动率(负向)")
    for w in [5, 20, 60]:
        add(f"rkurt_{w}", "volatility", lambda p, w=w: -ops.ts_mean(p.raw("rkurt"), w), f"{w}日已实现峰度(负向)")
        add(f"rvdown_ratio_{w}", "volatility",
            lambda p, w=w: -ops.ts_mean(ops.safe_div(p.raw("rv_down"), p.raw("rv_up")), w),
            f"{w}日下行/上行方差比(负向)")
        add(f"jump_ratio_{w}", "volatility",
            lambda p, w=w: -ops.ts_mean(ops.safe_div(p.raw("jump"), p.raw("rv")), w), f"{w}日跳跃占比(负向)")
        add(f"volofvol_{w}", "volatility", lambda p, w=w: -ops.ts_std(p.raw("rvol"), w), f"{w}日波动率的波动(负向)")
        add(f"rng_{w}", "volatility", lambda p, w=w: -ops.ts_mean(p.raw("rng"), w), f"{w}日日内振幅(负向)")
        add(f"downside_vol_{w}", "volatility", lambda p, w=w: -ops.ts_downside_std(_ret(p), w), f"{w}日下行波动(负向)")
        add(f"vol_change_{w}", "volatility",
            lambda p, w=w: -ops.safe_div(ops.ts_mean(p.raw("rvol"), 5), ops.ts_mean(p.raw("rvol"), w)),
            f"波动率{w}日趋势(负向)")
        add(f"onvol_{w}", "volatility", lambda p, w=w: -ops.ts_std(_on(p), w), f"{w}日隔夜波动(负向)")
        add(f"bpv_ratio_{w}", "volatility",
            lambda p, w=w: -ops.ts_mean(ops.safe_div(p.raw("bpv"), p.raw("rv")), w), f"{w}日连续波动占比(负向)")


# ================= 3. 成交量 / 流动性 =================
def _build_volume():
    for w in [5, 10, 20, 60]:
        add(f"turn_{w}", "volume", lambda p, w=w: -ops.ts_mean(_turn(p), w), f"{w}日换手代理(负向)")
        add(f"amihud_{w}", "volume", lambda p, w=w: ops.ts_mean(p.raw("amihud"), w), f"{w}日Amihud非流动性")
        add(f"amt_z_{w}", "volume", lambda p, w=w: -ops.ts_zscore(p.raw("amount"), w), f"{w}日异常放量(负向)")
        add(f"pv_corr_{w}", "volume",
            lambda p, w=w: -ops.ts_corr(_ret(p), ops.ts_pct(p.raw("amount"), 1), w), f"{w}日量价相关(负向)")
    for w in [5, 20, 60]:
        add(f"turn_std_{w}", "volume", lambda p, w=w: -ops.ts_std(_turn(p), w), f"{w}日换手波动(负向)")
        add(f"turn_trend_{w}", "volume",
            lambda p, w=w: -ops.safe_div(ops.ts_mean(_turn(p), 5), ops.ts_mean(_turn(p), w)), f"换手{w}日趋势(负向)")
        add(f"amt_vol_{w}", "volume",
            lambda p, w=w: -ops.ts_std(ops.signed_log(p.raw("amount")), w), f"{w}日成交额波动(负向)")
        add(f"zero_min_{w}", "volume", lambda p, w=w: -ops.ts_mean(p.raw("zero_min_ratio"), w), f"{w}日零成交分钟占比(负向)")
        add(f"ntrade_{w}", "volume", lambda p, w=w: ops.ts_mean(p.raw("n_trade_min"), w), f"{w}日活跃分钟数")
        add(f"pv_corr_lvl_{w}", "volume",
            lambda p, w=w: -ops.ts_corr(ops.ts_pct(p.raw("close_px"), 1), p.raw("amount"), w), f"{w}日价量水平相关(负向)")
        add(f"amihud_trend_{w}", "volume",
            lambda p, w=w: ops.safe_div(ops.ts_mean(p.raw("amihud"), 5), ops.ts_mean(p.raw("amihud"), w)),
            f"非流动性{w}日趋势")
        add(f"turn_skew_{w}", "volume", lambda p, w=w: -ops.ts_skew(_turn(p), w), f"{w}日换手偏度(负向)")


# ================= 4. 日内结构 =================
_INTRADAY_MATERIALS = [
    ("smart_money", -1, "聪明钱因子Q"),
    ("ret_last30", -1, "尾盘30分钟收益"),
    ("ret_last10", -1, "尾盘10分钟收益"),
    ("ret_open30", -1, "开盘30分钟收益"),
    ("ret_closeauc", -1, "收盘集合竞价收益"),
    ("ret_intraday", -1, "日内收益"),
    ("ret_morning", -1, "上午收益"),
    ("ret_afternoon", -1, "下午收益"),
    ("ret_mid", -1, "盘中收益"),
    ("amt_last30_share", +1, "尾盘30分钟成交占比"),
    ("amt_last10_share", +1, "尾盘10分钟成交占比"),
    ("amt_first30_share", -1, "开盘30分钟成交占比"),
    ("amt_morning_share", -1, "上午成交占比"),
    ("amt_openauc_share", -1, "开盘集合竞价成交占比"),
    ("amt_closeauc_share", +1, "收盘集合竞价成交占比"),
    ("amt_hhi", -1, "成交额集中度"),
    ("amt_gini", -1, "成交额基尼"),
    ("amt_time_wgt", +1, "成交额时间重心"),
    ("amt_top5_share", -1, "前5大分钟成交占比"),
    ("amt_cv", -1, "成交额变异系数"),
    ("amt_skew", -1, "成交额偏度"),
    ("corr_r_v", -1, "日内量价相关"),
    ("corr_absr_v", -1, "日内波动-量相关"),
    ("up_amt_share", -1, "上涨分钟成交占比"),
    ("up_min_ratio", -1, "上涨分钟占比"),
    ("vwap_dev", -1, "收盘相对VWAP偏离"),
    ("trend_eff", -1, "日内趋势效率"),
    ("t_max_px", +1, "日内最高价时点"),
    ("t_min_px", -1, "日内最低价时点"),
    ("t_max_amt", +1, "最大成交分钟时点"),
    ("r_ac1", -1, "分钟收益一阶自相关"),
    ("v_ac1", -1, "分钟成交额自相关"),
    ("bigmove_amt_share", -1, "大波动分钟成交占比"),
    ("gap_share", -1, "分钟跳空占比"),
    ("body_ratio", -1, "分钟实体占比"),
    ("max_abs_r", -1, "最大绝对分钟收益"),
    ("path_len", -1, "日内路径长度"),
]


def _build_intraday():
    for mat, sgn, desc in _INTRADAY_MATERIALS:
        for w in [5, 20, 60]:
            add(f"id_{mat}_m{w}", "intraday",
                lambda p, m=mat, w=w, s=sgn: s * ops.ts_mean(p.raw(m), w), f"{desc} {w}日均值")
    for k in range(8):
        for w in [5, 20]:
            add(f"seg{k}_ret_m{w}", "intraday",
                lambda p, k=k, w=w: -ops.ts_mean(p.raw(f"seg{k}_ret"), w), f"日内第{k}段收益{w}日均值(负向)")
            add(f"seg{k}_amt_m{w}", "intraday",
                lambda p, k=k, w=w: -ops.ts_mean(p.raw(f"seg{k}_amt"), w), f"日内第{k}段成交占比{w}日均值(负向)")
    # 时段差分结构
    for w in [5, 20]:
        add(f"tail_minus_open_{w}", "intraday",
            lambda p, w=w: ops.ts_mean(p.raw("ret_last30") - p.raw("ret_open30"), w), f"尾盘减开盘收益{w}日")
        add(f"pm_minus_am_{w}", "intraday",
            lambda p, w=w: ops.ts_mean(p.raw("ret_afternoon") - p.raw("ret_morning"), w), f"下午减上午收益{w}日")
        add(f"amt_tail_minus_open_{w}", "intraday",
            lambda p, w=w: ops.ts_mean(p.raw("amt_last30_share") - p.raw("amt_first30_share"), w),
            f"尾盘减开盘成交占比{w}日")
        add(f"smart_x_turn_{w}", "intraday",
            lambda p, w=w: -ops.cs_rank(ops.ts_mean(p.raw("smart_money"), w)) * ops.cs_rank(ops.ts_mean(_turn(p), w)),
            f"聪明钱×换手{w}日")


# ================= 5. 机械算子网格 =================
_MECH_MATERIALS = [
    "rvol", "rv", "rv_down", "rv_up", "rsj", "rskew", "rkurt", "jump", "bpv", "rng",
    "max_abs_r", "path_len", "trend_eff", "amihud", "amount", "n_trade_min", "zero_min_ratio",
    "amt_hhi", "amt_gini", "amt_cv", "amt_skew", "amt_top5_share", "amt_time_wgt",
    "amt_first30_share", "amt_last30_share", "amt_last10_share", "amt_morning_share",
    "amt_openauc_share", "amt_closeauc_share", "corr_r_v", "corr_absr_v", "corr_body_r",
    "up_amt_share", "up_min_ratio", "vwap_dev", "smart_money", "t_max_px", "t_min_px",
    "t_max_amt", "r_ac1", "v_ac1", "bigmove_amt_share", "gap_share", "body_ratio",
    "ret_intraday", "ret_open30", "ret_last30", "ret_last10", "ret_morning", "ret_afternoon",
    "ret_closeauc", "ret_mid",
]

_MECH_OPS = [
    ("std", lambda x, w: ops.ts_std(x, w)),
    ("z", lambda x, w: ops.ts_zscore(x, w)),
    ("d", lambda x, w: ops.ts_delta(x, w)),
    ("trend", lambda x, w: ops.safe_div(ops.ts_mean(x, 5), ops.ts_mean(x, w))),
    ("tsrank", lambda x, w: ops.ts_rank(x, w)),
]


def _build_mechanical():
    for mat in _MECH_MATERIALS:
        for opn, fn in _MECH_OPS:
            for w in [10, 20, 60]:
                if opn == "trend" and w == 10:
                    continue
                add(f"mx_{mat}_{opn}{w}", "mechanical",
                    lambda p, m=mat, f=fn, w=w: f(p.raw(m), w),
                    f"机械网格 {opn}({mat},{w})")


# ================= 6. 交互项 =================
_PAIRS = [
    ("rev_20", "turn_20"), ("rev_20", "rvol_20"), ("rev_5", "turn_5"),
    ("rev_20", "amihud_20"), ("mom_60s5", "rvol_20"), ("mom_60s5", "turn_20"),
    ("id_smart_money_m20", "turn_20"), ("id_smart_money_m20", "rev_20"),
    ("id_amt_last30_share_m20", "rev_20"), ("id_corr_r_v_m20", "rev_20"),
    ("rvol_20", "turn_20"), ("id_vwap_dev_m20", "rev_20"),
    ("rskew_20", "turn_20"), ("id_up_amt_share_m20", "rev_20"),
    ("max_ret_20", "turn_20"), ("ideal_rev_20", "rvol_20"),
]


def _build_interactions():
    for a, b in _PAIRS:
        if a not in REGISTRY or b not in REGISTRY:
            continue
        fa, fb = REGISTRY[a][1], REGISTRY[b][1]
        add(f"ix_{a}__X__{b}", "interaction",
            lambda p, fa=fa, fb=fb: ops.cs_rank(fa(p)) * ops.cs_rank(fb(p)), f"{a} × {b}")
        add(f"ix_{a}__P__{b}", "interaction",
            lambda p, fa=fa, fb=fb: ops.cs_rank(fa(p)) + ops.cs_rank(fb(p)), f"{a} + {b}")
        add(f"ix_{a}__C__{b}", "interaction",
            lambda p, fa=fa, fb=fb: ops.cs_neutralize(fa(p), fb(p)), f"{a} 对 {b} 正交化")


# ================= 7. 市场互动族(实验一,需先跑 build_market + merge_market) =================
_MKT_BASES = [
    ("mkt_beta", "日内市场beta"), ("mkt_corr", "日内与市场相关"),
    ("semibeta_n", "半贝塔:同跌共动"), ("semibeta_p", "半贝塔:同涨共动"),
    ("semibeta_mn", "半贝塔:市场跌我涨"), ("semibeta_mp", "半贝塔:市场涨我跌"),
    ("mkt_lead", "跟随市场强度"), ("mkt_lag", "领先市场强度"),
    ("idio_share", "特质波动占比"), ("idio_vol", "日内特质波动"),
    ("cojump", "与市场共同极端波动"), ("ex_ret_open30", "开盘30分超额"),
    ("ex_ret_pm", "下午超额"), ("ex_ret_last30", "尾盘30分超额"), ("ex_ret_day", "全日日内超额"),
]


def _build_market_family():
    from afactor.config import DAILY
    if not (DAILY / "mkt_beta.parquet").exists():
        return
    for b, desc in _MKT_BASES:
        for w in [5, 20, 60]:
            add(f"mi_{b}_m{w}", "market",
                lambda p, b=b, w=w: ops.ts_mean(p.raw(b), w), f"{desc} {w}日均值")
        for w in [20, 60]:
            add(f"mi_{b}_s{w}", "market",
                lambda p, b=b, w=w: ops.ts_std(p.raw(b), w), f"{desc} {w}日波动")
    for w in [5, 20, 60]:
        add(f"mi_sb_asym_m{w}", "market",
            lambda p, w=w: ops.ts_mean(p.raw("semibeta_n") - p.raw("semibeta_p"), w),
            f"半贝塔不对称(跌时共动-涨时共动){w}日")
        add(f"mi_leadlag_m{w}", "market",
            lambda p, w=w: ops.ts_mean(p.raw("mkt_lag") - p.raw("mkt_lead"), w),
            f"领先减跟随 {w}日")


def _apm_stat(p, w):
    """APM 代理:隔夜相对截面 vs 下午相对截面的持续差异(开源证券 APM 思路)。"""
    u = p["universe"]
    e_on = ops.cs_demean(p["overnight"].where(u))
    e_pm = ops.cs_demean(p.raw("ret_afternoon").where(u))
    diff = e_on - e_pm
    return ops.safe_div(ops.ts_mean(diff, w), ops.ts_std(diff, w)) * np.sqrt(w)


def _build_apm():
    for w in [20, 40]:
        add(f"apm_stat_w{w}", "market", lambda p, w=w: _apm_stat(p, w), f"APM隔夜-下午代理 {w}日")
        add(f"apm_pure_w{w}", "market",
            lambda p, w=w: ops.cs_neutralize_1d(_apm_stat(p, w), ops.ts_pct(p.raw("close_px"), 20)),
            f"APM对20日动量正交 {w}日")


# ================= 8. 量价结合:条件切割 + 跨维度交互(实验六) =================
_COND_VALUES = [
    ("ret_c2c", lambda p: p["ret_c2c"]), ("ret_intraday", lambda p: p.raw("ret_intraday")),
    ("overnight", lambda p: p["overnight"]), ("ret_last30", lambda p: p.raw("ret_last30")),
    ("smart_money", lambda p: p.raw("smart_money")), ("vwap_dev", lambda p: p.raw("vwap_dev")),
    ("rsj", lambda p: p.raw("rsj")),
]
_COND_CONDS = [
    ("amt", lambda p: p.raw("amount")), ("turn", lambda p: p["turnover_proxy"]),
    ("rvol", lambda p: p.raw("rvol")), ("tailamt", lambda p: p.raw("amt_last10_share")),
    ("amtz", lambda p: ops.ts_zscore(p.raw("amount"), 20)),
]


def _build_condvp():
    for vn, vf in _COND_VALUES:
        for cn, cf in _COND_CONDS:
            for w in [20, 40]:
                add(f"cv_{vn}_by_{cn}_{w}", "condvp",
                    lambda p, vf=vf, cf=cf, w=w: _rolling_split_by(vf(p), cf(p), w),
                    f"{vn} 在高{cn}日与低{cn}日的差({w}日)")
    price_reps = ["rev_20", "max_ret_5", "ivol_20", "rskew_20", "mom_overnight_20"]
    vol_reps = ["turn_20", "amt_vol_20", "amt_z_20", "amihud_20", "id_amt_last30_share_m20"]
    for a in price_reps:
        for b in vol_reps:
            if a not in REGISTRY or b not in REGISTRY:
                continue
            fa, fb = REGISTRY[a][1], REGISTRY[b][1]
            add(f"cx_{a}__X__{b}", "condvp",
                lambda p, fa=fa, fb=fb: ops.cs_rank(fa(p)) * ops.cs_rank(fb(p)), f"{a}×{b}")
            add(f"cx_{a}__P__{b}", "condvp",
                lambda p, fa=fa, fb=fb: ops.cs_rank(fa(p)) + ops.cs_rank(fb(p)), f"{a}+{b}")


# ================= 9. 统计行业族(实验二,需先跑 statind.build_labels) =================
def _build_statind():
    from afactor import statind as SI
    if not SI.LABEL_FILE.exists():
        return

    def _L(p):
        if "statind_L" not in p._derived:
            p._derived["statind_L"] = SI.load_labels().reindex(index=p.dates, columns=p.codes)
        return p._derived["statind_L"]

    for w in [5, 20, 60]:
        add(f"si_ind_mom_{w}", "statind",
            lambda p, w=w: SI.label_mean(ops.ts_pct(p.raw("close_px"), w), _L(p)),
            f"统计行业动量({w}日,组均收益)")
        add(f"si_rev_within_{w}", "statind",
            lambda p, w=w: -SI.label_demean(ops.ts_pct(p.raw("close_px"), w), _L(p)),
            f"统计行业内反转({w}日)")
    add("si_ind_disp_20", "statind",
        lambda p: SI.label_std(ops.ts_pct(p.raw("close_px"), 20), _L(p)), "组内20日收益分散度")
    add("si_amt_within_20", "statind",
        lambda p: -SI.label_demean(ops.ts_zscore(p.raw("amount"), 20), _L(p)), "组内异常放量(负)")
    for base in ["ivol_20", "amt_vol_20", "id_ret_intraday_m60", "id_v_ac1_m5"]:
        if base in REGISTRY:
            fn0 = REGISTRY[base][1]
            add(f"si_gd_{base}", "statind",
                lambda p, fn0=fn0: SI.label_demean(fn0(p), _L(p)), f"{base} 的统计行业内中性版")



# ================= 10. 基本面族(需 data_8:21 财务三表) =================
def _cut_tag() -> str:
    """面板缓存键的截止日后缀:默认截止日为空串(键与旧版完全相同),
    set_data_cutoff() 放宽/收紧后为 "@YYYYMMDD",避免同一个面板对象串用两个截止日的结果。"""
    from afactor.config import DATA_CUTOFF, data_cutoff
    cut = data_cutoff()
    return "" if cut == pd.Timestamp(DATA_CUTOFF) else f"@{cut:%Y%m%d}"


def _fu(p, col):
    """按需构建并缓存基本面日频面板(PIT)。

    2026-08-22 修正(P0-3):_fu 绕开 Panel.raw,TailPanel 的"全日字段自动 shift(1)"
    对基本面失效,导致 T 日晚间披露的财报在 T 日 14:50 就进入因子值。
    事件研究实测:尾盘口径 T→T+1 的 spearman +0.0297、五分组多空 +0.317%/日
    (主口径 open 窗口 −0.0004,p=0.91,不受影响)。
    修法:TailPanel 下把生效可见日整体推迟 1 个自然日(T 日公告 T+1 起生效)。
    """
    lag = 1 if getattr(p, "_is_tail", False) else 0
    tag = _cut_tag()
    key = f"__fu__{col}__{lag}{tag}"
    if key not in p._derived:
        from afactor import fundamental as FU
        rep = p._derived.get("__fu_rep__" + tag)
        if rep is None:
            rep = FU.build_report_level()
            p._derived["__fu_rep__" + tag] = rep
        p._derived[key] = FU._pit_daily(rep, col, p.dates, p.codes, lag_days=lag)
    return p._derived[key]


def _fu_event(p, col):
    """财报公告事件量(ear_car3 / ear_gap / days_since_ann)的日频面板。

    生效可见日:ear_car3 用 d0+2(公告后次日收盘已发生),ear_gap 用 d0+1;
    尾盘口径再整体推迟 1 日,与 _fu 保持一致。
    """
    lag = 1 if getattr(p, "_is_tail", False) else 0
    tag = _cut_tag()
    key = f"__ev__{col}__{lag}{tag}"
    if key in p._derived:
        return p._derived[key]
    from afactor import fundamental as FU
    ev = p._derived.get("__ev_raw__" + tag)
    if ev is None:
        ev = FU.build_event_features(p.dates, p.codes, p["ret_c2c"],
                                     p.raw("open_auction_px"), p.raw("close_px"))
        p._derived["__ev_raw__" + tag] = ev
    out = pd.DataFrame(np.nan, index=p.dates, columns=p.codes, dtype=np.float32)
    if col == "days_since_ann":
        d0 = ev.pivot_table(index="d0", columns="stock_code", values="ear_gap", aggfunc="last")
        mark = d0.notna().reindex(index=p.dates, columns=p.codes).fillna(False)
        cnt = (~mark).cumsum() - (~mark).cumsum().where(mark).ffill().fillna(0)
        out = cnt.where(mark.cumsum() > 0).astype(np.float32)
    else:
        eff = "eff_car" if col == "ear_car3" else "eff_gap"
        d = ev[["stock_code", eff, col]].dropna()
        d = d.rename(columns={eff: "eff"}).sort_values(["stock_code", "eff"])
        d = d.drop_duplicates(["stock_code", "eff"], keep="last")
        piv = d.pivot_table(index="eff", columns="stock_code", values=col, aggfunc="last")
        piv = piv.reindex(columns=p.codes)
        if lag:
            piv.index = piv.index + pd.Timedelta(days=1)
        out = piv.reindex(piv.index.union(p.dates)).ffill().reindex(p.dates).astype(np.float32)
    p._derived[key] = out
    return out


def _nonfin(p):
    """非金融股掩码:金融股(银行/保险/证券)的毛利率恒为 1、周转率恒在底部,
    这类因子在它们身上退化成行业哑变量(P1-5)。"""
    key = "__nonfin__" + _cut_tag()
    if key not in p._derived:
        from afactor import fundamental as FU
        p._derived[key] = FU.nonfinancial_mask(p.dates, p.codes)
    return p._derived[key]


def _safe_ratio(a, b):
    """比值:分母取绝对值并屏蔽近零,避免负分母翻转方向。"""
    return ops.safe_div(a, b.abs().where(b.abs() > 1e-6))


def _build_fundamental():
    from afactor.config import ROOT
    if not (ROOT / "data_8:21" / "stock_income_2015_now.pkl").exists():
        return
    cap = lambda p: p["market_cap"]

    # --- 估值(市值在分母)---
    for nm, col, desc in [("ep", "n_income_attr_p_ttm", "盈利市值比EP"),
                          ("sp", "revenue_ttm", "营收市值比SP"),
                          ("cfp", "n_cashflow_act_ttm", "经营现金流市值比CFP"),
                          ("opp", "operate_profit_ttm", "营业利润市值比")]:
        add(f"fu_{nm}", "fundamental",
            lambda p, c=col: _safe_ratio(_fu(p, c), cap(p)), f"{desc}(TTM/市值)")
    add("fu_bp", "fundamental",
        lambda p: _safe_ratio(_fu(p, "total_hldr_eqy_exc_min_int"), cap(p)), "账面市值比BP")

    # --- 盈利能力 ---
    add("fu_roe", "fundamental",
        lambda p: _safe_ratio(_fu(p, "n_income_attr_p_ttm"), _fu(p, "total_hldr_eqy_exc_min_int_avg")),
        "ROE(TTM归母净利/平均归母权益)")
    add("fu_roa", "fundamental",
        lambda p: _safe_ratio(_fu(p, "n_income_attr_p_ttm"), _fu(p, "total_assets_avg")), "ROA")
    add("fu_gross_margin", "fundamental",
        lambda p: _safe_ratio(_fu(p, "revenue_ttm") - _fu(p, "oper_cost_ttm"),
                              _fu(p, "revenue_ttm")).where(_nonfin(p)), "毛利率(仅非金融)")
    add("fu_net_margin", "fundamental",
        lambda p: _safe_ratio(_fu(p, "n_income_attr_p_ttm"),
                              _fu(p, "revenue_ttm")).where(_nonfin(p)), "净利率(仅非金融)")
    add("fu_op_margin", "fundamental",
        lambda p: _safe_ratio(_fu(p, "operate_profit_ttm"),
                              _fu(p, "revenue_ttm")).where(_nonfin(p)), "营业利润率(仅非金融)")

    # --- 成长 ---
    for nm, col, desc in [("rev_yoy", "revenue_yoy", "营收同比"),
                          ("profit_yoy", "n_income_attr_p_yoy", "归母净利同比"),
                          ("op_yoy", "operate_profit_yoy", "营业利润同比"),
                          ("asset_yoy", "total_assets_yoy", "总资产同比"),
                          ("equity_yoy", "total_hldr_eqy_exc_min_int_yoy", "净资产同比")]:
        add(f"fu_{nm}", "fundamental", lambda p, c=col: _fu(p, c), desc + "(对称增长率,值域±2)")
    # 成长加速度(同比的环比变化)
    for nm, col in [("rev_accel", "revenue_yoy"), ("profit_accel", "n_income_attr_p_yoy")]:
        add(f"fu_{nm}", "fundamental",
            lambda p, c=col: _fu(p, c) - ops.ts_delay(_fu(p, c), 60),
            f"{col} 的季度加速度(同比已为对称口径,值域 ±2)")

    # --- 质量 ---
    add("fu_accrual", "fundamental",
        lambda p: -_safe_ratio(_fu(p, "n_income_attr_p_ttm") - _fu(p, "n_cashflow_act_ttm"),
                               _fu(p, "total_assets_avg")), "应计比率(负向:应计高则差)")
    add("fu_cfo_ni", "fundamental",
        lambda p: _safe_ratio(_fu(p, "n_cashflow_act_ttm"),
                              _fu(p, "n_income_attr_p_ttm").where(_fu(p, "n_income_attr_p_ttm") > 0)),
        "经营现金流/净利润(仅盈利股;亏损股该比值测的是'净利是否接近零')")
    add("fu_leverage", "fundamental",
        lambda p: -_safe_ratio(_fu(p, "total_liab"),
                               _fu(p, "total_assets")).where(_nonfin(p)), "资产负债率(负向,仅非金融)")
    add("fu_asset_turn", "fundamental",
        lambda p: _safe_ratio(_fu(p, "revenue_ttm"),
                              _fu(p, "total_assets_avg")).where(_nonfin(p)), "总资产周转率(仅非金融)")
    add("fu_cash_ratio", "fundamental",
        lambda p: _safe_ratio(_fu(p, "money_cap"), _fu(p, "total_assets")), "货币资金占比")
    add("fu_goodwill", "fundamental",
        lambda p: -_safe_ratio(_fu(p, "goodwill"), _fu(p, "total_assets")), "商誉占比(负向)")

    # --- 规模本身(对照用,预期是最强的"马甲源")---
    add("fu_size", "fundamental", lambda p: -p["log_cap"], "对数市值(负向=小市值)")

    # --- 量价 × 基本面 条件切割(第二轮已证明"用状态切割价格行为"有效)---
    _FU_CONDS = [("ep", lambda p: _safe_ratio(_fu(p, "n_income_attr_p_ttm"), cap(p))),
                 ("roe", lambda p: _safe_ratio(_fu(p, "n_income_attr_p_ttm"),
                                               _fu(p, "total_hldr_eqy_exc_min_int_avg"))),
                 ("size", lambda p: p["log_cap"])]
    for vn, vf in [("ret_c2c", lambda p: p["ret_c2c"]),
                   ("ret_intraday", lambda p: p.raw("ret_intraday")),
                   ("rvol", lambda p: p.raw("rvol"))]:
        for cn, cf in _FU_CONDS:
            add(f"fx_{vn}_by_{cn}_20", "fundamental",
                lambda p, vf=vf, cf=cf: _rolling_split_by(vf(p), cf(p), 20),
                f"{vn} 在高{cn}与低{cn}股票间的差(20日)")
    # 基本面 × 量价 秩交互
    for a, b in [("fu_ep", "ivol_20"), ("fu_roe", "ivol_20"), ("fu_bp", "rev_20"),
                 ("fu_ep", "amt_vol_20"), ("fu_roe", "id_ret_intraday_m60"),
                 ("fu_accrual", "turn_20"), ("fu_rev_yoy", "ivol_20")]:
        if a in REGISTRY and b in REGISTRY:
            fa, fb = REGISTRY[a][1], REGISTRY[b][1]
            add(f"fx_{a}__P__{b}", "fundamental",
                lambda p, fa=fa, fb=fb: ops.cs_rank(fa(p)) + ops.cs_rank(fb(p)), f"{a}+{b}")


# ================= 13. 第四轮新数据族(真实市值/换手、盈余意外、龙虎榜) =================
def _nd(p, kind, key):
    """新数据面板的懒加载缓存。kind ∈ {surprise, toplist}。"""
    # P2/M-4 决策:事件族与财报族口径对齐,主口径 lag=1、尾盘 lag=1。
    # T 日盘后公布的信息在 T+1 开盘前与 T+1 14:50 都已是公开信息,
    # 故两个口径同用 lag=1(_next_trading_day 已用 side="right" 排除公告当天)。
    lag = 1
    ck = f"__nd__{kind}__{lag}{_cut_tag()}"
    if ck not in p._derived:
        from afactor import newdata as ND
        p._derived[ck] = (ND.surprise_panels(p.dates, p.codes, lag) if kind == "surprise"
                          else ND.toplist_panels(p.dates, p.codes, lag))
    return p._derived[ck][key]


def _real(p, key):
    """真实市值/换手率面板。

    2026-08-22 修正:这些是**当日收盘后**才知道的量,尾盘口径(T 日 14:50)下
    必须滞后一天,否则用到当日收盘数据 = 前视(与 P0-3 同类)。
    缓存键含 lag,防止修复后被旧缓存串用。
    """
    lag = 1 if getattr(p, "_is_tail", False) else 0
    ck = f"__real__{key}__{lag}{_cut_tag()}"
    if ck not in p._derived:
        from afactor import newdata as ND
        fn = {"mv": ND.total_mv, "turn": ND.turnover_rate,
              "circ": ND.circ_mv, "fs": ND.float_share}[key]
        v = fn(p.dates, p.codes)
        p._derived[ck] = v.shift(lag) if lag else v
    return p._derived[ck]


def _build_newdata():
    from afactor.config import ROOT
    if not (ROOT / "data_8:21" / "daily_basic.csv").exists():
        return

    # --- 真实换手率(与成交额代理秩相关仅 0.35,是未检验的独立维度)---
    for w in [1, 5, 20, 60]:
        add(f"rt_turn_{w}", "newdata",
            lambda p, w=w: -(ops.ts_mean(_real(p, "turn"), w) if w > 1 else _real(p, "turn")),
            f"真实换手率 {w}日均(负向)")
    for w in [5, 20, 60]:
        add(f"rt_turn_std_{w}", "newdata",
            lambda p, w=w: -ops.ts_std(_real(p, "turn"), w), f"真实换手率 {w}日波动(负向)")
    for w in [20, 60]:      # w=5 时分子分母相同,因子恒为 -1(L-3)
        add(f"rt_turn_trend_{w}", "newdata",
            lambda p, w=w: -ops.safe_div(ops.ts_mean(_real(p, "turn"), 5),
                                         ops.ts_mean(_real(p, "turn"), w)), f"换手率 {w}日趋势(负向)")
    add("rt_turn_z20", "newdata", lambda p: -ops.ts_zscore(_real(p, "turn"), 20), "换手率异常(负向)")
    # --- 真实市值 / 流通市值 ---
    add("rt_size", "newdata", lambda p: -ops.signed_log(_real(p, "mv")), "真实总市值(负向=小市值)")
    add("rt_size_circ", "newdata", lambda p: -ops.signed_log(_real(p, "circ")), "真实流通市值(负向)")
    add("rt_float_ratio", "newdata",
        lambda p: ops.safe_div(_real(p, "circ"), _real(p, "mv")).where(
            ops.safe_div(_real(p, "circ"), _real(p, "mv")) <= 1.0),
        "流通股本占比(注:circ_mv≡close×float_share,本因子实为日频流通股本/季频总股本之比;"
        "源数据无 free_share,不是自由流通度;A+H 股语义另论)")
    # --- 盈余意外三层 ---
    add("sp_fc_pchange", "newdata", lambda p: _nd(p, "surprise", "fc_pchange").clip(-300, 300),
        "业绩预告同比中值(公司自己的预测)")
    add("sp_ex_yoy_np", "newdata", lambda p: _nd(p, "surprise", "ex_yoy_np").clip(-300, 300),
        "业绩快报净利同比")
    add("sp_ex_yoy_sales", "newdata", lambda p: _nd(p, "surprise", "ex_yoy_sales").clip(-200, 200),
        "业绩快报营收同比")
    add("sp_ex_roe", "newdata", lambda p: _nd(p, "surprise", "ex_roe").clip(-50, 50), "业绩快报ROE")
    add("sp_ex_vs_fc", "newdata", lambda p: _nd(p, "surprise", "ex_vs_fc"),
        "快报相对预告的偏离(第二层意外)")
    for w in [20, 40, 60]:
        add(f"sp_fc_win{w}", "newdata",
            lambda p, w=w: _nd(p, "surprise", "fc_pchange").clip(-300, 300).where(
                _nd(p, "surprise", "days_since_surprise") <= w),
            f"预告意外仅在公告后 {w} 交易日内暴露")
    # --- 龙虎榜 ---
    add("tl_net_ratio", "newdata", lambda p: _nd(p, "toplist", "tl_net_ratio"), "龙虎榜净买入占比")
    add("tl_count20", "newdata", lambda p: -_nd(p, "toplist", "tl_count20"), "20日上榜次数(负向)")
    add("tl_net20", "newdata", lambda p: _nd(p, "toplist", "tl_net20"), "20日净买入占比累计")
    add("tl_updown20", "newdata", lambda p: -_nd(p, "toplist", "tl_updown20"),
        "20日涨幅榜减跌幅榜次数(负向)")
    add("tl_onlist_recent", "newdata",
        lambda p: -_nd(p, "toplist", "tl_on_list").rolling(5, min_periods=1).max(), "近5日曾上榜(负向)")
    # --- 与量价交互 ---
    for a, b in [("rt_turn_20", "ivol_20"), ("rt_size", "ivol_20"), ("sp_fc_pchange", "ivol_20"),
                 ("rt_turn_20", "rev_20"), ("tl_net20", "ivol_20")]:
        if a in REGISTRY and b in REGISTRY:
            fa, fb = REGISTRY[a][1], REGISTRY[b][1]
            add(f"nx_{a}__P__{b}", "newdata",
                lambda p, fa=fa, fb=fb: ops.cs_rank(fa(p)) + ops.cs_rank(fb(p)), f"{a}+{b}")


# ================= 12. 传统技术指标族(用户建议:KDJ/MACD/布林带等) =================
def _build_ta():
    """经典技术指标。无日频高低价,KDJ/威廉/CCI 用滚动收盘极值近似 HHV/LLV。
    全部只用价格比值与形态,不用绝对价格水平(前复权)。"""
    C = lambda p: p.raw("close_px")

    def ema(x, n): return x.ewm(span=n, min_periods=n // 2).mean()

    # MACD
    def macd_hist(p, f=12, s_=26, sig=9):
        c = C(p); dif = ema(c, f) - ema(c, s_); dea = ema(dif, sig)
        return ops.safe_div(dif - dea, c)                    # 用价格归一,消除价格水平
    add("ta_macd_hist", "ta", lambda p: macd_hist(p), "MACD 柱(归一)")
    add("ta_macd_dif", "ta", lambda p: ops.safe_div(ema(C(p), 12) - ema(C(p), 26), C(p)), "MACD DIF(归一)")
    add("ta_macd_hist_d3", "ta", lambda p: ops.ts_delta(macd_hist(p), 3), "MACD 柱 3 日变化")
    # RSI
    def rsi(p, n):
        r = p["ret_c2c"]
        up = r.clip(lower=0).rolling(n, min_periods=n // 2).mean()
        dn = (-r.clip(upper=0)).rolling(n, min_periods=n // 2).mean()
        return 100 - 100 / (1 + ops.safe_div(up, dn))
    for n in (6, 14, 24):
        add(f"ta_rsi_{n}", "ta", lambda p, n=n: rsi(p, n), f"RSI({n})")
    # 布林带
    for n in (20, 60):
        def bb(p, n=n):
            c = C(p); m = ops.ts_mean(c, n); sd = ops.ts_std(c, n)
            return ops.safe_div(c - m, 2 * sd)              # %b 中心化:>0.5 上轨外
        add(f"ta_boll_pctb_{n}", "ta", bb, f"布林 %b({n})")
        add(f"ta_boll_width_{n}", "ta",
            lambda p, n=n: ops.safe_div(ops.ts_std(C(p), n), ops.ts_mean(C(p), n)), f"布林带宽({n})")
    # KDJ(HHV/LLV 用收盘极值近似)
    def rsv(p, n=9):
        c = C(p); hh = ops.ts_max(c, n); ll = ops.ts_min(c, n)
        return 100 * ops.safe_div(c - ll, hh - ll)
    def kdj(p, n=9):
        r = rsv(p, n); k = r.ewm(alpha=1 / 3, min_periods=3).mean(); d = k.ewm(alpha=1 / 3, min_periods=3).mean()
        return k, d, 3 * k - 2 * d
    add("ta_kdj_k", "ta", lambda p: kdj(p)[0], "KDJ K(收盘极值近似)")
    add("ta_kdj_j", "ta", lambda p: kdj(p)[2], "KDJ J")
    add("ta_kdj_kd", "ta", lambda p: kdj(p)[0] - kdj(p)[1], "KDJ K-D")
    add("ta_wr_14", "ta", lambda p: -rsv(p, 14), "威廉 %R(14,近似)")
    # 均线系统
    for n in (5, 10, 20, 60, 120):
        add(f"ta_bias_{n}", "ta", lambda p, n=n: ops.safe_div(C(p), ops.ts_mean(C(p), n)) - 1, f"均线乖离({n})")
    add("ta_ma_cross_5_20", "ta", lambda p: ops.safe_div(ops.ts_mean(C(p), 5), ops.ts_mean(C(p), 20)) - 1, "MA5/MA20")
    add("ta_ma_cross_20_60", "ta", lambda p: ops.safe_div(ops.ts_mean(C(p), 20), ops.ts_mean(C(p), 60)) - 1, "MA20/MA60")
    add("ta_ma_align", "ta",
        lambda p: ((ops.ts_mean(C(p), 5) > ops.ts_mean(C(p), 20)).astype(float)
                   + (ops.ts_mean(C(p), 20) > ops.ts_mean(C(p), 60)).astype(float)
                   + (ops.ts_mean(C(p), 60) > ops.ts_mean(C(p), 120)).astype(float)).where(C(p).notna()),
        "均线多头排列计数(0-3)")
    # CCI(典型价用收盘近似)
    def cci(p, n=20):
        c = C(p); m = ops.ts_mean(c, n)
        md = (c - m).abs().rolling(n, min_periods=n // 2).mean()
        return ops.safe_div(c - m, 0.015 * md)
    add("ta_cci_20", "ta", lambda p: cci(p), "CCI(20,近似)")
    # 唐奇安位置 / 通道
    for n in (20, 60):
        add(f"ta_donchian_{n}", "ta",
            lambda p, n=n: ops.safe_div(C(p) - ops.ts_min(C(p), n), ops.ts_max(C(p), n) - ops.ts_min(C(p), n)),
            f"唐奇安通道位置({n})")
    # OBV 斜率(用成交额按涨跌加符号)
    add("ta_obv_slope_20", "ta",
        lambda p: ops.safe_div(ops.ts_sum(np.sign(p["ret_c2c"]) * p.raw("amount"), 20),
                               ops.ts_sum(p.raw("amount"), 20)), "OBV 斜率(20,归一)")
    # ATR 近似(日内振幅 rng 的均值)与 ATR 相对位置
    add("ta_atr_ratio", "ta", lambda p: ops.safe_div(ops.ts_mean(p.raw("rng"), 5), ops.ts_mean(p.raw("rng"), 20)), "ATR5/ATR20")
    # 动量振荡:ROC
    for n in (10, 20):
        add(f"ta_roc_{n}", "ta", lambda p, n=n: ops.ts_pct(C(p), n), f"ROC({n})")


# ================= 11. PEAD 盈余惯性族(第三轮) =================
def _build_pead():
    from afactor.config import ROOT
    if not (ROOT / "data_8:21" / "stock_income_2015_now.pkl").exists():
        return
    # 会计意外
    add("pead_sue_bt", "pead", lambda p: _fu(p, "sue_bt"),
        "SUE(Bernard-Thomas 季节性随机游走+漂移标准化)")
    add("pead_sue_ta", "pead", lambda p: _fu(p, "sue_ta"),
        "SUE(总资产标准化,Livnat-Mendenhall 2006)")
    add("pead_sur_rev", "pead", lambda p: _fu(p, "revenue_yoy"), "营收意外(Jegadeesh-Livnat 2006)")
    add("pead_sur_op", "pead", lambda p: _fu(p, "operate_profit_yoy"), "营业利润意外")
    add("pead_sur_cfo", "pead", lambda p: _fu(p, "n_cashflow_act_yoy"), "经营现金流意外")
    # 质量拆分:利润好于营业利润 = 靠非经常性损益,排雷信号
    add("pead_ni_op_gap", "pead",
        lambda p: -(ops.cs_rank(_fu(p, "n_income_attr_p_yoy")) -
                    ops.cs_rank(_fu(p, "operate_profit_yoy"))).where(_nonfin(p)),
        "净利同比减营业利润同比(负向:靠非经常性损益的不可持续)")
    add("pead_sue_cfo_gap", "pead",
        lambda p: -(ops.cs_rank(_fu(p, "n_income_attr_p_yoy")) -
                    ops.cs_rank(_fu(p, "n_cashflow_act_yoy"))),
        "利润意外减现金流意外(负向:应计驱动的假超预期)")
    # 事件反应
    add("pead_ear_car3", "pead", lambda p: _fu_event(p, "ear_car3"),
        "公告窗口[-1,+1]市场调整累计超额(EAR)")
    add("pead_ear_gap", "pead", lambda p: _fu_event(p, "ear_gap"),
        "公告后首日跳空(市场调整;开源证券'净利润断层')")
    add("pead_jump_combo", "pead",
        lambda p: ops.cs_rank(_fu(p, "sue_bt")) + ops.cs_rank(_fu_event(p, "ear_gap")),
        "净利润断层 = SUE + 公告跳空(开源证券)")
    add("pead_underreaction", "pead",
        lambda p: ops.cs_rank(_fu(p, "sue_bt")) - ops.cs_rank(_fu_event(p, "ear_car3")),
        "SUE 高但公告窗口没涨 = 反应不足,剩余漂移最多")
    # 漂移窗口限定:PEAD 是有限窗口事件效应
    for w in [20, 40, 60]:
        add(f"pead_sue_win{w}", "pead",
            lambda p, w=w: _fu(p, "sue_bt").where(_fu_event(p, "days_since_ann") <= w),
            f"SUE 仅在公告后 {w} 个交易日内暴露(Bernard-Thomas 漂移窗口)")
    add("pead_gap_win40", "pead",
        lambda p: _fu_event(p, "ear_gap").where(_fu_event(p, "days_since_ann") <= 40),
        "公告跳空仅在公告后 40 日内暴露")
    # 与量价交互(本项目已验证"状态切割"有效)
    for a, b in [("pead_sue_bt", "ivol_20"), ("pead_ear_gap", "ivol_20"),
                 ("pead_sue_bt", "amt_vol_20"), ("pead_jump_combo", "ivol_20")]:
        if a in REGISTRY and b in REGISTRY:
            fa, fb = REGISTRY[a][1], REGISTRY[b][1]
            add(f"px_{a}__P__{b}", "pead",
                lambda p, fa=fa, fb=fb: ops.cs_rank(fa(p)) + ops.cs_rank(fb(p)), f"{a}+{b}")


def _build_literature():
    """加载两轮研报调研产出的因子规格(literature/ 与 literature2/)。"""
    import glob
    from afactor import literature as LIT
    from afactor.config import ROOT
    for sub, pref in [("literature", "lit_"), ("literature2", "lit2_")]:
        specs = []
        for f in sorted(glob.glob(str(ROOT / "research" / sub / "spec*.json"))):
            specs += LIT.load_specs(f)
        if specs:
            LIT.register_specs(specs, REGISTRY, prefix=pref)


def build_all(with_literature: bool = True):
    if REGISTRY:
        return REGISTRY
    _build_momentum(); _build_volatility(); _build_volume()
    _build_intraday(); _build_mechanical(); _build_interactions()
    if with_literature:
        _build_literature()
    _build_market_family(); _build_apm(); _build_condvp(); _build_statind()
    _build_fundamental(); _build_pead(); _build_ta(); _build_newdata()
    _build_newmech()
    return REGISTRY


if __name__ == "__main__":
    r = build_all()
    fam = pd.Series([v[0] for v in r.values()]).value_counts()
    print(f"因子总数 {len(r)}"); print(fam.to_string())


# ================= 新机制族(2026-08-22,用户挖掘系统提供的研报方向)=================
# 这三族是查证后**现有 1951 个因子里完全没有**的机制,不是已有维度的算子变体:
#   1. 量价滞后互相关(中金高频手册 corr_pvl「量领先价」)—— 我原有的 mi_mkt_lead
#      是**市场层面**滞后,不是个股的量→价领先,机制不同
#   2. 筹码分布(中金 doc_vol_pdf90_std「筹码分位波动」)—— 成交量在**价格轴**上的
#      分布,与我全部按时间轴聚合的日内因子正交
#   3. QRS 阻力支撑相对强度 —— 日内高低点回归斜率
# 全部由已存的 seg0-7 日内 profile 派生,**不需要重新处理分钟数据**。
_NEWMECH: dict = {}


def _seg_base(p):
    """从 8 段日内 profile 派生的共用基础量(每个 Panel 只算一次)。

    字段语义已核对:seg{k}_ret 为分段收益(8 段之和逐位等于 ret_intraday);
    seg{k}_amt 为分段成交额**占比**(和≈0.99,已归一化)。
    尾盘口径下 TailPanel.raw 会把 seg* 整体滞后一天(无 pre_seg 版本),
    即使用 T-1 日的日内形态,无前视。
    """
    key = ("segbase", id(p), bool(getattr(p, "_is_tail", False)))
    if key in _NEWMECH:
        return _NEWMECH[key]
    idx, col = p.dates, p.codes
    R = np.stack([p.raw(f"seg{k}_ret").reindex(index=idx, columns=col).to_numpy(np.float32)
                  for k in range(8)])                      # (8,T,N) 分段收益
    A = np.stack([p.raw(f"seg{k}_amt").reindex(index=idx, columns=col).to_numpy(np.float32)
                  for k in range(8)])                      # (8,T,N) 分段成交额占比
    with np.errstate(invalid="ignore", divide="ignore"):
        W = A / np.nansum(A, axis=0, keepdims=True)        # 重新归一到严格 1
        P = np.nancumsum(np.nan_to_num(R), axis=0)         # (8,T,N) 日内累计价格路径
        P = np.where(np.isfinite(R).cumsum(axis=0) > 0, P, np.nan)

        # ---- 筹码分布:成交量在价格轴上的加权矩 ----
        mu = np.nansum(W * P, axis=0)                                   # 加权均价位置
        dev = P - mu
        var = np.nansum(W * dev ** 2, axis=0)
        sd = np.sqrt(np.maximum(var, 0))
        sk = np.nansum(W * dev ** 3, axis=0) / np.maximum(sd ** 3, 1e-12)
        up = np.nansum(W * np.maximum(dev, 0) ** 2, axis=0)             # 上半分散
        dn = np.nansum(W * np.minimum(dev, 0) ** 2, axis=0)             # 下半分散
        pc = P[-1]                                                       # 收盘价格位置
        profit = np.nansum(W * (P < pc), axis=0)                        # 获利盘比例
        # 加权分位:按价格排序后累计权重,取 p90 与中位之差(pdf90 的直接对应物)
        order = np.argsort(np.where(np.isfinite(P), P, np.inf), axis=0)
        Ps = np.take_along_axis(P, order, axis=0)
        Ws = np.take_along_axis(np.nan_to_num(W), order, axis=0)
        cw = np.cumsum(Ws, axis=0)
        def _q(level):
            hit = cw >= level
            first = np.argmax(hit, axis=0)
            return np.take_along_axis(Ps, first[None], axis=0)[0]
        p90, p50, p10 = _q(0.90), _q(0.50), _q(0.10)

        # ---- QRS:日内高低点(8 点采样,含开盘 0 点)----
        hi = np.fmax(np.nanmax(P, axis=0), 0.0)
        lo = np.fmin(np.nanmin(P, axis=0), 0.0)

        # ---- 量价滞后互相关:跨 8 段的 corr(量[k], 收益[k+L]) ----
        def _xcorr(X, Y, L):
            a, b = X[:8 - L], Y[L:]
            am = np.nanmean(a, axis=0); bm = np.nanmean(b, axis=0)
            ca, cb = a - am, b - bm
            num = np.nansum(ca * cb, axis=0)
            den = np.sqrt(np.nansum(ca ** 2, axis=0) * np.nansum(cb ** 2, axis=0))
            return num / np.where(den > 1e-12, den, np.nan)
        v2r1, v2r2 = _xcorr(W, R, 1), _xcorr(W, R, 2)      # 量领先价
        r2v1 = _xcorr(R, W, 1)                              # 价领先量

    del R, A, W, P, dev, order, Ps, Ws, cw
    mk = lambda a: pd.DataFrame(a, index=idx, columns=col).astype(np.float32)
    out = dict(chip_mean=mk(mu), chip_std=mk(sd), chip_skew=mk(sk),
               chip_up=mk(np.sqrt(up)), chip_dn=mk(np.sqrt(dn)), chip_profit=mk(profit),
               chip_p90gap=mk(p90 - p50), chip_p10gap=mk(p50 - p10),
               chip_iqr=mk(p90 - p10),
               qrs_hi=mk(hi), qrs_lo=mk(lo),
               lead_v2r1=mk(v2r1), lead_v2r2=mk(v2r2), lead_r2v1=mk(r2v1),
               lead_net=mk(v2r1 - r2v1))
    _NEWMECH[key] = out
    return out


def _build_newmech():
    fam = "newmech"

    # ---- 1. 量价滞后互相关(日内 8 段 + 日频跨日两个层面)----
    for base, sgn, desc in [("lead_v2r1", -1, "量领先价(滞后1段)"),
                            ("lead_v2r2", -1, "量领先价(滞后2段)"),
                            ("lead_r2v1", -1, "价领先量(滞后1段)"),
                            ("lead_net", -1, "量价领先净方向")]:
        for w in (5, 20, 60):
            add(f"nm_{base}_m{w}", fam,
                (lambda b, ww: lambda p: ops.ts_mean(_seg_base(p)[b], ww))(base, w),
                f"{desc} {w}日均")
        add(f"nm_{base}_std20", fam,
            (lambda b: lambda p: ops.ts_std(_seg_base(p)[b], 20))(base), f"{desc} 20日波动")

    def _lamt(p):
        return ops.signed_log(p.raw("amount"))
    for w in (10, 20, 60):
        add(f"nm_dlead_v2r_{w}", fam,
            (lambda ww: lambda p: ops.ts_corr(p["ret_c2c"], ops.ts_delay(_lamt(p), 1), ww))(w),
            f"日频量领先价 {w}日")
        add(f"nm_dlead_r2v_{w}", fam,
            (lambda ww: lambda p: ops.ts_corr(_lamt(p), ops.ts_delay(p["ret_c2c"], 1), ww))(w),
            f"日频价领先量 {w}日")
        add(f"nm_dlead_net_{w}", fam,
            (lambda ww: lambda p: ops.ts_corr(p["ret_c2c"], ops.ts_delay(_lamt(p), 1), ww)
             - ops.ts_corr(_lamt(p), ops.ts_delay(p["ret_c2c"], 1), ww))(w),
            f"日频领先净方向 {w}日")

    # ---- 2. 筹码分布 ----
    for base, sgn, desc in [("chip_mean", -1, "筹码均价位置"), ("chip_std", -1, "筹码分散度"),
                            ("chip_skew", -1, "筹码偏度"), ("chip_up", -1, "筹码上半分散"),
                            ("chip_dn", -1, "筹码下半分散"), ("chip_profit", -1, "获利盘比例"),
                            ("chip_p90gap", -1, "筹码p90-中位"), ("chip_p10gap", -1, "筹码中位-p10"),
                            ("chip_iqr", -1, "筹码p90-p10")]:
        for w in (5, 20, 60):
            add(f"nm_{base}_m{w}", fam,
                (lambda b, ww: lambda p: ops.ts_mean(_seg_base(p)[b], ww))(base, w),
                f"{desc} {w}日均")
        # 中金 doc_vol_pdf90_std 的直接对应物:筹码分位的**时序波动**
        for w in (10, 20):
            add(f"nm_{base}_std{w}", fam,
                (lambda b, ww: lambda p: ops.ts_std(_seg_base(p)[b], ww))(base, w),
                f"{desc} {w}日波动")

    # ---- 3. QRS 阻力支撑相对强度 ----
    def _qrs(p, w, mode):
        s = _seg_base(p)
        hi, lo = s["qrs_hi"], s["qrs_lo"]
        cov = ops.ts_cov(hi, lo, w)
        vl = ops.ts_std(lo, w) ** 2
        beta = ops.safe_div(cov, vl)
        if mode == "beta":
            return beta
        r = ops.ts_corr(hi, lo, w)
        return r ** 2 if mode == "r2" else beta * (r ** 2)
    for w in (18, 60):
        for mode, desc in [("beta", "QRS斜率"), ("r2", "QRS拟合优度"), ("robust", "QRS稳健化")]:
            add(f"nm_qrs_{mode}_{w}", fam,
                (lambda ww, mm: lambda p: _qrs(p, ww, mm))(w, mode), f"{desc} {w}日")
    add("nm_qrs_hl_ratio_20", fam,
        lambda p: ops.ts_mean(ops.safe_div(_seg_base(p)["qrs_hi"],
                                           -_seg_base(p)["qrs_lo"]), 20), "上下影强度比 20日")
