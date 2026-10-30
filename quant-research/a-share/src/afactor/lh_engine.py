"""长历史(2005-2025)换手率版排雷的回测适配层(前缀 lh;预注册 research/LONG_HISTORY_PREREG.md,2026-10-11)。

只让适配器去适应引擎,引擎(strategy.simulate / benchmark)不改语义;唯一的引擎扩展是默认关闭的
stamp_fn 钩子(分段印花税,预注册 §14.1)。第二步(风控层)直接复用本模块:

    from afactor import lh_engine as LE
    X = LE.load_inputs()                                  # 第一步 prep 落盘的输入(mmap)
    ad = LE.adapter(X, discount=0.0)                      # 引擎面板适配器(计价价 V / 开盘成交额 / 不可成交)
    W = LE.weights(X, "S1", buffer=(0.20, 0.35))          # 主配置 O0 的目标权重(T x N)
    res = LE.run(W, ad, X, rebalance=5)                   # 5 相位平均,Y=1 一跳 + 史实印花税

## 口径(预注册条目)
- 计价价 V(§6.1):交易日 = 后复权开盘价;停牌日 = 最近一个交易日后复权收盘(复牌跳空完整计入);
  敏感性 valuation="index":停牌期间 V 按自建全 A 等权收盘指数(截至 te-1)漂移,复牌日衔接开盘价。
- 退市(§6.5):stock_basic 退市日 od(<= 2025-12-31)起 V = 最后收盘 x (1 - 折价),此后恒定;
  od 起开盘成交额置 1e15、不可成交标记置 False,使冻结持仓在下一调仓日按冻结价清出。
- 不可成交(§6.2):成交日 te 停牌(开盘成交额 NaN/0)或"开盘即触及涨跌停"(面板 open_at_up/open_at_dn,
  无涨跌幅限制日不判触板);敏感性 dead="ow" = 面板一字板(最高 == 最低 且 |日收益| > 4.5%)。
- 开盘 10 分钟成交额(§6.3)= ratio x 全日成交额,ratio 主口径 0.11406(LongPanel "pool"),敏感性 0.061。
- 股票池(§4):当日有成交;上市满 min_list 个有效交易日(从面板首个有效交易日起计);
  ADV20 >= 当日合格股票 ADV20 的 (1 - q) 分位(q = 0.92 主口径);可选剔除当日 ST、只留幸存股。
- 信号(§3):S1 = -turn20、S2 = -turn60(最少 30 个观测)、S3 = +amihud20;停牌日 NaN 不计入均值。
- 成本(§6.4):佣金 1bp;印花税史实 6 段(含 2008-09-19 前买方税);一跳半价差(不复权收盘 close_raw)
  + 平方根冲击 Y=1(σ、ADV 截至 te-1);old = 只扣佣金 + 印花税;pess = Y 1.5 + 两跳。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from afactor.config import TMP, RESEARCH_END, STAMP_TAX_BPS_BEFORE, STAMP_TAX_BPS_AFTER, STAMP_TAX_CHANGE_DATE

TD = 242
INPUT_DIR = TMP / "lh_step1_inputs"
EVAL_START = "2006-01-04"
DEV = ("2006-01-04", "2014-12-31")
TEST = ("2015-01-05", "2025-12-31")
CAPITAL = 1e7
OPEN10_RATIO_MAIN = 0.11405877023935318      # LongPanel meta open10_amt_ratio_pool(预注册 §6.3 的 0.114)
OPEN10_RATIO_LO = 0.061                      # 10% 分位敏感性(预注册写 0.061)
FROZEN_AMT = 1e15                            # 退市冻结后"足够大"的窗口成交额

# §8 窗口(主口径 A = 上证收盘极值;B = 中证1000 收盘极值;orig = 任务书原窗口,补充行)
WINDOWS_A = {
    "BEAR08": ("2007-10-16", "2008-11-04"), "BEAR15": ("2015-06-12", "2016-01-28"),
    "BEAR18": ("2018-01-24", "2019-01-03"), "BEAR22": ("2021-09-13", "2022-04-26"),
    "BEAR24": ("2023-05-08", "2024-02-05"),
    "BULL06": ("2006-01-04", "2007-10-16"), "BULL14": ("2014-07-01", "2015-06-12"),
    "BULL19": ("2019-01-03", "2019-04-19"),
}
WINDOWS_B = {
    "BEAR08": ("2008-01-15", "2008-11-04"), "BEAR15": ("2015-06-12", "2015-09-15"),
    "BEAR18": ("2017-10-10", "2018-10-18"), "BEAR22": ("2021-12-01", "2022-04-26"),
    "BEAR24": ("2023-05-04", "2024-02-05"),
}
WINDOWS_ORIG = {"BEAR22_orig": ("2021-12-13", "2022-10-31"), "BEAR24_orig": ("2023-07-31", "2024-02-05")}
V_TROUGHS = {"V08": "2008-11-04", "V16": "2016-01-28", "V19": "2019-01-03", "V24": "2024-02-05",
             "V22_supp": "2022-04-26"}
BEARS = ("BEAR08", "BEAR15", "BEAR18", "BEAR22", "BEAR24")

# §6.4 印花税史实(按成交日生效,含首日):(起始日, 买方, 卖方)
STAMP_HIST = (("1990-01-01", 0.002, 0.002), ("2005-01-24", 0.001, 0.001), ("2007-05-30", 0.003, 0.003),
              ("2008-04-24", 0.001, 0.001), ("2008-09-19", 0.0, 0.001), ("2023-08-28", 0.0, 0.0005))


# ================================================================ 印花税钩子
def stamp_arrays(dates: pd.DatetimeIndex, extra_bps: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    d = pd.DatetimeIndex(dates).values
    b = np.zeros(len(d)); s = np.zeros(len(d))
    for start, rb, rs in STAMP_HIST:
        m = d >= np.datetime64(start)
        b[m] = rb; s[m] = rs
    return b + extra_bps * 1e-4, s + extra_bps * 1e-4


class StampFn:
    """stamp_fn(te) -> (买方税率, 卖方税率)。extra_bps 把额外佣金(如 5bp 敏感性的 +4bp)并进两边(记账时计入 stamp 列)。"""

    def __init__(self, dates, extra_bps: float = 0.0):
        self.b, self.s = stamp_arrays(dates, extra_bps)

    def __call__(self, te: int):
        return float(self.b[te]), float(self.s[te])


class StampFnCurrent:
    """复现引擎现行两段税率(卖出 10bp / 2023-08-28 起 5bp,无买方税),用于"钩子路径逐位等于默认路径"的检查。"""

    def __init__(self, dates):
        self.sw = int(np.searchsorted(pd.DatetimeIndex(dates).values, np.datetime64(STAMP_TAX_CHANGE_DATE)))

    def __call__(self, te: int):
        return 0.0, (STAMP_TAX_BPS_BEFORE if te < self.sw else STAMP_TAX_BPS_AFTER) * 1e-4


# ================================================================ 适配器
class LHAdapter:
    """引擎面板接口的最小实现:raw('twap_open10'/'amt_open10'/'close_px'),p['one_word_up'/'one_word_down'],
    dates / codes / _derived(CostModel 的估计面板放在这里)。数据全部来自构造参数,不读盘。"""
    _is_tail = False
    sealed = False

    def __init__(self, V: pd.DataFrame, amt_open10: pd.DataFrame, dead: pd.DataFrame,
                 cost_panels: dict | None = None):
        self._cache = {"twap_open10": V, "amt_open10": amt_open10, "close_px": V}
        self._flags = {"one_word_up": dead,
                       "one_word_down": pd.DataFrame(np.zeros(dead.shape, bool), index=dead.index,
                                                     columns=dead.columns)}
        self._derived: dict = {}
        if cost_panels is not None:
            for key in ("tick",):
                self._derived[f"__eng_cost__20_10_0.01_{key}"] = cost_panels

    def raw(self, name):
        return self._cache[name]

    def __getitem__(self, name):
        if name in self._flags:
            return self._flags[name]
        return self._cache[name]

    @property
    def dates(self):
        return self._cache["twap_open10"].index

    @property
    def codes(self):
        return self._cache["twap_open10"].columns


# ================================================================ 输入准备(prep)
def _rowwise_quantile_mask(adv: np.ndarray, elig: np.ndarray, keep_frac: float) -> np.ndarray:
    """当日合格股票(elig)里 ADV20 >= (1-keep_frac) 分位者。keep_frac >= 1 时不设门槛。"""
    if keep_frac >= 1.0:
        return elig.copy()
    x = np.where(elig, adv, np.nan).astype(np.float64)
    out = np.zeros_like(elig)
    q = 1.0 - keep_frac
    ok_rows = elig.sum(axis=1) > 0
    thr = np.full(adv.shape[0], np.nan)
    with np.errstate(all="ignore"):
        thr[ok_rows] = np.nanquantile(x[ok_rows], q, axis=1)
    out = elig & (x >= thr[:, None])
    return out


def prepare_inputs(out_dir: Path = INPUT_DIR, log=print) -> dict:
    """从 LongPanel 读原料,算好所有配置共用的输入并落盘为 .npy(workers 以 mmap 只读加载)。
    不计算任何策略收益。返回 meta。"""
    from afactor.lh_panel import LongPanel
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = LongPanel(suspend_fill=True, open10_ratio="pool")
    dates, codes = p.dates, p.codes
    T, N = len(dates), len(codes)
    f32 = lambda df: df.to_numpy(np.float32)
    trade = p.raw("trade").to_numpy(bool)
    amount = f32(p.raw("amount"))
    close = f32(p.raw("close_px"))
    valid = trade & (amount > 0) & np.isfinite(close)
    dl = np.cumsum(valid, axis=0).astype(np.float32)
    dl = np.where(valid, dl, np.nan).astype(np.float32)
    adv20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy(np.float32)
    log(f"[prep] T={T} N={N} valid={valid.mean():.3f}")

    # ---- 信号 ----
    turn = f32(p.raw("turn"))
    turn = np.where(valid & (turn > 0), turn, np.nan).astype(np.float32)
    tdf = pd.DataFrame(turn)
    s1 = -tdf.rolling(20, min_periods=10).mean().to_numpy(np.float32)
    s2 = -tdf.rolling(60, min_periods=30).mean().to_numpy(np.float32)
    pre = f32(p.raw("preclose_px"))
    with np.errstate(all="ignore"):
        rcc = close / pre - 1.0
        ami = np.abs(rcc) / amount * 1e9
    ami = np.where(valid & np.isfinite(ami), ami, np.nan).astype(np.float32)
    s3 = pd.DataFrame(ami).rolling(20, min_periods=10).mean().to_numpy(np.float32)
    del turn, tdf, pre, rcc, ami

    # ---- 自建全 A 等权收盘指数(§12.1;条件 1-3:当日有成交、上市满 6 日;收益 = 持有收益 ret,复牌含跳空)----
    ret = f32(p.raw("ret"))
    m13 = valid & (dl >= 6)
    with np.errstate(all="ignore"):
        ew = np.nanmean(np.where(m13 & np.isfinite(ret), ret, np.nan), axis=1)
    ew = np.nan_to_num(ew, nan=0.0)
    ew_level = np.cumprod(1.0 + ew)
    del ret

    # ---- 计价价与退市冻结 ----
    V0 = f32(p.raw("twap_open10"))
    info = p.code_info
    out = info["outDate"].fillna("").astype(str)
    is_del = ((out != "") & (out <= RESEARCH_END)).to_numpy()
    od_idx = np.full(N, -1)
    lt_idx = np.full(N, -1)
    lastc = np.full(N, np.nan, np.float32)
    dvals = dates.values
    lt_dates = pd.to_datetime(info["last_trade"]).values
    for j in np.flatnonzero(is_del):
        od = int(np.searchsorted(dvals, np.datetime64(pd.Timestamp(out.iloc[j]))))
        lt = int(np.searchsorted(dvals, lt_dates[j]))
        if od >= T or lt >= T:
            continue
        if not valid[lt, j]:                                   # 最后一行不是成交日:取最后成交日
            vv = np.flatnonzero(valid[:lt + 1, j])
            if not len(vv):
                continue
            lt = int(vv[-1])
        od_idx[j], lt_idx[j], lastc[j] = od, lt, close[lt, j]
    frozen = np.zeros((T, N), bool)
    for j in np.flatnonzero(od_idx >= 0):
        frozen[od_idx[j]:, j] = True
        V0[lt_idx[j] + 1:od_idx[j], j] = lastc[j]              # 最后成交日到退市日之间:沿用最后收盘(不可卖)
    np.save(out_dir / "frozen.npy", frozen)
    for d in (0.0, 0.5, 1.0):
        V = V0.copy()
        for j in np.flatnonzero(od_idx >= 0):
            V[od_idx[j]:, j] = lastc[j] * (1.0 - d)
        np.save(out_dir / f"V_d{int(d * 100)}.npy", V)
    # 停牌指数法估值(敏感性):停牌行(上市期间、首个成交日之后、退市日之前)V = 最近收盘 x I[te-1] / I[最近成交日]
    lc = pd.DataFrame(np.where(valid, close, np.nan)).ffill().to_numpy(np.float32)
    I_prev = np.concatenate([[1.0], ew_level[:-1]])
    I_at_trade = pd.DataFrame(np.where(valid, ew_level[:, None], np.nan)).ffill().to_numpy(np.float64)
    listed = p.raw("listed").to_numpy(bool)
    susp = listed & ~valid & np.isfinite(lc) & ~frozen
    Vidx = np.where(susp, lc * (I_prev[:, None] / I_at_trade), V0).astype(np.float32)
    for j in np.flatnonzero(od_idx >= 0):
        Vidx[od_idx[j]:, j] = lastc[j]
    np.save(out_dir / "V_idx_d0.npy", Vidx)
    del Vidx, I_at_trade, lc, V0

    # ---- 开盘成交额与不可成交 ----
    for tag, r in (("r114", OPEN10_RATIO_MAIN), ("r061", OPEN10_RATIO_LO)):
        a = (np.nan_to_num(amount, nan=np.nan) * np.float32(r)).astype(np.float32)
        a = np.where(frozen, np.float32(FROZEN_AMT), a).astype(np.float32)
        np.save(out_dir / f"amt_{tag}.npy", a)
    dead_open = (p.raw("open_at_up").to_numpy(bool) | p.raw("open_at_dn").to_numpy(bool)) & ~frozen
    dead_ow = (p.raw("one_word_up").to_numpy(bool) | p.raw("one_word_down").to_numpy(bool)) & ~frozen
    np.save(out_dir / "dead_open.npy", dead_open)
    np.save(out_dir / "dead_ow.npy", dead_ow)

    # ---- 股票池(§4)----
    is_st = p.raw("is_st").to_numpy(bool)
    surv2025 = ~is_del
    # 研究面板(cache/daily)里的股票 = 2020-2025 幸存股(S-a / B3 用)
    import pyarrow.parquet as pq
    from afactor.config import DAILY
    panel_codes = set(pq.ParquetFile(DAILY / "close_px.parquet").schema_arrow.names)
    survpanel = np.array([c in panel_codes for c in codes])
    base_elig = valid & np.isfinite(adv20)
    pools = {
        "q92": _rowwise_quantile_mask(adv20, base_elig & (dl >= 6), 0.92),
        "q80": _rowwise_quantile_mask(adv20, base_elig & (dl >= 6), 0.80),
        "q100": _rowwise_quantile_mask(adv20, base_elig & (dl >= 6), 1.00),
        "q92_list60": _rowwise_quantile_mask(adv20, base_elig & (dl >= 60), 0.92),
        "q92_surv2025": _rowwise_quantile_mask(adv20, base_elig & (dl >= 6) & surv2025[None, :], 0.92),
        "q92_survpanel": _rowwise_quantile_mask(adv20, base_elig & (dl >= 6) & survpanel[None, :], 0.92),
    }
    pools["q92_exST"] = pools["q92"] & ~is_st
    for k, v in pools.items():
        np.save(out_dir / f"pool_{k}.npy", v)
    np.save(out_dir / "score_S1.npy", s1)
    np.save(out_dir / "score_S2.npy", s2)
    np.save(out_dir / "score_S3.npy", s3)
    np.save(out_dir / "is_st.npy", is_st)

    # ---- 成本估计面板(一跳价差用 close_raw;σ、ADV 截至 te-1),与 LongPanel.cost_model 同式 ----
    p.cost_model(impact_y=1.0)
    key = "__eng_cost__20_10_0.01_tick"
    cp = p._derived[key]
    for k in ("hs", "sigma", "adv"):
        np.save(out_dir / f"cost_{k}.npy", np.asarray(cp[k], np.float32))

    meta = dict(dates=[str(x.date()) for x in dates], codes=list(map(str, codes)),
                is_delisted=is_del.tolist(), survpanel=survpanel.tolist(),
                od_idx=od_idx.tolist(), lt_idx=lt_idx.tolist(),
                ew_allA_ret=ew.tolist(),
                pool_sizes={k: float(v.sum(axis=1).mean()) for k, v in pools.items()},
                n_delisted_le2025=int(is_del.sum()), open10_ratio_main=OPEN10_RATIO_MAIN)
    (out_dir / "meta.json").write_text(json.dumps(meta))
    log(f"[prep] 池日均股票数: " + ", ".join(f"{k} {v:.0f}" for k, v in meta["pool_sizes"].items()))
    return meta


# ================================================================ 加载与组装
class Inputs:
    def __init__(self, d: Path = INPUT_DIR, mmap: bool = True):
        self.dir = Path(d)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.dates = pd.DatetimeIndex(pd.to_datetime(self.meta["dates"]))
        self.codes = pd.Index(self.meta["codes"])
        self._mm = "r" if mmap else None
        self._c: dict = {}

    def arr(self, name: str) -> np.ndarray:
        if name not in self._c:
            self._c[name] = np.load(self.dir / f"{name}.npy", mmap_mode=self._mm)
        return self._c[name]

    def df(self, name: str, rows=None) -> pd.DataFrame:
        a = self.arr(name)
        idx = self.dates
        if rows is not None:
            a = a[rows]; idx = idx[rows]
        return pd.DataFrame(a, index=idx, columns=self.codes, copy=False)

    @property
    def ew_allA_ret(self) -> pd.Series:
        return pd.Series(self.meta["ew_allA_ret"], index=self.dates)


def load_inputs(d: Path = INPUT_DIR, mmap: bool = True) -> Inputs:
    return Inputs(d, mmap)


def adapter(X: Inputs, discount: float = 0.0, valuation: str = "close", dead: str = "open",
            ratio: str = "r114", rows=None) -> LHAdapter:
    vname = "V_idx_d0" if valuation == "index" else f"V_d{int(round(discount * 100))}"
    if valuation == "index" and discount != 0.0:
        raise ValueError("指数法估值只做主折价 0%")
    sl = rows
    V = X.df(vname, sl)
    # ratio / dead 的 "free"/"susp_free"/"none" 只用于引擎断言 E3 的诊断(去掉可成交约束),不进任何策略配置
    if ratio in ("free", "susp_free"):
        Vv = V.to_numpy()
        ok = np.isfinite(Vv) & (Vv > 0)
        if ratio == "free":
            a = np.where(ok, np.float32(FROZEN_AMT), np.nan).astype(np.float32)
        else:
            a0 = X.arr("amt_r114")[sl] if sl is not None else X.arr("amt_r114")
            a = np.where(~np.isfinite(a0) & ok, np.float32(FROZEN_AMT), a0).astype(np.float32)
        amt = pd.DataFrame(a, index=V.index, columns=V.columns)
    else:
        amt = X.df(f"amt_{ratio}", sl)
    if dead == "none":
        dd = pd.DataFrame(np.zeros(V.shape, bool), index=V.index, columns=V.columns)
    else:
        dd = X.df("dead_open" if dead == "open" else "dead_ow", sl)
    cp = {k: (X.arr(f"cost_{k}")[sl] if sl is not None else X.arr(f"cost_{k}")) for k in ("hs", "sigma", "adv")}
    return LHAdapter(V, amt, dd, cost_panels=cp)


def pool(X: Inputs, name: str = "q92", rows=None) -> pd.DataFrame:
    return X.df(f"pool_{name}", rows)


def weights(X: Inputs, signal: str = "S1", buffer=(0.20, 0.35), pool_name: str = "q92", exclude_frac: float = 0.3,
            rows=None, score: pd.DataFrame | None = None, date_mask: np.ndarray | None = None) -> np.ndarray:
    """§5 排雷 + 缓冲带(沿用 backtest.weights_exclude_buffer);buffer=None 时无缓冲剔 exclude_frac。
    date_mask 与 strategy.build_weights 同义(池在 mask 外置空,缓冲带状态从 mask 起点重新开始);长历史主口径不用。"""
    from afactor import strategy as ST
    U = pool(X, pool_name, rows)
    S = score if score is not None else X.df(f"score_{signal}", rows)
    ad = _Shape(U.index, U.columns)
    return ST.build_weights(S, ad, "V1000", scheme="exclude", exclude_frac=exclude_frac, buffer=buffer,
                            universe_override=U, date_mask=date_mask)


class _Shape:
    def __init__(self, dates, codes):
        self.dates, self.codes = dates, codes


def cost_model(ad: LHAdapter, kind: str = "Y1"):
    """Y1 = 一跳 + Y=1(主);pess = Y 1.5 + 两跳;old = None(只扣佣金 + 印花税)。"""
    from afactor.impact import CostModel
    if kind == "old":
        return None
    if kind == "Y1":
        return CostModel(ad, impact_y=1.0, spread_mult=1.0)
    if kind == "pess":
        return CostModel(ad, impact_y=1.5, spread_mult=2.0)
    raise ValueError(kind)


def run(W: np.ndarray, ad: LHAdapter, rebalance: int = 5, cost: str = "Y1", commission_bps: float = 1.0,
        date_mask: np.ndarray | None = None, record: bool = False, stamp: str = "hist", phases=None) -> dict:
    """相位平均回测(逻辑与 strategy.run_weights 相同,另返回逐相位日收益)。"""
    from afactor import strategy as ST
    cm = cost_model(ad, cost)
    if stamp == "hist":
        sf = StampFn(ad.dates, extra_bps=commission_bps - 1.0)
    elif stamp == "current":
        sf = StampFnCurrent(ad.dates)
    else:
        sf = None
    k = max(int(rebalance), 1)
    phs = list(range(k)) if phases is None else list(phases)
    subs = [ST.simulate(W, ad, "open10", CAPITAL, rebalance=k, date_mask=date_mask, phase=ph,
                        cost_model=cm, record=record, stamp_fn=sf) for ph in phs]
    subs = [x for x in subs if x.get("rets") is not None and len(x["rets"]) > 5]
    out = ST._aggregate_phases(subs)
    out["phase_rets"] = pd.concat([x["rets"].rename(f"ph{i}") for i, x in enumerate(subs)], axis=1)
    if record:
        out["record"] = ST._merge_records([x["record"] for x in subs])
    return out


def bench(ad: LHAdapter, U: pd.DataFrame, rebalance: int = 5, date_mask=None) -> pd.Series:
    """B1 同频池等权(§7):同一决策池、同一计价价 V、k 相位平均、不计成本。"""
    from afactor import strategy as ST
    return ST.benchmark(ad, "V1000", "open10", date_mask=date_mask, rebalance=rebalance,
                        universe_override=U)["rets"]


# ================================================================ 统计
def nw_t(x: np.ndarray, q: int) -> float:
    from afactor.strategy import lr_variance_ratio
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30 or x.std() <= 0:
        return np.nan
    vr = lr_variance_ratio(x, q)
    t = x.mean() / x.std(ddof=1) * np.sqrt(n)
    return float(t / np.sqrt(vr)) if np.isfinite(vr) and vr > 0 else np.nan


def ann_geo(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float((1.0 + r).prod() ** (TD / len(r)) - 1.0)


def max_dd(r: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    nav = (1.0 + r.fillna(0.0)).cumprod()
    dd = nav / nav.cummax() - 1.0
    if len(dd) == 0:
        return np.nan, None, None
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()
    return float(dd.min()), peak, trough


def underwater(r: pd.Series) -> dict:
    nav = (1.0 + r.fillna(0.0)).cumprod().to_numpy()
    peak = np.maximum.accumulate(nav)
    below = nav < peak - 1e-12
    longest, cur, unrec = 0, 0, False
    for b in below:
        cur = cur + 1 if b else 0
        longest = max(longest, cur)
    unrec = bool(below[-1]) if len(below) else False
    # 当前(期末)水下段是否为最长段
    tail = 0
    for b in below[::-1]:
        if b:
            tail += 1
        else:
            break
    return dict(uw_longest_days=int(longest), uw_frac=float(below.mean()) if len(below) else np.nan,
                uw_unrecovered_at_end=unrec, uw_tail_days=int(tail))


def perf(r: pd.Series) -> dict:
    from afactor.strategy import lr_variance_ratio
    r = r.dropna()
    ann = ann_geo(r)
    vol = float(r.std() * np.sqrt(TD))
    vr = lr_variance_ratio(r.to_numpy(np.float64), 20)
    mdd, pk, tr = max_dd(r)
    out = dict(n_days=len(r), ann=ann, vol=vol, vol_adj=float(vol * np.sqrt(vr)) if np.isfinite(vr) else np.nan,
               mdd=mdd, mdd_peak=str(pk.date()) if pk is not None else "", mdd_trough=str(tr.date()) if tr is not None else "",
               calmar=ann / abs(mdd) if mdd < 0 else np.nan)
    out.update(underwater(r))
    return out


def rel_block(rs: pd.Series, rb: pd.Series, prefix: str = "") -> dict:
    a, b = rs.align(rb, join="inner")
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    d = a - b
    rel = (1.0 + a).cumprod() / (1.0 + b).cumprod()
    rdd = float((rel / rel.cummax() - 1.0).min()) if len(rel) else np.nan
    rtr = (rel / rel.cummax() - 1.0).idxmin() if len(rel) else None
    te = float(d.std() * np.sqrt(TD)) if len(d) > 1 else np.nan
    ex_ar = float(d.mean() * TD) if len(d) else np.nan
    return {f"{prefix}excess_geo": ann_geo(a) - ann_geo(b), f"{prefix}excess_arith": ex_ar,
            f"{prefix}nw_t5": nw_t(d.to_numpy(), 5), f"{prefix}nw_t10": nw_t(d.to_numpy(), 10),
            f"{prefix}nw_t20": nw_t(d.to_numpy(), 20), f"{prefix}te": te,
            f"{prefix}ir": ex_ar / te if te and te > 0 else np.nan,
            f"{prefix}rel_mdd": rdd, f"{prefix}rel_mdd_trough": str(rtr.date()) if rtr is not None else ""}


def yearly(rs: pd.Series, rb: pd.Series) -> pd.DataFrame:
    a, b = rs.align(rb, join="inner")
    g = a.index.year
    ys = (1 + a).groupby(g).prod() - 1
    yb = (1 + b).groupby(g).prod() - 1
    out = pd.DataFrame({"strategy": ys, "bench": yb})
    out["excess"] = out["strategy"] - out["bench"]
    return out


def year_t(ex: pd.Series) -> float:
    ex = ex.dropna()
    n = len(ex)
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(n)) if n > 2 and ex.std() > 0 else np.nan


def slice_eval(r: pd.Series, a: str, b: str) -> pd.Series:
    return r.loc[pd.Timestamp(a): pd.Timestamp(b)]


def window_rows(dates: pd.DatetimeIndex, a: str, b: str) -> np.ndarray:
    """窗口 [a, b] 的信号日行:a <= t < b(成交日 te = t+1 ∈ (a, b],即 a+1 开盘到 b+1 开盘)。"""
    ia = int(dates.searchsorted(pd.Timestamp(a)))
    ib = int(dates.searchsorted(pd.Timestamp(b)))
    return np.arange(ia, ib)


def v_window(dates: pd.DatetimeIndex, trough: str, n: int) -> tuple[str, str]:
    i = int(dates.searchsorted(pd.Timestamp(trough)))
    j = min(i + n, len(dates) - 1)
    return str(dates[i].date()), str(dates[j].date())
