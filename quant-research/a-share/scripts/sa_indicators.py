"""结构假设清单的观测指标(前缀 sa;报告 research/STRUCTURAL_ASSUMPTIONS.md)。

这是**描述与监测**,不是择时回测:
  - 指标只用 <= 2025-12-31 的数据(LongPanel / 第一步每日序列 / stock_basic 均按截止日截断);
  - 每个指标的"危险方向"与阈值规则(90/95 分位或 10/5 分位,按指标自身 2005-2025 历史)写在 SPEC 里,
    在看到任何指标读数之前定死;阈值不参考任何策略收益;
  - 策略收益衍生的序列(排雷滚动超额、跟踪误差、滚动 t、实际成本)只作描述,不参与状态判定;
  - 不计算任何"按指标调仓"的收益。

用法:
    PYTHONPATH=src python3 scripts/sa_indicators.py                # 全部(约 30 秒)
    PYTHONPATH=src python3 scripts/sa_indicators.py --no-fig       # 不出图
    PYTHONPATH=src python3 scripts/sa_indicators.py --freeze-spec  # 首次:把 SPEC/阈值/应对表的指纹写入
                                                                   # research/ledger/sa_prereg_manifest.json(已存在则拒绝覆盖)
    之后每次运行都对照该指纹,结果记在 sa_checks.csv item=prereg_fingerprint_match。

审查更正(2026-10-11,见报告末"审查记录"):
  - A3_hold_micro20_share 由 desc 改为 gate:原归类与下面 SPEC 注释里写明的 gate 入选条件不一致;
    原归类保留在 PRE_REVIEW_ROLE,只用于报告"按原归类"的敏感性(sa_current 的 status_pre_review 列、
    sa_trigger_history 的 *_pre_review 列);
  - 新增描述指标 A3_delist_forced_rate(只计名称含 ST/退 或退市前连续 20 日收盘 < 1 元的退市);
  - 新增审查核对行(sa_checks.csv item 以 rv_ 开头)与预注册指纹。

产出:
    research/ledger/sa_indicators.csv  指标 x 日期 长表(value / pct_full / pct_exp)
    research/ledger/sa_current.csv     截至 2025-12-31 的读数、历史分位、阈值与状态
    research/ledger/sa_events.csv      2008/2015/2018/2024 前后的读数
    research/ledger/sa_annual.csv      IPO / 退市 / 退市股在退市前一年的排雷归属
    research/ledger/sa_rules.csv       制度时间标注(项目代码内编码的 + 外部公开信息)
    research/ledger/sa_checks.csv      自检与交叉验证
    research/ledger/sa_trigger_history.csv  按实时(扩展窗)分位回放的月末状态与对应仓位上限档位(不算收益)
    reports/sa_indicators.png
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afactor.config import RESEARCH_END  # noqa: E402
from afactor import lh_engine as LE  # noqa: E402

LEDGER = ROOT / "research" / "ledger"
REPORTS = ROOT / "reports"
TD = 242
CUT = pd.Timestamp(RESEARCH_END)

# ============================================================== 规格(先写死,再计算)
# role: gate = 参与状态判定;desc = 只作描述。danger: high = 越高越危险;low = 越低越危险。
# 门槛:high → pct >= 95 警示、>= 90 关注;low → pct <= 5 警示、<= 10 关注(pct = 该指标自身全部历史中的分位)。
# 入选 gate 的条件:历史从 2006 年或更早开始、不由策略收益计算;用作另一 gate 指标的对照序列、同一序列的
# 单月分量、制度标注序列,只作描述。
# 注意:A2_ls_12m / A2_ls_24m 形式上不由策略收益计算,但与排雷相对 fp 的超额高度同步(月度相关见
# sa_checks.csv item=rv_corr_ls1m_vs_excess_fp_1m),实质上是排雷近 12/24 个月相对表现的监控;A2 不动仓位上限。
THR = {"high": (90.0, 95.0), "low": (10.0, 5.0)}   # (关注, 警示) 分位线
SPEC = [
    # id, assumption, role, danger, freq, unit, 中文名, 定义
    ("A1_turn_level", "A1", "gate", "low", "D", "%",
     "全A换手率水平", "全A(当日有成交、上市满6日)个股 20 日均换手率的截面中位数"),
    ("A1_turn_disp", "A1", "gate", "low", "D", "log",
     "决策池换手率分化度", "决策池(q92)内 20 日均换手率 log(80分位/20分位)"),
    ("A1_runup_spread", "A1", "gate", "low", "M", "log",
     "热门股过去一年涨幅差", "月末决策池内 高换手五分位 减 低换手五分位 的过去 250 日对数收益中位数之差"),
    ("A1_bp_spread", "A1", "desc", "low", "M", "log",
     "低/高换手估值差(B/P)", "月末决策池内 log(低换手五分位 B/P 中位数) - log(高换手五分位 B/P 中位数);2015-04 月末起(财报表只从 2015Q1 开始)"),
    ("A2_ls_12m", "A2", "gate", "low", "M", "log",
     "换手率五分位多空 12 个月", "月末按 20 日换手率分五组(决策池),低换手组 减 高换手组 下月买入持有对数收益(跳过 1 日),滚动 12 个月合计;毛收益"),
    ("A2_ls_24m", "A2", "gate", "low", "M", "log",
     "换手率五分位多空 24 个月", "同上,滚动 24 个月合计"),
    ("A2_ls_1m", "A2", "desc", "low", "M", "log", "换手率五分位多空 单月", "同上,单月"),
    ("A2_excess_b1_12m", "A2", "desc", "low", "D", "log",
     "排雷相对池等权 12 个月", "第一步主配置 O0 相对同频池等权 B1 的滚动 242 日对数超额(策略收益,只作描述)"),
    ("A2_excess_b1_24m", "A2", "desc", "low", "D", "log",
     "排雷相对池等权 24 个月", "同上,滚动 484 日"),
    ("A2_excess_fp_12m", "A2", "desc", "low", "D", "log",
     "排雷相对全池等权(fp)12 个月", "O0 相对同引擎全池等权 fp(付成本、可投资)的滚动 242 日对数超额(只作描述)"),
    ("A2_excess_fp_24m", "A2", "desc", "low", "D", "log",
     "排雷相对全池等权(fp)24 个月", "同上,滚动 484 日"),
    ("A3_delist_rate", "A3", "gate", "high", "A", "%",
     "年退市率", "stock_basic 当年退市数 / 年初在市数(沪深,不含北交所)"),
    ("A3_ipo_count", "A3", "gate", "high", "A", "只",
     "年 IPO 数", "stock_basic 当年上市数(沪深,不含北交所)"),
    ("A3_micro20_rel_12m", "A3", "gate", "low", "M", "log",
     "最小 20% 市值相对全A 12 个月", "月末按流通市值取全A最小 20%,下月买入持有,相对全A等权的对数收益,滚动 12 个月合计"),
    ("A3_delist_forced_rate", "A3", "desc", "high", "A", "%",
     "年强制退市率(描述)", "同年退市率,但只计 stock_basic 名称含 ST/退,或退市前最后 60 个交易日内有连续 20 日不复权收盘 < 1 元"
     "(面值退市)的退市;吸收合并、主动退市、代码变更等计入 other(sa_annual.csv)"),
    ("A3_hold_micro20_share", "A3", "gate", "high", "D", "比例",
     "排雷持仓中最小 20% 市值的占比", "O0 目标持仓(W>0)中属于全A流通市值最小 20% 的比例"),
    ("A3_pool_micro20_share", "A3", "desc", "high", "D", "比例",
     "决策池中最小 20% 市值的占比", "决策池中属于全A流通市值最小 20% 的比例(与上一行对照)"),
    ("A4_lowturn_rcorr", "A4", "gate", "high", "M", "相关",
     "低换手股残差平均相关", "月末低换手五分位(决策池)过去 60 日个股收益对全A等权回归残差的两两平均相关(只取 60 日全部有成交的股票)"),
    ("A4_pool_rcorr", "A4", "desc", "high", "M", "相关",
     "决策池残差平均相关", "同上,对全决策池(对照)"),
    ("A4_lowturn_corr_raw", "A4", "desc", "high", "M", "相关",
     "低换手股原始平均相关", "同 A4_lowturn_rcorr,但不剔除市场"),
    ("A4_micro20_amt_share", "A4", "gate", "high", "D", "比例",
     "最小 20% 市值成交占比", "全A流通市值最小 20% 股票的成交额 / 全A成交额,60 日平均"),
    ("A4_hold_turn_ratio", "A4", "gate", "high", "D", "倍",
     "排雷持仓/池 换手率比", "O0 目标持仓等权平均 20 日换手率 / 决策池等权平均"),
    ("A4_hold_vol_ratio", "A4", "gate", "high", "D", "倍",
     "排雷持仓/池 波动率比", "O0 目标持仓等权平均 60 日个股波动 / 决策池等权平均"),
    ("A4_te_60d", "A4", "desc", "high", "D", "年化",
     "排雷相对 B1 跟踪误差 60 日", "O0 - B1 日超额的滚动 60 日标准差 x sqrt(242)(策略收益衍生,只作描述)"),
    ("A5_pool_halfspread", "A5", "gate", "high", "D", "bp",
     "决策池一跳半价差中位数", "决策池内 0.5 x 0.01 元 / 不复权收盘(截至 t-1)的中位数,即成本模型的价差下界"),
    ("A5_cost_12m", "A5", "desc", "high", "D", "%NAV",
     "排雷实际成本 12 个月", "O0 逐日成本(佣金 + 史实印花税 + 一跳价差 + 平方根冲击)滚动 242 日合计(只作描述)"),
    ("A5_stamp_sell_bps", "A5", "desc", "high", "D", "bp",
     "卖方印花税", "史实分段(lh_engine.STAMP_HIST)"),
    ("A5_lim20_share", "A5", "desc", "high", "D", "比例",
     "决策池 20% 涨跌幅股票占比", "决策池中涨跌幅限制 >= 15% 的股票比例(科创板 2019-07、创业板 2020-08-24 起)"),
    ("A6_excess_t_5y", "A6", "desc", "low", "M", "t",
     "排雷相对 fp 滚动 5 年 NW t", "O0 - fp 日超额,滚动 1210 日 Newey-West t(q=5)(只作描述)"),
]
SPEC_DF = pd.DataFrame(SPEC, columns=["indicator", "assumption", "role", "danger", "freq", "unit", "name_cn", "definition"])
# 审查前的归类(只作敏感性;A3_delist_forced_rate 审查前不存在,按 desc 处理,不影响任何状态)
PRE_REVIEW_ROLE = {"A3_hold_micro20_share": "desc"}
SHARED = {"A6": ["A3_delist_rate"]}          # A6 引用 A3 的退市率(不重复计数)
STATIC_FLOOR = {"A6": ("关注", "既有证据:真样本外 2006-2019 相对 B1 NW t 1.51;COMP_v5 截距 +2.06pp 无长历史证据")}
MIN_HIST = {"D": 3 * TD, "M": 36, "A": 5}    # 扩展窗分位的最少历史

EVENTS = [  # 事件, 起, 止(预注册窗口,lh_step1_windows / STRESS_TEST W1)
    ("BEAR08", "2007-10-16", "2008-11-04"),
    ("BEAR15", "2015-06-12", "2016-01-28"),
    ("BEAR18", "2018-01-24", "2019-01-03"),
    ("BEAR24", "2023-05-08", "2024-02-05"),
    ("DMA24_W1", "2024-01-02", "2024-02-07"),
]

RULES = [  # date, category, event, source
    ("2005-01-24", "印花税", "双边 2‰ → 1‰", "项目代码 lh_engine.STAMP_HIST"),
    ("2007-05-30", "印花税", "双边 1‰ → 3‰", "项目代码 lh_engine.STAMP_HIST"),
    ("2008-04-24", "印花税", "双边 3‰ → 1‰", "项目代码 lh_engine.STAMP_HIST"),
    ("2008-09-19", "印花税", "改为卖方单边 1‰", "项目代码 lh_engine.STAMP_HIST"),
    ("2023-08-28", "印花税", "卖方 1‰ → 0.5‰", "项目代码 lh_engine.STAMP_HIST / config"),
    ("2014-01-01", "涨跌停", "新股首日 44% 上限(面板规则 IPO_44_START)", "项目代码 LongPanel _meta.json rules"),
    ("2019-07-22", "涨跌停/注册制", "科创板开市,20% 涨跌幅,注册制", "外部公开信息(面板按 lim_up_rate 记录)"),
    ("2020-08-24", "涨跌停/注册制", "创业板注册制首批,20% 涨跌幅", "项目代码 _meta.json GEM_20PCT_START"),
    ("2023-04-10", "注册制", "主板注册制首批上市(新股前 5 日无涨跌幅)", "项目代码 _meta.json MAIN_REG_START"),
    ("2025-07-07", "涨跌停", "主板 ST 涨跌幅 5% → 10%", "项目代码 _meta.json ST_10PCT_START"),
    ("2010-04-16", "衍生品", "沪深300 股指期货 IF 上市", "项目 LH_FUTURES_BASIS / lh_step2"),
    ("2015-04-16", "衍生品", "中证500 股指期货 IC 上市", "项目 LH_FUTURES_BASIS / lh_step2"),
    ("2022-07-22", "衍生品", "中证1000 股指期货 IM 上市", "项目 LH_FUTURES_BASIS / lh_step2"),
    ("2020-12-31", "退市", "沪深交易所发布退市新规(面值、财务类指标收紧)", "外部公开信息(未在项目数据中核验)"),
    ("2024-04-12", "退市/监管", "新'国九条'与证监会严格执行退市制度意见", "外部公开信息(STRESS_TEST W2 窗口起点)"),
    ("2024-04-30", "退市", "沪深交易所发布修订后的股票上市规则(退市新规)", "外部公开信息(未在项目数据中核验)"),
    ("2024-05-15", "量化监管", "证监会发布程序化交易管理规定(试行),2024-10-08 施行", "外部公开信息(未在项目数据中核验)"),
    ("1995-01-01", "交易制度", "T+1 交收制度(样本期内未变)", "外部公开信息"),
]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================== 工具
def pct_full(v: np.ndarray) -> np.ndarray:
    """每个值在该指标全部历史中的分位(0-100,kind='mean')。"""
    v = np.asarray(v, np.float64)
    ok = np.isfinite(v)
    s = np.sort(v[ok])
    n = len(s)
    out = np.full(len(v), np.nan)
    if n == 0:
        return out
    lo = np.searchsorted(s, v[ok], "left")
    hi = np.searchsorted(s, v[ok], "right")
    out[ok] = (lo + hi) / 2.0 / n * 100.0
    return out


def pct_expanding(v: np.ndarray, min_n: int) -> np.ndarray:
    """扩展窗分位:只用 <= t 的历史(含自身),历史不足 min_n 为 NaN —— 实时能看到的读数。"""
    v = np.asarray(v, np.float64)
    out = np.full(len(v), np.nan)
    hist: list[float] = []
    for i, x in enumerate(v):
        if not np.isfinite(x):
            continue
        bisect.insort(hist, x)
        n = len(hist)
        if n >= min_n:
            lo = bisect.bisect_left(hist, x)
            hi = bisect.bisect_right(hist, x)
            out[i] = (lo + hi) / 2.0 / n * 100.0
    return out


def row_nanquantiles(x: np.ndarray, mask: np.ndarray, qs) -> np.ndarray:
    y = np.where(mask, x, np.nan).astype(np.float64)
    out = np.full((len(qs), x.shape[0]), np.nan)
    ok = np.isfinite(y).sum(axis=1) >= 20
    with np.errstate(all="ignore"):
        out[:, ok] = np.nanquantile(y[ok], qs, axis=1)
    return out


def row_masked_mean(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask & np.isfinite(x)
    s = np.where(m, x, 0.0).sum(axis=1, dtype=np.float64)
    n = m.sum(axis=1)
    with np.errstate(all="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def quintile(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """一行内按 x 升序分五组(1..5),m 之外为 0。"""
    q = np.zeros(len(x), np.int8)
    ok = m & np.isfinite(x)
    idx = np.flatnonzero(ok)
    if len(idx) < 50:
        return q
    r = pd.Series(x[idx]).rank(method="first").to_numpy()
    q[idx] = np.minimum(5, np.ceil(r / len(idx) * 5)).astype(np.int8)
    return q


def month_end_idx(dates: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(np.arange(len(dates)), index=dates)
    return s.groupby([dates.year, dates.month]).max().to_numpy()


def avg_pair_corr(R: np.ndarray, m: np.ndarray | None) -> float:
    """R: T x n(无缺失)。m 给出时先对 m 回归取残差。返回两两相关的平均(对角线除外)。"""
    T, n = R.shape
    if n < 30:
        return np.nan
    X = R - R.mean(axis=0)
    if m is not None:
        mc = m - m.mean()
        b = (X * mc[:, None]).sum(axis=0) / float((mc * mc).sum())   # 不用 matmul:macOS Accelerate 会报假的浮点警告
        X = X - np.outer(mc, b)
    sd = X.std(axis=0)
    keep = sd > 1e-12
    X = X[:, keep] / sd[keep]
    n = X.shape[1]
    if n < 30:
        return np.nan
    s = X.sum(axis=1)
    tot = float((s * s).sum()) / T
    return float((tot - n) / (n * (n - 1)))


# ============================================================== 数据
def load():
    from afactor.lh_panel import LongPanel
    t0 = time.time()
    X = LE.load_inputs()
    p = LongPanel()
    dates, codes = p.dates, p.codes
    assert list(map(str, codes)) == list(map(str, X.codes)), "LongPanel 与第一步输入的股票顺序不一致"
    assert (dates == X.dates).all(), "LongPanel 与第一步输入的日期不一致"
    assert dates.max() <= CUT
    f32 = lambda df: df.to_numpy(np.float32)
    D = dict(dates=dates, codes=codes, p=p, X=X)
    trade = p.raw("trade").to_numpy(bool)
    amount = f32(p.raw("amount"))
    close = f32(p.raw("close_px"))
    valid = trade & (amount > 0) & np.isfinite(close)
    dl = np.cumsum(valid, axis=0)
    D["valid"] = valid
    D["allA"] = valid & (dl >= 6)
    D["amount"] = np.where(valid, amount, 0.0).astype(np.float32)
    ret = f32(p.raw("ret"))
    D["ret"] = ret
    D["listed"] = p.raw("listed").to_numpy(bool)
    D["float_mv"] = f32(p["float_mv"])
    D["close_px"] = close
    D["close_raw"] = f32(p.raw("close_raw"))
    D["lim_up_rate"] = f32(p.raw("lim_up_rate"))
    D["pool"] = np.asarray(X.arr("pool_q92"), bool)
    D["turn20"] = -np.asarray(X.arr("score_S1"), np.float32)
    D["hs"] = np.asarray(X.arr("cost_hs"), np.float32)
    D["ew"] = X.ew_allA_ret.to_numpy(np.float64)
    log(f"数据载入 {time.time() - t0:.0f}s:T={len(dates)} N={len(codes)}")
    return D


# ============================================================== 指标
def compute(D) -> tuple[dict, dict, pd.DataFrame, list]:
    dates = D["dates"]
    T, N = len(dates), len(D["codes"])
    S: dict[str, pd.Series] = {}
    extra: dict = {}
    checks: list = []
    pool, allA, turn20 = D["pool"], D["allA"], D["turn20"]
    ME = month_end_idx(dates)
    ME = ME[ME >= 60]

    # ---------- A1 换手率水平、分化度(日频)----------
    q = row_nanquantiles(turn20, allA, [0.5])
    S["A1_turn_level"] = pd.Series(q[0], index=dates)
    q = row_nanquantiles(turn20, pool, [0.2, 0.8])
    with np.errstate(all="ignore"):
        S["A1_turn_disp"] = pd.Series(np.log(q[1] / q[0]), index=dates)
    log("A1 日频完成")

    # ---------- 共用:对数累计持有收益、月末分组 ----------
    ret = D["ret"]
    lr = np.log1p(np.nan_to_num(ret, nan=0.0).clip(-0.99, None)).astype(np.float64)
    logcum = np.cumsum(lr, axis=0)
    del lr
    fmv = D["float_mv"]
    me_dates = dates[ME]

    # 月末持有期收益:formation t = ME[k],持有 t+2 .. ME[k+1](跳过 1 日),买入持有,退市后按 0 收益冻结
    ls1m, ls_q1, ls_q5, mic_rel = [], [], [], []
    runup, rows_out = [], []
    rq = {}
    for k in range(len(ME) - 1):
        t, t1 = ME[k], ME[k + 1]
        if t + 1 >= t1:
            continue
        hold = np.exp(logcum[t1] - logcum[t + 1]) - 1.0
        qt = quintile(turn20[t], pool[t])
        rq[t] = qt
        r1 = np.nanmean(hold[qt == 1]) if (qt == 1).any() else np.nan
        r5 = np.nanmean(hold[qt == 5]) if (qt == 5).any() else np.nan
        ls_q1.append(r1); ls_q5.append(r5)
        ls1m.append(np.log1p(r1) - np.log1p(r5))
        # 最小 20% 流通市值(全A)
        m = allA[t] & np.isfinite(fmv[t]) & (fmv[t] > 0)
        if m.sum() > 100:
            thr = np.nanquantile(fmv[t][m], 0.2)
            mic = m & (fmv[t] <= thr)
            mic_rel.append(np.log1p(np.nanmean(hold[mic])) - np.log1p(np.nanmean(hold[m])))
        else:
            mic_rel.append(np.nan)
        rows_out.append(t1)
    idx_hold = dates[np.array(rows_out)]
    s1m = pd.Series(ls1m, index=idx_hold)
    S["A2_ls_1m"] = s1m
    S["A2_ls_12m"] = s1m.rolling(12, min_periods=12).sum()
    S["A2_ls_24m"] = s1m.rolling(24, min_periods=24).sum()
    mr = pd.Series(mic_rel, index=idx_hold)
    S["A3_micro20_rel_12m"] = mr.rolling(12, min_periods=12).sum()
    extra["ls_legs"] = pd.DataFrame({"q1_lowturn": ls_q1, "q5_highturn": ls_q5, "ls_log": ls1m}, index=idx_hold)
    log("A2 / A3 月频收益完成")

    # 热门股过去一年涨幅差(月末,只用 <= t)
    listed = D["listed"]
    for t in ME:
        if t < 250:
            runup.append(np.nan)
            continue
        qt = rq.get(t)
        if qt is None:
            qt = quintile(turn20[t], pool[t])
            rq[t] = qt
        r250 = logcum[t] - logcum[t - 250]
        ok = listed[t - 250] & np.isfinite(r250)
        a = r250[(qt == 5) & ok]; b = r250[(qt == 1) & ok]
        runup.append(np.median(a) - np.median(b) if len(a) > 30 and len(b) > 30 else np.nan)
    S["A1_runup_spread"] = pd.Series(runup, index=me_dates)

    # ---------- A4 低换手股残差平均相关(月末)----------
    ew = D["ew"]
    valid = D["valid"]
    rc_low, rc_pool, rc_raw = [], [], []
    for t in ME:
        rows = slice(t - 59, t + 1)
        full = valid[rows].all(axis=0) & np.isfinite(ret[rows]).all(axis=0)
        qt = rq.get(t)
        if qt is None:
            qt = quintile(turn20[t], pool[t]); rq[t] = qt
        m = ew[rows]
        low = (qt == 1) & full
        pl = pool[t] & full
        R = ret[rows][:, low].astype(np.float64)
        rc_low.append(avg_pair_corr(R, m))
        rc_raw.append(avg_pair_corr(R, None))
        Rp = ret[rows][:, pl].astype(np.float64)
        rc_pool.append(avg_pair_corr(Rp, m))
    S["A4_lowturn_rcorr"] = pd.Series(rc_low, index=me_dates)
    S["A4_pool_rcorr"] = pd.Series(rc_pool, index=me_dates)
    S["A4_lowturn_corr_raw"] = pd.Series(rc_raw, index=me_dates)
    log("A1 涨幅差 / A4 相关完成")

    # ---------- A4 / A3 最小 20% 市值:成交占比、持仓占比(日频)----------
    amt = D["amount"]
    m_all = allA & np.isfinite(fmv) & (fmv > 0)
    thr = row_nanquantiles(fmv, m_all, [0.2])[0]
    micro = m_all & (fmv <= thr[:, None])
    tot = np.where(m_all, amt, 0).sum(axis=1, dtype=np.float64)
    mic_amt = np.where(micro, amt, 0).sum(axis=1, dtype=np.float64)
    with np.errstate(all="ignore"):
        share = np.where(tot > 0, mic_amt / tot, np.nan)
    S["A4_micro20_amt_share"] = pd.Series(share, index=dates).rolling(60, min_periods=40).mean()

    log("计算 O0 目标持仓 W(LE.weights)")
    W = LE.weights(D["X"], "S1")
    H = np.asarray(W) > 0
    del W
    nH = H.sum(axis=1)
    with np.errstate(all="ignore"):
        S["A3_hold_micro20_share"] = pd.Series(np.where(nH > 0, (H & micro).sum(1) / np.maximum(nH, 1), np.nan), index=dates)
        nP = pool.sum(axis=1)
        S["A3_pool_micro20_share"] = pd.Series(np.where(nP > 0, (pool & micro).sum(1) / np.maximum(nP, 1), np.nan), index=dates)
    ht = row_masked_mean(turn20, H); pt = row_masked_mean(turn20, pool)
    with np.errstate(all="ignore"):
        S["A4_hold_turn_ratio"] = pd.Series(ht / pt, index=dates)
    rv = np.where(valid, ret, np.nan).astype(np.float32)
    vol60 = pd.DataFrame(rv).rolling(60, min_periods=40).std().to_numpy(np.float32)
    del rv
    hv = row_masked_mean(vol60, H); pv = row_masked_mean(vol60, pool)
    with np.errstate(all="ignore"):
        S["A4_hold_vol_ratio"] = pd.Series(hv / pv, index=dates)
    del vol60
    # ST 占比(只在 2016-08-09 起有完整 ST 来源)
    is_st = np.asarray(D["X"].arr("is_st"), bool)
    with np.errstate(all="ignore"):
        st_h = np.where(nH > 0, (H & is_st).sum(1) / np.maximum(nH, 1), np.nan)
        st_p = np.where(nP > 0, (pool & is_st).sum(1) / np.maximum(nP, 1), np.nan)
    extra["st_share"] = pd.DataFrame({"hold_st_share": st_h, "pool_st_share": st_p}, index=dates).loc["2016-08-09":]
    extra["H"] = H
    log("A3 / A4 日频完成")

    # ---------- A5 价差、涨跌幅、印花税 ----------
    hs = D["hs"]
    q = row_nanquantiles(hs, pool, [0.5])[0]
    S["A5_pool_halfspread"] = pd.Series(q * 1e4, index=dates)
    lim = D["lim_up_rate"]
    with np.errstate(all="ignore"):
        S["A5_lim20_share"] = pd.Series(np.where(nP > 0, (pool & (lim >= 0.15)).sum(1) / np.maximum(nP, 1), np.nan), index=dates)
    _, sell = LE.stamp_arrays(dates)
    S["A5_stamp_sell_bps"] = pd.Series(sell * 1e4, index=dates)

    # ---------- B/P 估值差(2015-04 月末起)----------
    try:
        S["A1_bp_spread"], cov = bp_spread(D, ME, rq)
        extra["bp_cov"] = cov
    except Exception as e:  # 财报表缺失时不阻断其余指标
        log(f"B/P 估值差失败:{e!r}")
        checks.append(("bp_spread_error", repr(e), "", ""))
    log("A5 / B/P 完成")

    # ---------- 策略收益衍生(只作描述)----------
    d1 = pd.read_parquet(ROOT / "cache" / "_tmp" / "lh_step1_daily.parquet")
    assert d1.index.max() <= CUT
    d1 = d1.reindex(dates)
    ex_b1 = np.log1p(d1["strat_main"]) - np.log1p(d1["bench_b1"])
    ex_fp = np.log1p(d1["strat_main"]) - np.log1p(d1["strat_fullpool_ew"])
    ev = d1["strat_main"].notna()
    ex_b1 = ex_b1.where(ev); ex_fp = ex_fp.where(ev)
    S["A2_excess_b1_12m"] = ex_b1.rolling(TD, min_periods=TD - 5).sum()
    S["A2_excess_b1_24m"] = ex_b1.rolling(2 * TD, min_periods=2 * TD - 5).sum()
    S["A2_excess_fp_12m"] = ex_fp.rolling(TD, min_periods=TD - 5).sum()
    S["A2_excess_fp_24m"] = ex_fp.rolling(2 * TD, min_periods=2 * TD - 5).sum()
    S["A4_te_60d"] = (d1["strat_main"] - d1["bench_b1"]).rolling(60, min_periods=50).std() * np.sqrt(TD)
    S["A5_cost_12m"] = d1["main_cost_total"].rolling(TD, min_periods=TD - 5).sum() * 100.0
    arith_fp = (d1["strat_main"] - d1["strat_fullpool_ew"]).to_numpy()
    tt = []
    for t in ME:
        lo = t - 5 * TD + 1
        tt.append(LE.nw_t(arith_fp[lo:t + 1], 5) if lo >= 0 and np.isfinite(arith_fp[lo:t + 1]).sum() >= 5 * TD - 20 else np.nan)
    S["A6_excess_t_5y"] = pd.Series(tt, index=me_dates)
    extra["d1"] = d1
    extra["rq"] = rq
    extra["ME"] = ME

    # 只保留评价期 2005-2025 的有效值
    for k in list(S):
        S[k] = S[k].loc[: CUT].astype(np.float64)
    return S, extra, d1, checks


def bp_spread(D, ME, rq):
    """B/P = 归母权益 / (报告期总股本 x 公告日不复权收盘 x 后复权价格比);价格比把公告后的送转、分红从股本口径里抵掉。"""
    from afactor import fundamental as FU
    dates, codes = D["dates"], D["codes"]
    rep = FU.build_report_level().copy()
    rep["avail_ord"] = (rep["avail"] - pd.Timestamp("1970-01-01")).dt.days.astype(np.float64)
    me = ME[dates[ME] >= pd.Timestamp("2015-04-30")]
    md = dates[me]
    eq = FU._pit_daily(rep, "total_hldr_eqy_exc_min_int", md, codes).to_numpy(np.float64)
    sh = FU._pit_daily(rep, "total_share", md, codes).to_numpy(np.float64)
    av = FU._pit_daily(rep, "avail_ord", md, codes).to_numpy(np.float64)
    cpx = pd.DataFrame(np.where(D["valid"], D["close_px"], np.nan)).ffill().to_numpy(np.float64)
    craw = pd.DataFrame(np.where(D["valid"], D["close_raw"], np.nan)).ffill().to_numpy(np.float64)
    dnum = ((dates - pd.Timestamp("1970-01-01")).days).to_numpy()
    out, cov = [], []
    for i, t in enumerate(me):
        a = av[i]
        ok = np.isfinite(a) & np.isfinite(eq[i]) & np.isfinite(sh[i]) & (sh[i] > 0) & (eq[i] > 0)
        ok &= (dnum[t] - np.where(np.isfinite(a), a, 0)) <= 400            # 报告不超过 400 天
        k0 = np.searchsorted(dnum, np.where(ok, a, 0), "left").clip(0, t)
        cols = np.arange(len(codes))
        p0 = cpx[k0, cols]; r0 = craw[k0, cols]; pt = cpx[t]
        with np.errstate(all="ignore"):
            mv = sh[i] * r0 * (pt / p0)
            bp = eq[i] / mv
        ok &= np.isfinite(bp) & (bp > 0)
        qt = rq.get(t)
        if qt is None:
            qt = quintile(D["turn20"][t], D["pool"][t]); rq[t] = qt
        a1 = bp[(qt == 1) & ok]; a5 = bp[(qt == 5) & ok]
        n_pool = int(D["pool"][t].sum())
        cov.append((dates[t], float((D["pool"][t] & ok).sum() / max(n_pool, 1))))
        out.append(np.log(np.median(a1)) - np.log(np.median(a5)) if len(a1) > 50 and len(a5) > 50 else np.nan)
    return pd.Series(out, index=dates[me]), pd.DataFrame(cov, columns=["date", "bp_coverage_pool"]).set_index("date")


# ============================================================== 年度(IPO / 退市)
def annual(D, extra) -> pd.DataFrame:
    from afactor import newdata as ND
    sb = ND._stock_basic()                       # 读入即按截止日截断(上市日 > 截止日丢弃,之后的退市日置空)
    sb = sb[sb["ts_code"].str[-2:].isin(["SH", "SZ"])].drop_duplicates("ts_code")
    ld = pd.to_datetime(sb["list_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    dd = pd.to_datetime(sb["delist_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    assert (ld.dropna() <= CUT).all() and (dd.dropna() <= CUT).all()
    # 退市类型(描述):名称含 ST/退 → forced_name;否则退市前最后 60 个有效交易日内有连续 >= 20 日
    # 不复权收盘 < 1 元 → forced_face(面值退市);其余(吸收合并、主动退市、代码变更等)→ other。
    # name 是下载时快照,但已退市股票的名称停在退市时,不含截止日之后的信息。
    col = {c: j for j, c in enumerate(D["codes"])}
    craw, valid = D["close_raw"], D["valid"]

    def dl_kind(code: str, name: str) -> str:
        if isinstance(name, str) and ("ST" in name.upper() or "退" in name):
            return "forced_name"
        j = col.get(code)
        if j is None:
            return "other"
        rows_ = np.flatnonzero(valid[:, j])[-60:]
        below = craw[rows_, j] < 1.0
        run = best = 0
        for b in below:
            run = run + 1 if b else 0
            best = max(best, run)
        return "forced_face" if best >= 20 else "other"

    kind = pd.Series("", index=sb.index)
    is_dl = dd.notna()
    kind[is_dl] = [dl_kind(c, n) for c, n in zip(sb.loc[is_dl, "ts_code"], sb.loc[is_dl, "name"])]
    rows = []
    for y in range(2005, 2026):
        y0 = pd.Timestamp(f"{y}-01-01"); y1 = pd.Timestamp(f"{y}-12-31")
        n_start = int(((ld < y0) & (dd.isna() | (dd >= y0))).sum())
        ipo = int(((ld >= y0) & (ld <= y1)).sum())
        iny = (dd >= y0) & (dd <= y1)
        dl = int(iny.sum())
        k = kind[iny]
        n_forced = int(k.isin(["forced_name", "forced_face"]).sum())
        oth = sb.loc[iny & (kind == "other"), ["ts_code", "name"]]
        rows.append(dict(year=y, listed_at_start=n_start, ipo=ipo, delist=dl,
                         delist_rate_pct=100.0 * dl / n_start if n_start else np.nan,
                         delist_forced_name=int((k == "forced_name").sum()),
                         delist_forced_face=int((k == "forced_face").sum()),
                         delist_other=int((k == "other").sum()),
                         delist_forced_rate_pct=100.0 * n_forced / n_start if n_start else np.nan,
                         delist_other_list=";".join(f"{c}{n}" for c, n in zip(oth["ts_code"], oth["name"]))))
    A = pd.DataFrame(rows).set_index("year")
    # 退市股在最后交易日前 242 个交易日处于排雷的哪一侧(LongPanel,含 293 只 2005-2025 退市股)
    p = D["p"]; info = p.code_info; dates = D["dates"]
    out = info["outDate"].fillna("").astype(str)
    is_del = (out != "") & (out <= RESEARCH_END)
    H = extra["H"]; pool = D["pool"]
    cnt = {y: dict(kept=0, excluded=0, not_in_pool=0, too_early=0, base_sum=0.0, base_var=0.0) for y in A.index}
    nP = pool.sum(axis=1).astype(np.float64)
    nHP = (H & pool).sum(axis=1).astype(np.float64)
    for c in info.index[is_del.to_numpy()]:
        lt = pd.Timestamp(info.loc[c, "last_trade"])
        y = pd.Timestamp(out.loc[c]).year
        if y not in cnt:
            continue
        if pd.isna(lt):                          # 面板内从未成交(2005 年初即退市的 2 只)
            cnt[y]["too_early"] += 1
            continue
        k = int(np.searchsorted(dates.values, np.datetime64(lt))) - TD
        j = col[c]
        if k < 25:
            cnt[y]["too_early"] += 1
        elif not pool[k, j]:
            cnt[y]["not_in_pool"] += 1
        else:
            # 时间匹配的对照:同一天 (持仓 ∩ 池) / 池,即"随机抽一只池内股票落在保留侧"的概率
            b = nHP[k] / nP[k]
            cnt[y]["base_sum"] += b
            cnt[y]["base_var"] += b * (1.0 - b)
            if H[k, j]:
                cnt[y]["kept"] += 1
            else:
                cnt[y]["excluded"] += 1
    C = pd.DataFrame(cnt).T.add_prefix("delisted_1y_before_")
    A = A.join(C)
    A["delisted_panel_total"] = C[[f"delisted_1y_before_{x}" for x in ("kept", "excluded", "not_in_pool", "too_early")]].sum(axis=1)
    return A


# ============================================================== 汇总
def summarize(S: dict, A: pd.DataFrame):
    # 年度指标并入 S
    S["A3_delist_rate"] = pd.Series(A["delist_rate_pct"].to_numpy(), index=pd.to_datetime([f"{y}-12-31" for y in A.index]))
    S["A3_ipo_count"] = pd.Series(A["ipo"].to_numpy(np.float64), index=pd.to_datetime([f"{y}-12-31" for y in A.index]))
    S["A3_delist_forced_rate"] = pd.Series(A["delist_forced_rate_pct"].to_numpy(), index=pd.to_datetime([f"{y}-12-31" for y in A.index]))
    long_rows, cur_rows = [], []
    spec = SPEC_DF.set_index("indicator")
    for ind in SPEC_DF["indicator"]:
        s = S.get(ind)
        if s is None:
            continue
        s = s.dropna()
        s = s[s.index >= pd.Timestamp("2005-01-01")]
        if s.empty:
            continue
        sp = spec.loc[ind]
        pf = pct_full(s.to_numpy())
        pe = pct_expanding(s.to_numpy(), MIN_HIST[sp["freq"]])
        long_rows.append(pd.DataFrame({"indicator": ind, "assumption": sp["assumption"], "role": sp["role"],
                                       "freq": sp["freq"], "date": s.index, "value": s.to_numpy(),
                                       "pct_full": pf, "pct_exp": pe}))
        v = s.to_numpy()
        q05, q10, q90, q95 = np.quantile(v, [0.05, 0.10, 0.90, 0.95])
        last_p = pf[-1]
        w_, a_ = THR[sp["danger"]]
        if sp["danger"] == "high":
            rule_st = "警示" if last_p >= a_ else ("关注" if last_p >= w_ else "正常")
            thr_watch, thr_alarm = q90, q95
        else:
            rule_st = "警示" if last_p <= a_ else ("关注" if last_p <= w_ else "正常")
            thr_watch, thr_alarm = q10, q05
        st = rule_st if sp["role"] == "gate" else "描述"
        cur_rows.append(dict(assumption=sp["assumption"], indicator=ind, name_cn=sp["name_cn"], role=sp["role"],
                             role_pre_review=PRE_REVIEW_ROLE.get(ind, sp["role"]),
                             danger=sp["danger"], freq=sp["freq"], unit=sp["unit"],
                             hist_start=s.index[0].date(), hist_end=s.index[-1].date(), n_obs=len(s),
                             last_date=s.index[-1].date(), value=v[-1], pct_full=last_p, pct_exp=pe[-1],
                             hist_min=v.min(), hist_median=np.median(v), hist_max=v.max(),
                             q05=q05, q10=q10, q90=q90, q95=q95,
                             thr_watch=thr_watch, thr_alarm=thr_alarm, status=st, rule_status=rule_st,
                             definition=sp["definition"]))
    L = pd.concat(long_rows, ignore_index=True)
    C = pd.DataFrame(cur_rows)
    # 假设层状态 = gate 指标中最差者;A6 引用 A3 的退市率 + 静态下限。
    # status 按现行(审查更正后)归类;status_pre_review 按审查前归类(敏感性)。
    order = {"正常": 0, "关注": 1, "警示": 2}
    Ci = C.set_index("indicator")

    def assumption_status(a: str, role_col: str):
        inds = list(C[(C.assumption == a) & (C[role_col] == "gate")].indicator) + SHARED.get(a, [])
        sts = [Ci.loc[i, "rule_status"] for i in inds]
        worst = max(sts, key=lambda x: order[x]) if sts else "正常"
        reason = "; ".join(f"{i}={Ci.loc[i, 'rule_status']}(p{Ci.loc[i, 'pct_full']:.0f})" for i in inds)
        if a in STATIC_FLOOR and order[STATIC_FLOOR[a][0]] > order[worst]:
            worst = STATIC_FLOOR[a][0]
            reason = (reason + "; " if reason else "") + "静态下限:" + STATIC_FLOOR[a][1]
        return worst, reason

    rows = []
    for a in ["A1", "A2", "A3", "A4", "A5", "A6"]:
        worst, reason = assumption_status(a, "role")
        worst0, _ = assumption_status(a, "role_pre_review")
        rows.append(dict(assumption=a, indicator="__assumption_status__", name_cn="假设层状态", role="summary",
                         status=worst, status_pre_review=worst0, definition=reason))
    C = pd.concat([C, pd.DataFrame(rows)], ignore_index=True)
    return L, C


def events(L: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for ev, a, b in EVENTS:
        ia = int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(a))))
        pre = dates[max(ia - 126, 0)]
        pts = [("pre6m", pre), ("start", pd.Timestamp(a)), ("end", pd.Timestamp(b))]
        for ind, g in L.groupby("indicator", sort=False):
            g = g.set_index("date").sort_index()
            for pt, d in pts:
                h = g.loc[:d]
                if h.empty:
                    rows.append(dict(event=ev, indicator=ind, point=pt, ref_date=d.date(), asof=None,
                                     value=np.nan, pct_full=np.nan, pct_exp=np.nan))
                    continue
                r = h.iloc[-1]
                rows.append(dict(event=ev, indicator=ind, point=pt, ref_date=d.date(), asof=h.index[-1].date(),
                                 value=r["value"], pct_full=r["pct_full"], pct_exp=r["pct_exp"]))
            # 窗口内及其后 3 个月(63 个交易日)内,危险方向的极值(描述用)
            ib = int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(b))))
            hi_d = dates[min(ib + 63, len(dates) - 1)]
            w = g.loc[pd.Timestamp(a): hi_d]
            if len(w):
                dn = SPEC_DF.set_index("indicator").loc[ind, "danger"]
                k = w["value"].idxmax() if dn == "high" else w["value"].idxmin()
                r = w.loc[k]
                rows.append(dict(event=ev, indicator=ind, point="win_extreme", ref_date=f"{a}~{hi_d.date()}",
                                 asof=k.date(), value=r["value"], pct_full=r["pct_full"], pct_exp=r["pct_exp"]))
    return pd.DataFrame(rows)


def breach_spells(L: pd.DataFrame) -> pd.DataFrame:
    """gate 指标按"实时"扩展窗分位越过 95/5 线的连续区段(描述用,不做收益检验)。"""
    spec = SPEC_DF.set_index("indicator")
    rows = []
    for ind, g in L[L.role == "gate"].groupby("indicator", sort=False):
        dn = spec.loc[ind, "danger"]
        pe = g["pct_exp"].to_numpy()
        hit = (pe >= THR["high"][1]) if dn == "high" else (pe <= THR["low"][1])
        hit = np.where(np.isfinite(pe), hit, False)
        d = g["date"].to_numpy()
        i = 0
        while i < len(hit):
            if hit[i]:
                j = i
                while j + 1 < len(hit) and hit[j + 1]:
                    j += 1
                rows.append(dict(indicator=ind, start=pd.Timestamp(d[i]).date(), end=pd.Timestamp(d[j]).date(),
                                 n_obs=j - i + 1, freq=spec.loc[ind, "freq"]))
                i = j + 1
            else:
                i += 1
    return pd.DataFrame(rows)


TAIL_ASSUMPTIONS = ("A3", "A4", "A5")      # 破坏时主要威胁绝对/尾部风险 → 应对是下调仓位上限
CAP_TIERS = {0: 0.50, 1: 0.40, 2: 0.30}    # 处于"警示"的 A3/A4/A5 假设个数 → 仓位上限(lh_step2 静态网格的相邻档)
RESPONSES = [  # 报告 §1.4 的应对表(纳入预注册指纹)
    ("任一 gate 指标进入关注", "复核:重跑本脚本,对照该假设的证据账本,记录结论;仓位不变"),
    ("A1/A2/A6 进入警示", "复核,仓位上限不变;A2 两条同时警示时另评估排雷相对池等权的超额是否仍存在;参数改动另立预注册"),
    ("A3/A4/A5 中一条进入警示", "仓位上限 50% -> 40%,同时复核;下一个月末不再警示即恢复 50%;年度指标等下一个年度读数"),
    ("A3/A4/A5 中两条及以上同时警示", "仓位上限 -> 30%"),
    ("制度事件(印花税/交易费上调、T+0、涨跌幅改变、退市或发行制度改变)", "不看分位,生效前复核并按 §6 机械重算盈亏平衡"),
]
PREREG = LEDGER / "sa_prereg_manifest.json"


def spec_fingerprint() -> tuple[str, dict]:
    """SPEC + 阈值 + 状态/应对规则的指纹。改动其中任何一项,指纹都会变。"""
    payload = dict(SPEC=[list(r) for r in SPEC], PRE_REVIEW_ROLE=PRE_REVIEW_ROLE, THR={k: list(v) for k, v in THR.items()},
                   SHARED=SHARED, STATIC_FLOOR={k: list(v) for k, v in STATIC_FLOOR.items()}, MIN_HIST=MIN_HIST,
                   TAIL_ASSUMPTIONS=list(TAIL_ASSUMPTIONS), CAP_TIERS={str(k): v for k, v in CAP_TIERS.items()},
                   RESPONSES=[list(r) for r in RESPONSES], EVENTS=[list(e) for e in EVENTS])
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest(), payload


def trigger_history(L: pd.DataFrame, dates: pd.DatetimeIndex, roles: dict | None = None) -> pd.DataFrame:
    """按"实时"扩展窗分位(只用 <= t 的历史)回放预先写好的状态规则,月末抽样。
    只记录状态与对应的仓位上限档位,**不计算任何收益**(不是择时回测)。
    roles: 覆盖 SPEC 的 role(用于"按审查前归类"的敏感性)。"""
    spec = SPEC_DF.set_index("indicator")
    role = {i: (roles or {}).get(i, spec.loc[i, "role"]) for i in spec.index}
    me = dates[month_end_idx(dates)]
    gates = [i for i in spec.index if role[i] == "gate" and i in set(L.indicator)]
    st = {}
    for ind in gates:
        g = L[L.indicator == ind].set_index("date")["pct_exp"].sort_index()
        pe = g.reindex(g.index.union(me)).ffill().reindex(me)
        # 年度值记在当年 12-31,在日期 >= 12-31 的第一个月末可见(12 月最后交易日早于 12-31 时即次年 1 月末),下一个年度值出来前沿用
        dn = spec.loc[ind, "danger"]
        w_, a_ = THR[dn]
        if dn == "high":
            s_ = np.where(pe >= a_, 2, np.where(pe >= w_, 1, 0))
        else:
            s_ = np.where(pe <= a_, 2, np.where(pe <= w_, 1, 0))
        st[ind] = pd.Series(np.where(pe.notna(), s_, -1), index=me)     # -1 = 实时历史不足
    S = pd.DataFrame(st)
    out = pd.DataFrame(index=me)
    for a in ["A1", "A2", "A3", "A4", "A5"]:
        cols = [c for c in S.columns if spec.loc[c, "assumption"] == a]
        out[a] = S[cols].max(axis=1)
        out[a + "_n_avail"] = (S[cols] >= 0).sum(axis=1)
    n_alarm = sum((out[a] == 2).astype(int) for a in TAIL_ASSUMPTIONS)
    out["n_tail_alarm"] = n_alarm
    out["cap_implied"] = n_alarm.clip(upper=2).map(CAP_TIERS)
    out["gates_alarm"] = S.apply(lambda r: ";".join(c for c in S.columns if r[c] == 2), axis=1)
    out["gates_watch"] = S.apply(lambda r: ";".join(c for c in S.columns if r[c] == 1), axis=1)
    out.index.name = "date"
    return out


def cross_checks(S, extra, checks):
    out = list(checks)
    # 1) 多空月收益 vs Barra 纯流动性因子月收益(纯因子收益从 2021-01-15 开始,研究面板,新市值)
    try:
        bf = pd.read_csv(LEDGER / "barra_factor_returns.csv", header=[0, 1], index_col=0)
        bf.index = pd.to_datetime(bf.index)
        assert bf.index.max() <= CUT
        liq = bf[("open10", "Liquidity 流动性")].dropna()
        lm = liq.groupby([liq.index.year, liq.index.month]).sum()
        ls = S["A2_ls_1m"].dropna()
        lsm = pd.Series(ls.to_numpy(), index=pd.MultiIndex.from_arrays([ls.index.year, ls.index.month]))
        j = pd.concat([lsm.rename("ls"), lm.rename("liq")], axis=1).dropna()
        out.append(("corr_ls1m_vs_barra_liquidity_2021_2025", float(j.corr().iloc[0, 1]), len(j),
                    f"月度,{j.index[0][0]}-{j.index[0][1]:02d} ~ {j.index[-1][0]}-{j.index[-1][1]:02d}"
                    f"(纯因子收益首日 {liq.index.min().date()},首月不完整);期望为负"))
    except Exception as e:
        out.append(("corr_ls1m_vs_barra_liquidity_2021_2025", np.nan, 0, repr(e)))
    # 2) 市场层多空 12m vs 排雷 12m 超额(月末抽样,描述)
    ls12 = S["A2_ls_12m"].dropna()
    ex = S["A2_excess_fp_12m"].reindex(ls12.index, method="ffill")
    j = pd.concat([ls12, ex], axis=1).dropna()
    out.append(("corr_ls12m_vs_excess_fp_12m_monthend", float(j.corr().iloc[0, 1]), len(j), "重叠窗口,描述"))
    ex = S["A2_excess_b1_12m"].reindex(ls12.index, method="ffill")
    j = pd.concat([ls12, ex], axis=1).dropna()
    out.append(("corr_ls12m_vs_excess_b1_12m_monthend", float(j.corr().iloc[0, 1]), len(j), "重叠窗口,描述"))
    # 3) 多空腿全期年化
    legs = extra["ls_legs"].dropna()
    out.append(("ls_monthly_mean_log_ann", float(legs["ls_log"].mean() * 12), len(legs), "低换手组 - 高换手组,毛,月度对数 x 12"))
    out.append(("ls_monthly_share_positive", float((legs["ls_log"] > 0).mean()), len(legs), ""))
    # 4) B/P 覆盖
    if "bp_cov" in extra:
        c = extra["bp_cov"]["bp_coverage_pool"]
        out.append(("bp_coverage_pool_min", float(c.min()), len(c), "决策池中有 B/P 的比例(月末)最小值"))
        out.append(("bp_coverage_pool_median", float(c.median()), len(c), ""))
    # 5) 持仓与池的 ST 占比(2016-08-09 起)
    st = extra["st_share"].dropna()
    out.append(("hold_st_share_mean_2016_2025", float(st["hold_st_share"].mean()), len(st), ""))
    out.append(("pool_st_share_mean_2016_2025", float(st["pool_st_share"].mean()), len(st), ""))
    out.append(("hold_st_share_last", float(st["hold_st_share"].iloc[-1]), 1, str(st.index[-1].date())))
    out.append(("pool_st_share_last", float(st["pool_st_share"].iloc[-1]), 1, str(st.index[-1].date())))
    return pd.DataFrame(out, columns=["item", "value", "n", "note"])


def _spans(idx) -> str:
    """月末日期列表 -> 'YYYY-MM~YYYY-MM' 连续段(按月相邻合并)。"""
    ms = sorted({(d.year, d.month) for d in idx})
    out, i = [], 0
    while i < len(ms):
        j = i
        while j + 1 < len(ms) and (ms[j + 1][0] * 12 + ms[j + 1][1]) - (ms[j][0] * 12 + ms[j][1]) == 1:
            j += 1
        a, b = ms[i], ms[j]
        out.append(f"{a[0]}-{a[1]:02d}" + ("" if i == j else f"~{b[0]}-{b[1]:02d}"))
        i = j + 1
    return ", ".join(out)


def review_checks(S, L, A, TH, TH0, d1, dates, H=None, pool=None) -> list:
    """审查(2026-10-11)要求补的核对行,item 以 rv_ 开头。全部是机械计算,不做收益检验。"""
    out = []
    spec = SPEC_DF.set_index("indicator")
    # 1) A2 实质是排雷自身相对表现的监控:月度多空 vs 月度排雷相对 fp 对数超额(不重叠)
    ls = S["A2_ls_1m"].dropna()
    lsm = pd.Series(ls.to_numpy(), index=pd.MultiIndex.from_arrays([ls.index.year, ls.index.month]))
    ex = (np.log1p(d1["strat_main"]) - np.log1p(d1["strat_fullpool_ew"])).dropna()
    exm = ex.groupby([ex.index.year, ex.index.month]).sum()
    j = pd.concat([lsm.rename("ls"), exm.rename("ex")], axis=1).dropna()
    out.append(("rv_corr_ls1m_vs_excess_fp_1m", float(j.corr().iloc[0, 1]), len(j),
                "月度不重叠;排雷 O0 相对 fp 的日对数超额按自然月求和(多空按持有期月末归月)"))
    # 2) 2006-01-01 ~ 2007-10-15(BEAR08 起点前)各 gate 指标处于危险区(pct_full 越过关注线)的比例
    for ind in [i for i in spec.index if spec.loc[i, "role"] == "gate"]:
        g = L[L.indicator == ind].set_index("date").sort_index()
        w = g.loc[pd.Timestamp("2006-01-01"): pd.Timestamp("2007-10-15")]
        if w.empty:
            out.append((f"rv_bull06_danger_share|{ind}", np.nan, 0, "窗口内无读数"))
            continue
        dn = spec.loc[ind, "danger"]
        hit = (w["pct_full"] >= THR["high"][0]) if dn == "high" else (w["pct_full"] <= THR["low"][0])
        out.append((f"rv_bull06_danger_share|{ind}", float(hit.mean()), len(w),
                    f"2006-01-01~2007-10-15 读数中处于危险区(全历史分位越过关注线)的比例;中位分位 p{w['pct_full'].median():.0f}"))
    # 3) 热门股涨幅差的全历史最低点与最低 5% 的年份分布
    g = L[L.indicator == "A1_runup_spread"].set_index("date").sort_index()
    k = g["value"].idxmin()
    low5 = g[g["pct_full"] <= 5.0]
    out.append(("rv_runup_spread_min", float(g.loc[k, "value"]), 1, f"{k.date()}(全历史最低)"))
    out.append(("rv_runup_spread_bottom5pct_months", float(len(low5)), len(g),
                "按年:" + ", ".join(f"{y}:{n}" for y, n in low5.groupby(low5.index.year).size().items())))
    # 4) A2 两条同时实时警示的月末
    p12 = L[L.indicator == "A2_ls_12m"].set_index("date")["pct_exp"]
    p24 = L[L.indicator == "A2_ls_24m"].set_index("date")["pct_exp"]
    both = pd.concat([p12, p24], axis=1).dropna()
    both = both[(both.iloc[:, 0] <= THR["low"][1]) & (both.iloc[:, 1] <= THR["low"][1])]
    out.append(("rv_a2_double_alarm_months", float(len(both)), len(both), _spans(both.index)))
    # 5) 换手率版(LongPanel 第一步 O0)在 STRESS_TEST W1 的表现;W1 按成交日 te = t+1 ∈ [2024-01-02, 2024-02-07] 归属
    te = pd.Series(list(dates[1:]) + [pd.NaT], index=dates)
    sel = (te >= pd.Timestamp("2024-01-02")) & (te <= pd.Timestamp("2024-02-07"))
    r = d1.reindex(dates)[sel.to_numpy()]
    cum = {c: float((1 + r[c].fillna(0.0)).prod() - 1) for c in ("strat_main", "strat_fullpool_ew", "bench_b1")}
    for c, v in cum.items():
        out.append((f"rv_w1_turnover_version|{c}", v, int(sel.sum()), "W1 te 2024-01-02~02-07(信号日 t 收益在 t+1 开盘 -> t+2 开盘实现)"))
    out.append(("rv_w1_turnover_version|strat_minus_fp", cum["strat_main"] - cum["strat_fullpool_ew"], int(sel.sum()), "算术差"))
    # 6) 退市股退市前一年在保留侧:对照用时间匹配的 (持仓∩池)/池
    for a_, b_ in ((2006, 2025), (2022, 2025)):
        x = A.loc[a_:b_]
        kept = x["delisted_1y_before_kept"].sum()
        n = kept + x["delisted_1y_before_excluded"].sum()
        e = x["delisted_1y_before_base_sum"].sum()
        v = x["delisted_1y_before_base_var"].sum()
        out.append((f"rv_delisted_kept_vs_timematched_{a_}_{b_}", float(kept / n), int(n),
                    f"保留侧 {int(kept)}/{int(n)};时间匹配对照 {e / n:.3f};z = {(kept - e) / np.sqrt(v):.2f}"))
    # 7) 目标持仓规模与占池比例(§1.1 口径说明);持仓/池 微盘占比之比(描述)
    if H is not None and pool is not None:
        nH = H.sum(axis=1); nP = pool.sum(axis=1); ok = nP > 0
        out.append(("rv_target_holdings_avg", float(nH[ok].mean()), int(ok.sum()),
                    f"O0 目标持仓 W>0 日均只数;(持仓∩池)/池 日均 {np.mean((H & pool).sum(1)[ok] / nP[ok]):.3f};"
                    f"持仓在池外的比例 {np.mean((H & ~pool).sum(1)[ok] / np.maximum(nH[ok], 1)):.3f};池日均 {nP[ok].mean():.0f}"))
    rt = (S["A3_hold_micro20_share"] / S["A3_pool_micro20_share"]).dropna()
    out.append(("rv_hold_over_pool_micro20_last", float(rt.iloc[-1]), len(rt),
                f"全历史分位 p{pct_full(rt.to_numpy())[-1]:.0f}"))
    # 8) 实时回放档位汇总:现行归类(main)与审查前归类(pre_review)
    st2 = pd.read_csv(LEDGER / "lh_step2_summary.csv")
    st2 = st2[(st2["period"] == "full") & (st2["cash"] == 0.0)].set_index("rule")
    ann = {0.50: float(st2.loc["O1_f0500", "ann"]), 0.40: float(st2.loc["O1_f0400", "ann"]), 0.30: float(st2.loc["O1_f0300", "ann"])}
    for tag, T in (("main", TH), ("pre_review", TH0)):
        cap = T["cap_implied"]
        w = cap.loc["2008-01-01":"2025-12-31"]
        out.append((f"rv_replay_cap40_months|{tag}", float((w == 0.40).sum()), len(w), "2008-2025 月末"))
        out.append((f"rv_replay_cap30_months|{tag}", float((w == 0.30).sum()), len(w), _spans(w.index[w == 0.30])))
        for a_, b_ in (("2005", "2009"), ("2010", "2014"), ("2015", "2019"), ("2020", "2025")):
            ww = cap.loc[f"{a_}-01-01":f"{b_}-12-31"]
            out.append((f"rv_replay_cap_by_period|{tag}|{a_}_{b_}", float((ww == 0.40).sum()), len(ww),
                        f"40%:{int((ww == 0.40).sum())} 30%:{int((ww == 0.30).sum())}"))
        x = cap.loc["2008-01-01":"2009-12-31"]
        out.append((f"rv_replay_cap_below50_2008_2009|{tag}", float((x < 0.5).sum()), len(x), ""))
        x = cap.loc["2015-01-01":"2015-12-31"]
        f = x[x < 0.5]
        out.append((f"rv_replay_first_cut_2015|{tag}", float(f.iloc[0]) if len(f) else np.nan, len(x),
                    f"{f.index[0].date()} {T.loc[f.index[0], 'gates_alarm']}" if len(f) else "2015 年无降档"))
        d_ = pd.Timestamp("2023-12-29")
        out.append((f"rv_replay_cap_2023-12-29|{tag}", float(cap.loc[d_]), 1, str(T.loc[d_, "gates_alarm"])))
        wavg_ann = float(w.map(ann).mean())
        out.append((f"rv_replay_governance_cost_pp|{tag}", (ann[0.50] - wavg_ann) * 100, len(w),
                    f"按时间占比加权的静态前沿年化(lh_step2_summary O1_f0500/0400/0300 full cash=0),不是规则回测;平均上限 {w.mean():.3f}"))
    # 9) 成本与印花税的机械推算(沿用第一步记账口径;线性外推,不考虑交易行为变化)
    gr = pd.read_csv(LEDGER / "lh_step1_grid.csv").set_index("config")
    sm = pd.read_csv(LEDGER / "lh_step1_summary.csv")
    sm = sm[(sm["config"] == "main")].set_index("period")
    c_old, c_y1, c_ps = (float(gr.loc[c, "cost_ann_full"]) for c in ("main_old", "main", "main_pess"))
    e_old, e_y1, e_ps = (float(gr.loc[c, "full_excess"]) for c in ("main_old", "main", "main_pess"))
    t_y1 = float(gr.loc["main", "full_nw_t5"])
    s1 = (e_y1 - e_old) / (c_y1 - c_old); s2 = (e_ps - e_y1) / (c_ps - c_y1); sl = (s1 + s2) / 2
    be = c_y1 + e_y1 / (-sl)
    cost12 = float(S["A5_cost_12m"].dropna().iloc[-1]) / 100.0
    out.append(("rv_mech_cost_slope_old_to_Y1", s1, 1, "lh_step1_grid main_old -> main"))
    out.append(("rv_mech_cost_slope_Y1_to_pess", s2, 1, "lh_step1_grid main -> main_pess"))
    out.append(("rv_mech_breakeven_total_cost", be, 1,
                f"Y1 点 + 超额 / 平均斜率;为 2006-2025 平均成本 {c_y1:.4f} 的 {be / c_y1:.2f} 倍、近 12 个月实际成本 {cost12:.4f} 的 {be / cost12:.2f} 倍"))
    to_full = float(sm.loc["full", "turnover_ann_full"]); to_test = float(sm.loc["test", "turnover_ann"])
    stamp_hist = float(sm.loc["full", "cost_stamp_ann"])
    out.append(("rv_mech_stamp_hist_avg_ann", stamp_hist, 1, "2006-2025 史实印花税年均(已含在 +3.96pp 里)"))
    for tag, mult in (("sell5bp_current", 0.0005), ("sell10bp", 0.0010), ("bilateral30bp", 0.0060)):   # mult x 单边换手 = 年印花税
        c_ = mult * to_full
        inc = c_ - stamp_hist
        e_ = e_y1 + sl * inc
        out.append((f"rv_mech_stamp|{tag}", c_, 1,
                    f"年印花税 {c_:.4f}(单边换手 {to_full:.3f});相对史实均值增量 {inc * 100:+.2f}pp;超额约 {e_ * 100:+.2f}pp"
                    f"(斜率 {sl:.2f});标准误不变时 NW t 约 {t_y1 * e_ / e_y1:.2f}"))
    out.append(("rv_mech_stamp_per_10bp_sell", 0.001 * to_full, 1, f"每 +10bp 卖方印花税;检验窗换手 {to_test:.3f} 时 {0.001 * to_test:.4f}"))
    return out


# ============================================================== 图
def figure(L: pd.DataFrame, C: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    for fn in ("PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"):
        if any(fn == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [fn]
            break
    plt.rcParams["axes.unicode_minus"] = False
    INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    BLUE, SURF = "#2a78d6", "#fcfcfb"
    WARN, CRIT = "#fab219", "#d03b3b"
    SHADE = "#ecebe6"
    panels = [ind for ind in SPEC_DF.indicator if ind in set(L.indicator) and SPEC_DF.set_index("indicator").loc[ind, "role"] == "gate"]
    panels += ["A1_bp_spread", "A2_excess_fp_24m", "A5_stamp_sell_bps"]
    panels = [p for p in panels if p in set(L.indicator)]
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(17, 3.2 * nrow + 1.2), facecolor=SURF)
    axes = np.atleast_2d(axes)
    cur = C.set_index("indicator")
    spec = SPEC_DF.set_index("indicator")
    for ax in axes.flat:
        ax.set_visible(False)
    for k, ind in enumerate(panels):
        ax = axes.flat[k]
        ax.set_visible(True)
        ax.set_facecolor(SURF)
        g = L[L.indicator == ind].sort_values("date")
        sp = spec.loc[ind]
        for _, a, b in EVENTS[:4]:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=SHADE, lw=0, zorder=0)
        if sp["freq"] == "A":
            ax.step(g["date"], g["value"], where="pre", color=BLUE, lw=1.6, zorder=3)
            ax.plot(g["date"], g["value"], "o", ms=3.2, color=BLUE, zorder=3)
        else:
            ax.plot(g["date"], g["value"], color=BLUE, lw=1.1 if sp["freq"] == "D" else 1.5, zorder=3)
        r = cur.loc[ind]
        if ind != "A5_stamp_sell_bps":
            ax.axhline(r["thr_watch"], color=WARN, lw=1.2, zorder=2)
            ax.axhline(r["thr_alarm"], color=CRIT, lw=1.2, zorder=2)
        last = g.iloc[-1]
        ax.plot([last["date"]], [last["value"]], "o", ms=7, color=BLUE, mec=SURF, mew=1.5, zorder=4)
        tag = r["status"] if r["role"] == "gate" else "描述"
        ax.set_title(f"{ind.split('_')[0]} {sp['name_cn']}", loc="left", fontsize=9.5, color=INK, pad=14)
        dn = "低" if sp["danger"] == "low" else "高"
        sub = f"2025 末 p{r['pct_full']:.0f}  ·  {tag}" + (f"  ·  越{dn}越危险" if r["role"] == "gate" else "")
        ax.text(0.0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for sname, s in ax.spines.items():
            s.set_visible(sname == "bottom")
            s.set_color(BASE)
        ax.set_xlim(pd.Timestamp("2005-01-01"), pd.Timestamp("2026-03-01"))
    handles = [Line2D([], [], color=BLUE, lw=1.6, label="指标值(圆点 = 截至 2025-12-31 读数)"),
               Line2D([], [], color=WARN, lw=1.2, label="关注线:危险方向 90 分位(越低越危险者为 10 分位)"),
               Line2D([], [], color=CRIT, lw=1.2, label="警示线:危险方向 95 分位(越低越危险者为 5 分位)"),
               Patch(color=SHADE, label="熊市段 2008 / 2015 / 2018 / 2024(lh_step1_windows)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle("排雷结构假设的观测指标(2005-2025;分位按各指标自身全部历史;描述用,非择时信号)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=140, facecolor=SURF)
    plt.close(fig)


# ============================================================== 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--freeze-spec", action="store_true", help="把当前 SPEC/阈值/应对表的指纹写入 sa_prereg_manifest.json(已存在则拒绝)")
    args = ap.parse_args()
    t0 = time.time()
    fp_now, payload = spec_fingerprint()
    if args.freeze_spec:
        if PREREG.exists():
            raise SystemExit(f"{PREREG} 已存在,拒绝覆盖;改规则须另立新清单并在报告中登记")
        PREREG.write_text(json.dumps(dict(
            sha256=fp_now, frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            note=("审查(2026-10-11)后冻结。它不能证明审查前的 SPEC 是在看到读数之前写定的;"
                  "只作为此后复核的锚:之后任何对 SPEC、阈值、应对表、事件窗口的修改都会使指纹不符,"
                  "并在 sa_checks.csv item=prereg_fingerprint_match 中显示。"),
            diff_vs_pre_review=["A3_hold_micro20_share: role desc -> gate(按 SPEC 注释写明的 gate 条件更正)",
                                "新增描述指标 A3_delist_forced_rate(不参与状态)",
                                "阈值从硬编码改为 THR 常量(数值不变:90/95、10/5)",
                                "应对表写入代码常量 RESPONSES(内容与报告 §1.4 一致,未改)"],
            payload=payload), ensure_ascii=False, indent=1))
        log(f"已写入预注册指纹 {PREREG.name}:{fp_now[:12]}…")
    D = load()
    S, extra, d1, checks = compute(D)
    A = annual(D, extra)
    L, C = summarize(S, A)
    E = events(L, D["dates"])
    B = breach_spells(L)
    TH = trigger_history(L, D["dates"])
    TH0 = trigger_history(L, D["dates"], roles=PRE_REVIEW_ROLE)
    for c in ("A3", "n_tail_alarm", "cap_implied", "gates_alarm"):
        TH[c + "_pre_review"] = TH0[c]
    K = cross_checks(S, extra, checks)
    for row in review_checks(S, L, A, TH, TH0, d1, D["dates"], H=extra["H"], pool=D["pool"]):
        K.loc[len(K)] = row
    if PREREG.exists():
        fp_saved = json.loads(PREREG.read_text())["sha256"]
        K.loc[len(K)] = ("prereg_fingerprint_match", float(fp_saved == fp_now), 1, f"当前 {fp_now[:16]} / 清单 {fp_saved[:16]}")
        if fp_saved != fp_now:
            log("警告:SPEC/阈值/应对表与 sa_prereg_manifest.json 的指纹不符")
    else:
        K.loc[len(K)] = ("prereg_fingerprint_match", np.nan, 0, f"清单不存在;当前 {fp_now[:16]}")
    K.loc[len(K)] = ("breach_spells", np.nan, len(B), "gate 指标扩展窗分位越过 95/5 的连续区段数;明细见下方 spell| 行")
    for _, r in B.iterrows():
        K.loc[len(K)] = (f"spell|{r['indicator']}", r["n_obs"], r["n_obs"], f"{r['start']}~{r['end']}({r['freq']})")
    # 截止日自检
    assert L["date"].max() <= CUT
    K.loc[len(K)] = ("max_date_in_ledger", np.nan, 1, str(L["date"].max().date()))
    K.loc[len(K)] = ("runtime_sec", time.time() - t0, 1, "")
    LEDGER.mkdir(parents=True, exist_ok=True)
    L.assign(date=L["date"].dt.date).to_csv(LEDGER / "sa_indicators.csv", index=False)
    C.to_csv(LEDGER / "sa_current.csv", index=False)
    E.to_csv(LEDGER / "sa_events.csv", index=False)
    A.reset_index().to_csv(LEDGER / "sa_annual.csv", index=False)
    pd.DataFrame(RULES, columns=["date", "category", "event", "source"]).sort_values("date").to_csv(LEDGER / "sa_rules.csv", index=False)
    K.to_csv(LEDGER / "sa_checks.csv", index=False)
    TH.reset_index().assign(date=lambda x: x["date"].dt.date).to_csv(LEDGER / "sa_trigger_history.csv", index=False)
    log(f"账本写出:{len(L)} 行长表,{len(C)} 行当前读数")
    if not args.no_fig:
        figure(L, C, REPORTS / "sa_indicators.png")
        log("图 reports/sa_indicators.png")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(C[["assumption", "indicator", "role", "danger", "last_date", "value", "pct_full", "pct_exp", "status"]].to_string())
    log(f"完成,用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
