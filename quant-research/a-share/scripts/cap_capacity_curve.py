"""冲击成本下的容量曲线(前缀 cap,2026-10-10):这些策略能装多少钱?

用上游引擎(strategy.run_weights + impact.CostModel,见 research/ENGINE_IMPACT_MODEL.md)把
四个策略在 8 档资金 × 2 个执行口径 × 3 档冲击系数上逐格回测,读出"净超额随资金下降"的曲线,
按预注册定义给出半衰容量与盈亏平衡容量。

## 被测策略(股票池固定 V1000 = ADV>=2000 万,**不随资金变**,把换池与容量效应分开)
  COMP_v5_exclude  COMP_v5 标准排雷:缓冲 20/35、5 日调仓(约 2900 只等权,日换手约 0.02)
  COMP_v5_top20    COMP_v5 Top20 等权、5 日调仓
  COMP_v5_top50    COMP_v5 Top50 等权、5 日调仓
  B_gbt_top20      B×GBT walk-forward 样本外打分 Top20、5 日调仓(2023-2025)
⚠ COMP_v5 的 24 个成员在 2020-2025 研究期内选出,COMP_v5 的任何组合结果都是**信号样本内**,
  测的是组合构建 / 成本层,不是信号的样本外能力;B_gbt 在 2023-2025 是样本外。

## 网格(预注册,全部报告;写在 research/ledger/capacity_prereg.json,先于网格运行落盘)
  AUM ∈ {1e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10} 元 × 执行口径 {open10, vwap1} × Y ∈ {0.5, 1.0, 1.5}
  价差 = 一跳下限(CostModel 默认 spread_method="tick")。
  另三组敏感性(同样全部报告):
    legacy      只扣佣金+印花税、无价差无冲击,参与率约束照常 -> 分离"参与率(现金拖累)"与"显式成本"
    Y1.0_2tick  Y=1、两跳价差(spread_mult=2)-> 价差的上沿
    Y1.0_ride   Y=1、初始资金 = AUM 之后随净值滚动(participation_on_nav=True,冲击 Q 用 capital×净值)

## 资金口径(主口径 = 恒定 AUM;为什么)
容量问的是"规模为 X 时还剩多少超额"。若用初始资金滚动,5 年净值翻倍的策略在检验窗的实际规模是 2X,
而亏钱的高换手方案规模缩水 —— 规模被策略自己的收益污染,开发窗 / 检验窗也不可比。
所以主口径固定每天的资金为 X(等价于盈利按比例分出、亏损按比例补足,资金进出不计成本):
  - 参与率上限 = 10% × te 窗口成交额 / X(participation_on_nav=False,引擎旧口径正是按 capital 折算);
  - 冲击 Q = |Δw| × X:用 ConstAUMCost 包装 CostModel,把引擎传入的 nav 固定为 1。
两者用同一个 X,口径自洽。随净值滚动的版本作敏感性(Y1.0_ride)。
⚠ record 里的 trade_value / aum / max_part 列按 capital×净值 计,恒定 AUM 口径下不用这些列;
  成交额一律用 turnover × X 折算。

## 相位与"分批"换算
5 日调仓跑满 5 个相位取日收益平均;引擎对**每个相位都按全额资金**模拟(上游约定)。
所以本表的"AUM = X"对应"整本组合每周在固定一天整体调仓(对 5 个调仓日取平均)"。
若实盘把资金拆成 5 个错开调仓的子组合(每份 X/5,相位平均的实盘对应),它在 AUM = X 时的结果
**恒等于**本表 AUM = X/5 那一格 —— 容量 ×5,无需重跑。报告两种口径。

## 时序与窗口
  COMP_v5:模拟从 2020-07-01 起(2020H2 只作预热:建仓、让组合进入稳态,不报告),
           报告 2021-2025 全期、开发窗 2021-2023、检验窗 2024-2025(同一条连续路径切片,不在窗口起点重新建仓)。
  B_gbt  :打分只有 2023-2025,从 2023-01-03 空仓起步(无预热),报告 2023-2025、2023、2024-2025。
  收益打在信号日 t(open10:twap_open10[t+2]/twap_open10[t+1]-1;vwap1:vwap[t+2]/vwap[t+1]-1)。
  冲击的 σ / ADV 与半价差由 CostModel 按成交日 te 只用到 te-1(上游已测,审查扰动探针逐位复核)。
  ⚠ 参与率上限**不是**事前量(2026-10-10 审查 P1,旧版此处写反了):引擎 ndarray 目标分支按
    cap_w = 10% × **te 当天实际**窗口成交额 / X 截量,等价于"10% POV 执行算法"—— 成交量在执行中才揭示。
    POV 实盘应按窗口 VWAP 成交,引擎却按窗口 TWAP 计价;且窗口成交额的"意外"与 te->te+1 收益截面负相关
    (放量的股票次日偏弱),POV 的成交随当天放量 / 缩量变动,持仓路径因此带上执行日才揭示的信息。
    随机打分安慰剂(零收益现金口径)显示这一假定单独就能移动 1~2pp/年,主要来自 POV 平均成交更满(仓位效应,见 checks)。
    因此全部 384 格另跑一遍 sizing="exante"(见下),报告两者并列,不单报 POV。

## 截量口径 sizing(预注册补充,审查后加入,capacity_prereg.json 的 supplements)
  pov     引擎默认:参与率上限 = 10% × te 实际窗口成交额(预注册主网格,结果与旧账本逐位一致)
  exante  参与率上限 = 10% × 截至 te-1 的 20 日窗口成交额中位数(min_periods 10;与 strategy.decision_window_amount
          同一估计量)。实现:在 exante_sizing() 上下文里把 ST.exec_prices 返回的成交额换成估计值,
          **可成交集合 can 不变**(估计缺失或 <=0 而实际 >0 时退回实际值,只为不改 can;2021-2025 池内占 2.9e-5)。
          引擎代码与默认行为不动(规则 2)。冲击模型仍用全日 σ/ADV,不因 te 当天缩量多收费 —— 事前下单在
          缩量日的真实参与率会超过 10%(checks 里量化),这一侧对 exante 有利。真值在两者之间。

## 现金计价敏感性(审查 P1)
  引擎把参与率截量留下的现金按零收益计,基准始终满仓。另报两档:现金腿按池等权收益补仓(ret + cash×pool,
  期货 / ETF 补仓位的乐观端,忽略基差与成本)、按中证1000 代理收益补仓(ret + cash×csi1000p)。一阶近似:
  不改变持仓路径(现金腿的漂移是二阶)。excess_pool_cashpool / excess_pool_cashcsi 及对应容量列。

## 基准(2026-10-10 第二轮修订 fixa 后;预注册补充 fixa_2026-10-10)
  pool      **同频(5 日)再平衡**的 V1000 池等权(ST.benchmark(..., rebalance=5),5 相位平均,同执行价、不含成本)
            —— **容量定义用这一个**。修订前用每天重置等权的 ST.benchmark(k=1):5 日调仓的组合其间让权重漂移,
            日收益正自相关时 k=1 基准系统性吃亏,"与选股无关的再平衡频率差"被记成策略超额(vwap1 约 +2pp/年、
            open10 约 +0.5pp/年,research/ledger/fixa_fullpool_checks.csv)。两个口径都改(容量定义需口径一致)。
  pool_d1   修订前的 k=1 池等权(预注册原口径),账本保留 *_pool_d1 列只作前后对照
  csi1000p  中证1000 PIT 代理,**成交时点权重**(IP.exec_time_bench_returns):收盘权重先漂移到成交价时点再乘成交价收益
  csi1000p_legacy  修订前的 IP.engine_consistent_bench_returns(收盘权重 × 成交价,系统性偏低),只作前后对照
  csi1000r  真实 000852 收盘(价格指数,截到 2025-12-31)对齐到信号日:open10 用 close(t+1)/close(t)
            (scripts/run_ir_analysis.py 同法);vwap1 的持有区间 vwap(t+1)->vwap(t+2) 与 close(t+1)->close(t+2)
            共享隔夜跳空,实测相关峰值在该对齐上(见 checks),故 vwap1 用 close(t+2)/close(t+1)。
            价格指数不含股息(中证1000 约 1.23pp/年,CSI1000_PROXY.md §5.1),只作量级参照。

## 容量定义(预注册)
  (a) 半衰容量:净超额(几何年化差,vs 池等权)降到 1000 万那一格一半时的 AUM,log10(AUM) 线性插值;
  (b) 盈亏平衡容量:净超额降到 0 时的 AUM(同法插值;1000 万已 <=0 时向 100 万方向找)。
  网格外:> 1e10 记"右删失",< 1e6 记"左删失"。不确定性:把算术超额曲线整体平移 ±1 个 NW 标准误
  (1000 万格,带宽 10 = max(5, 2h))后的盈亏平衡 AUM,作为区间。

## 统计
日频重叠收益的 t 用 Newey-West(带宽 max(5, 2h) = 10);TE / IR 用 ST.excess_stats 的 NW20 *_adj 口径
(vwap1 平均价口径日收益正自相关,日频 Sharpe/IR 高估,只看 *_adj)。

## 回归断言(每一格都检查,不过就 raise)
  exposure <= 1(引擎逐日硬断言 + 相位明细最大值复核);corr(策略, 池基准.shift(k)) 峰值 k=0
  (run 阶段对修订前基准逐格断言;summarize 对同频基准逐格复核 align_peak_k_samefreq);
  全池等权常数打分 vs 池基准:超额 + 成本 ∈ [-0.5, +1.5]pp(1e6 / 1e7,两口径,legacy 与 Y=1);
  日调仓对 k=1 基准(原判据)+ 5 日调仓对同频基准(fixa 新增,2020H2 预热、报告 2021-2025 切片);
  另:COMP_v5 排雷 1000 万 Y=1 按 eng_cost_demo 原配置重跑,必须与 eng_cost_demo.csv 逐位相等;
  B_gbt Top20 逐年旧成本复现 ml_results_open_h5.csv(±1pp)。

## 用法
  python3 scripts/cap_capacity_curve.py prereg            # 写预注册(已存在则核对一致)
  python3 scripts/cap_capacity_curve.py prereg_supp       # 写预注册补充(审查后加入的 exante 截量等;已存在则核对)
  python3 scripts/cap_capacity_curve.py run open10 [pov|exante]   # 重计算:一个口径 192 格,约 7 分钟
  python3 scripts/cap_capacity_curve.py run vwap1  [pov|exante]
  python3 scripts/cap_capacity_curve.py checks            # 引擎判据、复现核对、流动性诊断、安慰剂、冷启动
  python3 scripts/cap_capacity_curve.py summarize         # 账本 + 图
  python3 scripts/cap_capacity_curve.py prereg_fixa       # 第二轮修订(fixa)的预注册补充(换基准,已存在则核对)
  python3 scripts/cap_capacity_curve.py bench             # 只重建基准缓存(网格日收益与基准无关,不必重跑 run)
产出:research/ledger/capacity_{prereg.json,grid.csv,summary.csv,checks.csv,diagnostics.csv,decomp.csv}、
     reports/capacity_curve.png、reports/cap_cost_breakdown.png;中间结果 cache/_tmp/cap_*.parquet。

只读研究期面板 Panel()(2020-01-02 ~ 2025-12-31);指数收盘经 IP.index_close 先截到 2025-12-31。
**数据不含已退市股(约 209 只),所有收益数字有幸存者偏差,方向为高估。**
"""
from __future__ import annotations
import os, sys, json, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, TMP, REPORTS, RESEARCH_END
from afactor.panel import Panel
from afactor import strategy as ST, impact as IM, index_proxy as IP
from afactor.strategy import TRADING_DAYS

pd.set_option("display.width", 250)

AUMS = [1e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10]
REF_AUM = 1e7
MODES = ["open10", "vwap1"]
PRIMARY_VARIANTS = ["Y0.5", "Y1.0", "Y1.5"]
EXTRA_VARIANTS = ["legacy", "Y1.0_2tick", "Y1.0_ride"]
VARIANTS = PRIMARY_VARIANTS + EXTRA_VARIANTS
REBALANCE = 5
NW_T_BW = 10                       # max(5, 2h),h = 5 日调仓
VERSION = "V1000"

STRATS = {
    "COMP_v5_exclude": dict(score="comp", plan=dict(scheme="exclude", buffer=(0.20, 0.35)),
                            sim=("2020-07-01", RESEARCH_END), signal_oos=False,
                            windows={"2021-2025": (2021, 2025, "full"), "2021-2023": (2021, 2023, "dev"),
                                     "2024-2025": (2024, 2025, "test")}),
    "COMP_v5_top20": dict(score="comp", plan=dict(scheme="top", n=20),
                          sim=("2020-07-01", RESEARCH_END), signal_oos=False,
                          windows={"2021-2025": (2021, 2025, "full"), "2021-2023": (2021, 2023, "dev"),
                                   "2024-2025": (2024, 2025, "test")}),
    "COMP_v5_top50": dict(score="comp", plan=dict(scheme="top", n=50),
                          sim=("2020-07-01", RESEARCH_END), signal_oos=False,
                          windows={"2021-2025": (2021, 2025, "full"), "2021-2023": (2021, 2023, "dev"),
                                   "2024-2025": (2024, 2025, "test")}),
    "B_gbt_top20": dict(score="bgbt", plan=dict(scheme="top", n=20),
                        sim=("2023-01-01", RESEARCH_END), signal_oos=True,
                        windows={"2023-2025": (2023, 2025, "full"), "2023": (2023, 2023, "dev"),
                                 "2024-2025": (2024, 2025, "test")}),
}

PREREG = LEDGER / "capacity_prereg.json"
BENCH_CACHE = str(TMP / "cap_bench_{}.parquet")
SIZINGS = ["pov", "exante"]                 # 截量口径(见模块 docstring);pov = 预注册主网格
EXANTE_WINDOW, EXANTE_MINP = 20, 10         # 与 strategy.decision_window_amount 默认一致


def daily_cache(mode: str, sizing: str = "pov") -> str:
    """pov 沿用旧文件名(与审查前的缓存同名,便于逐位对比);exante 加后缀。"""
    return str(TMP / (f"cap_daily_{mode}.parquet" if sizing == "pov" else f"cap_daily_{mode}_{sizing}.parquet"))


def cell_cache(mode: str, sizing: str = "pov") -> str:
    return str(TMP / (f"cap_cells_{mode}.parquet" if sizing == "pov" else f"cap_cells_{mode}_{sizing}.parquet"))


# record.daily 里保留的列(相位平均)
KEEP_COLS = ["ret", "gross_ret", "cost_total", "cost_commission", "cost_stamp", "cost_spread", "cost_impact",
             "exposure", "cash", "turnover", "unfilled", "desired_turnover", "part_cut", "n_hold", "nav_prev"]


# ====================================================================== 截量口径:事前估计
_ORIG_EXEC_PRICES = ST.exec_prices        # 导入时的原函数;exante_amount 必须用它,避免在上下文内递归
_EXANTE_CACHE: dict = {}


def exante_amount(p: Panel, mode: str) -> tuple[pd.DataFrame, dict]:
    """exante 截量用的"窗口成交额":行 te = 截至 te-1 的 20 日窗口成交额中位数(min_periods 10)。

    与 strategy.decision_window_amount 同一估计量。为了**不改变可成交集合 can**(引擎按 win_amt[te] > 0 判定,
    这是旧引擎"停牌一般事先公告"的约定,不在本修复范围),估计缺失或 <= 0 而实际 > 0 时退回实际值;
    实际 <= 0 或 NaN 时原样保留实际值(can 仍为 False)。返回 (成交额表, 统计)。
    缓存按 (id(p), mode),随进程生灭。"""
    key = (id(p), mode)
    _cache = _EXANTE_CACHE
    if key not in _cache:
        _, amt = _ORIG_EXEC_PRICES(p, mode)
        a = amt.astype(np.float64)
        est = a.rolling(EXANTE_WINDOW, min_periods=EXANTE_MINP).median().shift(1)
        good = np.isfinite(est) & (est > 0)
        used = est.where(good & (a > 0), a)
        av, uv = a.to_numpy(), used.to_numpy()
        assert np.array_equal(av > 0, uv > 0), "exante 成交额改变了可成交集合 can"
        fb = (av > 0) & ~good.to_numpy()
        _cache[key] = (used, dict(fallback_n=int(fb.sum()), actual_pos_n=int((av > 0).sum())))
    return _cache[key]


class exante_sizing:
    """上下文管理器:块内 ST.exec_prices 返回的成交额换成事前估计(价格不变),退出时恢复原函数。

    simulate() 在函数体内按模块全局名调用 exec_prices,所以替换 ST.exec_prices 即生效;
    CostModel 直接读 p.raw("amount") 算 ADV,不经过 exec_prices,不受影响(冲击口径不变)。
    benchmark() 只取价格,也不受影响。只在本进程内生效,不改任何库代码。"""

    def __init__(self, p: Panel, on: bool = True):
        self.p, self.on, self._orig = p, on, None

    def __enter__(self):
        if self.on:
            self._orig = ST.exec_prices
            orig, p0 = self._orig, self.p

            def _exec_prices_exante(p_, mode):
                px, amt = orig(p_, mode)
                if p_ is not p0:
                    raise RuntimeError("exante_sizing 只对构造时的面板生效")
                return px, exante_amount(p0, mode)[0]
            ST.exec_prices = _exec_prices_exante
        return self

    def __exit__(self, *exc):
        if self._orig is not None:
            ST.exec_prices = self._orig
        return False


# ====================================================================== 成本口径包装
class ConstAUMCost:
    """恒定 AUM 口径的成本模型包装:冲击 Q = |Δw| × capital,与净值无关。

    引擎调用 cost_model.extra_cost(te, |Δw|, capital, nav_prev);底层 CostModel 用 capital×nav 算 Q
    (随净值滚动)。容量曲线主口径要求每天规模固定为 X,所以这里把 nav 固定为 1 —— 与
    participation_on_nav=False(参与率上限 = 10%×窗口成交额 / capital)同一个 X,口径自洽。
    其余接口(cost_coefficients 等)不经过这里,引擎只用 extra_cost 与 describe。"""

    def __init__(self, base: IM.CostModel):
        self.base = base

    def extra_cost(self, te, abs_delta, capital, nav):
        return self.base.extra_cost(te, abs_delta, capital, 1.0)

    def describe(self) -> dict:
        d = self.base.describe()
        d["aum_mode"] = "const"
        return d


def cost_setup(p: Panel, variant: str, models: dict):
    """variant -> (cost_model, participation_on_nav)。models 缓存同一面板上的 CostModel(共享估计面板)。"""
    if variant == "legacy":
        return None, False
    if variant in ("Y0.5", "Y1.0", "Y1.5"):
        y = float(variant[1:])
        if y not in models:
            models[y] = IM.CostModel(p, impact_y=y)
        return ConstAUMCost(models[y]), False
    if variant == "Y1.0_2tick":
        if "2tick" not in models:
            models["2tick"] = IM.CostModel(p, impact_y=1.0, spread_mult=2.0)
        return ConstAUMCost(models["2tick"]), False
    if variant == "Y1.0_ride":
        if 1.0 not in models:
            models[1.0] = IM.CostModel(p, impact_y=1.0)
        return models[1.0], True
    raise ValueError(variant)


# ====================================================================== 数据
def sim_mask(p: Panel, strat: str) -> np.ndarray:
    a, b = STRATS[strat]["sim"]
    return np.asarray((p.dates >= pd.Timestamp(a)) & (p.dates <= pd.Timestamp(b)))


def load_scores(p: Panel) -> dict:
    comp = pd.read_parquet(TMP / "comp_v5_score_open10.parquet")
    bg = pd.DataFrame(np.nan, index=p.dates, columns=p.codes, dtype=np.float32)
    for y in (2023, 2024, 2025):
        s = pd.read_parquet(LEDGER / "scores" / f"B_gbt_open_h5_{y}.parquet")
        idx = s.index.intersection(p.dates)
        bg.loc[idx, :] = s.reindex(index=idx, columns=p.codes).to_numpy(np.float32)
    return dict(comp=comp, bgbt=bg)


def strategy_weights(p: Panel, scores: dict, strat: str) -> np.ndarray:
    cfg = STRATS[strat]
    plan = cfg["plan"]
    m = sim_mask(p, strat)
    return ST.build_weights(scores[cfg["score"]], p, VERSION, scheme=plan["scheme"], n=plan.get("n", 20),
                            buffer=plan.get("buffer"), date_mask=m)


def real_index_on_signal_days(p: Panel, mode: str, code: str = "000852") -> pd.Series:
    """真实指数(价格指数,先截到 2025-12-31)对齐到信号日。open10 与 IP.index_returns_on_signal_days 相同;
    vwap1 用 close(t+2)/close(t+1)(与 vwap(t+1)->vwap(t+2) 共享 t+1->t+2 的隔夜跳空,相关峰值在此,见 checks)。"""
    if mode != "vwap1":
        return IP.index_returns_on_signal_days(p, mode, code)
    lag = ST.EXEC_MODES[mode]["lag"]
    px = IP.index_close(code).reindex(p.dates).ffill()
    r = px.shift(-2) / px.shift(-1) - 1.0
    idx = p.dates[:len(p.dates) - 1 - lag]
    return r.reindex(idx).fillna(0.0).rename(code)


def bench_frame(p: Panel, mode: str) -> pd.DataFrame:
    """基准日收益表(信号日索引,全样本、不按策略 mask 截断;5 相位平均与锚点无关,见 ST.benchmark)。
    pool / csi1000p 是修订后(fixa)的主口径;pool_d1 / csi1000p_legacy 是修订前,只作前后对照。"""
    b = ST.benchmark(p, VERSION, mode, rebalance=REBALANCE)["rets"].rename("pool")
    b1 = ST.benchmark(p, VERSION, mode)["rets"].rename("pool_d1")
    pr = IP.exec_time_bench_returns(p, mode).rename("csi1000p")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pl = IP.engine_consistent_bench_returns(p, mode, "000852").rename("csi1000p_legacy")
    ri = real_index_on_signal_days(p, mode, "000852").rename("csi1000r")
    for x in (b1, pr, pl, ri):
        assert x.index.equals(b.index), f"基准索引不一致 {x.name}"
    return pd.concat([b, b1, pr, pl, ri], axis=1)


def cmd_bench():
    """只重建基准缓存(fixa):网格的策略日收益与基准无关,换基准不必重跑 run。"""
    cmd_prereg_fixa()
    p = Panel()
    for mode in MODES:
        bf = bench_frame(p, mode)
        bf.to_parquet(BENCH_CACHE.format(mode))
        m = bf.index.year >= 2021
        print(mode, {c: round(geo_ann(bf.loc[m, c]) * 100, 3) for c in bf.columns}, flush=True)


def align_profile(rs: pd.Series, rb: pd.Series, ks=(-2, -1, 0, 1, 2)) -> dict:
    a, b = rs.align(rb, join="inner")
    return {k: float(a.corr(b.shift(k))) for k in ks}


# ====================================================================== 预注册
def prereg_payload() -> dict:
    return dict(
        task="冲击成本下的容量曲线(前缀 cap)",
        written_before_grid_run=True,
        strategies={k: dict(score=v["score"], plan={kk: (list(vv) if isinstance(vv, tuple) else vv)
                                                      for kk, vv in v["plan"].items()},
                            rebalance=REBALANCE, sim=list(v["sim"]),
                            windows={w: list(x) for w, x in v["windows"].items()},
                            signal_out_of_sample=v["signal_oos"]) for k, v in STRATS.items()},
        universe="V1000 决策池(ADV>=2000 万),不随 AUM 变",
        aum_grid=AUMS, ref_aum=REF_AUM, modes=MODES,
        primary_variants=PRIMARY_VARIANTS, extra_variants=EXTRA_VARIANTS,
        spread="一跳下限(tick);Y1.0_2tick 为两跳",
        aum_accounting="主口径恒定 AUM(参与率与冲击 Q 都按 X);Y1.0_ride = 初始 X 随净值滚动",
        phase_note="5 相位平均,每相位按全额 X 模拟 = 整本组合每周调仓;5 份错开子组合的结果 = AUM X/5 那一格(容量 ×5)",
        capacity_def=dict(
            halflife="净超额(几何年化差 vs 池等权)降到 1e7 格一半时的 AUM,log10 线性插值;1e7 格 <=0 时无定义",
            breakeven="净超额降到 0 时的 AUM,同法插值;1e7 格已 <=0 时向 1e6 找;网格外记删失",
            uncertainty="算术超额曲线整体平移 ±1 个 NW(带宽 10)标准误后的盈亏平衡 AUM"),
        primary_benchmark="pool(V1000 池等权);csi1000p(同执行价 PIT 代理,全收益)与 csi1000r(真实价格指数)作参照",
        stats=dict(t="Newey-West 带宽 10 = max(5, 2h)", te_ir="excess_stats NW20 *_adj"),
        windows_rule="COMP_v5 开发窗 2021-2023 / 检验窗 2024-2025(2020H2 预热);B_gbt 2023-2025 全样本外",
        caveats=["COMP_v5 信号样本内", "无退市股(幸存者偏差,收益高估)", "冲击系数为文献量级未校准"],
    )


def prereg_supplement_payload() -> dict:
    """预注册补充(2026-10-10 审查后加入,**不是**事前预注册:写于看过审查探针之后、跑补充网格之前)。
    原主网格的定义与结论口径不变;补充项全部报告,不挑选。"""
    return dict(
        id="review_2026-10-10",
        post_hoc=True,
        reason=("审查 P1:参与率上限按 te 实际窗口成交额截量(10% POV 假定,按 TWAP 计价)不是事前量;"
                "审查探针显示它在约束区单独贡献 +1~2pp/年。审查 P1:现金按零收益计、基准满仓,决定了 legacy 等盈亏平衡。"),
        sizing=dict(
            grid="全部 384 格(4 策略 × 2 口径 × 6 成本设定 × 8 档资金)、5 相位,与主网格同一组设定",
            pov="引擎默认(主网格):cap = 10% × te 实际窗口成交额 / X",
            exante=(f"cap = 10% × 截至 te-1 的 {EXANTE_WINDOW} 日窗口成交额中位数(min_periods {EXANTE_MINP})/ X;"
                    "can 不变(估计缺失或 <=0 而实际 >0 时退回实际值);引擎代码不动,脚本内替换 exec_prices 的成交额"),
            report="头条半衰 / 盈亏平衡给 'POV ~ 事前' 区间;两者都按同一容量定义插值"),
        cash_valuation=dict(
            zero="主口径:现金零收益",
            pool="现金腿按 V1000 池等权收益(ret + cash×pool),乐观端",
            csi1000p="现金腿按中证1000 PIT 代理收益(ret + cash×csi1000p)",
            report="两档都给几何净超额与半衰 / 盈亏平衡;一阶近似,不改持仓路径"),
        arithmetic="恒定 AUM 下按元计的损益对应算术年化超额;头条表与几何并列,容量定义仍按预注册的几何口径",
        checks=["随机打分 Top20 安慰剂:pov − exante 配对差(8 个种子,两口径,legacy,1e6/1e7/3e7/1e8/3e8)",
                "B×GBT 2023-01-03 冷启动的相位空仓偏差:相位明细直接计算 + COMP_v5 Top20 替身(冷启动 vs 预热)",
                "exante 下实际参与率(按 te 实际窗口成交额)超过 10% 的成交额占比",
                "窗口成交额意外(实际 / 事前估计)与 te->te+1 收益的截面秩相关(机制)"],
    )


def prereg_fixa_payload() -> dict:
    """第二轮修订(fixa)的预注册补充:**只换基准**,写于看过审计探针之后、重算账本之前(post_hoc)。"""
    return dict(
        id="fixa_2026-10-10",
        post_hoc=True,
        reason=("第二轮审计(高级残留问题 1/2):k=1 池等权基准每天重置等权,5 日调仓组合对它的超额含与选股无关的"
                "再平衡频率差(常数打分全池等权、旧成本,vwap1 约 +2pp/年、open10 约 +0.5pp/年);"
                "中证1000 代理旧基准(收盘权重 × 成交价收益)系统性偏低(open10 约 0.8pp/年、vwap1 约 2.5pp/年)"),
        benchmark=dict(
            pool="ST.benchmark(p, 'V1000', mode, rebalance=5):同频再平衡池等权、5 相位平均 —— 容量定义改用它,两个口径都改",
            pool_d1="ST.benchmark(p, 'V1000', mode):预注册原口径,账本保留 *_pool_d1 列只作前后对照",
            csi1000p="IP.exec_time_bench_returns(p, mode):成交时点权重的中证1000 代理",
            csi1000p_legacy="IP.engine_consistent_bench_returns(p, mode):修订前,账本保留只作对照"),
        unchanged="网格、成本设定、截量口径、窗口、容量定义(半衰 / 盈亏平衡 / ±1SE 的算法)全部不变;策略日收益不重跑",
        checks=["全池等权常数打分 5 日调仓 vs 同频基准:超额 + 成本 ∈ [-0.5, +1.5]pp(2020H2 预热,2021-2025 切片)",
                "summarize 逐格复核 corr(策略, 同频基准.shift(k)) 峰值 k=0",
                "账本 *_pool_d1 列与修订前账本的 excess_pool 一致(证明只换了基准)"],
    )


def cmd_prereg_fixa():
    pay = json.loads(json.dumps(prereg_fixa_payload(), ensure_ascii=False))
    old = json.load(open(PREREG))
    sup = old.get("supplements", [])
    hit = [s for s in sup if s.get("id") == pay["id"]]
    if hit:
        if {k: v for k, v in hit[0].items() if k != "written_at"} != pay:
            raise RuntimeError("fixa 预注册补充已存在且与当前脚本不一致:不得在看到结果后改设定")
        print(f"fixa 预注册补充已存在且一致(written_at {hit[0].get('written_at')})")
        return
    pay["written_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    old["supplements"] = sup + [pay]
    with open(PREREG, "w") as f:
        json.dump(old, f, ensure_ascii=False, indent=1)
    print(f"已写 fixa 预注册补充 -> {PREREG}")


def cmd_prereg_supp():
    pay = json.loads(json.dumps(prereg_supplement_payload(), ensure_ascii=False))
    old = json.load(open(PREREG))
    sup = old.get("supplements", [])
    hit = [s for s in sup if s.get("id") == pay["id"]]
    if hit:
        if {k: v for k, v in hit[0].items() if k != "written_at"} != pay:
            raise RuntimeError("预注册补充已存在且与当前脚本不一致:不得在看到结果后改设定")
        print(f"预注册补充已存在且一致(written_at {hit[0].get('written_at')})")
        return
    pay["written_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    old["supplements"] = sup + [pay]
    with open(PREREG, "w") as f:
        json.dump(old, f, ensure_ascii=False, indent=1)
    print(f"已写预注册补充 {pay['id']} -> {PREREG}")


def cmd_prereg():
    pay = prereg_payload()
    if PREREG.exists():
        old = json.load(open(PREREG))
        old_cmp = {k: v for k, v in old.items() if k not in ("written_at", "supplements")}
        if old_cmp != json.loads(json.dumps(pay, ensure_ascii=False)):
            raise RuntimeError("预注册已存在且与当前脚本设定不一致:不得在看到结果后改设定")
        print(f"预注册已存在且一致:{PREREG}(written_at {old.get('written_at')})")
        return
    pay["written_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    with open(PREREG, "w") as f:
        json.dump(pay, f, ensure_ascii=False, indent=1)
    print(f"已写预注册 {PREREG}")


# ====================================================================== 重计算:一个口径的全部格
def run_cell(p: Panel, W: np.ndarray, mode: str, aum: float, variant: str, m: np.ndarray, models: dict,
             sizing: str = "pov"):
    cm, pon = cost_setup(p, variant, models)       # CostModel 在上下文外构造(它不读 exec_prices,双保险)
    with exante_sizing(p, on=(sizing == "exante")):
        res = ST.run_weights(W, p, mode, aum, rebalance=REBALANCE, date_mask=m, cost_model=cm,
                             record=True, participation_on_nav=pon)
    rec = res["record"]
    daily = rec["daily"][KEEP_COLS].copy()
    by = rec["daily_by_phase"]
    # part_bound_frac = 被截掉的 |Δw| 之和 / 意图 |Δw| 之和(**成交额口径**,不是股票数占比;审查 P2);
    # n_part_bound = 触及上限的股票数。
    rb = by[by["rebalance"].astype(bool)].groupby(level=0)[["part_bound_frac", "unfilled", "desired_turnover",
                                                          "part_cut", "n_part_bound"]].mean()
    daily = daily.join(rb.add_prefix("rb_"))
    daily["exp_max_phase"] = by.groupby(level=0)["exposure"].max()
    # 相位平均的 record.ret 必须等于引擎返回的 rets(同一组相位的均值)
    dret = float(np.nanmax(np.abs(daily["ret"].to_numpy() - res["rets"].reindex(daily.index).to_numpy())))
    daily["ret"] = res["rets"].reindex(daily.index)
    return res, daily, dret


def cmd_run(mode: str, sizing: str = "pov"):
    assert mode in MODES and sizing in SIZINGS
    if sizing != "pov":
        cmd_prereg_supp()                      # 补充网格必须先有补充预注册(已存在则核对一致)
    t0 = time.time()
    p = Panel()
    scores = load_scores(p)
    bench = bench_frame(p, mode)
    bench.to_parquet(BENCH_CACHE.format(mode))
    if sizing == "exante":
        st_ = exante_amount(p, mode)[1]
        print(f"exante 估计缺失退回实际值:{st_['fallback_n']} / {st_['actual_pos_n']} 个实际>0 的股票日(全面板)", flush=True)
    models: dict = {}
    dailies, cells = [], []
    for strat in STRATS:
        m = sim_mask(p, strat)
        W = strategy_weights(p, scores, strat)
        for variant in VARIANTS:
            for aum in AUMS:
                t1 = time.time()
                res, daily, dret = run_cell(p, W, mode, aum, variant, m, models, sizing)
                prof = align_profile(res["rets"], bench["pool"])
                peak = max(prof, key=prof.get)
                exp_max = float(daily["exp_max_phase"].max())
                if peak != 0:
                    raise AssertionError(f"{strat} {mode} {variant} {aum:.0e}: 对齐峰值 k={peak} {prof}")
                if exp_max > 1.0 + 1e-9 or res["exposure"] > 1.0 + 1e-9:
                    raise AssertionError(f"{strat} {mode} {variant} {aum:.0e}: exposure>1 ({exp_max})")
                if dret > 1e-12:
                    raise AssertionError(f"record.ret 与 rets 不一致 {dret}")
                cid = f"{strat}|{mode}|{variant}|{aum:.0e}"
                daily = daily.assign(cell=cid, strategy=strat, mode=mode, variant=variant, aum=aum, sizing=sizing)
                dailies.append(daily.reset_index())
                cells.append(dict(cell=cid, strategy=strat, mode=mode, variant=variant, aum=aum, sizing=sizing,
                                  ann_return_sim=res["ann_return"], exposure=res["exposure"],
                                  turnover=res["turnover"], holdings=res["holdings"], n_days=res["n_days"],
                                  exp_max=exp_max, align_peak_k=peak,
                                  **{f"align_k{k:+d}": v for k, v in prof.items()},
                                  rec_ret_maxdiff=dret, secs=time.time() - t1))
                print(f"  {cid:42s} ann {res['ann_return']*100:7.2f}%  exp {res['exposure']:.3f}  "
                      f"turn {res['turnover']:.4f}  peak k={peak}  {time.time()-t1:.1f}s", flush=True)
        del W
    D = pd.concat(dailies, ignore_index=True)
    D.to_parquet(daily_cache(mode, sizing))
    pd.DataFrame(cells).to_parquet(cell_cache(mode, sizing))
    print(f"done {mode} {sizing}: {len(cells)} 格, {time.time()-t0:.0f}s -> {daily_cache(mode, sizing)}")


# ====================================================================== 判据 / 复现 / 诊断
def cmd_checks():
    t0 = time.time()
    p = Panel()
    scores = load_scores(p)
    yrs = np.asarray(p.dates.year)
    rows = []

    # (1) eng_cost_demo 原配置逐位复现(证明本脚本调用引擎的方式与上游一致)
    demo = pd.read_csv(LEDGER / "eng_cost_demo.csv")
    cm1 = IM.CostModel(p, impact_y=1.0)
    for mode in MODES:
        m = (yrs >= 2021) & (yrs <= 2025)
        r = ST.run_strategy(scores["comp"], p, VERSION, mode, scheme="exclude", buffer=(0.20, 0.35),
                            rebalance=5, date_mask=m, cost_model=cm1)
        ref = demo.query("mode==@mode and window=='2021-2025' and plan=='exclude_std' and cost_model=='Y1.0_tick'")
        d = float(r["ann_return"] - ref["ann_return"].iloc[0])
        rows.append(dict(check="reproduce_eng_cost_demo", mode=mode, item="COMP_v5 排雷 1e7 Y1.0 tick 2021-2025",
                         value=r["ann_return"], reference=float(ref["ann_return"].iloc[0]), diff=d,
                         passed=abs(d) < 1e-12, note="eng_cost_demo.csv 同配置(participation 按初始资金,冲击按 capital×净值)"))
        print(rows[-1], flush=True)

    # (2) B_gbt Top20 逐年旧成本复现 ml_results_open_h5.csv(加载口径核对)
    ml = pd.read_csv(LEDGER / "ml_results_open_h5.csv").query(
        "feature_set=='B' and model=='gbt' and version=='V1000' and plan=='Top20周频'")
    for y in (2023, 2024, 2025):
        m = yrs == y
        r = ST.run_strategy(scores["bgbt"], p, VERSION, "open10", scheme="top", n=20, rebalance=5, date_mask=m)
        b = ST.benchmark(p, VERSION, "open10", date_mask=m)
        ref = ml.query("oos_year==@y").iloc[0]
        for item, v, rv in (("ann", r["ann_return"], ref["ann"]),
                            ("excess", r["ann_return"] - b["ann_return"], ref["excess"]),
                            ("exposure", r["exposure"], ref["exposure"])):
            rows.append(dict(check="reproduce_ml_results_B_gbt_top20", mode="open10", item=f"{y} {item}",
                             value=float(v), reference=float(rv), diff=float(v - rv),
                             passed=abs(v - rv) < 0.01, note="容差 1pp;逐年掩码、旧成本、1e7(与账本同调用)"))
            print(rows[-1], flush=True)

    # (3) 全池等权常数打分 vs 池基准:超额 + 成本 ≈ 0(可成交性残差 0.1~0.9pp 为已知非 bug)
    ones = pd.DataFrame(1.0, index=p.dates, columns=p.codes, dtype=np.float32)
    models: dict = {}
    for mode in MODES:
        m = (yrs >= 2021) & (yrs <= 2025)
        W = ST.build_weights(ones, p, VERSION, scheme="exclude", exclude_frac=0.0, date_mask=m)
        b = ST.benchmark(p, VERSION, mode, date_mask=m)
        for variant in ("legacy", "Y1.0"):
            cm, pon = cost_setup(p, variant, models)
            for aum in (1e6, 1e7):
                r = ST.run_weights(W, p, mode, aum, rebalance=1, date_mask=m, cost_model=cm,
                                   participation_on_nav=pon)
                cost_pp = r["cost_bps"] * 1e-4 * TRADING_DAYS * 100
                exc_pp = (r["ann_return"] - b["ann_return"]) * 100
                resid = exc_pp + cost_pp
                prof = align_profile(r["rets"], b["rets"])
                rows.append(dict(check="fullpool_resid_pp", mode=mode, item=f"{variant} {aum:.0e} 2021-2025",
                                 value=resid, reference=0.0, diff=resid,
                                 passed=(-0.5 <= resid <= 1.5) and max(prof, key=prof.get) == 0 and r["exposure"] <= 1,
                                 note=f"excess={exc_pp:.3f}pp cost={cost_pp:.3f}pp exposure={r['exposure']:.4f} "
                                      f"peak_k={max(prof, key=prof.get)}"))
                print(rows[-1], flush=True)
    # (3b) fixa:5 日调仓的全池等权 vs **同频**池等权基准(纪律 4)。2020H2 预热、报告 2021-2025 切片
    #      (不在 2021-01-04 冷启动:否则相位 k 前 k 天空仓,把 2021 年 1 月的空仓伪超额混进判据)。
    #      对照行 fullpool_resid_pp_r5_vs_d1 = 同一组合对 k=1 基准(修订前口径),显示再平衡频率差,不是判据。
    m_sim = sim_mask(p, "COMP_v5_exclude")
    W5 = ST.build_weights(ones, p, VERSION, scheme="exclude", exclude_frac=0.0, date_mask=m_sim)
    for mode in MODES:
        bb = {"samefreq": ST.benchmark(p, VERSION, mode, rebalance=REBALANCE)["rets"],
              "d1": ST.benchmark(p, VERSION, mode)["rets"]}
        for variant in ("legacy", "Y1.0"):
            cm, pon = cost_setup(p, variant, models)
            for aum in (1e6, 1e7):
                r = ST.run_weights(W5, p, mode, aum, rebalance=REBALANCE, date_mask=m_sim, cost_model=cm,
                                   participation_on_nav=pon, record=True)
                rs = r["rets"][r["rets"].index.year >= 2021]
                cost = r["record"]["daily"]["cost_total"].reindex(rs.index)
                for tag, b_ in bb.items():
                    b2 = b_.reindex(rs.index)
                    ex_g = (geo_ann(rs) - geo_ann(b2)) * 100
                    ex_a = float((rs - b2).mean() * TRADING_DAYS * 100)
                    c_pp = float(cost.mean() * TRADING_DAYS * 100)
                    prof = align_profile(r["rets"], b_)
                    pk = max(prof, key=prof.get)
                    chk = "fullpool_resid_pp_samefreq" if tag == "samefreq" else "fullpool_resid_pp_r5_vs_d1"
                    rows.append(dict(check=chk, mode=mode, item=f"{variant} {aum:.0e} 2021-2025 rebalance=5",
                                     value=ex_g + c_pp, reference=0.0, diff=ex_g + c_pp,
                                     passed=((-0.5 <= ex_g + c_pp <= 1.5) and pk == 0 and r["exposure"] <= 1)
                                     if tag == "samefreq" else True,
                                     note=(f"excess_geo={ex_g:.3f}pp excess_arith={ex_a:.3f}pp cost={c_pp:.3f}pp "
                                           f"resid_arith={ex_a + c_pp:.3f}pp exposure={r['exposure']:.4f} peak_k={pk};"
                                           + ("同频基准(判据)" if tag == "samefreq" else
                                              "对照:同一组合对 k=1 基准(修订前口径),不是判据"))))
                    print(rows[-1], flush=True)

    # (4) 指数对齐:真实 000852 与池基准的相关峰值(vwap1 改用 t+1->t+2 收盘后应在 k=0)
    for mode in MODES:
        b = ST.benchmark(p, VERSION, mode, rebalance=REBALANCE)["rets"]
        for tag, ri in (("index_returns_on_signal_days", IP.index_returns_on_signal_days(p, mode, "000852")),
                        ("cap_real_index_alignment", real_index_on_signal_days(p, mode, "000852")),
                        ("csi1000_proxy_same_exec", IP.engine_consistent_bench_returns(p, mode, "000852")),
                        ("csi1000_proxy_exec_time", IP.exec_time_bench_returns(p, mode))):
            mm = (b.index.year >= 2021)
            prof = align_profile(b[mm], ri[mm])
            pk = max(prof, key=prof.get)
            rows.append(dict(check="index_alignment_peak", mode=mode, item=tag, value=float(pk), reference=0.0,
                             diff=float(pk), passed=(pk == 0) or tag == "index_returns_on_signal_days",
                             note=" ".join(f"k={k:+d}:{v:.3f}" for k, v in prof.items())))
            print(rows[-1], flush=True)
    # (5) 起点敏感性(不是判据,是披露):COMP_v5 冷启动 2021-01-04 vs 2020H2 预热后切片。
    #     排雷缓冲带有滞后(跌破 20% 剔、回到 35% 才纳入),持仓状态依赖起点,同一信号不同起点的路径不同。
    m_cold = (yrs >= 2021) & (yrs <= 2025)
    for mode in MODES:
        b = ST.benchmark(p, VERSION, mode, date_mask=m_cold, rebalance=REBALANCE)
        for strat in ("COMP_v5_exclude", "COMP_v5_top20", "COMP_v5_top50"):
            plan = STRATS[strat]["plan"]
            W_cold = ST.build_weights(scores["comp"], p, VERSION, scheme=plan["scheme"], n=plan.get("n", 20),
                                      buffer=plan.get("buffer"), date_mask=m_cold)
            W_warm = strategy_weights(p, scores, strat)
            for variant in ("legacy", "Y1.0"):
                cm, pon = cost_setup(p, variant, models)
                r_c = ST.run_weights(W_cold, p, mode, REF_AUM, rebalance=REBALANCE, date_mask=m_cold,
                                     cost_model=cm, participation_on_nav=pon)
                r_w = ST.run_weights(W_warm, p, mode, REF_AUM, rebalance=REBALANCE, date_mask=sim_mask(p, strat),
                                     cost_model=cm, participation_on_nav=pon)
                rw = r_w["rets"][r_w["rets"].index.year >= 2021]
                e_c = r_c["ann_return"] - b["ann_return"]
                e_w = geo_ann(rw) - geo_ann(b["rets"].reindex(rw.index))
                # 差异落在哪:逐日对数收益差按月累加(冷启动 5 个相位在 2021 年初有 0~4 天空仓 + 建仓成本)
                ld = np.log1p(rw) - np.log1p(r_c["rets"].reindex(rw.index))
                jan = float(ld[(ld.index.year == 2021) & (ld.index.month == 1)].sum())
                rows.append(dict(check="start_date_sensitivity", mode=mode, item=f"{strat} {variant} 1e7 2021-2025",
                                 value=e_w, reference=e_c, diff=e_w - e_c, passed=True,
                                 note=(f"value=2020H2 预热后切片(本报告主口径)超额,reference=2021-01-04 冷启动超额;披露项。"
                                       f"累计对数收益差 {ld.sum()*100:.2f}pp,其中 2021-01 {jan*100:.2f}pp,"
                                       f"其余月份 {(ld.sum()-jan)*100:.2f}pp")))
                print(rows[-1], flush=True)

    # (6)~(9) 审查后补充(预注册补充 review_2026-10-10;披露项,不是判据)
    rows += exante_mechanism_rows(p)
    rows += bgbt_coldstart_rows(p, scores, models)
    rows += exante_participation_rows(p, scores, models)
    rows += placebo_rows(p)

    pd.DataFrame(rows).to_csv(LEDGER / "capacity_checks.csv", index=False)
    bad = [r for r in rows if not r["passed"]]
    print(f"checks: {len(rows)} 项, 未通过 {len(bad)}")
    for r in bad:
        print("  FAIL", r)

    diag = liquidity_diagnostics(p)
    diag.to_csv(LEDGER / "capacity_diagnostics.csv", index=False)
    print(diag.to_string())
    print(f"done checks {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------- 审查后补充的披露项
def _nw_t(x: np.ndarray, bw: int) -> float:
    """均值的 Newey-West t(Bartlett 核)。"""
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan
    u = x - x.mean()
    lrv = u @ u / n
    for L in range(1, bw + 1):
        lrv += 2.0 * (1.0 - L / (bw + 1)) * (u[L:] @ u[:-L]) / n
    return float(x.mean() / np.sqrt(lrv / n)) if lrv > 0 else np.nan


def exante_mechanism_rows(p: Panel) -> list:
    """(6) exante 估计的覆盖率 + 机制:窗口成交额"意外"(te 实际 / 事前估计)与 te->te+1 持有收益的截面秩相关。
    若显著为负,说明放量股次日偏弱;"按 te 实际成交额截量"(POV)的成交随当天放量 / 缩量变动,持仓路径因此带上
    执行日才揭示的信息 —— 与信号无关的执行口径差异。
    池按信号日 t = te - lag 的 V1000;2021-2025;t 用 NW(带宽 5,1 日收益不重叠)。"""
    rows = []
    U = ST.universe(p, VERSION).to_numpy(bool)
    yrs = np.asarray(p.dates.year)
    for mode in MODES:
        lag = ST.EXEC_MODES[mode]["lag"]
        px_df, amt_df = _ORIG_EXEC_PRICES(p, mode)
        a = amt_df.to_numpy(np.float64)
        est = amt_df.astype(np.float64).rolling(EXANTE_WINDOW, min_periods=EXANTE_MINP).median().shift(1).to_numpy()
        used = exante_amount(p, mode)[0].to_numpy(np.float64)
        Ute = np.zeros_like(U)
        Ute[lag:] = U[:len(U) - lag]
        sel = Ute & (yrs >= 2021)[:, None]
        pos = a > 0
        fb = pos & ~(np.isfinite(est) & (est > 0))
        rows.append(dict(check="exante_fallback_share", mode=mode, item="池内(信号日池)te 实际>0 的股票日中,事前估计缺失退回实际值的比例",
                         value=float(fb[sel & pos].mean()), reference=0.0, diff=float(fb[sel & pos].mean()), passed=True,
                         note=f"n={int((fb & sel).sum())};全面板 {exante_amount(p, mode)[1]['fallback_n']} / "
                              f"{exante_amount(p, mode)[1]['actual_pos_n']}(多为新股,不在池内)"))
        rr = a[sel & pos & np.isfinite(est) & (est > 0)] / est[sel & pos & np.isfinite(est) & (est > 0)]
        q = np.quantile(rr, [0.05, 0.5, 0.95])
        rows.append(dict(check="exante_actual_over_est", mode=mode, item="池内 te 实际窗口成交额 / 事前估计 的中位数",
                         value=float(q[1]), reference=1.0, diff=float(q[1] - 1.0), passed=True,
                         note=f"5%/95% 分位 {q[0]:.3f}/{q[2]:.3f};实际 < 估计 的比例 {(rr < 1).mean():.3f}"))
        px = px_df.to_numpy(np.float64)
        g = np.full_like(px, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            g[:-1] = px[1:] / px[:-1] - 1.0                 # 行 te:te -> te+1 持有收益(与引擎同口径)
            x = np.log(a / used)
        ics, dts = [], []
        for te in np.flatnonzero(sel.any(axis=1)):
            mk = sel[te] & np.isfinite(x[te]) & np.isfinite(g[te]) & pos[te]
            if mk.sum() < 100:
                continue
            ics.append(pd.Series(x[te][mk]).rank().corr(pd.Series(g[te][mk]).rank()))
            dts.append(p.dates[te])
        ic = np.asarray(ics)
        rows.append(dict(check="exante_mechanism_rankcorr", mode=mode,
                         item="rank corr(log 实际/事前 窗口成交额, te->te+1 收益),池内逐日均值 2021-2025",
                         value=float(ic.mean()), reference=0.0, diff=float(ic.mean()), passed=True,
                         note=f"NW t(带宽 5)={_nw_t(ic, 5):.2f};{len(ic)} 天;负值 = 放量股次日偏弱、缩量股偏强;POV 成交随当天放量 / 缩量变动"))
        print(rows[-3:], flush=True)
    return rows


def bgbt_coldstart_rows(p: Panel, scores: dict, models: dict) -> list:
    """(7) B×GBT 从 2023-01-03 空仓冷启动(无预热)的相位空仓偏差(审查 P2)。

    引擎把相位锚在 mask 首日:相位 k 在前 k 个信号日空仓。B×GBT 没有 2023 年之前的打分,无法预热,所以量化:
      (a) 相位明细直接算:相位 k 的空仓日 i<k 若按相位 0 当天的毛收益持有,少赚 Σ_k Σ_{i<k} log(1+r0_i) / 5;
          另给池基准版本 Σ_k Σ_{i<k} log(1+pool_i) / 5。这是一次性的对数收益缺口,摊到 2023 单年 ≈ 同值,
          摊到 2023-2025 ≈ 1/3。
      (b) 替身:COMP_v5 Top20 2023-01-03 冷启动 vs 2020H2 预热路径,2023 与 2023-2025 窗口的超额差(冷 − 预热)。"""
    rows = []
    strat = "B_gbt_top20"
    m = sim_mask(p, strat)
    W = strategy_weights(p, scores, strat)
    for mode in MODES:
        bench = ST.benchmark(p, VERSION, mode, rebalance=REBALANCE)["rets"]
        for variant in ("legacy", "Y1.0"):
            cm, pon = cost_setup(p, variant, models)
            res = ST.run_weights(W, p, mode, REF_AUM, rebalance=REBALANCE, date_mask=m, cost_model=cm,
                                 record=True, participation_on_nav=pon)
            by = res["record"]["daily_by_phase"]
            d0 = sorted(by.index.unique())[:REBALANCE]
            r0 = by[by["phase"] == 0]["gross_ret"].reindex(d0).to_numpy()
            exp_ = {k: by[by["phase"] == k]["exposure"].reindex(d0).to_numpy() for k in range(REBALANCE)}
            # 核对:相位 k 在前 k 天空仓
            ok = all(np.allclose(exp_[k][:k], 0.0) and exp_[k][k] > 0.5 for k in range(1, REBALANCE))
            sf_strat = sum(np.log1p(r0[:k]).sum() for k in range(REBALANCE)) / REBALANCE
            pb = bench.reindex(d0).to_numpy()
            sf_pool = sum(np.log1p(pb[:k]).sum() for k in range(REBALANCE)) / REBALANCE
            rows.append(dict(check="bgbt_coldstart_shortfall", mode=mode, item=f"B×GBT Top20 {variant} 1e7 冷启动缺口(相位明细)",
                             value=-sf_strat, reference=-sf_pool, diff=-sf_strat + sf_pool, passed=bool(ok),
                             note=(f"value=−Σ_kΣ_(i<k)log(1+相位0毛收益)/5(对数,一次性;负=冷启动低估),"
                                   f"reference=同式用池基准;前 5 个信号日相位0毛收益(%) {np.round(r0*100, 2).tolist()},"
                                   f"池基准(%) {np.round(pb*100, 2).tolist()};passed=相位 k 前 k 天空仓核对")))
            print(rows[-1], flush=True)
    # (b) 替身
    m_cold = np.asarray((p.dates >= pd.Timestamp("2023-01-01")) & (p.dates <= pd.Timestamp(RESEARCH_END)))
    plan = STRATS["COMP_v5_top20"]["plan"]
    Wc = ST.build_weights(scores["comp"], p, VERSION, scheme="top", n=plan["n"], date_mask=m_cold)
    Ww = strategy_weights(p, scores, "COMP_v5_top20")
    for mode in MODES:
        bench = ST.benchmark(p, VERSION, mode, rebalance=REBALANCE)["rets"]
        for variant in ("legacy", "Y1.0"):
            cm, pon = cost_setup(p, variant, models)
            rc = ST.run_weights(Wc, p, mode, REF_AUM, rebalance=REBALANCE, date_mask=m_cold, cost_model=cm,
                                participation_on_nav=pon)["rets"]
            rw = ST.run_weights(Ww, p, mode, REF_AUM, rebalance=REBALANCE, date_mask=sim_mask(p, "COMP_v5_top20"),
                                cost_model=cm, participation_on_nav=pon)["rets"]
            for y0, y1 in ((2023, 2023), (2023, 2025)):
                a_ = rc[(rc.index.year >= y0) & (rc.index.year <= y1)]
                b_ = rw[(rw.index.year >= y0) & (rw.index.year <= y1)]
                ec = geo_ann(a_) - geo_ann(bench.reindex(a_.index))
                ew = geo_ann(b_) - geo_ann(bench.reindex(b_.index))
                rows.append(dict(check="bgbt_coldstart_proxy", mode=mode,
                                 item=f"COMP_v5 Top20 替身 {variant} 1e7 {y0}-{y1}:冷启动 − 预热 超额",
                                 value=ec - ew, reference=0.0, diff=ec - ew, passed=True,
                                 note=f"冷启动(2023-01-03 起)超额 {ec*100:.2f}pp,预热(2020H2 起)超额 {ew*100:.2f}pp;"
                                      "负 = 冷启动低估;B×GBT 无法预热,用它估计 B×GBT 的偏差量级"))
                print(rows[-1], flush=True)
    return rows


def exante_participation_rows(p: Panel, scores: dict, models: dict) -> list:
    """(8) exante 截量下的**实际**参与率(按 te 实际窗口成交额):事前下单在缩量日会超过 10%,
    而冲击模型用全日 σ/ADV、不因 te 缩量多收费 —— 这一侧对 exante 有利。单相位 0,Y=1,从持仓重建 |Δw|。"""
    rows = []
    cases = (("COMP_v5_exclude", "open10", 1e10), ("COMP_v5_exclude", "vwap1", 1e10), ("COMP_v5_top20", "open10", 1e8),
             ("COMP_v5_top20", "vwap1", 1e8), ("B_gbt_top20", "open10", 1e8), ("B_gbt_top20", "vwap1", 1e8))
    Wc = {}
    for strat, mode, aum in cases:
        if strat not in Wc:
            Wc[strat] = strategy_weights(p, scores, strat)
        m = sim_mask(p, strat)
        lag = ST.EXEC_MODES[mode]["lag"]
        px_df, amt_df = _ORIG_EXEC_PRICES(p, mode)
        px, act = px_df.to_numpy(np.float64), amt_df.to_numpy(np.float64)
        cm, pon = cost_setup(p, "Y1.0", models)
        with exante_sizing(p):
            res = ST.simulate(Wc[strat], p, mode, aum, rebalance=REBALANCE, date_mask=m, phase=0, cost_model=cm,
                              record=True, participation_on_nav=pon)
        dl = res["record"]["daily"]; H = res["record"]["holdings"].astype(np.float64)
        tix = np.array([p.dates.get_loc(x) for x in dl.index])
        vals, parts, recon = [], [], []
        for j in range(1, len(dl)):
            if not bool(dl["rebalance"].iloc[j]) or dl.index[j].year < 2021:
                continue
            tep, te = tix[j - 1] + lag, tix[j] + lag
            ok = np.isfinite(px[tep]) & np.isfinite(px[tep + 1]) & (px[tep] > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                rr = np.where(ok, px[tep + 1] / px[tep] - 1.0, 0.0)
            hp = H[j - 1] * (1.0 + np.nan_to_num(rr)) / (1.0 + dl["ret"].iloc[j - 1])
            if hp.sum() > 1 + 1e-12:
                hp = hp / hp.sum()
            d = np.abs(H[j] - hp)
            recon.append(abs(d.sum() / 2 - dl["turnover"].iloc[j]))
            k = d > 1e-10
            with np.errstate(invalid="ignore", divide="ignore"):
                part = d[k] * aum / act[te][k]
            vals.append(d[k] * aum); parts.append(part)
        v, pt = np.concatenate(vals), np.concatenate(parts)
        fin = np.isfinite(pt)
        w = v[fin] / v[fin].sum()
        rows.append(dict(check="exante_realized_participation", mode=mode, item=f"{strat} {aum:.0e} Y1.0 相位0:成交额加权 实际参与率>10% 的占比",
                         value=float(w[pt[fin] > 0.10 + 1e-9].sum()), reference=0.0, diff=float(w[pt[fin] > 0.10 + 1e-9].sum()),
                         passed=bool(max(recon) < 1e-6),
                         note=(f">20% 占比 {w[pt[fin] > 0.20].sum():.3f};实际参与率 成交额加权中位 "
                               f"{float(np.interp(0.5, np.cumsum(w[np.argsort(pt[fin])]), np.sort(pt[fin]))):.3f}、"
                               f"95% 分位 {float(np.interp(0.95, np.cumsum(w[np.argsort(pt[fin])]), np.sort(pt[fin]))):.3f};"
                               f"重建换手误差 max {max(recon):.1e}(passed=重建 <1e-6);pov 口径按构造 ≤10%")))
        print(rows[-1], flush=True)
    return rows


PLACEBO_SEEDS = 8
PLACEBO_AUMS = (1e6, 1e7, 3e7, 1e8, 3e8)


def placebo_rows(p: Panel) -> list:
    """(9) 随机打分 Top20 安慰剂(与任何信号无关):同一组随机目标,pov 截量 − exante 截量 的配对超额差。
    系统性 > 0 = POV 截量本身(用 te 实际成交额)带来的、与信号无关的增益。legacy 成本(只看截量效应),
    2020H2 预热、报告 2021-2025;逐种子明细写 capacity_placebo.csv。时间序列 t:8 个种子平均后的日配对差,NW 带宽 10。"""
    rows, det = [], []
    m = sim_mask(p, "COMP_v5_top20")
    T, N = len(p.dates), len(p.codes)
    for mode in MODES:
        bench = ST.benchmark(p, VERSION, mode, rebalance=REBALANCE)["rets"]       # fixa:同频基准
        dsum = {a: None for a in PLACEBO_AUMS}
        for seed in range(PLACEBO_SEEDS):
            rng = np.random.default_rng(seed)
            sc = pd.DataFrame(rng.standard_normal((T, N)).astype(np.float32), index=p.dates, columns=p.codes)
            W = ST.build_weights(sc, p, VERSION, scheme="top", n=20, date_mask=m)
            del sc
            for aum in PLACEBO_AUMS:
                out = {}
                for sz in SIZINGS:
                    with exante_sizing(p, on=(sz == "exante")):
                        r = ST.run_weights(W, p, mode, aum, rebalance=REBALANCE, date_mask=m, cost_model=None,
                                           record=True)
                    rs = r["rets"][r["rets"].index.year >= 2021]
                    cash = r["record"]["daily"]["cash"].reindex(rs.index)
                    rb_ = bench.reindex(rs.index)
                    out[sz] = rs
                    # 现金按池收益补仓后的超额:把"pov 截量多买进(现金少)"的仓位效应剥掉,剩下的才是选股(缩量)效应
                    det.append(dict(mode=mode, seed=seed, aum=aum, sizing=sz,
                                    excess_pool=geo_ann(rs) - geo_ann(rb_),
                                    excess_pool_arith=float((rs - rb_).mean() * TRADING_DAYS),
                                    excess_pool_cashpool=geo_ann(rs + cash * rb_) - geo_ann(rb_),
                                    cash_mean=float(cash.mean()), exposure=r["exposure"], turnover=r["turnover"]))
                    del r
                dd = out["pov"] - out["exante"]
                dsum[aum] = dd if dsum[aum] is None else dsum[aum] + dd
            print(f"placebo {mode} seed {seed} done", flush=True)
        D = pd.DataFrame(det)
        D = D[D["mode"] == mode]
        for aum in PLACEBO_AUMS:
            a = D[(D.aum == aum) & (D.sizing == "pov")].set_index("seed")
            b = D[(D.aum == aum) & (D.sizing == "exante")].set_index("seed")
            dg = (a["excess_pool"] - b["excess_pool"])
            dc = (a["excess_pool_cashpool"] - b["excess_pool_cashpool"])
            tt = _nw_t((dsum[aum] / PLACEBO_SEEDS).to_numpy(), NW_T_BW)
            rows.append(dict(check="placebo_random_top20_pov_minus_exante", mode=mode,
                             item=f"legacy {aum:.0e} 2021-2025,{PLACEBO_SEEDS} 个随机种子",
                             value=float(dg.mean()), reference=0.0, diff=float(dg.mean()), passed=True,
                             note=(f"几何超额差 均值 {dg.mean()*100:+.2f}pp、种子间 sd {dg.std()*100:.2f}pp、"
                                   f"min/max {dg.min()*100:+.2f}/{dg.max()*100:+.2f}pp、>0 的种子 {int((dg > 0).sum())}/{len(dg)};"
                                   f"日配对差(种子平均)NW t={tt:.2f};pov 超额均值 {a['excess_pool'].mean()*100:+.2f}pp,"
                                   f"exante {b['excess_pool'].mean()*100:+.2f}pp;仓位 pov {a['exposure'].mean():.3f} / "
                                   f"exante {b['exposure'].mean():.3f};现金按池收益补仓后差 均值 {dc.mean()*100:+.2f}pp、"
                                   f"sd {dc.std()*100:.2f}pp、>0 的种子 {int((dc > 0).sum())}/{len(dc)}")))
            print(rows[-1], flush=True)
    pd.DataFrame(det).to_csv(LEDGER / "capacity_placebo.csv", index=False)
    return rows


def liquidity_diagnostics(p: Panel) -> pd.DataFrame:
    """决策池流动性逐年诊断 + 开盘 10 分钟窗口的"方差份额 / 成交额份额"。

    为什么要后者:平方根定律 Y·σ_H·sqrt(Q/V_H) 里 σ_H、V_H 应是同一执行时段的波动与成交额。
    引擎对 open10 用的是全日 σ 与全日 ADV。若窗口占全日方差的份额 f_var、占成交额的份额 f_vol,
    则窗口口径冲击 = 全日口径 × sqrt(f_var / f_vol)。f_var 用 Parkinson 区间方差之比估计
    (ln(open10_high/open10_low)² / ln(high/low)²,两者都由分钟 open/close 的极值构成,同口径),
    f_vol = amt_open10 / amount。只是量级诊断,不进入回测。"""
    U = ST.universe(p, VERSION).to_numpy(bool)
    amt = p.raw("amount").to_numpy(np.float64)
    a10 = p.raw("amt_open10").to_numpy(np.float64)
    adv = p["adv20"].to_numpy(np.float64)
    hi, lo = p.raw("high_px").to_numpy(np.float64), p.raw("low_px").to_numpy(np.float64)
    h10, l10 = p.raw("open10_high").to_numpy(np.float64), p.raw("open10_low").to_numpy(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pk_day = np.log(hi / lo) ** 2
        pk_win = np.log(h10 / l10) ** 2
        fvol = a10 / amt
    yrs = np.asarray(p.dates.year)
    rows = []
    for y in range(2021, 2026):
        sel = yrs == y
        Uy = U[sel]
        ok = Uy & np.isfinite(pk_day[sel]) & np.isfinite(pk_win[sel]) & (pk_day[sel] > 0) & np.isfinite(fvol[sel]) \
            & (amt[sel] > 0) & (fvol[sel] > 0)
        fvar_i = pk_win[sel][ok] / pk_day[sel][ok]
        fvol_i = fvol[sel][ok]
        rows += [
            dict(year=y, item="pool_n_median", value=float(np.median(Uy.sum(axis=1))), note="V1000 决策池每日股票数中位"),
            dict(year=y, item="pool_adv20_median_yuan", value=float(np.nanmedian(np.where(Uy, adv[sel], np.nan))),
                 note="池内个股 20 日均成交额中位(元)"),
            dict(year=y, item="pool_amount_sum_median_yuan",
                 value=float(np.median(np.nansum(np.where(Uy, amt[sel], 0.0), axis=1))), note="池内全日成交额合计,日中位(元)"),
            dict(year=y, item="pool_amt_open10_sum_median_yuan",
                 value=float(np.median(np.nansum(np.where(Uy, a10[sel], 0.0), axis=1))),
                 note="池内开盘 09:31-09:40 窗口成交额合计,日中位(元)"),
            dict(year=y, item="market_amount_sum_median_yuan",
                 value=float(np.median(np.nansum(amt[sel], axis=1))), note="面板全部股票全日成交额合计,日中位(元;不含退市股)"),
            dict(year=y, item="fvol_open10_median", value=float(np.median(fvol_i)),
                 note="池内个股日 amt_open10/amount 中位"),
            dict(year=y, item="fvol_open10_ratio_of_sums",
                 value=float(np.nansum(np.where(Uy, a10[sel], 0.0)) / np.nansum(np.where(Uy, amt[sel], 0.0))),
                 note="池内 Σamt_open10 / Σamount"),
            dict(year=y, item="fvar_open10_parkinson_median", value=float(np.median(fvar_i)),
                 note="池内个股日 ln(H10/L10)²/ln(H/L)² 中位"),
            dict(year=y, item="fvar_open10_parkinson_ratio_of_means", value=float(pk_win[sel][ok].mean() / pk_day[sel][ok].mean()),
                 note="池内 mean ln(H10/L10)² / mean ln(H/L)²"),
            dict(year=y, item="window_impact_scale_sqrt_fvar_over_fvol",
                 value=float(np.sqrt((pk_win[sel][ok].mean() / pk_day[sel][ok].mean())
                                     / (np.nansum(np.where(Uy, a10[sel], 0.0)) / np.nansum(np.where(Uy, amt[sel], 0.0))))),
                 note="sqrt(f_var/f_vol),均值口径:窗口口径冲击 / 全日口径冲击 的量级"),
            dict(year=y, item="window_impact_scale_median_stockday", value=float(np.median(np.sqrt(fvar_i / fvol_i))),
                 note="逐股票日 sqrt(f_var/f_vol) 中位"),
        ]
    return pd.DataFrame(rows)


# ====================================================================== 汇总
def geo_ann(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float((1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0)


def window_stats(d: pd.DataFrame, bench: pd.DataFrame, aum: float) -> dict:
    """一格一窗口的统计。d = 该格日频明细(索引信号日),bench = 三个基准的日收益。"""
    rs = d["ret"]
    out = dict(n_days=len(rs), ann_return=geo_ann(rs))
    # fixa:pool = 同频池等权、csi1000p = 成交时点代理(主口径);pool_d1 / csi1000p_legacy = 修订前,只作对照
    for bn in ("pool", "csi1000p", "csi1000r", "pool_d1", "csi1000p_legacy"):
        rb = bench[bn].reindex(rs.index)
        out[f"bench_{bn}_ann"] = geo_ann(rb)
        out[f"excess_{bn}"] = out["ann_return"] - out[f"bench_{bn}_ann"]
    es1 = ST.excess_stats(rs, bench["pool_d1"].reindex(rs.index), bandwidth=20, t_bandwidth=NW_T_BW)
    out.update(excess_pool_d1_arith=es1.get("excess_arith", np.nan), t_nw_pool_d1=es1.get("t_nw", np.nan))
    es = ST.excess_stats(rs, bench["pool"].reindex(rs.index), bandwidth=20, t_bandwidth=NW_T_BW)
    out.update(excess_pool_arith=es.get("excess_arith", np.nan), t_nw=es.get("t_nw", np.nan),
               te_adj=es.get("te_adj", np.nan), ir_adj=es.get("ir_adj", np.nan))
    out["se_excess_arith"] = out["excess_pool_arith"] / out["t_nw"] if out["t_nw"] else np.nan
    es2 = ST.excess_stats(rs, bench["csi1000p"].reindex(rs.index), bandwidth=20, t_bandwidth=NW_T_BW)
    out.update(t_nw_csi1000p=es2.get("t_nw", np.nan), ir_adj_csi1000p=es2.get("ir_adj", np.nan))
    gross = d["gross_ret"]
    out["gross_excess_pool_arith"] = float((gross - bench["pool"].reindex(rs.index)).mean() * TRADING_DAYS)
    for c in ("commission", "stamp", "spread", "impact", "total"):
        out[f"cost_{c}_bp"] = float(d[f"cost_{c}"].mean() * TRADING_DAYS * 1e4)
    out["cash_mean"] = float(d["cash"].mean())
    out["cash_drag_pp"] = float((d["cash"] * bench["pool"].reindex(rs.index)).mean() * TRADING_DAYS * 100)
    # 现金计价敏感性(审查 P1):现金腿按池等权 / 中证1000 代理收益补仓。相位平均的 cash 是各相位 cash 的均值,
    # ret + cash×b 对相位线性,所以等于"每个相位都补仓"后的相位平均(一阶:不改持仓路径)。
    for tag, bn in (("cashpool", "pool"), ("cashcsi", "csi1000p")):
        rb_ = bench[bn].reindex(rs.index)
        r_adj = rs + d["cash"] * rb_
        out[f"ann_return_{tag}"] = geo_ann(r_adj)
        out[f"excess_pool_{tag}"] = out[f"ann_return_{tag}"] - out["bench_pool_ann"]
        out[f"excess_pool_arith_{tag}"] = float((r_adj - bench["pool"].reindex(rs.index)).mean() * TRADING_DAYS)
    out["unfilled_rebal"] = float(d["rb_unfilled"].mean())
    # 成交额口径:调仓日被参与率截掉的 |Δw| / 意图 |Δw|(不是股票数占比,审查 P2);n_part_bound = 被截股票数
    out["part_bound_frac_rebal"] = float(d["rb_part_bound_frac"].mean())
    out["n_part_bound_rebal"] = float(d["rb_n_part_bound"].mean()) if "rb_n_part_bound" in d else np.nan
    out["turnover_daily"] = float(d["turnover"].mean())
    out["turnover_ann"] = out["turnover_daily"] * TRADING_DAYS
    traded = float((2.0 * d["turnover"]).sum())
    out["impact_per_traded_bp"] = float(d["cost_impact"].sum() / traded * 1e4) if traded > 0 else np.nan
    out["spread_per_traded_bp"] = float(d["cost_spread"].sum() / traded * 1e4) if traded > 0 else np.nan
    out["traded_value_ann_yuan"] = out["turnover_ann"] * 2.0 * aum           # 恒定 AUM 口径的年双边成交额
    out["holdings"] = float(d["n_hold"].mean())
    out["exposure_max"] = float(d["exp_max_phase"].max())
    nav = (1.0 + rs).cumprod()
    out["max_dd"] = float((nav / nav.cummax() - 1.0).min())
    out["vr_lr"] = ST.lr_variance_ratio(rs.to_numpy(np.float64))
    vol = float(rs.std() * np.sqrt(TRADING_DAYS))
    sh = float(rs.mean() / rs.std() * np.sqrt(TRADING_DAYS)) if rs.std() > 0 else np.nan
    out["sharpe_adj"] = sh / np.sqrt(out["vr_lr"]) if np.isfinite(out["vr_lr"]) and out["vr_lr"] > 0 else np.nan
    out["vol_adj"] = vol * np.sqrt(out["vr_lr"]) if np.isfinite(out["vr_lr"]) and out["vr_lr"] > 0 else np.nan
    out["aum_end_ride"] = float(aum * d["nav_prev"].iloc[-1] * (1 + rs.iloc[-1]))   # 只对 Y1.0_ride 有意义
    return out


def interp_cross(aums, ex, target: float, ref_idx: int):
    """从 ref_idx 出发找净超额首次 <= target 的位置,log10(AUM) 线性插值。
    返回 (aum, flag)。ex[ref] <= target 时向左找(盈亏平衡在 ref 之下)。"""
    la = np.log10(np.asarray(aums, float))
    ex = np.asarray(ex, float)
    if not np.isfinite(ex[ref_idx]):
        return np.nan, "nan"
    if ex[ref_idx] > target:
        for i in range(ref_idx + 1, len(ex)):
            if ex[i] <= target:
                x = la[i - 1] + (ex[i - 1] - target) / (ex[i - 1] - ex[i]) * (la[i] - la[i - 1])
                return float(10 ** x), "interp"
        return np.nan, "gt_max"
    for i in range(ref_idx - 1, -1, -1):
        if ex[i] > target:
            x = la[i] + (ex[i] - target) / (ex[i] - ex[i + 1]) * (la[i + 1] - la[i])
            return float(10 ** x), "interp_below_ref"
    return np.nan, "lt_min"


# 容量曲线:(列名后缀, 净超额列)。pool / csi1000p 是预注册定义;cashpool / cashcsi 是现金计价敏感性
# (vs 池等权,现金腿补仓);arith 是按元计的算术口径(恒定 AUM 下的元损益,审查 P2)。
CAP_CURVES = (("pool", "excess_pool"), ("csi1000p", "excess_csi1000p"),
              ("cashpool", "excess_pool_cashpool"), ("cashcsi", "excess_pool_cashcsi"),
              ("arith", "excess_pool_arith"),
              # fixa:修订前的两个基准(k=1 池等权、收盘权重代理),只作前后对照
              ("pool_d1", "excess_pool_d1"), ("csi1000p_legacy", "excess_csi1000p_legacy"))
GRID_KEYS = ["strategy", "mode", "variant", "sizing", "window", "window_role"]


def capacity_rows(G: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ref_idx = AUMS.index(REF_AUM)
    keys = GRID_KEYS
    for k, g in G.groupby(keys, sort=False):
        g = g.set_index("aum").reindex(AUMS)
        row = dict(zip(keys, k))
        for bn, col in CAP_CURVES:
            ex = g[col].to_numpy()
            e_ref = ex[ref_idx]
            row[f"excess_{bn}_1e6"] = ex[0]
            row[f"excess_{bn}_1e7"] = e_ref
            if np.isfinite(e_ref) and e_ref > 0:
                a, f = interp_cross(AUMS, ex, 0.5 * e_ref, ref_idx)
            else:
                a, f = np.nan, "ref_nonpositive"
            row[f"cap_half_{bn}"], row[f"cap_half_{bn}_flag"] = a, f
            a, f = interp_cross(AUMS, ex, 0.0, ref_idx)
            row[f"cap_be_{bn}"], row[f"cap_be_{bn}_flag"] = a, f
        # ±1 NW 标准误区间(算术超额 vs 池等权;曲线整体平移,假定成本 / 参与率的差分是确定的)
        ar = g["excess_pool_arith"].to_numpy()
        se = float(g["se_excess_arith"].iloc[ref_idx])
        row["excess_pool_arith_1e7"], row["se_arith_1e7"], row["t_nw_1e7"] = ar[ref_idx], se, float(g["t_nw"].iloc[ref_idx])
        for tag, sh in (("lo", -se), ("mid", 0.0), ("hi", +se)):
            a, f = interp_cross(AUMS, ar + sh, 0.0, ref_idx)
            row[f"cap_be_arith_{tag}"], row[f"cap_be_arith_{tag}_flag"] = a, f
        for c in ("cap_half_pool", "cap_be_pool", "cap_half_csi1000p", "cap_be_csi1000p"):
            row[f"{c}_tranche5"] = row[c] * 5 if np.isfinite(row[c]) else np.nan
        row["turnover_ann_1e7"] = float(g["turnover_ann"].iloc[ref_idx])
        row["holdings_1e7"] = float(g["holdings"].iloc[ref_idx])
        row["cash_1e7"] = float(g["cash_mean"].iloc[ref_idx])
        # 年化"元超额" = X × 算术净超额(恒定 AUM 下每年按元计的超额损益;**非预注册**,只在网格点上取最大)。
        # 现金按池收益补仓时净超额随 X 趋于 0(组合退化成基准),盈亏平衡不再是有意义的容量指标,
        # 元超额峰值所在的 X 才是"再加钱也不多赚"的规模(审查 P1 现金计价的补充读法)。
        xs = np.asarray(AUMS, float)
        for tag, col in (("cash0", "excess_pool_arith"), ("cashpool", "excess_pool_arith_cashpool")):
            ya = xs * g[col].to_numpy()
            i = int(np.nanargmax(ya))
            row[f"yuan_alpha_max_{tag}"], row[f"aum_at_yuan_alpha_max_{tag}"] = float(ya[i]), float(xs[i])
        # 盈亏平衡点附近的现金比例(log-AUM 线性插值;删失时取端点)——判断盈亏平衡是不是"现金拖累决定"的
        la = np.log10(np.asarray(AUMS, float)); cash = g["cash_mean"].to_numpy()
        for c in ("cap_be_pool", "cap_be_cashpool"):
            a = row[c]
            row[f"cash_at_{c}"] = (float(np.interp(np.log10(a), la, cash)) if np.isfinite(a) else
                                   float(cash[-1]) if row[f"{c}_flag"] == "gt_max" else
                                   float(cash[0]) if row[f"{c}_flag"] == "lt_min" else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def decomp_rows(G: pd.DataFrame) -> pd.DataFrame:
    """净超额相对 1e7 的损失 = 毛超额变化(参与率 / 现金拖累 / 未成交)- 显式成本变化,全部算术年化 pp。"""
    rows = []
    keys = GRID_KEYS
    for k, g in G.groupby(keys, sort=False):
        g = g.set_index("aum").reindex(AUMS)
        r0 = g.loc[REF_AUM]
        for aum in AUMS:
            r = g.loc[aum]
            d_net = (r["excess_pool_arith"] - r0["excess_pool_arith"]) * 100
            d_gross = (r["gross_excess_pool_arith"] - r0["gross_excess_pool_arith"]) * 100
            d_cost = (r["cost_total_bp"] - r0["cost_total_bp"]) / 100
            rows.append(dict(zip(keys, k), aum=aum, d_net_pp=d_net, d_gross_pp=d_gross, d_cost_pp=d_cost,
                             d_spread_pp=(r["cost_spread_bp"] - r0["cost_spread_bp"]) / 100,
                             d_impact_pp=(r["cost_impact_bp"] - r0["cost_impact_bp"]) / 100,
                             d_cash_drag_pp=r["cash_drag_pp"] - r0["cash_drag_pp"],
                             d_cash=r["cash_mean"] - r0["cash_mean"],
                             identity_resid_pp=d_net - (d_gross - d_cost)))
    return pd.DataFrame(rows)


def scaling_rows(G: pd.DataFrame) -> pd.DataFrame:
    """量纲分析(**非预注册**,只用来解释网格结果、给右删失的格子一个外推量级)。

    净超额 = 毛超额 - 成交额 ×(线性成本 + 冲击),都按"每一元单边成交额"计:
        α_tr  = 毛超额(算术,vs 池等权)/ 年双边换手      —— 每成交一元能换来的超额
        c_lin = (佣金 + 印花税 + 半价差)/ 年双边换手      —— 与规模无关
        c_imp(X) = 冲击 / 年双边换手 ∝ Y σ sqrt(X × 单笔权重 / ADV) ∝ sqrt(X)(交易路径不变时)
    令 α_tr = c_lin + c_imp(X_ref)·sqrt(X/X_ref) 得 X_be ≈ X_ref × ((α_tr - c_lin)/c_imp(X_ref))²。
    参考点取 1e6(参与率几乎不约束,交易路径 = 无摩擦路径)。

    ⚠ 方向(审查 P2,旧 docstring 写反):参与率约束一旦生效,引擎会截掉 / 推迟交易,换手下降、冲击随之少于
    "交易路径不变"的外推,所以外推**低估**引擎路径下的净超额;排雷 open10 实测网格在 1e9 / 3e9 / 1e10 都高于
    外推(pred_net_arith_* vs grid_net_arith_*),外推的 109 亿只是下界。集中持仓被截量时还会多出现金拖累,
    两股力量方向相反,外推对它们只作量级参考。
    每元冲击因子 = (每元冲击 / 排雷每元冲击)²:混合了"持仓集中度"(单笔相对 ADV 的大小,∝ 1/持仓数)与
    "选中个股的 σ / ADV"(审查 P2:B×GBT 与 COMP Top20 同为 20 只、换手相近,每元冲击仍高约 1.3 倍,
    这部分来自个股特征而非集中度)。impact_vs_comp_top20 给出同窗口同持仓数的比值。"""
    rows = []
    full = G[(G.window_role.isin(["full", "dev", "test"])) & G.variant.isin(PRIMARY_VARIANTS)]
    for (s, m, v, sz, w), g in full.groupby(["strategy", "mode", "variant", "sizing", "window"], sort=False):
        g = g.set_index("aum").reindex(AUMS)
        r1, r7 = g.loc[1e6], g.loc[1e7]
        tr1 = 2.0 * r1["turnover_ann"]
        alpha_tr = r1["gross_excess_pool_arith"] * 1e4 / tr1
        c_lin = (r1["cost_commission_bp"] + r1["cost_stamp_bp"] + r1["cost_spread_bp"]) / tr1
        c_imp1 = r1["cost_impact_bp"] / tr1
        x_be = 1e6 * ((alpha_tr - c_lin) / c_imp1) ** 2 if alpha_tr > c_lin and c_imp1 > 0 else np.nan
        expo = np.log10(r7["impact_per_traded_bp"] / r1["impact_per_traded_bp"]) if r1["impact_per_traded_bp"] > 0 else np.nan
        row = dict(strategy=s, mode=m, variant=v, sizing=sz, window=w, turnover_ann_1e6=r1["turnover_ann"],
                   holdings_1e6=r1["holdings"], gross_excess_arith_1e6_pp=r1["gross_excess_pool_arith"] * 100,
                   alpha_per_traded_bp=alpha_tr, lin_cost_per_traded_bp=c_lin,
                   spread_per_traded_bp=r1["spread_per_traded_bp"], impact_per_traded_bp_1e6=c_imp1,
                   impact_per_traded_bp_1e7=r7["impact_per_traded_bp"],
                   impact_exponent_1e6_1e7=expo, part_bound_frac_rebal_1e6=r1["part_bound_frac_rebal"],
                   part_bound_frac_rebal_1e7=r7["part_bound_frac_rebal"],
                   x_be_sqrt_extrap=x_be,
                   note="非预注册量纲外推;参与率约束时外推低估引擎路径(截量减少交易与冲击),集中持仓另有现金拖累")
        # 外推 vs 网格(算术净超额,pp/年):pred = 双边换手(1e6) × (α_tr - c_lin - c_imp1·sqrt(X/1e6))
        for a in (1e9, 3e9, 1e10):
            row[f"pred_net_arith_{a:.0e}"] = tr1 * (alpha_tr - c_lin - c_imp1 * np.sqrt(a / 1e6)) / 100
            row[f"grid_net_arith_{a:.0e}"] = g.loc[a, "excess_pool_arith"] * 100
            row[f"turnover_ann_{a:.0e}"] = g.loc[a, "turnover_ann"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out["margin_bp"] = out["alpha_per_traded_bp"] - out["lin_cost_per_traded_bp"]
    # 相对排雷方案(同口径、同 Y、同截量、同窗口名)的容量缩小倍数拆分:X_be(排雷)/X_be = 每元冲击因子 × 毛利因子
    #   每元冲击因子 = (c_imp / c_imp_排雷)²   —— 每元冲击之比(集中度 + 选中个股的 σ/ADV)
    #   毛利因子     = (margin_排雷 / margin)²  —— 每元成交的"超额减线性成本"之比(换手稀释 + 价差)
    # 窗口名不匹配(B_gbt 的 2023-2025)或毛利 <= 0 时为 NaN。
    kcols = ["mode", "variant", "sizing", "window"]
    key = pd.MultiIndex.from_frame(out[kcols])
    ex = out[out.strategy == "COMP_v5_exclude"].set_index(kcols)
    out["impact_penalty_vs_exclude"] = (out["impact_per_traded_bp_1e6"]
                                        / ex["impact_per_traded_bp_1e6"].reindex(key).to_numpy()) ** 2
    m_ex = ex["margin_bp"].reindex(key).to_numpy()
    out["margin_penalty_vs_exclude"] = np.where(out["margin_bp"] > 0, (m_ex / out["margin_bp"]) ** 2, np.nan)
    # 同持仓数(20 只)、同窗口下 B×GBT / COMP_v5 Top20 的每元冲击之比:剥掉集中度后剩下的个股特征部分
    t20 = out[out.strategy == "COMP_v5_top20"].set_index(kcols)["impact_per_traded_bp_1e6"]
    out["impact_vs_comp_top20"] = out["impact_per_traded_bp_1e6"] / t20.reindex(key).to_numpy()
    return out


def grid_check_rows(G: pd.DataFrame) -> list:
    """网格层面的回归判据,追加到 capacity_checks.csv(check 以 grid_ 开头)。"""
    rows = [dict(check="grid_exposure_le_1", mode="all", item="全部格子相位明细最大仓位",
                 value=float(G["exposure_max"].max()), reference=1.0, diff=float(G["exposure_max"].max() - 1.0),
                 passed=bool(G["exposure_max"].max() <= 1 + 1e-9), note=f"{len(G)} 行(格 × 窗口)"),
            dict(check="grid_align_peak_k0", mode="all", item="corr(策略, 池基准.shift(k)) 峰值 k=0 的格子比例",
                 value=float((G["align_peak_k"] == 0).mean()), reference=1.0,
                 diff=float((G["align_peak_k"] == 0).mean() - 1.0), passed=bool((G["align_peak_k"] == 0).all()),
                 note="run 阶段逐格断言(修订前 k=1 基准);此处复核"),
            dict(check="grid_align_peak_k0_samefreq", mode="all",
                 item="corr(策略, 同频池基准.shift(k)) 峰值 k=0 的格子比例(fixa)",
                 value=float((G["align_peak_k_samefreq"] == 0).mean()), reference=1.0,
                 diff=float((G["align_peak_k_samefreq"] == 0).mean() - 1.0),
                 passed=bool((G["align_peak_k_samefreq"] == 0).all()), note="summarize 逐格重算")]
    f = G[G.window_role == "full"]
    for sz in SIZINGS:
        fs = f[f.sizing == sz]
        a = fs[fs.variant == "legacy"].set_index(["strategy", "mode", "aum"])["turnover_daily"]
        b = fs[fs.variant == "Y1.0"].set_index(["strategy", "mode", "aum"])["turnover_daily"]
        rel = ((b - a.reindex(b.index)) / a.reindex(b.index)).abs()
        rows.append(dict(check="grid_same_trades_legacy_vs_Y1", mode="all",
                         item=f"[{sz}] 恒定 AUM 下 legacy 与 Y1.0 日换手相对差的最大值",
                         value=float(rel.max()), reference=0.0, diff=float(rel.max()), passed=bool(rel.max() < 0.05),
                         note="成本只从收益里扣、不改变目标与截量;差异来自净值路径对漂移的二阶影响"))
        v = fs[(fs["mode"] == "vwap1") & (fs.variant == "Y1.0")].set_index(["strategy", "aum"])["impact_per_traded_bp"]
        for s in STRATS:
            e = float(np.log10(v.loc[(s, 1e7)] / v.loc[(s, 1e6)]))
            rows.append(dict(check="grid_impact_sqrt_scaling", mode="vwap1", item=f"[{sz}] {s} 每元冲击 1e6->1e7 的弹性",
                             value=e, reference=0.5, diff=e - 0.5, passed=bool(abs(e - 0.5) < 0.05),
                             note="平方根定律:交易路径不变时每元冲击 ∝ sqrt(AUM)"))
    # exante vs pov 在 1e6(参与率几乎不约束)应几乎相同:差异只来自偶发截量。判据(补充预注册前定):|差| < 0.1pp
    k = ["strategy", "mode", "variant", "window", "aum"]
    po = G[G.sizing == "pov"].set_index(k)["excess_pool"]
    ea = G[G.sizing == "exante"].set_index(k)["excess_pool"].reindex(po.index)
    d6 = (po - ea)[po.index.get_level_values("aum") == 1e6].abs()
    rows.append(dict(check="grid_exante_vs_pov_at_1e6", mode="all", item="1e6 格 |pov − exante| 净超额最大值(全部策略/成本/窗口)",
                     value=float(d6.max()), reference=0.0, diff=float(d6.max()), passed=bool(d6.max() < 0.001),
                     note=f"中位 {float(d6.median())*100:.4f}pp;1e6 时参与率几乎不约束,两种截量口径应一致"))
    return rows


def cmd_summarize():
    t0 = time.time()
    grid = []
    for mode in MODES:
      B = pd.read_parquet(BENCH_CACHE.format(mode))
      for sizing in SIZINGS:
        D = pd.read_parquet(daily_cache(mode, sizing))
        C = pd.read_parquet(cell_cache(mode, sizing)).set_index("cell")
        for cid, d in D.groupby("cell", sort=False):
            d = d.set_index("date").sort_index()
            strat, variant, aum = d["strategy"].iloc[0], d["variant"].iloc[0], float(d["aum"].iloc[0])
            prof_sf = align_profile(d["ret"], B["pool"])           # fixa:对同频基准逐格复核对齐
            pk_sf = max(prof_sf, key=prof_sf.get)
            for wname, (y0, y1, role) in STRATS[strat]["windows"].items():
                dw = d[(d.index.year >= y0) & (d.index.year <= y1)]
                st = window_stats(dw, B, aum)
                grid.append(dict(strategy=strat, mode=mode, variant=variant, sizing=sizing,
                                 Y={"legacy": 0.0, "Y0.5": 0.5, "Y1.0": 1.0, "Y1.5": 1.5,
                                    "Y1.0_2tick": 1.0, "Y1.0_ride": 1.0}[variant],
                                 spread={"legacy": "none", "Y1.0_2tick": "2tick"}.get(variant, "tick"),
                                 aum_mode="ride" if variant == "Y1.0_ride" else "const",
                                 aum=aum, window=wname, window_role=role,
                                 signal_oos=STRATS[strat]["signal_oos"],
                                 align_peak_k=int(C.loc[cid, "align_peak_k"]), align_peak_k_samefreq=int(pk_sf),
                                 **st))
        del D
    G = pd.DataFrame(grid)
    # 相对 1e6 同格的现金增量 = 参与率约束带来的现金(1e6 下现金只来自一字板 / 停牌残仓)
    k4 = ["strategy", "mode", "variant", "sizing", "window"]
    base = G[G["aum"] == AUMS[0]].set_index(k4)["cash_mean"]
    G["cash_vs_1e6"] = G["cash_mean"] - base.reindex(pd.MultiIndex.from_frame(G[k4])).to_numpy()
    G["note"] = np.where(G["signal_oos"], "B_gbt walk-forward 样本外", "COMP_v5 信号样本内(只测组合/成本层)")
    G.to_csv(LEDGER / "capacity_grid.csv", index=False)
    S = capacity_rows(G)
    S.to_csv(LEDGER / "capacity_summary.csv", index=False)
    DC = decomp_rows(G)
    DC.to_csv(LEDGER / "capacity_decomp.csv", index=False)
    assert G["exposure_max"].max() <= 1.0 + 1e-9 and (G["align_peak_k"] == 0).all()
    assert (G["align_peak_k_samefreq"] == 0).all(), "有格子与同频基准错位"
    assert DC["identity_resid_pp"].abs().max() < 1e-6, "分解恒等式不成立"
    SC = scaling_rows(G)
    SC.to_csv(LEDGER / "capacity_scaling.csv", index=False)
    ck_path = LEDGER / "capacity_checks.csv"
    ck = pd.read_csv(ck_path) if ck_path.exists() else pd.DataFrame()
    if len(ck):
        ck = ck[~ck["check"].astype(str).str.startswith("grid_")]
    gck = pd.DataFrame(grid_check_rows(G))
    pd.concat([ck, gck], ignore_index=True).to_csv(ck_path, index=False)
    print(gck.to_string())
    print(SC.round(3).to_string())

    show = S[S.variant.isin(VARIANTS)][["strategy", "mode", "variant", "sizing", "window", "excess_pool_1e7",
                                        "cap_half_pool", "cap_half_pool_flag", "cap_be_pool", "cap_be_pool_flag",
                                        "cap_be_arith_lo", "cap_be_arith_hi", "cap_be_csi1000p", "cap_be_csi1000p_flag"]]
    print(show.to_string())
    make_plots(G, S)
    print(f"done summarize {time.time()-t0:.0f}s")


# ====================================================================== 图
COLORS = {"COMP_v5_exclude": "#2a78d6", "COMP_v5_top20": "#eb6834", "COMP_v5_top50": "#1baf7a",
          "B_gbt_top20": "#eda100"}
LABELS = {"COMP_v5_exclude": "COMP_v5 排雷 20/35", "COMP_v5_top20": "COMP_v5 Top20",
          "COMP_v5_top50": "COMP_v5 Top50", "B_gbt_top20": "B×GBT Top20 (OOS)"}


def _setup_fonts():
    import matplotlib
    from matplotlib import font_manager
    for f in ("PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti", "Arial Unicode MS", "Songti SC"):
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def _style_ax(ax, ink2, grid_c):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid_c)
    ax.tick_params(colors=ink2, labelsize=8.5)
    ax.grid(True, axis="y", color=grid_c, lw=0.7)
    ax.set_axisbelow(True)


AUM_LABELS = ["100万", "1000万", "3000万", "1亿", "3亿", "10亿", "30亿", "100亿"]


def make_plots(G: pd.DataFrame, S: pd.DataFrame):
    """两张图(单一 y 轴,不用双轴):
    capacity_curve.png     净超额 vs log AUM;实线 Y=1、色带 Y=0.5~1.5、点线 legacy(只扣佣金印花税)
    cap_cost_breakdown.png 上两行:Y=1 成本分解堆叠柱;下一行:平均现金比例(参与率约束)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    _setup_fonts()
    ink, ink2, grid_c, surf = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"
    full_all = G[G["window_role"] == "full"]
    full = full_all[full_all["sizing"] == "pov"]            # 主网格;exante 只在图 1 画虚线
    S = S[S["sizing"] == "pov"]

    # ---------------- 图 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), sharey=True, facecolor=surf)
    for ax, mode in zip(axes, MODES):
        ax.set_facecolor(surf)
        for strat in STRATS:
            g = full[(full.strategy == strat) & (full["mode"] == mode)]
            piv = g.pivot_table(index="aum", columns="variant", values="excess_pool").reindex(AUMS) * 100
            c = COLORS[strat]
            ax.fill_between(AUMS, piv["Y0.5"], piv["Y1.5"], color=c, alpha=0.15, linewidth=0)
            ax.plot(AUMS, piv["legacy"], color=c, lw=1.2, ls=":")
            ax.plot(AUMS, piv["Y1.0"], color=c, lw=2, marker="o", ms=5, mec=surf, mew=1)
            ge = full_all[(full_all.strategy == strat) & (full_all["mode"] == mode) & (full_all.sizing == "exante")
                          & (full_all.variant == "Y1.0")].set_index("aum").reindex(AUMS)
            ax.plot(AUMS, ge["excess_pool"] * 100, color=c, lw=1.4, ls="--")
            srow = S[(S.strategy == strat) & (S["mode"] == mode) & (S.variant == "Y1.0") & (S.window_role == "full")]
            if len(srow) and np.isfinite(srow["cap_be_pool"].iloc[0]):
                ax.plot([srow["cap_be_pool"].iloc[0]], [0], marker="v", ms=9, color=c, mec=ink, mew=0.8, zorder=5)
        ax.axhline(0, color=ink2, lw=1)
        ax.axvline(REF_AUM, color=ink2, lw=0.8, ls="--", alpha=0.5, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(AUMS)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(AUM_LABELS)
        _style_ax(ax, ink2, grid_c)
        ax.set_title("open10:T+1 开盘 09:31-09:40 TWAP" if mode == "open10" else "vwap1:T+1 全日 VWAP",
                     color=ink, fontsize=12, loc="left")
        ax.set_xlabel("AUM(恒定规模,对数轴;灰色竖线 = 1000 万参考点)", color=ink2, fontsize=9)
    axes[0].set_ylabel("净超额 vs V1000 同频(5 日)池等权(pp/年,几何)", color=ink2)
    h = [Line2D([], [], color=COLORS[s], lw=2, marker="o", ms=5,
                label=LABELS[s] + (" · 2023-25" if s.startswith("B_") else " · 2021-25")) for s in STRATS]
    h += [Line2D([], [], color=ink2, lw=2, label="实线:Y=1(基准)"),
          Patch(facecolor=ink2, alpha=0.2, label="色带:Y=0.5~1.5"),
          Line2D([], [], color=ink2, lw=1.4, ls="--", label="虚线:Y=1、参与率按事前估计截量"),
          Line2D([], [], color=ink2, lw=1.2, ls=":", label="点线:只扣佣金+印花税(参与率约束照常)"),
          Line2D([], [], color=ink2, lw=0, marker="v", ms=8, mec=ink, label="Y=1 盈亏平衡(网格内)")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=9, labelcolor=ink,
               bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("冲击成本下的容量曲线(一跳半价差 + 平方根冲击,参与率上限 10% 窗口成交额;实线 = te 实际成交额截量)",
                 color=ink, fontsize=12.5, x=0.01, ha="left")
    fig.text(0.01, 0.008, "股票池固定 V1000(ADV≥2000万);COMP_v5 为信号样本内,B×GBT 为 walk-forward 样本外;"
             "高 AUM 下集中持仓大部分是现金(按零收益计),净超额趋近 -基准收益;数据无退市股,收益偏高。",
             color=ink2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(REPORTS / "capacity_curve.png", dpi=150, facecolor=surf)
    plt.close(fig)

    # ---------------- 图 2:成本分解(上两行)+ 现金比例(下一行),各自单轴
    comp_cols = [("cost_commission_bp", "佣金", "#eda100"), ("cost_stamp_bp", "印花税", "#1baf7a"),
                 ("cost_spread_bp", "半价差(一跳)", "#eb6834"), ("cost_impact_bp", "平方根冲击", "#2a78d6")]
    fig = plt.figure(figsize=(15, 10), facecolor=surf)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.9], hspace=0.45, wspace=0.25)
    x = np.arange(len(AUMS))
    for i, mode in enumerate(MODES):
        for j, strat in enumerate(STRATS):
            ax = fig.add_subplot(gs[i, j])
            ax.set_facecolor(surf)
            g = full[(full.strategy == strat) & (full["mode"] == mode) & (full.variant == "Y1.0")].set_index("aum").reindex(AUMS)
            bottom = np.zeros(len(AUMS))
            for col, lab, c in comp_cols:
                v = g[col].to_numpy() / 100
                ax.bar(x, v, bottom=bottom, color=c, width=0.72, edgecolor=surf, linewidth=1.5, label=lab)
                bottom += v
            _style_ax(ax, ink2, grid_c)
            ax.set_xticks(x)
            ax.set_xticklabels(["1e6", "1e7", "3e7", "1e8", "3e8", "1e9", "3e9", "1e10"], fontsize=8)
            ax.set_title(f"{LABELS[strat]} · {mode}", fontsize=10, color=ink, loc="left")
            if j == 0:
                ax.set_ylabel("成本 pp/年", color=ink2)
    for k, mode in enumerate(MODES):
        ax = fig.add_subplot(gs[2, 2 * k:2 * k + 2])
        ax.set_facecolor(surf)
        for strat in STRATS:
            g = full[(full.strategy == strat) & (full["mode"] == mode) & (full.variant == "Y1.0")].set_index("aum").reindex(AUMS)
            ax.plot(x, g["cash_mean"].to_numpy() * 100, color=COLORS[strat], lw=2, marker="o", ms=5,
                    mec=surf, mew=1, label=LABELS[strat])
        _style_ax(ax, ink2, grid_c)
        ax.set_ylim(-3, 103)
        ax.set_xticks(x)
        ax.set_xticklabels(AUM_LABELS, fontsize=8.5)
        ax.set_title(f"平均现金比例(参与率约束未成交,%)· {mode}", fontsize=10, color=ink, loc="left")
        if k == 0:
            ax.set_ylabel("现金 %", color=ink2)
            ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    hb, lb = fig.axes[0].get_legend_handles_labels()
    fig.legend(hb, lb, loc="upper right", ncol=4, frameon=False, fontsize=9.5)
    fig.suptitle("Y=1、一跳价差下的成本分解(柱,pp/年)与平均现金比例;恒定 AUM,全期",
                 color=ink, fontsize=12, x=0.01, ha="left")
    fig.savefig(REPORTS / "cap_cost_breakdown.png", dpi=140, facecolor=surf, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summarize"
    if cmd == "prereg":
        cmd_prereg()
    elif cmd == "prereg_supp":
        cmd_prereg_supp()
    elif cmd == "run":
        cmd_prereg()
        cmd_run(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "pov")
    elif cmd == "checks":
        cmd_checks()
    elif cmd == "summarize":
        cmd_summarize()
    elif cmd == "prereg_fixa":
        cmd_prereg_fixa()
    elif cmd == "bench":
        cmd_bench()
    else:
        raise SystemExit(f"未知子命令 {cmd}")
