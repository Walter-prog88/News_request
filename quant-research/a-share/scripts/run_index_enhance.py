"""中证 1000 指数增强:约束优化器回测 + 评价(2026-10-10,前缀 ie)。

## 做什么
用 afactor.optimizer.IndexEnhancer(Grinold α + Barra 风险模型 + 引擎同一成本模型 + cvxpy)
在中证 1000 PIT 代理基准上做"约束下的指数增强",经 strategy.simulate 的 callable 接口逐日成交模拟,
与同信号的 Top20 等权 / 标准排雷对照,评价超额、跟踪误差、约束是否生效、事后归因与容量。

## 口径(全部预注册,见 src/afactor/optimizer.py 模块文档)
- 期间:风险模型首个预测日 2021-04-20 ~ 2025-12-31(2020 与 2021-01~04 只作预热:风险模型需要 60 个
  纯因子收益观测才出协方差,纯因子收益又要 252 日 Beta 窗口)。开发窗 2021-04-20~2023-12-31,
  检验窗 2024-01-01~2025-12-31;2024-01-02~2024-02-07 单列。
- 执行:open10(T 收盘信号 -> T+1 09:31-09:40 TWAP),1000 万,V1000 决策池,每 5 个交易日调仓。
- 成本:引擎与优化器共用 CostModel(impact_y=1,一跳价差,下界口径)。对照组同样计冲击。
- 基准:中证 1000 PIT 代理,权重先从信号日收盘漂移到成交时点(W[t]·px[t+1]/close[t] 归一)再乘引擎成交价收益
  (OPT.exec_time_bench_returns),全收益、无成本 = "在引擎成交价上计价的同一只指数"。
  **2026-10-10 审查 P0 修复**:初版用 IP.engine_consistent_bench_returns(收盘权重 W[t] 直接乘 px[t+1]→px[t+2]),
  每天把权重重置回过时一个隔夜的收盘权重,open10 下偏低约 0.75pp/年、vwap1 下偏低约 2.4pp/年
  (平均价口径的正自相关),容量曲线被抬高约 2pp。新旧基准与买入持有/引擎复制的对照见 ie_bench_compare.csv。
  真实 000852(价格指数,对齐到信号日)只作参考。
- 变体:V1 基准(5 相位,每个相位一条独立优化路径);V2 去风格约束;V3 TE 3%;V4 TE 7%;V5 成分股权重 ≥80%;
  V6 优化器忽略成本(引擎照收);V7 α 换成 B_gbt walk-forward 样本外打分(2023-2025);
  对照 Top20 等权 / 排雷缓冲 20/35(同信号 COMP_v5,5 相位);
  容量:V1 在 vwap1 口径下 AUM 1e7(锚点,额外加)/1e8/1e9/5e9,参与率按 capital×净值。
  V2~V7 与容量只跑相位 0(计算量),V1 的相位离散度单独报告。
- **信号样本内**:COMP_v5 的 24 个成员是在 2020-2025 研究期内选出的,用它跑的所有组合结果测的是
  组合构建层,不是信号的样本外能力。V7(B_gbt walk-forward)在 2023-2025 是样本外。
- 幸存者偏差:面板缺约 209 只已退市股。代理基准与策略都只在存活股里选 -> 二者收益都偏高;
  基准偏高约 +0.07pp/年量级(可测下界,CSI1000_PROXY.md),策略偏高的方向相同但量级未知
  (多头组合若曾持有后来退市的股票,真实亏损不在面板里)。超额的偏差方向不确定,报告里如实写。

## 用法
  python3 scripts/run_index_enhance.py --stage prep            # IC 缓存
  python3 scripts/run_index_enhance.py --stage run --job V1 --phase 0
  python3 scripts/run_index_enhance.py --stage all --procs 3   # 全部作业(最多 3 个并行子进程)+ 报告
  python3 scripts/run_index_enhance.py --stage report          # 只汇总
产出:research/ledger/ie_{variants,yearly,attribution,exposures,solver_log,engine_checks,paired,bench_compare}.csv,
      reports/ie_excess_nav.png、reports/ie_active_exposure.png;中间结果 cache/_tmp/ie_*。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from afactor.config import TMP, LEDGER, REPORTS, ROOT

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Solution may be inaccurate")

PFX = "ie"
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
TD = 242
START = pd.Timestamp("2021-04-20")          # 风险模型首个预测日(risk_model_summary.json first_forecast)
END = pd.Timestamp("2025-12-31")
V7_START = pd.Timestamp("2023-01-03")       # B_gbt walk-forward 样本外首日
DEV = (START, pd.Timestamp("2023-12-31"))
TEST = (pd.Timestamp("2024-01-01"), END)
SPECIAL = (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-07"))
COMP_SCORE = TMP / "comp_v5_score_open10.parquet"
GBT_SCORES = [LEDGER / "scores" / f"B_gbt_open_h5_{y}.parquet" for y in (2023, 2024, 2025)]

# ---- 作业表(参数在看任何结果前定死)
JOBS = {
    "V1": dict(desc="基准:TE5%、行业±2%、Size/NonlinSize±0.1、其余风格±0.5、个股+1%", cfg={},
               phases=[0, 1, 2, 3, 4]),
    "V2": dict(desc="去掉风格约束(只留行业与个股)", cfg=dict(size_bound=None, style_bound=None)),
    "V3": dict(desc="事前 TE 3%", cfg=dict(te_ann=0.03)),
    "V4": dict(desc="事前 TE 7%", cfg=dict(te_ann=0.07)),
    "V5": dict(desc="公募口径:成分股权重≥80%", cfg=dict(min_bench_weight=0.80)),
    "V6": dict(desc="优化器忽略成本(引擎照收)", cfg=dict(cost_mult=0.0)),
    "V7": dict(desc="α=B_gbt walk-forward 样本外打分(2023-2025)", cfg={}, score="gbt", start=V7_START),
    # 事后诊断变体:看过 V1 的归因(特质 ≈ 20 日反转)与偏差统计量(b=1.84)之后加的,不参与任何选择;
    # 界限 0.1 与 Size 相同,跑之前定死。依据 RISK_MODEL.md §7"应额外约束 20 日反转暴露"。
    "V8": dict(desc="事后诊断:V1 + 20日反转主动暴露|·|≤0.1(看过 V1 结果后加)", cfg={}, extra="rev20"),
    "CAP1e7": dict(desc="容量锚点:vwap1、1000万(额外,区分执行口径与规模效应)", cfg={}, mode="vwap1",
                   capital=1e7, pon=True),
    "CAP1e8": dict(desc="容量:vwap1、1亿", cfg={}, mode="vwap1", capital=1e8, pon=True),
    "CAP1e9": dict(desc="容量:vwap1、10亿", cfg={}, mode="vwap1", capital=1e9, pon=True),
    "CAP5e9": dict(desc="容量:vwap1、50亿", cfg={}, mode="vwap1", capital=5e9, pon=True),
}
CONTROLS = {
    "CTRL_top20": dict(desc="对照:COMP_v5 Top20 等权,5日调仓", kw=dict(scheme="top", n=20)),
    "CTRL_excl": dict(desc="对照:COMP_v5 标准排雷缓冲20/35,5日调仓", kw=dict(scheme="exclude", buffer=(0.20, 0.35))),
}
# 引擎判据(纪律 4):每个执行口径都要跑。vwap1 两条是审查 P0 后补的(初版只跑了 open10)。
CHECKS = {"CHK_proxy_replica": "open10", "CHK_pool_ew": "open10",
          "CHK_proxy_replica_vwap1": "vwap1", "CHK_pool_ew_vwap1": "vwap1"}


def chk_mode(name: str) -> str:
    return CHECKS.get(name, "open10")


def jspec(name: str) -> dict:
    s = dict(mode="open10", capital=1e7, pon=False, score="comp", start=START, phases=[0], extra=None)
    s.update(JOBS[name.partition("_ph")[0]])          # "V1_ph0" 这类单相位视图沿用母变体的设定
    return s


# ================================================================ 公共加载
_C: dict = {}


def P():
    if "p" not in _C:
        from afactor.panel import Panel
        _C["p"] = Panel()
    return _C["p"]


def date_mask(start, end=END) -> np.ndarray:
    d = P().dates
    return np.asarray((d >= start) & (d <= end))


def load_score(kind: str) -> pd.DataFrame:
    p = P()
    if kind == "comp":
        s = pd.read_parquet(COMP_SCORE)
    else:
        s = pd.concat([pd.read_parquet(f) for f in GBT_SCORES]).sort_index()
        s = s[~s.index.duplicated()]
    s = s.reindex(index=p.dates, columns=p.codes)
    assert s.index.max() <= END
    return s


def ic_dec(kind: str) -> tuple[pd.Series, pd.Series]:
    """(逐日 5 日 rank IC, 决策日可用的 IC 预测)。缓存 cache/_tmp/ie_ic_{kind}.parquet。"""
    from afactor import optimizer as OPT, strategy as ST
    f = TMP / f"{PFX}_ic_{kind}.parquet"
    if f.exists():
        df = pd.read_parquet(f)
        return df["ic"], df["ic_dec"]
    p = P()
    U = ST.universe(p, "V1000")
    ic = OPT.rank_ic_series(load_score(kind), p.fwd_ret(5, "open"), U)
    cfg = OPT.OptConfig()
    icd = OPT.ic_for_decision(ic, cfg.ic_window, cfg.ic_lag, cfg.ic_min_obs, 0.0)
    pd.DataFrame({"ic": ic, "ic_dec": icd}).to_parquet(f)
    return ic, icd


def rev20_z() -> pd.DataFrame:
    """20 日反转暴露 z = zscore(robust_clip(−(close_t/close_{t−20}−1), 5MAD), 决策池内市值加权去均值)。
    与 RISK_MODEL.md §5.3(e) 的缺失因子诊断同一构造;t 日收盘已知。决策池外 NaN。缓存 ie_rev20_z.parquet。"""
    from afactor import barra as BR, factors as FA, strategy as ST
    f = TMP / f"{PFX}_rev20_z.parquet"
    p = P()
    if f.exists():
        z = pd.read_parquet(f)
        if z.index.equals(p.dates) and z.columns.equals(p.codes):
            return z
    uu = ST.universe(p, "V1000").reindex(index=p.dates, columns=p.codes).fillna(False)
    mv = FA._real(p, "mv").reindex(index=p.dates, columns=p.codes)
    close = p["close_px"]
    x20 = -(close / close.shift(20) - 1.0)
    z = BR.zscore(BR.robust_clip(x20, uu, 5.0), uu, mv.where(uu)).astype(np.float32)
    z.to_parquet(f)
    return z


# ================================================================ 作业
def run_job(name: str, phase: int) -> None:
    from afactor import strategy as ST, index_proxy as IP, optimizer as OPT
    from afactor.barra import RiskModel
    from afactor.impact import CostModel
    t0 = time.time()
    sp = jspec(name)
    p = P()
    rm = RiskModel.load("open10")
    W = IP.proxy_weights(p, "000852")
    U = ST.universe(p, "V1000")
    score = load_score(sp["score"])
    _, icd = ic_dec(sp["score"])
    cfg = replace(OPT.OptConfig(), **sp["cfg"])
    cm = CostModel(p, impact_y=1.0)
    extra = {"Rev20": (rev20_z(), 0.10)} if sp["extra"] == "rev20" else None
    enh = OPT.IndexEnhancer(p, rm, W, score, icd, U, cfg, tag=name, extra_exposures=extra)
    mask = date_mask(sp["start"])
    print(f"[{name} ph{phase}] mode={sp['mode']} capital={sp['capital']:.0e} cfg={sp['cfg']} 载入 {time.time()-t0:.0f}s",
          flush=True)
    res = ST.simulate(enh, p, sp["mode"], sp["capital"], rebalance=5, date_mask=mask, phase=phase,
                      cost_model=cm, record=True, participation_on_nav=sp["pon"])
    prefix = f"{PFX}_{name}_ph{phase}"
    ST.save_record(res, prefix)
    lg = enh.log_frame()
    lg.to_parquet(TMP / f"{prefix}_solver.parquet")
    summ = {k: res[k] for k in ("ann_return", "sharpe", "max_dd", "turnover", "cost_bps", "holdings",
                                 "exposure", "n_days", "vol")}
    summ.update(job=name, phase=phase, n_solves=len(lg), n_failed=int((lg["status"] == "failed").sum()),
                elapsed_s=time.time() - t0, cfg=cfg.describe(), mode=sp["mode"], capital=sp["capital"])
    json.dump(summ, open(TMP / f"{prefix}_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(f"[{name} ph{phase}] 年化 {res['ann_return']:.2%} 换手 {res['turnover']:.4f} 求解 {len(lg)} 次 "
          f"失败 {summ['n_failed']} 用时 {time.time()-t0:.0f}s", flush=True)


def run_controls(force: bool = False) -> None:
    """对照组:同信号 COMP_v5 的 Top20 / 排雷,同窗口、同成本模型、5 相位。另跑引擎判据(open10 与 vwap1 各两条)。
    已有结果的作业跳过(force=True 重跑);它们与评价基准无关,基准修复后不需要重跑。"""
    from afactor import strategy as ST, index_proxy as IP
    from afactor.impact import CostModel
    p = P()
    cm = CostModel(p, impact_y=1.0)
    mask = date_mask(START)
    have = lambda nm: (not force) and (TMP / f"{PFX}_{nm}_daily.parquet").exists()
    for nm, c in CONTROLS.items():
        if have(nm):
            continue
        t0 = time.time()
        res = ST.run_strategy(load_score("comp"), p, "V1000", "open10", rebalance=5, date_mask=mask, cost_model=cm,
                              record=True, **c["kw"])
        ST.save_record(res, f"{PFX}_{nm}")
        print(f"[{nm}] 年化 {res['ann_return']:.2%} 换手 {res['turnover']:.4f} {time.time()-t0:.0f}s", flush=True)
    W = IP.proxy_weights(p, "000852").to_numpy(np.float64)
    const = pd.DataFrame(1.0, index=p.dates, columns=p.codes)
    for nm, mode in CHECKS.items():
        if have(nm):
            continue
        t0 = time.time()
        if nm.startswith("CHK_proxy_replica"):
            # 判据 1:代理基准权重直接喂引擎(ndarray 目标,5 相位)-> 相对同口径代理的超额应 ≈ −成本
            res = ST.run_weights(W, p, mode, 1e7, rebalance=5, date_mask=mask, cost_model=cm, record=True)
        else:
            # 判据 2:常数打分全池等权 vs 池等权基准(引擎回归断言③,任务书纪律 4)
            res = ST.run_strategy(const, p, "V1000", mode, scheme="top", n=100000, rebalance=5, date_mask=mask,
                                  cost_model=cm, record=True)
        ST.save_record(res, f"{PFX}_{nm}")
        print(f"[{nm}] {mode} 年化 {res['ann_return']:.2%} {time.time()-t0:.0f}s", flush=True)
    print("[controls] 完成", flush=True)


# ================================================================ 驱动
def all_tasks() -> list[tuple[str, int]]:
    tasks = []
    for nm in JOBS:
        for ph in jspec(nm)["phases"]:
            tasks.append((nm, ph))
    return tasks


def drive(procs: int, only: list[str] | None = None) -> None:
    ic_dec("comp"); ic_dec("gbt"); rev20_z()          # 先建 IC / 反转暴露缓存,子进程直接读
    tasks = [("CTRL", -1)] + all_tasks()
    if only:
        tasks = [x for x in tasks if x[0] in only]
    def done(x):
        if x[0] == "CTRL":
            return all((TMP / f"{PFX}_{k}_daily.parquet").exists()
                       for k in list(CONTROLS) + list(CHECKS))
        return (TMP / f"{PFX}_{x[0]}_ph{x[1]}_summary.json").exists()
    todo = [x for x in tasks if not done(x)]
    print(f"作业 {len(tasks)} 个,待跑 {len(todo)} 个,并行 {procs}", flush=True)
    running: list = []
    script = os.path.abspath(__file__)
    while todo or running:
        while todo and len(running) < procs:
            nm, ph = todo.pop(0)
            lf = LOGS / (f"{PFX}_controls.log" if nm == "CTRL" else f"{PFX}_{nm}_ph{ph}.log")
            args = [sys.executable, script, "--stage", "controls"] if nm == "CTRL" else \
                [sys.executable, script, "--stage", "run", "--job", nm, "--phase", str(ph)]
            fh = open(lf, "w")
            running.append((nm, ph, subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT), fh, time.time()))
            print(f"  启动 {nm} ph{ph} -> {lf.name}", flush=True)
        time.sleep(5)
        still = []
        for nm, ph, pr, fh, ts in running:
            if pr.poll() is None:
                still.append((nm, ph, pr, fh, ts))
            else:
                fh.close()
                print(f"  完成 {nm} ph{ph} rc={pr.returncode} {time.time()-ts:.0f}s", flush=True)
                if pr.returncode != 0:
                    raise RuntimeError(f"{nm} ph{ph} 失败,见日志")
        running = still


# ================================================================ 评价工具
def _win(s: pd.Series | pd.DataFrame, a, b):
    return s[(s.index >= a) & (s.index <= b)]


def nw_t(x, q=5) -> float:
    from afactor.barra import nw_t as _nw
    return _nw(np.asarray(x, float), q)


def perf(rs: pd.Series, rb: pd.Series, daily: pd.DataFrame | None, a, b, vwap: bool = False,
         rreal: pd.Series | None = None) -> dict:
    """窗口 [a, b] 内的相对表现。rs 策略净收益、rb 基准收益(同信号日索引)。"""
    from afactor import strategy as ST
    s, bb = rs.align(rb, join="inner")
    s, bb = _win(s, a, b), _win(bb, a, b)
    n = len(s)
    if n < 15:
        return dict(n_days=n)
    d = s - bb
    nav_s, nav_b = (1 + s).cumprod(), (1 + bb).cumprod()
    rel = nav_s / nav_b
    yrs = n / TD
    o = dict(n_days=n, start=str(s.index[0].date()), end=str(s.index[-1].date()),
             ann_ret=float(nav_s.iloc[-1] ** (1 / yrs) - 1), bench_ann=float(nav_b.iloc[-1] ** (1 / yrs) - 1),
             excess_geo=float(rel.iloc[-1] ** (1 / yrs) - 1), excess_arith=float(d.mean() * TD),
             cum_excess=float(rel.iloc[-1] - 1),
             max_rel_dd=float((rel / rel.cummax() - 1).min()), max_dd=float((nav_s / nav_s.cummax() - 1).min()),
             bench_max_dd=float((nav_b / nav_b.cummax() - 1).min()),
             vol=float(s.std() * np.sqrt(TD)), sharpe=float(s.mean() / s.std() * np.sqrt(TD)) if s.std() > 0 else np.nan)
    if n >= 60:
        es = ST.excess_stats(s, bb)
        o.update(te=es["te"], ir=es["ir"], te_adj=es["te_adj"], ir_adj=es["ir_adj"], t_nw=es["t_nw"],
                 excess_ac1=es["excess_ac1"])
    else:
        te = float(d.std() * np.sqrt(TD))
        o.update(te=te, ir=o["excess_arith"] / te if te > 0 else np.nan, te_adj=np.nan, ir_adj=np.nan,
                 t_nw=np.nan, excess_ac1=np.nan)
    m_s = (1 + s).groupby(s.index.to_period("M")).prod()
    m_b = (1 + bb).groupby(bb.index.to_period("M")).prod()
    mex = m_s / m_b - 1
    o.update(n_months=int(len(mex)), month_win=float((mex > 0).mean()),
             te_monthly=float(mex.std() * np.sqrt(12)) if len(mex) > 2 else np.nan)
    if daily is not None:
        dd = _win(daily, a, b)
        o.update(turnover=float(dd["turnover"].mean()), cost_ann=float(dd["cost_total"].mean() * TD),
                 cost_spread_ann=float(dd["cost_spread"].mean() * TD),
                 cost_impact_ann=float(dd["cost_impact"].mean() * TD),
                 cost_comm_stamp_ann=float((dd["cost_commission"] + dd["cost_stamp"]).mean() * TD),
                 exposure=float(dd["exposure"].mean()), n_hold=float(dd["n_hold"].mean()),
                 part_cut=float(dd["part_cut"].mean()), max_part=float(dd["max_part"].max()))
    if rreal is not None:
        s2, r2 = _win(rs, a, b).align(_win(rreal, a, b), join="inner")
        rel2 = (1 + s2).cumprod() / (1 + r2).cumprod()
        m2 = (1 + s2).groupby(s2.index.to_period("M")).prod() / (1 + r2).groupby(r2.index.to_period("M")).prod() - 1
        o.update(real_ann=float((1 + r2).prod() ** (TD / len(r2)) - 1),
                 excess_geo_real=float(rel2.iloc[-1] ** (TD / len(r2)) - 1),
                 te_real_daily=float((s2 - r2).std() * np.sqrt(TD)),
                 te_real_monthly=float(m2.std() * np.sqrt(12)) if len(m2) > 2 else np.nan,
                 month_win_real=float((m2 > 0).mean()))
    return o


def windows(start=START) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """全期 / 开发窗 / 检验窗。V7(2023 起)的"开发窗"只剩 2023 年,标成"2023(样本外首年)"。"""
    w = [("全期", start, END)]
    if start <= DEV[0]:
        w.append(("开发窗2021-23", DEV[0], DEV[1]))
    elif start <= DEV[1]:
        w.append((f"{start.year}(样本外首年)", start, DEV[1]))
    w += [("检验窗2024-25", TEST[0], TEST[1])]
    return w


def years(start=START) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    out = []
    for y in range(start.year, 2026):
        a = max(start, pd.Timestamp(f"{y}-01-01"))
        out.append((str(y) if a.month == 1 else f"{y}(自{a.strftime('%m-%d')})", a, pd.Timestamp(f"{y}-12-31")))
    out.append(("2024-01-02~02-07", SPECIAL[0], SPECIAL[1]))
    return out


def b_matrix(z: pd.Series) -> np.ndarray:
    """偏差统计量的错位起点矩阵:行 j = 第 j 个 5 日块,列 k = 起点 k(z[5j+k]);尾部不足一整块的丢弃。"""
    n = len(z) // 5
    return z.to_numpy(np.float64)[:5 * n].reshape(n, 5)


def b_boot(Zs: list, n_boot: int = 2000, block: int = 10, seed: int = 0) -> np.ndarray:
    """b 的移动块自助(审查 P2:初版只给了 b=1、正态假设下的 ±sqrt(2/n),低估约一半)。
    块长 10 个 5 日窗口(≈50 个交易日)以保留波动聚集;环形取块。多个矩阵共用同一组行索引 = 配对自助。
    返回 (n_boot, len(Zs)):每列是 b* = 5 个起点 std 的均值。"""
    rng = np.random.default_rng(seed)
    n = Zs[0].shape[0]
    nb = int(np.ceil(n / block))
    out = np.empty((n_boot, len(Zs)))
    for r in range(n_boot):
        st = rng.integers(0, n, nb)
        idx = ((st[:, None] + np.arange(block)[None, :]).ravel()[:n]) % n
        for j, Z in enumerate(Zs):
            out[r, j] = float(np.nanmean(np.nanstd(Z[idx], axis=0, ddof=1)))
    return out


def bench_compare(p, rb: dict, rb_legacy: dict, V: dict, Wb: pd.DataFrame) -> pd.DataFrame:
    """新旧评价基准的对照账本 ie_bench_compare.csv(审查 P0)。
    kind=bench:各口径基准年化(几何),及相对新基准的日差(算术年化 pp、NW t):
      c2c        收盘到收盘的代理(r[t] = Σ W[t]·R[t+1],与 index_proxy 同一全收益),执行口径无关的参照
      exec       新基准:成交时点权重(OPT.exec_time_bench_returns)
      legacy     初版基准:收盘权重 W[t] × 成交价收益(IP.engine_consistent_bench_returns)
      bh5        每 5 日按收盘权重 W[t] 重置、其间按成交价漂移的买入持有(5 相位均值,无成本、无可成交规则)
      replica_gross  代理权重经引擎复制(CHK_proxy_replica*,5 相位)的毛收益 = 净收益 + 成本
    kind=variant:各变体相对 exec / legacy / replica_gross 三个基准的几何超额、TE、IR、NW t(新旧对照的出处)。"""
    from afactor import strategy as ST
    T = len(p.dates)
    Wv = Wb.reindex(index=p.dates, columns=p.codes).fillna(0.0).to_numpy(np.float64)
    cp = p.raw("close_px").where(p["valid"]).ffill().reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
    R = np.full_like(cp, np.nan)
    R[:-1] = cp[1:] / cp[:-1] - 1.0
    R = np.where(np.isfinite(R), R, 0.0)
    idx = p.dates[:T - 2]
    c2c = pd.Series(np.where(Wv.sum(1) > 0, (Wv * R).sum(1), np.nan), index=p.dates).reindex(idx)
    del R, cp
    first = int(p.dates.searchsorted(START))
    benches = {}
    for m in ("open10", "vwap1"):
        px = ST.exec_prices(p, m)[0].reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
        G = np.full_like(px, np.nan)
        G[:T - 2] = px[2:] / px[1:-1] - 1.0
        Gz = np.where(np.isfinite(G), G, 0.0)
        del G, px
        acc = []
        for ph in range(5):
            out = np.full(T, np.nan)
            h = None
            for t in range(first, T - 2):
                if h is None or (t - first - ph) % 5 == 0:
                    h = Wv[t].copy()
                out[t] = float(np.dot(h, Gz[t]))
                h = h * (1.0 + Gz[t])
                h = h / h.sum()
            acc.append(out)
        bh5 = pd.Series(np.nanmean(np.vstack(acc), axis=0), index=p.dates).reindex(idx)
        del Gz
        rep_ = V["CHK_proxy_replica" + ("" if m == "open10" else "_vwap1")]
        rg = rep_["rets"] + rep_["daily"]["cost_total"]
        benches[m] = dict(exec=rb[m], legacy=rb_legacy[m], bh5=bh5, replica_gross=rg, c2c=c2c)
    rows = []
    for m, bd in benches.items():
        ref = bd["exec"]
        for bn, ser in bd.items():
            for wn, a, b in windows():
                x = _win(ser.reindex(ref.index), a, b).dropna()
                d = _win(ser.reindex(ref.index) - ref, a, b).dropna()
                rows.append(dict(kind="bench", mode=m, bench=bn, variant="", window=wn, n_days=len(x),
                                 ann_geo=float((1 + x).prod() ** (TD / len(x)) - 1),
                                 diff_vs_exec_arith_pp=float(d.mean() * TD * 100),
                                 diff_vs_exec_t_nw=nw_t(d, 5) if bn != "exec" else np.nan))
    for nm, v in V.items():
        if nm.startswith("CHK_"):
            continue
        m = v["spec"]["mode"]
        st = v["spec"].get("start", START)
        for bn in ("exec", "legacy", "replica_gross"):
            for wn, a, b in windows(st):
                o = perf(v["rets"], benches[m][bn], None, a, b)
                rows.append(dict(kind="variant", mode=m, bench=bn, variant=nm, window=wn, n_days=o.get("n_days"),
                                 excess_geo=o.get("excess_geo"), excess_arith=o.get("excess_arith"), te=o.get("te"),
                                 te_adj=o.get("te_adj"), ir=o.get("ir"), ir_adj=o.get("ir_adj"), t_nw=o.get("t_nw"),
                                 bench_ann=o.get("bench_ann")))
    return pd.DataFrame(rows)


# ================================================================ 结果装载
def load_variant(name: str) -> dict:
    """合并相位:日收益 = 相位均值,daily 数值列 = 相位均值,持仓 = 相位均值(与 ST._merge_records 同口径)。"""
    from afactor.strategy import _REC_COLS
    if name in CONTROLS or name.startswith("CHK_"):
        daily = pd.read_parquet(TMP / f"{PFX}_{name}_daily.parquet")
        byp = pd.read_parquet(TMP / f"{PFX}_{name}_daily_by_phase.parquet")
        H = pd.read_parquet(TMP / f"{PFX}_{name}_holdings.parquet")
        return dict(name=name, daily=daily, rets=daily["ret"], by_phase=byp, H=H, solver=None,
                    spec=dict(mode=chk_mode(name), capital=1e7, start=START))
    base, _, phs = name.partition("_ph")
    sp = jspec(base)
    if phs:                                               # 单相位视图,如 V1_ph0(与相位 0 的敏感性变体配对)
        sp = dict(sp, phases=[int(phs)])
    ds, Hs, sl = [], [], []
    for ph in sp["phases"]:
        pre = TMP / f"{PFX}_{base}_ph{ph}"
        d = pd.read_parquet(f"{pre}_daily.parquet")
        ds.append(d.assign(phase=ph))
        Hs.append(pd.read_parquet(f"{pre}_holdings.parquet"))
        sl.append(pd.read_parquet(f"{pre}_solver.parquet"))
    byp = pd.concat(ds)
    idx = ds[0].index
    for d in ds[1:]:
        idx = idx.union(d.index)
    num = [c for c in _REC_COLS if c != "rebalance"]
    daily = byp.groupby(level=0)[num].mean().reindex(idx)
    cols = Hs[0].columns
    for h in Hs[1:]:
        cols = cols.union(h.columns)
    H = sum(h.reindex(index=idx, columns=cols).fillna(0.0).astype(np.float64) for h in Hs) / len(Hs)
    return dict(name=name, daily=daily, rets=daily["ret"], by_phase=byp, H=H.astype(np.float32),
                solver=pd.concat(sl, ignore_index=True), spec=sp)


# ================================================================ 报告
def report() -> None:
    from afactor import strategy as ST, index_proxy as IP, barra as BR, factors as FA, optimizer as OPT
    from afactor.barra import RiskModel, psd_sqrt
    p = P()
    dates, codes = p.dates, p.codes
    rm = RiskModel.load("open10")
    Wb = IP.proxy_weights(p, "000852")
    # 评价基准:成交时点权重(审查 P0 修复);rb_legacy = 初版基准(收盘权重 × 成交价收益),只用于新旧对照
    rb, Wx = {}, {}
    for m in ("open10", "vwap1"):
        rb[m], Wx[m] = OPT.exec_time_bench_returns(p, m, Wb, return_weights=True)
    rb_legacy = {m: IP.engine_consistent_bench_returns(p, m, "000852") for m in ("open10", "vwap1")}
    rreal = IP.index_returns_on_signal_days(p, "open10", "000852")
    pool = {m: ST.benchmark(p, "V1000", m)["rets"] for m in ("open10", "vwap1")}
    names = list(JOBS) + ["V1_ph0"] + list(CONTROLS)
    V = {}
    for nm in names:
        try:
            V[nm] = load_variant(nm)
        except FileNotFoundError as e:
            print(f"  ⚠ 缺 {nm}: {e}")
    for nm in CHECKS:
        V[nm] = load_variant(nm)
    desc = {**{k: v["desc"] for k, v in JOBS.items()}, **{k: v["desc"] for k, v in CONTROLS.items()},
            "V1_ph0": "V1 的相位 0 单独一条路径(与只跑相位 0 的敏感性变体配对)",
            "CHK_proxy_replica": "判据:代理基准权重直接喂引擎", "CHK_pool_ew": "判据:常数打分全池等权",
            "CHK_proxy_replica_vwap1": "判据(vwap1):代理基准权重直接喂引擎",
            "CHK_pool_ew_vwap1": "判据(vwap1):常数打分全池等权"}

    # ---------------- 引擎判据(纪律 4)
    chk = []
    for nm, v in V.items():
        mode = v["spec"]["mode"]
        s = v["rets"]
        b = rb[mode].reindex(s.index)
        ks = (-2, -1, 0, 1, 2)
        cs = {k: float(s.corr(rb[mode].shift(k).reindex(s.index))) for k in ks}
        peak = max(ks, key=lambda k: cs[k])
        row = dict(variant=nm, check="corr_peak_vs_proxy", value=peak, detail=" ".join(f"{k:+d}:{cs[k]:.3f}" for k in ks),
                   ok=bool(peak == 0))
        chk.append(row)
        chk.append(dict(variant=nm, check="max_exposure", value=float(v["by_phase"]["exposure"].max()), detail="",
                        ok=bool(v["by_phase"]["exposure"].max() <= 1 + 1e-9)))
        if nm.startswith("CHK_pool_ew"):
            # 池等权基准 = 引擎的 ST.benchmark(每日等权再平衡);vwap1 下它与 5 日调仓的全池等权之间
            # 同样有平均价口径的正自相关偏差(审查 P0),这里如实记录,不改上游引擎
            pb = pool[mode].reindex(s.index)
            for y in range(2021, 2026):
                sel = s.index.year == y
                ex = float((s[sel] - pb[sel]).mean() * TD)
                c = float(v["daily"]["cost_total"][sel].mean() * TD)
                chk.append(dict(variant=nm, check=f"pool_ew_excess_plus_cost_{y}", value=ex + c,
                                detail=f"{mode} excess_arith={ex:.4%} cost={c:.4%}(残差应≈0;上游 open10 0.10~0.91pp)",
                                ok=bool(abs(ex + c) < 0.012)))
            # 2024 残差偏大时的定位:剔除 2024-01-02~02-07(一字跌停集中期,策略买不进跌停股、基准照买)
            sel = (s.index.year == 2024) & ~((s.index >= SPECIAL[0]) & (s.index <= SPECIAL[1]))
            ex = float((s[sel] - pb[sel]).mean() * TD); c = float(v["daily"]["cost_total"][sel].mean() * TD)
            chk.append(dict(variant=nm, check="pool_ew_excess_plus_cost_2024_ex_JanFeb", value=ex + c,
                            detail=f"excess_arith={ex:.4%} cost={c:.4%};剔除 {SPECIAL[0].date()}~{SPECIAL[1].date()}",
                            ok=bool(abs(ex + c) < 0.012)))
            selj = (s.index >= SPECIAL[0]) & (s.index <= SPECIAL[1])
            chk.append(dict(variant=nm, check="pool_ew_cum_resid_JanFeb2024", value=float(((s[selj] - pb[selj]) +
                            v["daily"]["cost_total"][selj]).sum()), detail="窗口内(超额+成本)逐日和(非年化)", ok=True))
        if nm.startswith("CHK_proxy_replica"):
            # 新基准(成交时点权重)与初版基准(收盘权重)各一套;初版那套就是审查 P0 指出的机械差
            for tag, bser in (("", rb[mode]), ("legacybench_", rb_legacy[mode])):
                b2 = bser.reindex(s.index)
                for wn, a, bb in windows() + years()[:-1]:
                    sel = (s.index >= a) & (s.index <= bb)
                    ex = float((s[sel] - b2[sel]).mean() * TD)
                    c = float(v["daily"]["cost_total"][sel].mean() * TD)
                    d_ = (s[sel] - b2[sel] + v["daily"]["cost_total"][sel]).dropna()
                    chk.append(dict(variant=nm, check=f"replica_excess_plus_cost_{tag}{wn}", value=ex + c,
                                    detail=f"{mode} excess_arith={ex:.4%} cost={c:.4%} "
                                           f"te={(s[sel]-b2[sel]).std()*np.sqrt(TD):.4%} t_nw={nw_t(d_, 5):.2f}",
                                    ok=bool(abs(ex + c) < 0.01)))
    chk = pd.DataFrame(chk)
    chk.to_csv(LEDGER / f"{PFX}_engine_checks.csv", index=False)
    print(chk[~chk["check"].isin(["max_exposure"])].to_string())
    for nm in V:
        if nm.startswith("CHK_"):
            continue
        r = chk[(chk.variant == nm) & (chk.check == "corr_peak_vs_proxy")]
        assert bool(r["ok"].all()), f"{nm} 与基准错位"

    # ---------------- 绩效:全期/开发/检验 + 逐年 + 特殊窗口
    rows, yrows = [], []
    for nm, v in V.items():
        if nm.startswith("CHK_"):
            continue
        mode = v["spec"]["mode"]
        st = v["spec"].get("start", START)
        sv = v["solver"]
        extra = dict(variant=nm, desc=desc[nm], mode=mode, capital=v["spec"]["capital"],
                     n_phases=int(v["by_phase"]["phase"].nunique()), signal=("B_gbt 样本外" if nm == "V7" else
                                                                             "COMP_v5 信号样本内"))
        if sv is not None:
            ok = sv[sv.status != "no_risk_model"]
            extra.update(n_solves=len(ok), n_failed=int((ok.status == "failed").sum()),
                         n_inaccurate=int((ok.status == "optimal_inaccurate").sum()),
                         n_scs=int((ok.solver == "SCS").sum()), n_cap_relaxed=int(ok.cap_relaxed.fillna(False).sum()),
                         solve_s_mean=float(ok.solve_s.mean()), solve_s_p95=float(ok.solve_s.quantile(0.95)),
                         n_cand_mean=float(ok.n_cand.mean()), te_ex_ante_mean=float(ok.te_ex_ante.mean()),
                         te_binding_frac=float((ok.te_ex_ante >= 0.995 * jspec(nm)["cfg"].get("te_ann", 0.05)).mean()),
                         ic_dec_mean=float(ok.ic.mean()), ic_zero_frac=float((ok.ic <= 0).mean()),
                         alpha_ex_ante_ann=float(ok.alpha_ex_ante.mean() * TD / 5),
                         n_hold_target=float(ok.n_hold.mean()), bench_w_in_target=float(ok.bench_w_in.mean()),
                         turnover_plan=float(ok.turnover_plan.mean()))
        for wn, a, b in windows(st):
            o = perf(v["rets"], rb[mode], v["daily"], a, b, vwap=(mode == "vwap1"),
                     rreal=rreal if mode == "open10" else None)
            # vwap1 下引擎的池等权基准自身有 +1.3~+2.7pp/年的平均价口径偏差(ie_engine_checks.csv
            # CHK_pool_ew_vwap1),相对池等权的数字不可用,置 NaN
            po = perf(v["rets"], pool[mode], None, a, b) if mode == "open10" else {}
            rows.append(dict(extra, window=wn, **o, excess_geo_vs_pool=po.get("excess_geo", np.nan),
                             te_vs_pool=po.get("te", np.nan)))
        for yn, a, b in years(st):
            o = perf(v["rets"], rb[mode], v["daily"], a, b, rreal=rreal if mode == "open10" else None)
            yrows.append(dict(variant=nm, phase="avg", period=yn, **o))
            if v["by_phase"]["phase"].nunique() > 1:
                for ph, g in v["by_phase"].groupby("phase"):
                    o = perf(g["ret"], rb[mode], None, a, b)
                    yrows.append(dict(variant=nm, phase=str(ph), period=yn, **o))
    # V1 在 V7 窗口(2023-2025)上的同窗对照
    for wn, a, b in [("同窗2023-25", V7_START, END), ("2023(样本外首年)", V7_START, pd.Timestamp("2023-12-31"))]:
        for nm in ("V1", "V1_ph0", "CTRL_top20", "CTRL_excl"):      # V1_ph0:与单相位的 V7 同口径(审查 P2)
            if nm in V:
                o = perf(V[nm]["rets"], rb["open10"], V[nm]["daily"], a, b, rreal=rreal)
                rows.append(dict(variant=nm, desc=desc[nm], mode="open10", capital=1e7, window=wn, **o,
                                 signal="COMP_v5 信号样本内"))
    var = pd.DataFrame(rows)
    yr = pd.DataFrame(yrows)
    # 相位离散度:每年 5 个相位超额的标准差 / 极差
    disp = []
    for nm in ("V1", "CTRL_top20", "CTRL_excl"):
        g = yr[(yr.variant == nm) & (yr.phase != "avg")]
        for per, gg in g.groupby("period"):
            disp.append(dict(variant=nm, phase="dispersion", period=per,
                             phase_std=float(gg.excess_geo.std(ddof=1)),          # 5 个相位年化超额(几何)的标准差
                             phase_range=float(gg.excess_geo.max() - gg.excess_geo.min()),   # 极差
                             n_days=int(gg.n_days.max())))
    yr = pd.concat([yr, pd.DataFrame(disp)], ignore_index=True)
    # 配对检验:只跑相位 0 的变体 vs V1 的相位 0(同一调仓日历),日收益差的算术年化与 NW t(q=5)
    prs = []
    if "V1_ph0" in V:
        r0 = V["V1_ph0"]["rets"]
        for nm in ["V2", "V3", "V4", "V5", "V6", "V7", "V8"]:
            if nm not in V:
                continue
            st = V[nm]["spec"].get("start", START)
            for wn, a, b in windows(st):
                d = (V[nm]["rets"] - r0).dropna()
                d = d[(d.index >= a) & (d.index <= b)]
                prs.append(dict(variant=nm, vs="V1_ph0", window=wn, n_days=len(d), diff_arith_ann=float(d.mean() * TD),
                                t_nw=nw_t(d, 5), te_diff=float(d.std() * np.sqrt(TD))))
        # V1 相位 0 vs V1 五相位均值:单相位变体若与 5 相位的 V1 比,要先扣掉这一项(审查 P2)
        for wn, a, b in windows():
            d = (r0 - V["V1"]["rets"]).dropna()
            d = d[(d.index >= a) & (d.index <= b)]
            prs.append(dict(variant="V1_ph0", vs="V1(5相位均值)", window=wn, n_days=len(d),
                            diff_arith_ann=float(d.mean() * TD), t_nw=nw_t(d, 5), te_diff=float(d.std() * np.sqrt(TD))))
        # V1 五个相位之间的相互差异(同一规则、不同日历):给"相位噪声"一个配对尺度
        byp = V["V1"]["by_phase"]
        for ph in range(1, 5):
            rp = byp[byp.phase == ph]["ret"]
            for wn, a, b in windows():
                d = (rp - r0).dropna()
                d = d[(d.index >= a) & (d.index <= b)]
                prs.append(dict(variant=f"V1_ph{ph}", vs="V1_ph0", window=wn, n_days=len(d),
                                diff_arith_ann=float(d.mean() * TD), t_nw=nw_t(d, 5), te_diff=float(d.std() * np.sqrt(TD))))
    pd.DataFrame(prs).to_csv(LEDGER / f"{PFX}_paired.csv", index=False)
    var.to_csv(LEDGER / f"{PFX}_variants.csv", index=False)
    yr.to_csv(LEDGER / f"{PFX}_yearly.csv", index=False)
    print("绩效写入", LEDGER / f"{PFX}_variants.csv")
    bc = bench_compare(p, rb, rb_legacy, V, Wb)
    bc.to_csv(LEDGER / f"{PFX}_bench_compare.csv", index=False)
    print(bc[bc.kind == "bench"][["mode", "bench", "window", "ann_geo", "diff_vs_exec_arith_pp",
                                   "diff_vs_exec_t_nw"]].to_string())

    # ---------------- 暴露、事前/事后 TE、归因(open10 变体与对照)
    T = len(dates)
    Wv = Wx["open10"]            # 成交时点的基准权重:A = 成交后持仓 − Wx,与新基准收益同一时点(审查 P0)
    Ls = {}

    def L_at(i):
        if i not in Ls:
            Ls[i] = psd_sqrt(rm.cov[i])[1]
        return Ls[i]

    R = BR.fwd_returns(p, "open10")
    Rv = R.to_numpy(np.float64)
    resid = rm.resid()
    U = ST.universe(p, "V1000").to_numpy(bool)
    mvv = FA._real(p, "mv").reindex(index=dates, columns=codes).to_numpy(np.float64)
    z20 = rev20_z().to_numpy(np.float64)
    # 缺失因子(20 日反转)收益 g_t:风险模型残差对 z20 的 sqrt(市值) 加权截面回归斜率(与 RISK_MODEL.md §5.3(e) 同法)
    T_R = len(R)
    g20 = np.full(T, np.nan)
    Rs = np.asarray(resid[:T_R], np.float64)
    first = int(np.searchsorted(dates, START))
    for t in range(first, T_R):
        m = U[t] & np.isfinite(Rs[t]) & np.isfinite(z20[t]) & np.isfinite(mvv[t]) & (mvv[t] > 0)
        if m.sum() < 200:
            continue
        ww = np.sqrt(np.sqrt(mvv[t, m]))
        A_ = np.column_stack([np.ones(m.sum()), z20[t, m]])
        g20[t] = np.linalg.lstsq(A_ * ww[:, None], Rs[t, m] * ww, rcond=None)[0][1]

    sty = list(rm.style_names)
    exp_rows, att_rows, ts_store = [], [], {}
    for nm, v in V.items():
        if v["spec"]["mode"] != "open10":
            continue
        H = v["H"].reindex(columns=codes).fillna(0.0)
        idx = H.index[(H.index >= v["spec"].get("start", START)) & (H.index <= END)]
        Hv = H.loc[idx].to_numpy(np.float64)
        pos = dates.get_indexer(idx)
        E = np.full((len(idx), 1 + rm.K + rm.J), np.nan)
        te_pred = np.full(len(idx), np.nan)
        e20 = np.full(len(idx), np.nan)
        unc = np.zeros(len(idx))
        for j, i in enumerate(pos):
            if not np.isfinite(rm.cov[i, 0, 0]):
                continue
            a = Hv[j] - Wv[i]
            m = np.asarray(rm.mask[i])
            am = a[m]
            X = np.nan_to_num(np.asarray(rm.X[i])[m].astype(np.float64))
            Hm = rm._onehot(rm.ind[m])
            d = np.asarray(rm.spec[i])[m].astype(np.float64)
            with np.errstate(all="ignore"):
                e = np.concatenate([[am.sum()], np.dot(X.T, am), np.dot(Hm.T, am)])
                y = np.dot(L_at(i).T, e)
            E[j] = e
            te_pred[j] = np.sqrt(np.sum(y * y) + np.sum(np.nan_to_num(d) * am * am)) * np.sqrt(TD / 5)
            unc[j] = float(np.abs(a[~m]).sum())
            e20[j] = float(np.nansum(np.nan_to_num(z20[i]) * a))
        Edf = pd.DataFrame(E, index=idx, columns=rm.factor_names)
        ts_store[nm] = dict(E=Edf, te_pred=pd.Series(te_pred, index=idx), e20=pd.Series(e20, index=idx))
        # 主动暴露统计(实际持仓,逐日)
        bounds = {s_: (0.10 if s_ in ("Size 市值", "NonlinSize 非线性市值") else 0.50) for s_ in sty}
        ind_cols = rm.ind_names
        st0 = v["spec"].get("start", START)
        warm = dates[min(int(dates.searchsorted(st0)) + 5, T - 1)]    # 建仓周:相位 1-4 前几天还是现金,暴露统计剔除
        for wn, a, b in windows(st0):
            sel = (idx >= max(a, warm)) & (idx <= b)
            if sel.sum() < 10:
                continue
            for f in ["Country"] + sty:
                x = Edf.loc[sel, f]
                bd = bounds.get(f, np.nan)
                exp_rows.append(dict(variant=nm, window=wn, source="实际持仓(逐日,相位合并)", factor=f,
                                     mean=float(x.mean()), mean_abs=float(x.abs().mean()), max_abs=float(x.abs().max()),
                                     p95_abs=float(x.abs().quantile(0.95)), bound=bd,
                                     frac_over_bound=float((x.abs() > bd + 1e-9).mean()) if np.isfinite(bd) else np.nan))
            xi = Edf.loc[sel, ind_cols].abs()
            mx = xi.max(axis=1)
            exp_rows.append(dict(variant=nm, window=wn, source="实际持仓(逐日,相位合并)", factor="行业 max|·|(30个取最大)",
                                 mean=float(mx.mean()), mean_abs=float(mx.mean()), max_abs=float(mx.max()),
                                 p95_abs=float(mx.quantile(0.95)), bound=0.02,
                                 frac_over_bound=float((mx > 0.02 + 1e-9).mean())))
            x = pd.Series(e20, index=idx)[sel]
            exp_rows.append(dict(variant=nm, window=wn, source="实际持仓(逐日,相位合并)", factor="Rev20 20日反转(未约束,诊断)",
                                 mean=float(x.mean()), mean_abs=float(x.abs().mean()), max_abs=float(x.abs().max()),
                                 p95_abs=float(x.abs().quantile(0.95)), bound=np.nan, frac_over_bound=np.nan))
            exp_rows.append(dict(variant=nm, window=wn, source="实际持仓(逐日,相位合并)", factor="覆盖池外主动权重",
                                 mean=float(unc[sel].mean()), mean_abs=float(unc[sel].mean()),
                                 max_abs=float(unc[sel].max()), p95_abs=float(np.quantile(unc[sel], 0.95)),
                                 bound=np.nan, frac_over_bound=np.nan))
            # 事前 TE(实际持仓的风险模型预测)vs 事后 TE
            d_ex = (v["rets"] - rb["open10"]).reindex(idx)[sel].dropna()
            tp = pd.Series(te_pred, index=idx)[sel]
            exp_rows.append(dict(variant=nm, window=wn, source="TE", factor="事前TE(实际持仓,均值)",
                                 mean=float(tp.mean()), max_abs=float(tp.max()), p95_abs=float(tp.quantile(0.95))))
            exp_rows.append(dict(variant=nm, window=wn, source="TE", factor="事后TE(日频×√242)",
                                 mean=float(d_ex.std() * np.sqrt(TD))))
            exp_rows.append(dict(variant=nm, window=wn, source="TE", factor="事后TE(NW20 长期方差)",
                                 mean=float(d_ex.std() * np.sqrt(TD) * np.sqrt(ST.lr_variance_ratio(d_ex.to_numpy())))))
            # 偏差统计量:z = 实现 5 日主动收益 / 事前 5 日主动波动,5 个错位起点各取不重叠窗口,std 取均值
            act = (v["rets"] - rb["open10"]).reindex(idx)
            act5 = act.rolling(5).sum().shift(-4)
            z = (act5 / (pd.Series(te_pred, index=idx) / np.sqrt(TD / 5)))[sel].dropna()
            bs = [float(np.std(z.iloc[k::5], ddof=1)) for k in range(5) if len(z.iloc[k::5]) > 10]
            ts_store[nm].setdefault("z", {})[wn] = z
            bt = b_boot([b_matrix(z)])[:, 0] if len(z) >= 100 else np.full(2, np.nan)
            bi = b_boot([b_matrix(z)], block=1)[:, 0] if len(z) >= 100 else np.full(2, np.nan)   # 敏感性:iid
            # bound = b=1、正态假设下的 95% 半宽 sqrt(2/n)(初版唯一给的带宽);ci_lo/ci_hi = 移动块自助 95% 区间
            exp_rows.append(dict(variant=nm, window=wn, source="TE", factor="偏差统计量 b(5日,不重叠,5起点均值)",
                                 mean=float(np.mean(bs)) if bs else np.nan,
                                 max_abs=float(np.max(bs)) if bs else np.nan, p95_abs=float(np.min(bs)) if bs else np.nan,
                                 bound=float(np.sqrt(2 / max(len(z) / 5, 1))),
                                 ci_lo=float(np.nanquantile(bt, 0.025)), ci_hi=float(np.nanquantile(bt, 0.975)),
                                 boot_se=float(np.nanstd(bt)), ci_lo_iid=float(np.nanquantile(bi, 0.025)),
                                 ci_hi_iid=float(np.nanquantile(bi, 0.975))))
        # 事前(优化器目标)暴露:solver log 里的 exp:* 列
        sv = v["solver"]
        if sv is not None and "exp:Size 市值" in sv:
            ok = sv[sv.status.isin(["optimal", "optimal_inaccurate"])]
            for f in sty:
                x = ok[f"exp:{f}"]
                bd = bounds[f]
                exp_rows.append(dict(variant=nm, window="全期", source="优化器目标(调仓日)", factor=f,
                                     mean=float(x.mean()), mean_abs=float(x.abs().mean()), max_abs=float(x.abs().max()),
                                     p95_abs=float(x.abs().quantile(0.95)), bound=bd,
                                     frac_over_bound=float((x.abs() > bd + 1e-5).mean())))
            x = ok["max_abs_ind"]
            exp_rows.append(dict(variant=nm, window="全期", source="优化器目标(调仓日)", factor="行业 max|·|(30个取最大)",
                                 mean=float(x.mean()), mean_abs=float(x.mean()), max_abs=float(x.max()),
                                 p95_abs=float(x.quantile(0.95)), bound=0.02, frac_over_bound=float((x > 0.02 + 1e-5).mean())))
            if "exp:Rev20" in ok:
                x = ok["exp:Rev20"]
                exp_rows.append(dict(variant=nm, window="全期", source="优化器目标(调仓日)", factor="Rev20(V8 约束)",
                                     mean=float(x.mean()), mean_abs=float(x.abs().mean()), max_abs=float(x.abs().max()),
                                     p95_abs=float(x.abs().quantile(0.95)), bound=0.10,
                                     frac_over_bound=float((x.abs() > 0.10 + 1e-5).mean())))
            x = ok["te_ex_ante"]
            exp_rows.append(dict(variant=nm, window="全期", source="优化器目标(调仓日)", factor="事前TE(目标)",
                                 mean=float(x.mean()), max_abs=float(x.max()), p95_abs=float(x.quantile(0.95)),
                                 bound=jspec(nm)["cfg"].get("te_ann", 0.05)))

        # ---- 事后归因:A = 实际持仓 − 代理基准(含覆盖池外),R = 信号日 open10 收益
        A = pd.DataFrame(Hv - Wv[pos], index=idx, columns=codes)
        att, fac = rm.attribute(A, R, by_factor=True)
        att["rev20"] = (pd.Series(e20, index=idx) * pd.Series(g20[pos], index=idx)).reindex(att.index)
        att["specific_ex_rev20"] = att["specific"] - att["rev20"].fillna(0.0)
        eng_ex = (v["rets"] - rb["open10"]).reindex(att.index)
        cost = v["daily"]["cost_total"].reindex(att.index)
        att["engine_excess"] = eng_ex
        att["cost"] = cost
        att["recon_resid"] = att["total"] - cost - eng_ex          # 归因总额(毛)− 成本 − 引擎超额,应≈0
        ts_store[nm]["att"] = att
        for wn, a, b in windows(v["spec"].get("start", START)) + ([("同窗2023-25", V7_START, END)] if nm != "V7" else []):
            sel = (att.index >= a) & (att.index <= b)
            if sel.sum() < 30:
                continue
            g = att[sel]
            for comp in ["total", "country", "style", "industry", "specific", "rev20", "specific_ex_rev20",
                         "cost", "engine_excess", "recon_resid"]:
                x = g[comp].dropna()
                att_rows.append(dict(variant=nm, window=wn, component=comp, ann_pp=float(x.mean() * TD * 100),
                                     t_nw=nw_t(x, 5), n_days=len(x)))
            # R²(主动特质 ~ 反转贡献)
            ok = g[["specific", "rev20"]].dropna()
            if len(ok) > 30:
                att_rows.append(dict(variant=nm, window=wn, component="R2_specific_on_rev20",
                                     ann_pp=float(np.corrcoef(ok["specific"], ok["rev20"])[0, 1] ** 2), t_nw=np.nan,
                                     n_days=len(ok)))
            gf = fac[(fac.index >= a) & (fac.index <= b)]
            for f in sty + ["行业合计"]:
                x = gf[rm.ind_names].sum(axis=1) if f == "行业合计" else gf[f]
                att_rows.append(dict(variant=nm, window=wn, component=f"style:{f}" if f != "行业合计" else "industry_sum_check",
                                     ann_pp=float(x.mean() * TD * 100), t_nw=nw_t(x, 5), n_days=len(x)))
    # b 的配对比较(同一组自助行索引)与窗口间比较(独立自助)。审查 P2:"优化组合比不优化的偏差更大"要有区间支持
    for a_, b_ in [("V1", "CTRL_excl"), ("V1", "CTRL_top20"), ("V1_ph0", "V8"), ("V1_ph0", "V5"), ("V1_ph0", "V3"),
                   ("V1_ph0", "V2"), ("V1_ph0", "V6"), ("V1_ph0", "CTRL_excl")]:
        for wn in ("全期", "开发窗2021-23", "检验窗2024-25"):
            z1 = ts_store.get(a_, {}).get("z", {}).get(wn)
            z2 = ts_store.get(b_, {}).get("z", {}).get(wn)
            if z1 is None or z2 is None:
                continue
            common = z1.index.intersection(z2.index)
            Z1, Z2 = b_matrix(z1.reindex(common)), b_matrix(z2.reindex(common))
            pt = float(np.nanmean(np.nanstd(Z1, axis=0, ddof=1)) - np.nanmean(np.nanstd(Z2, axis=0, ddof=1)))
            bt = b_boot([Z1, Z2])
            dd = bt[:, 0] - bt[:, 1]
            bi = b_boot([Z1, Z2], block=1)
            di = bi[:, 0] - bi[:, 1]
            exp_rows.append(dict(variant=f"{a_} − {b_}", window=wn, source="TE", factor="偏差统计量 b 差(配对移动块自助)",
                                 mean=pt, ci_lo=float(np.quantile(dd, 0.025)), ci_hi=float(np.quantile(dd, 0.975)),
                                 boot_se=float(dd.std()), p_boot_le0=float((dd <= 0).mean()),
                                 ci_lo_iid=float(np.quantile(di, 0.025)), ci_hi_iid=float(np.quantile(di, 0.975))))
    for nm in ("V1", "V1_ph0", "V8", "CTRL_excl", "CTRL_top20"):
        zz = ts_store.get(nm, {}).get("z", {})
        if "开发窗2021-23" in zz and "检验窗2024-25" in zz:
            Zd, Zt = b_matrix(zz["开发窗2021-23"]), b_matrix(zz["检验窗2024-25"])
            pt = float(np.nanmean(np.nanstd(Zt, axis=0, ddof=1)) - np.nanmean(np.nanstd(Zd, axis=0, ddof=1)))
            dd = b_boot([Zt], seed=2)[:, 0] - b_boot([Zd], seed=1)[:, 0]
            exp_rows.append(dict(variant=nm, window="检验窗 − 开发窗", source="TE",
                                 factor="偏差统计量 b 差(窗口间,独立移动块自助)", mean=pt,
                                 ci_lo=float(np.quantile(dd, 0.025)), ci_hi=float(np.quantile(dd, 0.975)),
                                 boot_se=float(dd.std()), p_boot_le0=float((dd <= 0).mean())))
    pd.DataFrame(exp_rows).to_csv(LEDGER / f"{PFX}_exposures.csv", index=False)
    pd.DataFrame(att_rows).to_csv(LEDGER / f"{PFX}_attribution.csv", index=False)
    print("暴露 / 归因写入")

    # ---------------- 求解日志
    sl = pd.concat([v["solver"] for nm, v in V.items() if v["solver"] is not None and "_ph" not in nm],
                   ignore_index=True)                       # V1_ph0 是 V1 相位 0 的视图,不重复计入
    keep = [c for c in sl.columns if not c.startswith("viol:")] + [c for c in sl.columns if c.startswith("viol:")]
    sl[keep].to_csv(LEDGER / f"{PFX}_solver_log.csv", index=False)

    # ---------------- 图
    plot_nav(V, rb, ts_store)
    plot_exposure(ts_store, V)
    print("完成")


# ================================================================ 图(配色:参考调色板 1-4 号槽,浅色底)
C_SURF, C_TXT, C_TXT2, C_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def _style(ax):
    import matplotlib.pyplot as plt  # noqa
    ax.set_facecolor(C_SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_TXT2, labelsize=9)
    ax.grid(axis="y", color=C_GRID, lw=0.8)
    ax.yaxis.label.set_color(C_TXT2)
    ax.title.set_color(C_TXT)


def _font():
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_nav(V, rb, ts_store):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _font()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), facecolor=C_SURF, gridspec_kw=dict(height_ratios=[3, 2]))
    ax = axes[0]
    series = [("V1", "V1 指增(TE5%,5相位)"), ("CTRL_top20", "Top20 等权"), ("CTRL_excl", "排雷缓冲20/35"),
              ("V7", "V7 B_gbt 样本外(2023起)")]
    for k, (nm, lab) in enumerate(series):
        if nm not in V:
            continue
        s = V[nm]["rets"]
        s = s[s.index >= V[nm]["spec"].get("start", START)]
        b = rb["open10"].reindex(s.index)
        rel = (1 + s).cumprod() / (1 + b).cumprod()
        ax.plot(rel.index, rel.values, color=SERIES[k], lw=2 if nm == "V1" else 1.5, label=lab)
        ax.annotate(f"{rel.iloc[-1]:.2f}", (rel.index[-1], rel.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=C_TXT2, fontsize=9, va="center")
    ax.axhline(1.0, color=C_TXT2, lw=0.8)
    ax.axvspan(SPECIAL[0], SPECIAL[1], color=C_GRID, alpha=0.6, lw=0)
    ax.axvline(TEST[0], color=C_TXT2, lw=0.8, ls="--")
    ax.text(TEST[0], ax.get_ylim()[1], " 检验窗 →", color=C_TXT2, fontsize=9, va="top")
    ax.set_ylabel("相对净值(策略/中证1000同口径代理)")
    ax.set_title("相对中证1000代理(成交时点权重)的累计超额(open10,1000万,含冲击成本;COMP_v5 为信号样本内)",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C_TXT)
    _style(ax)
    # 下:V1 事前 TE vs 事后滚动 60 日 TE
    ax = axes[1]
    if "V1" in ts_store:
        tp = ts_store["V1"]["te_pred"].iloc[5:]                 # 去掉建仓周(相位 1-4 还是现金)
        s = V["V1"]["rets"]
        d = (s - rb["open10"].reindex(s.index)).reindex(tp.index)
        roll = d.rolling(60).std() * np.sqrt(TD)
        ax.plot(tp.index, tp.values, color=SERIES[0], lw=2, label="事前 TE(风险模型,实际持仓)")
        ax.plot(roll.index, roll.values, color=SERIES[1], lw=1.5, label="事后 TE(滚动60日)")
        ax.axhline(0.05, color=C_TXT2, lw=0.8, ls="--")
        ax.text(tp.index[0], 0.05, "约束 5%", color=C_TXT2, fontsize=9, va="bottom")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title("V1 跟踪误差:事前 vs 事后(相位合并组合)", fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C_TXT)
    _style(ax)
    fig.tight_layout()
    fig.savefig(REPORTS / f"{PFX}_excess_nav.png", dpi=150, facecolor=C_SURF)
    plt.close(fig)


def plot_exposure(ts_store, V):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _font()
    if "V1" not in ts_store:
        return
    E = ts_store["V1"]["E"].iloc[5:]                            # 去掉建仓周(相位 1-4 还是现金)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), facecolor=C_SURF, sharex=True)
    ax = axes[0]
    for k, (f, lab) in enumerate([("Size 市值", "Size"), ("NonlinSize 非线性市值", "NonlinSize")]):
        ax.plot(E.index, E[f], color=SERIES[k], lw=1.5, label=lab)
    for nm, k in (("CTRL_excl", 2),):
        if nm in ts_store:
            ex = ts_store[nm]["E"].iloc[5:]
            ax.plot(ex.index, ex["Size 市值"], color=SERIES[k], lw=1.2, label="排雷 Size(对照,无约束)")
    ax.axhline(0.1, color=C_TXT2, lw=0.8, ls="--"); ax.axhline(-0.1, color=C_TXT2, lw=0.8, ls="--")
    ax.set_ylim(-0.75, 0.4)
    ax.set_title("主动暴露:Size / NonlinSize(V1 实际持仓,约束 ±0.1,虚线)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper center", labelcolor=C_TXT, ncol=3)
    _style(ax)
    ax = axes[1]
    for k, (f, lab) in enumerate([("Liquidity 流动性", "Liquidity"), ("ResidVol 残差波动", "ResidVol"),
                                  ("BookToPrice 价值", "BookToPrice")]):
        ax.plot(E.index, E[f], color=SERIES[k], lw=1.5, label=lab)
    if "e20" in ts_store["V1"]:
        e20 = ts_store["V1"]["e20"].iloc[5:]
        ax.plot(e20.index, e20.values, color=SERIES[3], lw=1.2, label="Rev20(未建模,未约束)")
    ax.axhline(0.5, color=C_TXT2, lw=0.8, ls="--"); ax.axhline(-0.5, color=C_TXT2, lw=0.8, ls="--")
    ax.set_ylim(-1.0, 1.05)
    ax.set_title("主动暴露:其余风格(约束 ±0.5)与未建模的 20 日反转", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=C_TXT, ncol=4)
    _style(ax)
    ax = axes[2]
    ind = [c for c in E.columns if c.startswith("行业:")]
    mx = E[ind].abs().max(axis=1)
    ax.plot(mx.index, mx.values, color=SERIES[0], lw=1.5, label="V1 行业 max|主动暴露|")
    if "CTRL_excl" in ts_store:
        m2 = ts_store["CTRL_excl"]["E"][ind].iloc[5:].abs().max(axis=1)
        ax.plot(m2.index, m2.values, color=SERIES[2], lw=1.2, label="排雷(对照)")
    ax.axhline(0.02, color=C_TXT2, lw=0.8, ls="--")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("行业主动暴露的最大绝对值(30 个申万一级,约束 2%)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C_TXT)
    _style(ax)
    fig.tight_layout()
    fig.savefig(REPORTS / f"{PFX}_active_exposure.png", dpi=150, facecolor=C_SURF)
    plt.close(fig)


# ================================================================ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prep", "run", "controls", "all", "report"])
    ap.add_argument("--job")
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.stage == "prep":
        for k in ("comp", "gbt"):
            ic, icd = ic_dec(k)
            print(k, "IC 均值", float(ic.mean()), "决策日 IC 预测均值", float(icd.mean()))
    elif a.stage == "run":
        run_job(a.job, a.phase)
    elif a.stage == "controls":
        run_controls()
    elif a.stage == "all":
        drive(a.procs, a.only)
        report()
    else:
        report()
