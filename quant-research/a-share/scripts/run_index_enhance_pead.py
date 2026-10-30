"""PEAD 信号接入中证 1000 指增优化器(2026-10-10,前缀 iepead)。

## 做什么
在上游 ie 的 V1 基准变体(scripts/run_index_enhance.py、src/afactor/optimizer.py)上,把 α 从 α_COMP 改成
α_COMP + α_PEAD(作业 P1),再跑一个"只用 PEAD α"的变体(P2),5 个相位平均,报告 PEAD 的边际贡献。
上游 PEAD 检验(research/PEAD_REPORT.md)的主比较 E−B 判定"证据不足",所以这里的一切结果**只能当诊断**。

## 预注册(写于计算 PEAD IC 与任何回测之前;不按结果调整,敏感性不做挑选)
信号
  - PEAD 打分 = cache/_tmp/pead_signal_open10.parquet(fu_op_yoy,营业利润单季对称同比,±2),经
    pead_engine.load_signal_panel(p, "open10") 读取。第 t 行 = t+1 日 09:31 成交前已知(约 20% 在 t 收盘后公开),
    open10 的 lag = 1,在 t+1 开盘 TWAP 成交,无前视;但**不得**把第 t 行当成 t 收盘已知的量(IC 的实现时点照样按 t+6)。
  - α_PEAD = IC_P,t · ω_i · z_P,i(Grinold,与 COMP 同一函数 optimizer.grinold_alpha):
      z_P = PEAD 打分在可投资池(V1000 决策池 ∩ 风险模型覆盖)内非缺失股票上的截面 z-score,截尾 ±3;
            非缺失 < 20 只时整行 z = 0(grinold_alpha 的现成门槛),事件池外 α_PEAD = 0;
      ω   = 风险模型 5 日特质波动(与 COMP 相同);
      IC_P = PEAD 打分的逐日 rank IC(决策池内、PEAD 非缺失 ≥ 20 只的日子;5 日前向收益
            twap_open10[s+6]/twap_open10[s+1]−1),250 行滚动均值,只用 s ≤ t−6 已实现部分,下限 0,
            不足 20 个观测记 0(optimizer.ic_for_decision 原样)。
  - P1:α = α_COMP + α_PEAD,两者直接相加、不正交化(afactor.optimizer_pead.AdditiveAlphaEnhancer)。
    其余与 V1 **完全相同**:OptConfig() 默认值(TE 5%、行业 ±2%、Size/NonlinSize ±0.1、其余风格 ±0.5、个股 +1%、
    κ=1、λ=1、候选集 = 成分 ∪ 可投资池 α 前 30% ∪ 持仓)、open10、1000 万、V1000、5 日调仓、
    CostModel(impact_y=1)、2021-04-20 ~ 2025-12-31、相位 0~4。
  - P2:α = α_PEAD 单独(IndexEnhancer 原样,打分换成 PEAD 面板、IC 换成 IC_P)。候选集规则同 V1
    ("有打分的可投资股票里 α 前 30%"),对 P2 来说"有打分"= 事件池。
  - P0(对照,任务书外、跑前加入):α ≡ 0,其余同 V1,5 相位。P2 的超额里有"零 α 优化器跟踪基准"本身的
    机械部分(可成交性约定、成本),P2 − P0 才是 PEAD 在跟踪框架里的增量。
    【审查后更正】P0 的实现是 IndexEnhancer(打分 = COMP_v5,IC ≡ 0)。IC ≤ 0 时上游候选集按 COMP 的 z 取
    "可投资池前 30%",所以 P0 的非成分候选全部来自 COMP z 前 30%(首解日 712 只),并不是与 COMP 无关的纯对照;
    P2 − P0 因此混入了"候选集不同"这一项。
  - P0c(审查后追加,**看过全部结果之后**加入,只作机制诊断):α ≡ 0 且打分全缺失 → 候选集 = 成分 ∪ 持仓,
    5 相位。回答两件事:零 α 优化器偏离基准是否只是 COMP 候选集造成的(P0 − P0c),以及 P2 相对一个
    "真正不含 COMP 信息"的零 α 对照如何(P2 − P0c)。不进预注册判据的结论,只报数。
  - Z0(金标准核对,任务书外、跑前加入):P1 的代码路径但 IC_P ≡ 0,只跑相位 0,必须与上游 ie_V1_ph0 逐位相同,
    证明 AdditiveAlphaEnhancer 没有改动 V1 的任何其他行为。
  - P1f / P2f(敏感性,**事后追加**):P1 / P2 但 IC_P 冻结为回测首日
    2021-04-20 的决策日 IC 预测(0.02596,只用 2020-01 ~ 2021-04 已实现的 IC),此后不再更新、不设下限问题。
    为什么加:PEAD 的逐日 rank IC 在 2023-2025 逐年均值都 ≤ 0(iepead_exposures.csv source=IC),按"下限 0"的
    滚动规则 IC_P 在 2023、2025 全年为 0、2024 年 97% 的日子为 0 —— P1 在检验窗几乎就是 V1、P2 就是 P0。
    冻结 IC 回答的是另一个问题:"如果研究者在回测开始时就相信 PEAD、事后不再按表现更新,结果如何",
    把"PEAD 没有价值"与"规则把它关掉了"分开。它用的 IC 只来自回测开始前,不含前视;但不是主规则,只作诊断。
    【审查后更正加入时点】原文写"看任何回测结果之前"加入,与运行日志冲突:首个 driver(logs/iepead_driver.log,
    清单"作业 16 个" = Z0/P0/P1/P2,无 P1f/P2f)在 03:49:44 已打印 P1 − V1、P2 − P0 的配对结果,
    第二个 driver(logs/iepead_driver_fixed.log,"作业 10 个" = P1f/P2f)创建于 03:49:52;各相位年化此前也已写入
    logs/iepead_<job>_ph<k>.log。因此 P1f/P2f 只能算**看过主结果之后追加**的事后敏感性,不是盲测。
评价(全部沿用 ie 的口径与函数)
  - 基准:OPT.exec_time_bench_returns(p, "open10", 代理权重)(成交时点权重的中证 1000 代理)。
  - 窗口:全期 2021-04-20~2025-12-31;开发窗 2021-04-20~2023-12-31;检验窗 2024-2025;逐年。
  - 边际贡献 = P1 − V1(同一相位配对,5 相位均值的日差序列):超额、TE、IR、换手、成本的变化。
    配对 t:5 相位**均值**日差序列的 Newey-West t,带宽 10 = max(5, 2h)(h = 5 日持有),另报带宽 5(与 ie_paired 可比)
    与"日历季度为独立单位"的季度 t。5 个相位共享行情、大部分持仓,**不当作 5 个独立样本**;逐相位差只报离散度。
  - 判据(诊断性):全期配对 NW t(带宽 10)≥ 2,且开发窗、检验窗点估计都 > 0 → "优化器里有可辨增量";
    否则"无可辨增量"。P2 − P0 同一判据。上游判定"证据不足",即使这里为正也只是诊断,不构成上线依据。
  - 归因:rm.attribute(主动权重 = 实际持仓 − 成交时点基准权重),外加 20 日反转贡献(与 ie 同法,
    市值用封存截断版 sealed_total_mv 重算)与"事件股特质":特质收益按股票拆成事件窗内(PEAD 面板最近 5 行内
    非缺失,= 任一相位最近一次调仓时在事件池)与其余两部分。问题"特质部分是否增加"看 P1 − V1 的特质、
    剔反转后特质、事件股特质三项的配对差。
信号样本内:COMP_v5 的 24 个成员与 fu_op_yoy 这一簇都是在 2020-2025 研究期内选出的,P0~P2 的全部结果都是
"信号样本内";检验窗只是事先固定的切分。幸存者偏差:面板缺约 209 只退市股;PEAD 主事件中 3.08% 的股票不在
行情面板(pead_event_survivorship.csv),事件股多为小盘,偏差方向对超额不确定。

## 封存
  - PEAD 面板、Panel 都只到 2025-12-31(断言);
  - CostModel 经 ND.raw_close 读 daily_basic:先把截到 ≤ 2025-12-31 的版本放进 newdata 缓存(与 pead_run_arms 同法),
    构建后断言缓存最大日期;
  - 报告阶段 20 日反转暴露的市值加权与回归权重用 sealed_total_mv(财报 avail ≤ 2025-12-31),
    与上游 ie_rev20_z 的差异记入 iepead_exposures.csv。
  - 风险模型(barra open10)、代理权重是上游缓存,按原样使用(局限里写明)。

## 用法
  python3 scripts/run_index_enhance_pead.py --stage prep                 # PEAD IC 缓存
  python3 scripts/run_index_enhance_pead.py --stage run --job P1 --phase 0
  python3 scripts/run_index_enhance_pead.py --stage all --procs 3        # JOBS 里全部作业(Z0 + P0/P1/P2/P1f/P2f/P0c × 5 相位)+ 报告
  python3 scripts/run_index_enhance_pead.py --stage report
  python3 scripts/run_index_enhance_pead.py --stage probe                # 审查后追加:P0 首次建仓的反事实
产出:research/ledger/iepead_{variants,yearly,paired,attribution,exposures,solver_log,p0_mechanism}.csv,
      reports/iepead_excess_nav.png;中间结果 cache/_tmp/iepead_*;日志 logs/iepead_*.log。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from afactor.config import TMP, LEDGER, REPORTS, ROOT
import run_index_enhance as IE                     # 复用 ie 的面板单例、打分、IC 缓存、评价函数(只读)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Solution may be inaccurate")

PFX = "iepead"
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
TD = 242
START, END = IE.START, IE.END
DEV, TEST = IE.DEV, IE.TEST
SEAL_END = pd.Timestamp("2025-12-31")
PEAD_MIN = 20                                      # IC 与 z 的最少事件股数(= grinold_alpha 的门槛)
NW_MAIN, NW_ALT = 10, 5                            # 配对 t 的 NW 带宽:max(5, 2h), h=5;另报 5 与 ie_paired 可比

JOBS = {
    "Z0": dict(desc="金标准核对:P1 代码路径、IC_PEAD≡0、相位0,必须与 ie_V1_ph0 逐位相同", kind="add",
               pead_ic="zero", phases=[0]),
    "P0": dict(desc="对照:α≡0(零 α 优化器跟踪基准),其余同 V1", kind="zero", phases=[0, 1, 2, 3, 4]),
    "P1": dict(desc="V1 + PEAD α(α_COMP + α_PEAD,其余同 V1)", kind="add", pead_ic="pead", phases=[0, 1, 2, 3, 4]),
    "P2": dict(desc="只用 PEAD α(其余同 V1)", kind="pead_only", phases=[0, 1, 2, 3, 4]),
    # 敏感性:IC_PEAD 冻结为回测首日的决策日 IC 预测(看过主作业结果之后追加,见模块文档"审查后更正加入时点")
    "P1f": dict(desc="敏感性:V1 + PEAD α,IC_PEAD 冻结为 2021-04-20 的决策日预测", kind="add", pead_ic="fixed",
                phases=[0, 1, 2, 3, 4]),
    "P2f": dict(desc="敏感性:只用 PEAD α,IC_PEAD 冻结为 2021-04-20 的决策日预测", kind="pead_only", pead_ic="fixed",
                phases=[0, 1, 2, 3, 4]),
    # 审查后追加(看过全部结果之后):纯零 α 对照,打分全缺失 -> 候选集 = 成分 ∪ 持仓,不含 COMP 前 30%
    "P0c": dict(desc="审查后追加:α≡0 且候选集只含成分与持仓(不含 COMP 前 30%),其余同 V1", kind="zero_nocand",
                phases=[0, 1, 2, 3, 4]),
}
IC_FIX_DATE = START                                # 冻结 IC 的取值日(决策日 IC 预测,只含 t−6 前已实现的 IC)


# ================================================================ 公共
def P():
    return IE.P()                                   # 与 IE 共用同一个 Panel 单例


def seal_newdata() -> None:
    """把截到 ≤ 2025-12-31 的 daily_basic 放进 newdata 缓存(ND.raw_close / float_share / total_mv 都从它取)。
    与 scripts/pead_build_batches.sealed_daily_basic 同法;CostModel 只用 raw_close 的研究期行,数值与不截断时相同
    (金标准 Z0 验证),截断是为了 2026 行根本不进内存。"""
    from afactor import newdata as ND
    db = ND._CACHE.get("db")
    if db is not None and db["trade_date"].max() <= SEAL_END:
        return
    d = pd.read_csv(ND.DATA / "daily_basic.csv",
                    usecols=["ts_code", "trade_date", "close", "turnover_rate",
                             "turnover_rate_f", "float_share", "circ_mv"])
    d["trade_date"] = ND._ymd(d["trade_date"])
    ND._CACHE["db"] = d[d["trade_date"] <= SEAL_END].copy()
    del d


def assert_sealed() -> None:
    from afactor import newdata as ND
    db = ND._CACHE.get("db")
    assert db is None or db["trade_date"].max() <= SEAL_END, "newdata 缓存里出现了封存期 daily_basic"


def pead_panel() -> pd.DataFrame:
    from afactor import pead_engine as PE
    p = P()
    S = PE.load_signal_panel(p, "open10")           # lag=1 口径原样返回;lag=0 会报错
    assert S.index.max() <= SEAL_END and p.dates.max() <= SEAL_END
    return S


def pead_ic() -> tuple[pd.Series, pd.Series]:
    """(逐日 PEAD rank IC, 决策日 IC 预测)。缓存 cache/_tmp/iepead_ic_pead.parquet。"""
    from afactor import optimizer as OPT, optimizer_pead as OPTP, strategy as ST
    f = TMP / f"{PFX}_ic_pead.parquet"
    if f.exists():
        df = pd.read_parquet(f)
        return df["ic"], df["ic_dec"]
    p = P()
    U = ST.universe(p, "V1000")
    S = pead_panel()
    fwd = p.fwd_ret(5, "open")
    ic = OPTP.event_rank_ic(S, fwd, U, min_stocks=PEAD_MIN)
    cfg = OPT.OptConfig()
    icd = OPT.ic_for_decision(ic, cfg.ic_window, cfg.ic_lag, cfg.ic_min_obs, 0.0)
    n = (S.notna() & U.reindex(index=S.index, columns=S.columns).fillna(False)).sum(axis=1)
    pd.DataFrame({"ic": ic, "ic_dec": icd, "n_event": n.reindex(ic.index)}).to_parquet(f)
    return ic, icd


def pead_ic_for(kind: str) -> pd.Series:
    """作业用的决策日 IC_PEAD:pead = 滚动(下限 0);zero = 恒 0(Z0);fixed = 冻结为 IC_FIX_DATE 的滚动预测。"""
    p = P()
    if kind == "zero":
        return pd.Series(0.0, index=p.dates)
    icd = pead_ic()[1].reindex(p.dates)
    if kind == "pead":
        return icd
    if kind == "fixed":
        v = float(icd.loc[IC_FIX_DATE])
        assert np.isfinite(v) and v > 0
        return pd.Series(v, index=p.dates)
    raise ValueError(kind)


# ================================================================ 作业
def run_job(name: str, phase: int) -> None:
    from afactor import strategy as ST, index_proxy as IP, optimizer as OPT, optimizer_pead as OPTP
    from afactor.barra import RiskModel
    from afactor.impact import CostModel
    t0 = time.time()
    sp = JOBS[name]
    p = P()
    seal_newdata()
    rm = RiskModel.load("open10")
    W = IP.proxy_weights(p, "000852")
    U = ST.universe(p, "V1000")
    comp = IE.load_score("comp")
    _, icd_c = IE.ic_dec("comp")
    cfg = OPT.OptConfig()                            # V1 = 默认值
    cm = CostModel(p, impact_y=1.0)
    assert_sealed()
    zero = pd.Series(0.0, index=p.dates)
    icp = pead_ic_for(sp.get("pead_ic", "pead"))
    if sp["kind"] == "add":
        enh = OPTP.AdditiveAlphaEnhancer(p, rm, W, comp, icd_c, U, cfg, tag=name,
                                         add_scores={"PEAD": (pead_panel(), icp)})
    elif sp["kind"] == "zero":
        # 注意:IC ≡ 0 时上游候选集按 COMP 的 z 取可投资池前 30%,P0 的非成分候选因此来自 COMP(审查后更正)
        enh = OPT.IndexEnhancer(p, rm, W, comp, zero, U, cfg, tag=name)
    elif sp["kind"] == "zero_nocand":
        # 打分全缺失:z ≡ 0、α ≡ 0、inv_idx 为空 -> 候选集 = 成分 ∪ 持仓(覆盖池内)
        blank = pd.DataFrame(np.nan, index=p.dates, columns=p.codes, dtype=np.float32)
        enh = OPT.IndexEnhancer(p, rm, W, blank, zero, U, cfg, tag=name)
    elif sp["kind"] == "pead_only":
        enh = OPT.IndexEnhancer(p, rm, W, pead_panel(), icp, U, cfg, tag=name)
    else:
        raise ValueError(sp["kind"])
    mask = IE.date_mask(START)
    print(f"[{name} ph{phase}] kind={sp['kind']} 载入 {time.time()-t0:.0f}s", flush=True)
    res = ST.simulate(enh, p, "open10", 1e7, rebalance=5, date_mask=mask, phase=phase,
                      cost_model=cm, record=True, participation_on_nav=False)
    prefix = f"{PFX}_{name}_ph{phase}"
    ST.save_record(res, prefix)
    lg = enh.log_frame()
    lg.to_parquet(TMP / f"{prefix}_solver.parquet")
    summ = {k: res[k] for k in ("ann_return", "sharpe", "max_dd", "turnover", "cost_bps", "holdings",
                                 "exposure", "n_days", "vol")}
    summ.update(job=name, phase=phase, n_solves=len(lg), n_failed=int((lg["status"] == "failed").sum()),
                elapsed_s=time.time() - t0, cfg=cfg.describe(), kind=sp["kind"])
    json.dump(summ, open(TMP / f"{prefix}_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(f"[{name} ph{phase}] 年化 {res['ann_return']:.2%} 换手 {res['turnover']:.4f} 求解 {len(lg)} 次 "
          f"失败 {summ['n_failed']} 用时 {time.time()-t0:.0f}s", flush=True)


def drive(procs: int, only: list[str] | None = None) -> None:
    pead_ic()
    tasks = [(nm, ph) for nm in JOBS for ph in JOBS[nm]["phases"]]
    if only:
        tasks = [x for x in tasks if x[0] in only]
    todo = [x for x in tasks if not (TMP / f"{PFX}_{x[0]}_ph{x[1]}_summary.json").exists()]
    print(f"作业 {len(tasks)} 个,待跑 {len(todo)} 个,并行 {procs}", flush=True)
    running: list = []
    script = os.path.abspath(__file__)
    while todo or running:
        while todo and len(running) < procs:
            nm, ph = todo.pop(0)
            lf = LOGS / f"{PFX}_{nm}_ph{ph}.log"
            fh = open(lf, "w")
            pr = subprocess.Popen([sys.executable, script, "--stage", "run", "--job", nm, "--phase", str(ph)],
                                  stdout=fh, stderr=subprocess.STDOUT)
            running.append((nm, ph, pr, fh, time.time()))
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


# ================================================================ 结果装载
def load_variant(name: str) -> dict:
    """V1 / V1_ph0 从上游 ie 缓存读(IE.load_variant);本前缀作业同法合并相位(相位均值)。"""
    from afactor.strategy import _REC_COLS
    if name in ("V1", "V1_ph0"):
        v = IE.load_variant(name)
        v["src"] = "ie"
        return v
    base, _, phs = name.partition("_ph")
    phases = JOBS[base]["phases"] if not phs else [int(phs)]
    ds, Hs, sl = [], [], []
    for ph in phases:
        pre = TMP / f"{PFX}_{base}_ph{ph}"
        ds.append(pd.read_parquet(f"{pre}_daily.parquet").assign(phase=ph))
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
                solver=pd.concat(sl, ignore_index=True), spec=dict(mode="open10", capital=1e7, start=START,
                                                                  phases=phases), src=PFX)


# ================================================================ 统计工具
def _win(s, a, b):
    return s[(s.index >= a) & (s.index <= b)]


def paired_stats(d: pd.Series) -> dict:
    """配对日差(已是 5 相位均值序列)的年化与三种 t。"""
    from afactor.barra import nw_t
    from afactor import pead_engine as PE
    d = d.dropna()
    if len(d) < 30:
        return dict(n_days=len(d))
    q = PE.quarterly_t(d)
    return dict(n_days=len(d), diff_arith_ann=float(d.mean() * TD), t_nw10=nw_t(d.to_numpy(), NW_MAIN),
                t_nw5=nw_t(d.to_numpy(), NW_ALT), n_q=q["n_q"], q_t=q["q_t"],
                te_diff=float(d.std() * np.sqrt(TD)))


def verdict(rows: list[dict]) -> str:
    """预注册判据:全期 t_nw10 ≥ 2 且开发窗、检验窗点估计都 > 0。"""
    r = {x["window"]: x for x in rows}
    try:
        ok = (r["全期"]["t_nw10"] >= 2 and r["开发窗2021-23"]["diff_arith_ann"] > 0
              and r["检验窗2024-25"]["diff_arith_ann"] > 0)
    except KeyError:
        return "缺数据"
    return "有可辨增量(诊断)" if ok else "无可辨增量"


# ================================================================ 报告
PAIRS = [("P1", "V1", "PEAD 的边际贡献(V1 + PEAD − V1)"),
         ("P2", "P0", "只用 PEAD α 相对零 α 优化器"),
         ("P1f", "V1", "敏感性:IC_PEAD 冻结时的边际贡献"),
         ("P2f", "P0", "敏感性:IC_PEAD 冻结时只用 PEAD vs 零 α"),
         ("V1", "P0", "参照:COMP α 相对零 α 优化器"),
         ("P1", "P0", "参照:COMP + PEAD 相对零 α 优化器"),
         # 审查后追加(看过全部结果之后,只作机制诊断,不作判定)
         ("P0", "P0c", "审查后追加:零 α 下 COMP 前 30% 候选集的作用(P0 − 纯零 α)"),
         ("P2", "P0c", "审查后追加:只用 PEAD vs 纯零 α(候选集只含成分与持仓)")]


def report() -> None:
    from afactor import strategy as ST, index_proxy as IP, barra as BR, optimizer as OPT
    from afactor.barra import RiskModel, psd_sqrt, nw_t
    from pead_build_batches import sealed_total_mv
    p = P()
    seal_newdata()
    dates, codes = p.dates, p.codes
    T = len(dates)
    rm = RiskModel.load("open10")
    Wb = IP.proxy_weights(p, "000852")
    rb, Wx = OPT.exec_time_bench_returns(p, "open10", Wb, return_weights=True)
    rreal = IP.index_returns_on_signal_days(p, "open10", "000852")
    V = {nm: load_variant(nm) for nm in ["V1", "V1_ph0", "Z0", "P0", "P1", "P2", "P1f", "P2f", "P0c"]}
    desc = {"V1": "上游 ie V1(COMP α,5 相位;缓存直接读取)", "V1_ph0": "上游 ie V1 相位 0",
            **{k: v["desc"] for k, v in JOBS.items()}}
    exp_rows: list[dict] = []

    # ---------------- 金标准:Z0 vs ie_V1_ph0
    z0, v0 = V["Z0"], V["V1_ph0"]
    dr = (z0["rets"] - v0["rets"]).abs()
    Hz, Hv0 = z0["H"].astype(np.float64), v0["H"].astype(np.float64)
    cols = Hz.columns.union(Hv0.columns)
    dh = (Hz.reindex(columns=cols).fillna(0) - Hv0.reindex(index=Hz.index, columns=cols).fillna(0)).abs().to_numpy().max()
    sz, sv = z0["solver"], v0["solver"]
    dobj = float(np.nanmax(np.abs(sz["obj"].to_numpy() - sv["obj"].to_numpy()))) if len(sz) == len(sv) else np.nan
    for k, val in (("max|Δ日收益|", float(dr.max())), ("max|Δ持仓权重|", float(dh)), ("max|Δ目标函数值|", dobj),
                   ("交易日数差", float(len(z0["rets"]) - len(v0["rets"]))), ("求解次数差", float(len(sz) - len(sv)))):
        exp_rows.append(dict(variant="Z0 vs ie_V1_ph0", window="全期", source="金标准", factor=k, mean=val,
                             ok=bool(val == 0.0)))
    print("金标准 Z0 vs ie_V1_ph0:", exp_rows[-5:], flush=True)

    # ---------------- 引擎判据(纪律 4)
    for nm in ("Z0", "P0", "P1", "P2", "P1f", "P2f", "P0c"):
        s = V[nm]["rets"]
        ks = (-2, -1, 0, 1, 2)
        cs = {k: float(s.corr(rb.shift(k).reindex(s.index))) for k in ks}
        peak = max(ks, key=lambda k: cs[k])
        exp_rows.append(dict(variant=nm, window="全期", source="引擎判据", factor="corr_peak_vs_proxy (k)",
                             mean=float(peak), detail=" ".join(f"{k:+d}:{cs[k]:.3f}" for k in ks), ok=bool(peak == 0)))
        mx = float(V[nm]["by_phase"]["exposure"].max())
        exp_rows.append(dict(variant=nm, window="全期", source="引擎判据", factor="max_exposure", mean=mx,
                             ok=bool(mx <= 1 + 1e-9)))
        assert peak == 0, f"{nm} 与基准错位"
        assert mx <= 1 + 1e-9, f"{nm} exposure > 1"

    # ---------------- PEAD IC 与覆盖
    ic, icd = pead_ic()
    icdf = pd.read_parquet(TMP / f"{PFX}_ic_pead.parquet")
    _, icd_c = IE.ic_dec("comp")
    for wn, a, b in IE.windows() + IE.years()[:-1]:
        x = _win(ic, a, b).dropna()
        y = _win(icd, a, b)
        n = _win(icdf["n_event"], a, b)
        exp_rows.append(dict(variant="PEAD 信号", window=wn, source="IC", factor="逐日 rank IC(事件股≥20 的日子)",
                             mean=float(x.mean()), n_days=len(x), t_nw10=nw_t(x.to_numpy(), NW_MAIN),
                             detail="5 日前向收益,决策池内;相邻日事件与收益窗口重叠,t 只作参考"))
        exp_rows.append(dict(variant="PEAD 信号", window=wn, source="IC", factor="决策日 IC 预测(下限 0 后)",
                             mean=float(y.mean()), frac_zero=float((y.fillna(0) <= 0).mean()), n_days=len(y)))
        exp_rows.append(dict(variant="COMP_v5", window=wn, source="IC", factor="决策日 IC 预测(下限 0 后)",
                             mean=float(_win(icd_c, a, b).mean())))
        exp_rows.append(dict(variant="PEAD 信号", window=wn, source="覆盖", factor="决策池内事件股数/日",
                             mean=float(n.mean()), p50=float(n.median()), frac_ge20=float((n >= PEAD_MIN).mean())))

    # ---------------- PEAD α 在调仓日的生效情况(求解日志,逐窗口/逐年)
    # 审查后修复:P2/P2f 用原版 IndexEnhancer,日志里没有 n_z 列,旧版把缺失当"生效",frac_active 只等于 IC>0 的比例
    # (P2f 记成 1.000)。现在 4 个变体统一用 PEAD 面板重算"可投资池(决策池 ∩ 风险模型覆盖)内事件股数",
    # 生效 = IC 预测 > 0 且事件股 ≥ 20(grinold_alpha 的门槛);P1/P1f 另与日志 n_z 交叉核对。
    Spd = pead_panel().to_numpy(np.float32)
    Ub = ST.universe(p, "V1000").reindex(index=dates, columns=codes).fillna(False).to_numpy(bool)

    def n_event_inv(ds) -> np.ndarray:
        pos_ = dates.get_indexer(ds)
        assert (pos_ >= 0).all()
        return np.array([int((Ub[i] & np.asarray(rm.mask[i]) & np.isfinite(Spd[i])).sum()) for i in pos_])

    for nm in ("P1", "P1f", "P2", "P2f"):
        sv = V[nm]["solver"]
        ok = sv[sv.status.isin(["optimal", "optimal_inaccurate"])].copy()
        ok["date"] = pd.to_datetime(ok["date"])
        ok = ok.set_index("date").sort_index()
        ne_ = pd.Series(n_event_inv(ok.index), index=ok.index)
        cover_ = ne_ >= PEAD_MIN
        if "ic:PEAD" in ok:                              # P1 / P1f:附加 α 的诊断列
            icp_, nz_, al_ = ok["ic:PEAD"].fillna(0), ok["n_z:PEAD"].fillna(0), ok["alpha_ex_ante:PEAD"].fillna(0)
            alt = ok["alpha_ex_ante"]
            aw_, ex_ = ok["aw:PEAD"], ok["exp:PEAD"]
            mism = int(((nz_ > 0) != cover_).sum())          # 日志 n_z 与面板重算是否一致
            exp_rows.append(dict(variant=nm, window="全期", source="求解日志(调仓日,5相位合计)",
                                 factor="核对:日志 n_z:PEAD>0 与面板重算事件股≥20 不一致的调仓日数", mean=float(mism),
                                 n_solves=len(ok), ok=bool(mism == 0)))
        else:                                            # P2 / P2f:α 全部来自 PEAD
            icp_, nz_, al_ = ok["ic"].fillna(0), None, ok["alpha_ex_ante"]
            alt, aw_, ex_ = ok["alpha_ex_ante"], pd.Series(np.nan, index=ok.index), pd.Series(np.nan, index=ok.index)
        for wn, a, b in IE.windows() + IE.years()[:-1]:
            g = (ok.index >= a) & (ok.index <= b)
            if g.sum() == 0:
                continue
            on = g & (icp_ > 0).to_numpy() & cover_.to_numpy()
            exp_rows.append(dict(variant=nm, window=wn, source="求解日志(调仓日,5相位合计)", factor="PEAD α 生效",
                                 n_solves=int(g.sum()), frac_ic_pos=float((icp_[g] > 0).mean()),
                                 frac_cover_ge20=float(cover_[g].mean()),
                                 frac_active=float(on.sum() / g.sum()),
                                 mean=float(al_[g].mean() * TD / 5), alpha_total_ann=float(alt[g].mean() * TD / 5),
                                 aw_active=float(aw_[on].mean()) if on.any() else np.nan,
                                 exp_active=float(ex_[on].mean()) if on.any() else np.nan,
                                 detail="frac_active = IC 预测>0 且可投资池内事件股≥20(面板重算,4 变体同口径);"
                                        "frac_ic_pos 不含事件股门槛;mean = 事前 PEAD α 年化(α_PEAD'(w−wb)×242/5);"
                                        "alpha_total_ann = 事前总 α 年化;aw/exp_active = 生效调仓日的事件股主动权重 / "
                                        "PEAD z 主动暴露(仅 P1/P1f)"))
    del Spd, Ub

    # ---------------- 零 α 优化器为什么不贴基准(审查后追加:求解日志里的建仓日与逐年/逐季事前诊断)
    # 风险项 = λ·Var5 = λ·te_ex_ante²·5/242(目标函数里与成本同为 5 日收益单位);cost_ex_ante = 本次计划交易成本
    lam = OPT.OptConfig().risk_aversion
    for nm in ("P0", "P0c", "P2", "V1"):
        sv = V[nm]["solver"]
        ok = sv[sv.status.isin(["optimal", "optimal_inaccurate"])].copy()
        ok["date"] = pd.to_datetime(ok["date"])
        if nm in ("P0", "P0c"):
            for ph, g in ok.groupby("phase"):
                f0 = g.sort_values("date").iloc[0]
                exp_rows.append(dict(variant=nm, window=f"相位{ph} 首次建仓 {f0['date']:%Y-%m-%d}", source="零α机制(求解日志)",
                                     factor="首次建仓日(从现金)", bench_w_in=float(f0["bench_w_in"]),
                                     te_ex_ante=float(f0["te_ex_ante"]), cost_ex_ante=float(f0["cost_ex_ante"]),
                                     risk_term_5d=float(lam * f0["te_ex_ante"] ** 2 * 5 / TD),
                                     n_hold=float(f0["n_hold"]), n_cand=float(f0["n_cand"]),
                                     detail="bench_w_in = 目标组合中成分股权重合计;risk_term_5d = λ·Var5"))
            ok["q"] = ok["date"].dt.to_period("Q").astype(str)
            for q, g in ok.groupby("q"):
                exp_rows.append(dict(variant=nm, window=q, source="零α机制(求解日志)", factor="逐季均值",
                                     bench_w_in=float(g["bench_w_in"].mean()), te_ex_ante=float(g["te_ex_ante"].mean()),
                                     n_hold=float(g["n_hold"].mean()), n_solves=len(g)))
        for wn, a, b in IE.windows() + IE.years()[:-1]:
            g = ok[(ok["date"] >= a) & (ok["date"] <= b)]
            exp_rows.append(dict(variant=nm, window=wn, source="零α机制(求解日志)", factor="窗口均值",
                                 bench_w_in=float(g["bench_w_in"].mean()), te_ex_ante=float(g["te_ex_ante"].mean()),
                                 cost_ex_ante=float(g["cost_ex_ante"].mean()), n_hold=float(g["n_hold"].mean()),
                                 n_cand=float(g["n_cand"].mean()), n_solves=len(g)))

    # ---------------- 绩效:全期/开发/检验 + 逐年(逐相位)
    rows, yrows = [], []
    for nm, v in V.items():
        sv = v["solver"]
        extra = dict(variant=nm, desc=desc[nm], mode="open10", capital=1e7,
                     n_phases=int(v["by_phase"]["phase"].nunique()), signal="信号样本内(COMP_v5 / fu_op_yoy)")
        if sv is not None:
            ok = sv[sv.status != "no_risk_model"]
            extra.update(n_solves=len(ok), n_failed=int((ok.status == "failed").sum()),
                         n_inaccurate=int((ok.status == "optimal_inaccurate").sum()),
                         n_scs=int((ok.solver == "SCS").sum()), n_cap_relaxed=int(ok.cap_relaxed.fillna(False).sum()),
                         te_ex_ante_mean=float(ok.te_ex_ante.mean()), ic_dec_mean=float(ok.ic.mean()),
                         alpha_ex_ante_ann=float(ok.alpha_ex_ante.mean() * TD / 5),
                         n_hold_target=float(ok.n_hold.mean()), turnover_plan=float(ok.turnover_plan.mean()),
                         n_cand_mean=float(ok.n_cand.mean()))
            if "alpha_ex_ante:PEAD" in ok:
                act = ok["n_z:PEAD"].fillna(0) > 0
                on = act & (ok["ic:PEAD"].fillna(0) > 0)
                extra.update(pead_ic_dec_mean=float(ok["ic:PEAD"].mean()),
                             pead_ic_zero_frac=float((ok["ic:PEAD"].fillna(0) <= 0).mean()),
                             pead_cover_frac=float(act.mean()), pead_active_frac=float(on.mean()),
                             pead_alpha_ex_ante_ann=float(ok["alpha_ex_ante:PEAD"].fillna(0).mean() * TD / 5),
                             pead_aw_active_mean=float(ok.loc[on, "aw:PEAD"].mean()) if on.any() else np.nan,
                             pead_exp_active_mean=float(ok.loc[on, "exp:PEAD"].mean()) if on.any() else np.nan)
        for wn, a, b in IE.windows():
            o = IE.perf(v["rets"], rb, v["daily"], a, b, rreal=rreal)
            rows.append(dict(extra, window=wn, **o))
        for yn, a, b in IE.years():
            o = IE.perf(v["rets"], rb, v["daily"], a, b, rreal=rreal)
            yrows.append(dict(variant=nm, phase="avg", period=yn, **o))
            if v["by_phase"]["phase"].nunique() > 1:
                for ph, g in v["by_phase"].groupby("phase"):
                    o = IE.perf(g["ret"], rb, None, a, b)
                    yrows.append(dict(variant=nm, phase=str(ph), period=yn, **o))
    var = pd.DataFrame(rows)
    yr = pd.DataFrame(yrows)
    var.to_csv(LEDGER / f"{PFX}_variants.csv", index=False)
    yr.to_csv(LEDGER / f"{PFX}_yearly.csv", index=False)
    print("绩效写入", LEDGER / f"{PFX}_variants.csv", flush=True)

    # ---------------- 配对:5 相位均值日差 + 逐相位离散度 + 指标差(边际贡献)
    prs = []
    for a_, b_, lab in PAIRS:
        d_all = (V[a_]["rets"] - V[b_]["rets"]).dropna()
        wrows = []
        for wn, a, b in IE.windows() + IE.years()[:-1]:
            st = paired_stats(_win(d_all, a, b))
            row = dict(pair=f"{a_} − {b_}", desc=lab, window=wn, kind="配对日差(5相位均值)", **st)
            # 指标差:两侧各自的绩效相减(超额几何、TE、IR、换手、成本、持股数)
            va = var[(var.variant == a_) & (var.window == wn)]
            vb = var[(var.variant == b_) & (var.window == wn)]
            if len(va) and len(vb):
                for m in ("excess_geo", "excess_arith", "te", "ir", "te_monthly", "turnover", "cost_ann",
                          "cost_impact_ann", "n_hold", "max_rel_dd"):
                    row[f"d_{m}"] = float(va[m].iloc[0] - vb[m].iloc[0])
                    row[f"{m}_a"] = float(va[m].iloc[0])
                    row[f"{m}_b"] = float(vb[m].iloc[0])
            prs.append(row)
            if wn in ("全期", "开发窗2021-23", "检验窗2024-25"):
                wrows.append(dict(window=wn, **st))
        # 逐相位(同一相位配对):只报离散度,不当独立样本
        ba, bb = V[a_]["by_phase"], V[b_]["by_phase"]
        if ba["phase"].nunique() > 1 and bb["phase"].nunique() > 1:
            for wn, a, b in IE.windows():
                ds = []
                for ph in sorted(ba["phase"].unique()):
                    dk = (ba[ba.phase == ph]["ret"] - bb[bb.phase == ph]["ret"]).dropna()
                    dk = _win(dk, a, b)
                    ds.append(float(dk.mean() * TD))
                    prs.append(dict(pair=f"{a_} − {b_}", desc=lab, window=wn, kind=f"相位{ph}", n_days=len(dk),
                                    diff_arith_ann=float(dk.mean() * TD), t_nw10=nw_t(dk.to_numpy(), NW_MAIN)))
                prs.append(dict(pair=f"{a_} − {b_}", desc=lab, window=wn, kind="相位离散度",
                                diff_arith_ann=float(np.mean(ds)), phase_std=float(np.std(ds, ddof=1)),
                                phase_min=float(np.min(ds)), phase_max=float(np.max(ds))))
        vd = verdict(wrows)
        if lab.startswith("审查后追加"):
            vd = f"(事后诊断,不作判定;机械套用判据 = {vd})"
        prs.append(dict(pair=f"{a_} − {b_}", desc=lab, window="判据", kind="预注册判据", verdict=vd))
    pr = pd.DataFrame(prs)
    pr.to_csv(LEDGER / f"{PFX}_paired.csv", index=False)
    print(pr[pr.kind.isin(["配对日差(5相位均值)", "预注册判据"]) & ~pr.window.str.match(r"^\d")]
          [["pair", "window", "diff_arith_ann", "t_nw10", "t_nw5", "q_t", "d_te", "d_ir", "d_turnover", "d_cost_ann",
            "verdict"]].to_string(), flush=True)

    # ---------------- 归因、暴露(实际持仓)
    att_rows = attribution(p, rm, Wx, rb, V, exp_rows, sealed_total_mv)
    pd.DataFrame(att_rows).to_csv(LEDGER / f"{PFX}_attribution.csv", index=False)
    pd.DataFrame(exp_rows).to_csv(LEDGER / f"{PFX}_exposures.csv", index=False)
    sl = pd.concat([V[nm]["solver"] for nm in ("Z0", "P0", "P1", "P2", "P1f", "P2f", "P0c")], ignore_index=True)
    sl.to_csv(LEDGER / f"{PFX}_solver_log.csv", index=False)
    plot(V, rb, icd, icd_c)
    print("完成", flush=True)


def rev20_sealed(p, sealed_total_mv) -> tuple[pd.DataFrame, np.ndarray]:
    """20 日反转暴露,与 IE.rev20_z 同一构造,但市值用封存截断版。返回 (z20, mv ndarray)。
    缓存 cache/_tmp/iepead_rev20_z.parquet;与上游 ie_rev20_z 的最大差记入 exposures。"""
    from afactor import barra as BR, strategy as ST
    f = TMP / f"{PFX}_rev20_z.parquet"
    fm = TMP / f"{PFX}_mv_sealed.parquet"
    if f.exists() and fm.exists():
        z = pd.read_parquet(f); mv = pd.read_parquet(fm)
        if z.index.equals(p.dates) and z.columns.equals(p.codes):
            return z, mv.reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
    seal_newdata()
    mv = sealed_total_mv(p.dates, p.codes).reindex(index=p.dates, columns=p.codes)
    assert_sealed()
    uu = ST.universe(p, "V1000").reindex(index=p.dates, columns=p.codes).fillna(False)
    close = p["close_px"]
    x20 = -(close / close.shift(20) - 1.0)
    z = BR.zscore(BR.robust_clip(x20, uu, 5.0), uu, mv.where(uu)).astype(np.float32)
    z.to_parquet(f)
    mv.astype(np.float32).to_parquet(fm)
    return z, mv.to_numpy(np.float64)


def attribution(p, rm, Wx, rb, V, exp_rows, sealed_total_mv) -> list[dict]:
    """事后归因(与 ie 同法)+ 事件股特质拆分 + 配对差。A = 实际持仓 − 成交时点基准权重,R = 信号日 open10 收益。"""
    from afactor import barra as BR, strategy as ST
    from afactor.barra import psd_sqrt, nw_t
    dates, codes = p.dates, p.codes
    T = len(dates)
    z20df, mvv = rev20_sealed(p, sealed_total_mv)
    z20 = z20df.to_numpy(np.float64)
    zie = pd.read_parquet(TMP / "ie_rev20_z.parquet").reindex(index=dates, columns=codes).to_numpy(np.float64)
    sel = np.asarray(dates >= START)
    dz = np.abs(z20[sel] - zie[sel])
    exp_rows.append(dict(variant="封存核对", window="全期", source="封存", factor="max|z20(封存市值) − ie_rev20_z|",
                         mean=float(np.nanmax(dz)), p95_abs=float(np.nanquantile(dz, 0.95)),
                         detail="ie 用 FA._real(p,'mv')(财报保留最晚公告版本,可能含 2026 修订);本文用 sealed_total_mv"))
    del zie, dz
    R = BR.fwd_returns(p, "open10")
    Rv = R.to_numpy(np.float64)
    resid = np.asarray(rm.resid()[:len(R)], np.float64)
    U = ST.universe(p, "V1000").reindex(index=dates, columns=codes).fillna(False).to_numpy(bool)
    # 20 日反转因子收益 g20:风险模型残差对 z20 的 sqrt(sqrt(市值)) 加权截面回归斜率(与 ie 同法)
    g20 = np.full(T, np.nan)
    first = int(np.searchsorted(dates, START))
    for t in range(first, len(R)):
        m = U[t] & np.isfinite(resid[t]) & np.isfinite(z20[t]) & np.isfinite(mvv[t]) & (mvv[t] > 0)
        if m.sum() < 200:
            continue
        ww = np.sqrt(np.sqrt(mvv[t, m]))
        A_ = np.column_stack([np.ones(m.sum()), z20[t, m]])
        g20[t] = np.linalg.lstsq(A_ * ww[:, None], resid[t, m] * ww, rcond=None)[0][1]
    del resid
    # 事件窗掩码:PEAD 面板最近 5 行(d−4..d)内非缺失 = 任一相位最近一次调仓时在事件池
    S = pead_panel().to_numpy(np.float64)
    fin = np.isfinite(S).astype(np.int32)
    cs = np.cumsum(fin, axis=0)
    E = cs.copy()
    E[5:] = cs[5:] - cs[:-5]
    E = E > 0
    # z_PEAD(决策口径,可投资池内 ≥20 只才非零)用于事后主动暴露
    Ls = {}
    sty = list(rm.style_names)
    store = {}
    att_rows = []
    for nm in ("V1", "P0", "P1", "P2", "P1f", "P2f", "P0c"):
        v = V[nm]
        H = v["H"].reindex(columns=codes).fillna(0.0)
        idx = H.index[(H.index >= START) & (H.index <= END)]
        Hv = H.loc[idx].to_numpy(np.float64)
        pos = dates.get_indexer(idx)
        A = Hv - Wx[pos]
        Adf = pd.DataFrame(A, index=idx, columns=codes)
        att, fac = rm.attribute(Adf, R, by_factor=True)
        # 事件股特质 + 主动暴露(逐日)
        spE = pd.Series(np.nan, index=att.index)
        spN = pd.Series(np.nan, index=att.index)          # 审查后追加:非成分股(成交时点基准权重 = 0)上的特质
        awN = np.full(len(idx), np.nan)                   # 非成分股持仓权重 Σw(= 非成分股主动权重)
        awE = np.full(len(idx), np.nan); e20 = np.full(len(idx), np.nan); ezp = np.full(len(idx), np.nan)
        te_pred = np.full(len(idx), np.nan)
        Est = np.full((len(idx), rm.K), np.nan)
        apos = {d: j for j, d in enumerate(att.index)}
        for j, i in enumerate(pos):
            a = A[j]
            Ei = E[i]
            Nc = ~(Wx[i] > 0)                              # 非成分股(成交时点代理权重为 0)
            awE[j] = float(a[Ei].sum())
            awN[j] = float(a[Nc].sum())
            e20[j] = float(np.nansum(np.nan_to_num(z20[i]) * a))
            covok = np.isfinite(rm.cov[i, 0, 0])
            m = np.asarray(rm.mask[i]) if covok else np.isfinite(np.asarray(rm.X[i])[:, 0])
            Xall = np.nan_to_num(np.asarray(rm.X[i]).astype(np.float64))
            Est[j] = Xall[m].T @ a[m]
            # z_PEAD:可投资池(决策池 ∩ 覆盖)内非缺失 ≥ 20 只才非零(与优化器同一口径)
            inv = U[i] & m & np.isfinite(S[i])
            if inv.sum() >= PEAD_MIN:
                s_ = S[i, inv]; sd = s_.std()
                if sd > 0:
                    zp = np.clip((s_ - s_.mean()) / sd, -3, 3)
                    ezp[j] = float(zp @ a[inv])
            if covok:
                if i not in Ls:
                    Ls[i] = psd_sqrt(rm.cov[i])[1]
                Hm = rm._onehot(rm.ind[m]); am = a[m]
                ee = np.concatenate([[am.sum()], Xall[m].T @ am, Hm.T @ am])
                y = Ls[i].T @ ee
                d_ = np.nan_to_num(np.asarray(rm.spec[i])[m].astype(np.float64))
                te_pred[j] = np.sqrt(np.sum(y * y) + np.sum(d_ * am * am)) * np.sqrt(TD / 5)
            d = idx[j]
            if d in apos and i < len(Rv):
                r = np.nan_to_num(Rv[i])
                mE = m & Ei
                Xe = Xall[mE]; He = rm._onehot(rm.ind[mE]); aE = a[mE]
                ee = np.concatenate([[aE.sum()], Xe.T @ aE, He.T @ aE])
                f = rm.F.loc[d].to_numpy(np.float64)
                spE.loc[d] = float(a[Ei] @ r[Ei]) - float(ee @ f)
                mN = m & Nc                                 # 同一构造:主动收益 − 因子部分(覆盖池内)
                XN = Xall[mN]; HN = rm._onehot(rm.ind[mN]); aN = a[mN]
                eN = np.concatenate([[aN.sum()], XN.T @ aN, HN.T @ aN])
                spN.loc[d] = float(a[Nc] @ r[Nc]) - float(eN @ f)
        att["rev20"] = (pd.Series(e20, index=idx) * pd.Series(g20[pos], index=idx)).reindex(att.index)
        att["specific_ex_rev20"] = att["specific"] - att["rev20"].fillna(0.0)
        att["specific_event"] = spE
        att["specific_nonevent"] = att["specific"] - spE
        att["specific_noncons"] = spN
        att["specific_cons"] = att["specific"] - spN
        att["cost"] = v["daily"]["cost_total"].reindex(att.index)
        att["engine_excess"] = (v["rets"] - rb).reindex(att.index)
        att["recon_resid"] = att["total"] - att["cost"] - att["engine_excess"]
        store[nm] = dict(att=att, fac=fac)
        warm = dates[min(int(dates.searchsorted(START)) + 5, T - 1)]      # 建仓周剔除(相位 1-4 还是现金)
        for wn, a0, b0 in IE.windows() + IE.years()[:-1]:
            sel = (idx >= max(a0, warm)) & (idx <= b0)
            fns = (("事件窗股票主动权重 Σa", awE), ("PEAD z 主动暴露 Σ z·a(事件池≥20 的日子)", ezp),
                   ("Rev20 主动暴露", e20), ("事前TE(实际持仓,风险模型)", te_pred),
                   ("非成分股持仓权重 Σw(成交时点基准权重=0)", awN))
            if wn[0].isdigit():                            # 逐年只报非成分股权重(审查后追加),其余维持原窗口
                fns = fns[-1:]
            for fn, x in fns:
                xx = pd.Series(x, index=idx)[sel].dropna()
                exp_rows.append(dict(variant=nm, window=wn, source="实际持仓(逐日,相位合并)", factor=fn,
                                     mean=float(xx.mean()) if len(xx) else np.nan,
                                     mean_abs=float(xx.abs().mean()) if len(xx) else np.nan, n_days=len(xx)))
            if wn[0].isdigit():
                continue
            for k, s_ in enumerate(sty):
                xx = pd.Series(Est[:, k], index=idx)[sel].dropna()
                exp_rows.append(dict(variant=nm, window=wn, source="实际持仓(逐日,相位合并)", factor=s_,
                                     mean=float(xx.mean()), mean_abs=float(xx.abs().mean()), n_days=len(xx)))
            dd = (v["rets"] - rb).reindex(idx)[sel].dropna()
            exp_rows.append(dict(variant=nm, window=wn, source="实际持仓(逐日,相位合并)", factor="事后TE(日频×√242)",
                                 mean=float(dd.std() * np.sqrt(TD)), n_days=len(dd)))
        comps = ["total", "country", "style", "industry", "specific", "specific_event", "specific_nonevent", "rev20",
                 "specific_ex_rev20", "specific_cons", "specific_noncons", "cost", "engine_excess", "recon_resid"]
        for wn, a0, b0 in IE.windows():
            g = att[(att.index >= a0) & (att.index <= b0)]
            for c in comps:
                x = g[c].dropna()
                att_rows.append(dict(variant=nm, window=wn, component=c, ann_pp=float(x.mean() * TD * 100),
                                     t_nw5=nw_t(x.to_numpy(), 5), t_nw10=nw_t(x.to_numpy(), NW_MAIN), n_days=len(x)))
            gf = fac[(fac.index >= a0) & (fac.index <= b0)]
            for s_ in sty:
                att_rows.append(dict(variant=nm, window=wn, component=f"style:{s_}",
                                     ann_pp=float(gf[s_].mean() * TD * 100), t_nw5=nw_t(gf[s_].to_numpy(), 5),
                                     n_days=len(gf)))
    # 配对差(同一组日期)
    for a_, b_ in (("P1", "V1"), ("P2", "P0"), ("P1", "P0"), ("P1f", "V1"), ("P2f", "P0"), ("P0", "P0c"), ("P2", "P0c")):
        A1, A2 = store[a_]["att"], store[b_]["att"]
        common = A1.index.intersection(A2.index)
        for wn, a0, b0 in IE.windows():
            cm_ = common[(common >= a0) & (common <= b0)]
            for c in ["total", "style", "industry", "specific", "specific_event", "specific_nonevent", "rev20",
                      "specific_ex_rev20", "specific_cons", "specific_noncons", "cost", "engine_excess"]:
                x = (A1.loc[cm_, c] - A2.loc[cm_, c]).dropna()
                att_rows.append(dict(variant=f"{a_} − {b_}", window=wn, component=c, ann_pp=float(x.mean() * TD * 100),
                                     t_nw5=nw_t(x.to_numpy(), 5), t_nw10=nw_t(x.to_numpy(), NW_MAIN), n_days=len(x)))
            F1, F2 = store[a_]["fac"], store[b_]["fac"]
            for s_ in sty:
                x = (F1.loc[cm_, s_] - F2.loc[cm_, s_]).dropna()
                att_rows.append(dict(variant=f"{a_} − {b_}", window=wn, component=f"style:{s_}",
                                     ann_pp=float(x.mean() * TD * 100), t_nw5=nw_t(x.to_numpy(), 5),
                                     t_nw10=nw_t(x.to_numpy(), NW_MAIN), n_days=len(x)))
    # 与上游 ie_attribution.csv 的 V1 对账(本文用封存市值重算 rev20,其余应完全一致)
    try:
        ia = pd.read_csv(LEDGER / "ie_attribution.csv")
        ia = ia[ia.variant == "V1"].set_index(["window", "component"])["ann_pp"]
        mine = pd.DataFrame([r for r in att_rows if r["variant"] == "V1"]).set_index(["window", "component"])["ann_pp"]
        common = ia.index.intersection(mine.index)
        dd = (mine.reindex(common) - ia.reindex(common)).abs()
        for (wn, c), val in dd.items():
            if not c.startswith("style:"):
                exp_rows.append(dict(variant="V1 归因对账(本文 − ie_attribution.csv)", window=wn, source="对账",
                                     factor=c, mean=float(val)))
    except FileNotFoundError:
        pass
    # COMP 与 PEAD 的截面相关(加法组合的"预测误差不相关"假设的诊断)
    comp = IE.load_score("comp").to_numpy(np.float64)
    cors = []
    for i in range(first, T):
        m = np.asarray(rm.mask[i]) if np.isfinite(rm.cov[i, 0, 0]) else None
        if m is None:
            continue
        inv = U[i] & m & np.isfinite(S[i]) & np.isfinite(comp[i])
        if inv.sum() < PEAD_MIN:
            continue
        cors.append((dates[i], float(np.corrcoef(S[i, inv], comp[i, inv])[0, 1]),
                     float(pd.Series(S[i, inv]).rank().corr(pd.Series(comp[i, inv]).rank()))))
    cdf = pd.DataFrame(cors, columns=["date", "pearson", "spearman"]).set_index("date")
    for wn, a0, b0 in IE.windows():
        x = _win(cdf, a0, b0)
        exp_rows.append(dict(variant="PEAD vs COMP_v5", window=wn, source="信号相关", factor="事件股内截面相关(日均)",
                             mean=float(x["pearson"].mean()), spearman=float(x["spearman"].mean()), n_days=len(x)))
    return att_rows


# ================================================================ 零 α 机制探针(审查后追加)
def p0_probe() -> None:
    """P0 每个相位首次从现金建仓那一次求解的反事实(审查发现 1:P0 偏离基准的机制)。

    用与 run_job("P0") 完全相同的输入构造优化器,只截一小段日期让引擎走到首次调仓,截获 ActiveProblem,然后在
    同一个问题上比较:w*(P0 实际解)、wb(持有基准)、λ ∈ {10, 100, 1000} 重解、"候选集只含成分股"重解
    (= P0c 的首次建仓问题,与 P0c 求解日志交叉核对)。目标 = −成本 − λ·Var5(α ≡ 0)。
    写 research/ledger/iepead_p0_mechanism.csv。只读研究期数据,和作业同样做封存截断与断言。"""
    from dataclasses import replace
    from afactor import strategy as ST, index_proxy as IP, optimizer as OPT
    from afactor.barra import RiskModel
    from afactor.impact import CostModel
    p = P()
    seal_newdata()
    rm = RiskModel.load("open10")
    W = IP.proxy_weights(p, "000852")
    U = ST.universe(p, "V1000")
    comp = IE.load_score("comp")
    cfg = OPT.OptConfig()
    cm = CostModel(p, impact_y=1.0)
    assert_sealed()
    zero = pd.Series(0.0, index=p.dates)
    d = p.dates
    i0 = int(d.searchsorted(START))
    mask = np.asarray((d >= START) & (d <= d[i0 + 8]))
    rows = []
    orig = OPT.solve
    for ph in range(5):
        enh = OPT.IndexEnhancer(p, rm, W, comp, zero, U, cfg, tag="probe")
        cap: dict = {}

        def spy(prob, cfg_):
            r = orig(prob, cfg_)
            if "prob" not in cap:
                cap["prob"], cap["res"] = prob, r
            return r
        OPT.solve = spy
        try:
            ST.simulate(enh, p, "open10", 1e7, rebalance=5, date_mask=mask, phase=ph, cost_model=cm,
                        record=False, participation_on_nav=False)
        finally:
            OPT.solve = orig
        prob, res = cap["prob"], cap["res"]
        day = pd.Timestamp(enh.log[0]["date"])
        assert prob.h.sum() == 0, "首次调仓应从现金建仓"
        b = prob.bench

        def one(pb, w, case, c_=cfg, ref_bench=None):
            vf, vs = pb.risk_parts(w)
            c = pb.cost(w)
            bb = pb.bench if ref_bench is None else ref_bench
            eff = pb.lin_buy + pb.c15 * np.sqrt(np.clip(w, 0, None))   # 有效买入成本率(从现金:Δw = w)
            return dict(phase=ph, date=day, case=case, risk_aversion=c_.risk_aversion,
                        bench_w_in=float(w[bb].sum()), n_hold=int((w > 1e-6).sum()), n_cand=int(pb.n),
                        n_cand_bench=int(bb.sum()), te_ann=float(np.sqrt(max(vf + vs, 0)) * np.sqrt(TD / 5)),
                        cost_5d=c, risk_term_5d=float(c_.risk_aversion * (vf + vs)),
                        obj_at_lambda1=float(-c - cfg.risk_aversion * (vf + vs)),
                        feasible=bool(OPT.violations_ok(OPT.check_solution(pb, w, c_, True))),
                        violated=",".join(k for k, v_ in OPT.check_solution(pb, w, c_, True).items()
                                          if v_ > OPT._TOL.get(k, 0.0)),
                        eff_buy_bp_bench=float((eff * w)[bb].sum() / max(w[bb].sum(), 1e-12) * 1e4),
                        eff_buy_bp_nonbench=float((eff * w)[~bb].sum() / w[~bb].sum() * 1e4) if w[~bb].sum() > 0
                        else np.nan)
        rows.append(one(prob, res["w"], "P0 实际首解 w*(λ=1)"))
        rows.append(one(prob, prob.wb.copy(), "反事实:持有基准 wb"))
        for lam in (10.0, 100.0, 1000.0):
            c2 = replace(cfg, risk_aversion=lam)
            r2 = OPT.solve(prob, c2)
            rows.append(one(prob, r2["w"], f"反事实:λ={lam:g} 重解", c2))
        bi = np.flatnonzero(b)
        sub = OPT.ActiveProblem(alpha=prob.alpha[bi], wb=prob.wb[bi], h=prob.h[bi], X=prob.X[bi], H=prob.H[bi],
                                L=prob.L, d=prob.d[bi], lin_buy=prob.lin_buy[bi], lin_sell=prob.lin_sell[bi],
                                c15=prob.c15[bi], fixed=prob.fixed[bi],
                                cap=None if prob.cap is None else prob.cap[bi], bench=prob.bench[bi],
                                style_names=prob.style_names, total=prob.total)
        r3 = OPT.solve(sub, cfg)
        rows.append(one(sub, r3["w"], "反事实:候选集只含成分股 重解(= P0c 首次建仓)"))
        # 成分 vs 非成分候选的线性成本系数(佣金 + 半价差)
        rows.append(dict(phase=ph, date=day, case="成本系数:lin_buy 中位数(bp)",
                         eff_buy_bp_bench=float(np.median(prob.lin_buy[b]) * 1e4),
                         eff_buy_bp_nonbench=float(np.median(prob.lin_buy[~b]) * 1e4),
                         n_cand=int(prob.n), n_cand_bench=int(b.sum())))
        print(f"[probe ph{ph}] {day:%Y-%m-%d} w* bench_w_in {rows[-7]['bench_w_in']:.3f} TE {rows[-7]['te_ann']:.4f} "
              f"cost {rows[-7]['cost_5d']*1e4:.2f}bp vs 持有基准 {rows[-6]['cost_5d']*1e4:.2f}bp", flush=True)
    df = pd.DataFrame(rows)
    # 与 P0 / P0c 求解日志交叉核对(首次建仓日的 bench_w_in、te_ex_ante)
    chk = []
    for nm, case in (("P0", "P0 实际首解 w*(λ=1)"), ("P0c", "反事实:候选集只含成分股 重解(= P0c 首次建仓)")):
        for ph in range(5):
            f = TMP / f"{PFX}_{nm}_ph{ph}_solver.parquet"
            if not f.exists():
                continue
            lg = pd.read_parquet(f)
            lg = lg[lg.status.isin(["optimal", "optimal_inaccurate"])].sort_values("date").iloc[0]
            me = df[(df.phase == ph) & (df.case == case)].iloc[0]
            chk.append(dict(phase=ph, date=pd.Timestamp(lg["date"]), case=f"核对:{nm} 求解日志首解 − 探针",
                            bench_w_in=float(lg["bench_w_in"] - me["bench_w_in"]),
                            te_ann=float(lg["te_ex_ante"] - me["te_ann"]),
                            cost_5d=float(lg["cost_ex_ante"] - me["cost_5d"])))
    df = pd.concat([df, pd.DataFrame(chk)], ignore_index=True)
    df.to_csv(LEDGER / f"{PFX}_p0_mechanism.csv", index=False)
    print(df.to_string(), flush=True)


# ================================================================ 图
def plot(V, rb, icd, icd_c) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    IE._font()
    C_SURF, C_TXT2 = IE.C_SURF, IE.C_TXT2
    SER = IE.SERIES
    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), facecolor=C_SURF,
                             gridspec_kw=dict(height_ratios=[3, 2, 1.6]), sharex=True)
    ax = axes[0]
    for k, (nm, lab) in enumerate([("V1", "V1:COMP α"), ("P1", "P1:COMP + PEAD α"), ("P2", "P2:只用 PEAD α"),
                                   ("P0", "P0:零 α(对照)"), ("P1f", "P1f:COMP + PEAD α(IC 冻结,敏感性)")]):
        s = V[nm]["rets"]
        s = s[s.index >= START]
        rel = (1 + s).cumprod() / (1 + rb.reindex(s.index)).cumprod()
        ax.plot(rel.index, rel.values, color=SER[k], lw=2 if nm in ("V1", "P1") else 1.4, label=lab)
        ax.annotate(f"{rel.iloc[-1]:.2f}", (rel.index[-1], rel.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=C_TXT2, fontsize=9, va="center")
    ax.axhline(1.0, color=C_TXT2, lw=0.8)
    ax.axvline(TEST[0], color=C_TXT2, lw=0.8, ls="--")
    ax.text(TEST[0], ax.get_ylim()[1], " 检验窗 →", color=C_TXT2, fontsize=9, va="top")
    ax.set_ylabel("相对净值(策略 / 中证1000同口径代理)")
    ax.set_title("相对中证1000代理的累计超额(open10,1000万,含冲击,5相位平均;信号样本内)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    IE._style(ax)
    ax = axes[1]
    for a_, b_, lab, c in [("P1", "V1", "P1 − V1(PEAD 边际贡献)", SER[1]), ("P2", "P0", "P2 − P0(只用 PEAD vs 零 α)", SER[2]),
                           ("P1f", "V1", "P1f − V1(IC 冻结,敏感性)", SER[4])]:
        d = (V[a_]["rets"] - V[b_]["rets"]).dropna()
        d = d[d.index >= START]
        ax.plot(d.index, d.cumsum().values * 100, color=c, lw=1.6, label=lab)
    ax.axhline(0, color=C_TXT2, lw=0.8)
    ax.axvline(TEST[0], color=C_TXT2, lw=0.8, ls="--")
    ax.set_ylabel("累计日差(pp,算术)")
    ax.set_title("配对差的累计值(同相位配对,5 相位平均)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    IE._style(ax)
    ax = axes[2]
    x = icd[icd.index >= START]
    y = icd_c[icd_c.index >= START]
    ax.plot(x.index, x.values, color=SER[1], lw=1.4, label="PEAD 决策日 IC 预测(下限 0)")
    ax.plot(y.index, y.values, color=SER[0], lw=1.4, label="COMP_v5 决策日 IC 预测")
    ax.set_title("Grinold α 用的 IC(250 日滚动、只用 t−6 已实现)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    IE._style(ax)
    fig.tight_layout()
    fig.savefig(REPORTS / f"{PFX}_excess_nav.png", dpi=150, facecolor=C_SURF)
    plt.close(fig)


# ================================================================ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prep", "run", "all", "report", "probe"])
    ap.add_argument("--job")
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.stage == "prep":
        ic, icd = pead_ic()
        print("PEAD IC 有效日", int(ic.notna().sum()), "均值", float(ic.mean()), "决策日 IC 预测均值", float(icd.mean()))
    elif a.stage == "run":
        run_job(a.job, a.phase)
    elif a.stage == "probe":
        p0_probe()
    elif a.stage == "all":
        drive(a.procs, a.only)
        report()
    else:
        report()
