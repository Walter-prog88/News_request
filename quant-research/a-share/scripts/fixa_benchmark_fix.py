"""第二轮修订(前缀 fixa,2026-10-10):修复审计的两个"高"级残留问题,都出在**基准构造**上。

问题 1  k=1 池等权基准(ST.benchmark)每天重置等权,5 日调仓的组合其间让权重漂移。个股相对池均值的收益有正的
        截面序列相关时(vwap1 是全日平均价口径,滞后 1 期截面相关 0.175;open10 0.037、tail1450 0.036,
        research/ledger/fixa_r2_mech.csv;复核前这里引用的是池等权序列自相关 0.308 / 0.068,统计量不对,只作旁证),
        "每天卖相对涨的、买相对跌的回到等权"系统性吃亏,两者之差是与选股无关的"再平衡频率差",被记成策略超额。
        修复 = ST.benchmark(..., rebalance=k):同频再平衡的池等权基准(k 个相位平均,不计成本、不加可成交约束)。
问题 2  中证1000 代理旧基准 IP.engine_consistent_bench_returns 用信号日收盘权重直接乘成交价收益,
        每天把权重重置回过时一个隔夜的收盘权重,系统性偏低。
        修复 = IP.exec_time_bench_returns(成交时点权重,由 optimizer 搬入 index_proxy);旧函数加 DeprecationWarning。

本脚本只做"重算与对照"(各报告的正文改动见各自的"第二轮修订(fixa)"小节):
  capdiff     容量报告修订前后的关键数字(修订前账本快照 vs 修订后账本)-> research/ledger/fixa_capacity_before_after.csv
  bench       基准对照(两个池等权 × 两个代理 × 真实指数;分窗口 / 逐年 / 相位)
              -> research/ledger/fixa_bench_compare.csv
              全池等权引擎判据(常数打分、旧成本、k ∈ {1,5}、两口径;2020H2 起的连续路径切片,
              另复现审计探针的 2021-01-04 冷启动口径)-> research/ledger/fixa_fullpool_checks.csv
  downstream  下游报告里"与上游 0.10~0.91pp 同量级"之类跨口径引用的同频重算:
                IE 判据(读 cache/_tmp/ie_CHK_pool_ew*_daily.parquet,不重跑优化器)
                PEAD 引擎门槛(重跑 pead_engine.run_account 的全池等权,约 1 分钟)
                STRESS 全池等权(重跑 ST.run_strategy,复刻版已证明与它逐位相同)
              -> research/ledger/fixa_downstream_checks.csv
              ENGINE §7 / §3.1:eng_cost_demo 的策略年化不变,只换基准;排雷超额 IR 重算
              -> research/ledger/fixa_eng_cost_demo.csv、fixa_eng_excess_ir.csv
              CSI1000_PROXY §7:推荐基准对真实指数的跟踪统计、池等权 − 代理的量级
              -> research/ledger/fixa_proxy_checks.csv

封存:只读研究期面板 Panel()(<= 2025-12-31);真实指数经 IP.index_close 先截断;不读 2026 数据、不碰 cache/sealed。
COMP_v5 为信号样本内(2020-2025 选出),本脚本里用到它的数字只说明基准口径的影响,不是信号能力。
数据不含已退市股(约 209 只),所有收益数字有幸存者偏差(方向为高估),基准与策略两侧都有。

    python3 scripts/fixa_benchmark_fix.py bench
    python3 scripts/fixa_benchmark_fix.py downstream
    python3 scripts/fixa_benchmark_fix.py capdiff      # 先跑 cap_capacity_curve.py bench / checks / summarize
    python3 scripts/fixa_benchmark_fix.py review2 [mech|barra|warm]

review2(第二轮修订的复核审查,1 条 P1 + 4 条 P2,同日):
  warm   P1:ENGINE §3.1 / §7 的同频超额改用 2020H2 预热的连续路径切片,并给冷启动对照 -> fixa_r2_eng_warm.csv
  mech   P2:再平衡频率差的机制(截面相对收益的序列相关与一阶展开预测)+ tail1450 等口径的同频 − 每日重置量级
         (研究期 Panel 与 TailPanel)-> fixa_r2_mech.csv
  barra  P2:BARRA_ATTRIBUTION 换同频基准后的总超额 / 流动性 / alpha 变化(OLS 对 y 线性,Δ 精确)-> fixa_r2_barra_delta.csv
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from afactor.config import LEDGER, TMP
from afactor.panel import Panel
from afactor import strategy as ST, index_proxy as IP

TD = ST.TRADING_DAYS
V = "V1000"
MODES = ("open10", "vwap1")
K = 5
WINDOWS = {"2021-2025": (2021, 2025), "2021-2023": (2021, 2023), "2024-2025": (2024, 2025),
           **{str(y): (y, y) for y in range(2020, 2026)}}


def geo(r: pd.Series) -> float:
    r = r.dropna()
    return float((1.0 + r).prod() ** (TD / len(r)) - 1.0) if len(r) else np.nan


def nw_t(x, bw: int = 5) -> float:
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan
    u = x - x.mean()
    lrv = float((u * u).sum()) / n                     # 不用 @(numpy 2.0 + Accelerate 的 matmul 报假的浮点警告)
    for L in range(1, bw + 1):
        lrv += 2.0 * (1.0 - L / (bw + 1)) * float((u[L:] * u[:-L]).sum()) / n
    return float(x.mean() / np.sqrt(lrv / n)) if lrv > 0 else np.nan


def ysel(idx: pd.DatetimeIndex, y0: int, y1: int) -> np.ndarray:
    return np.asarray((idx.year >= y0) & (idx.year <= y1))


def real_index(p: Panel, mode: str) -> pd.Series:
    """真实 000852(价格指数)对齐到信号日:open10 同 IP.index_returns_on_signal_days;vwap1 用 close(t+2)/close(t+1)
    (与 scripts/cap_capacity_curve.real_index_on_signal_days 同法)。"""
    if mode != "vwap1":
        return IP.index_returns_on_signal_days(p, mode, "000852")
    lag = ST.EXEC_MODES[mode]["lag"]
    px = IP.index_close("000852").reindex(p.dates).ffill()
    r = px.shift(-2) / px.shift(-1) - 1.0
    return r.reindex(p.dates[:len(p.dates) - 1 - lag]).fillna(0.0)


def legacy_proxy(p: Panel, mode: str) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return IP.engine_consistent_bench_returns(p, mode, "000852")


# ====================================================================== bench
def cmd_bench():
    t0 = time.time()
    p = Panel()
    rows = []
    for mode in MODES:
        b1 = ST.benchmark(p, V, mode)["rets"]
        b5 = ST.benchmark(p, V, mode, rebalance=K)["rets"]
        R5 = ST._pool_ew_drift_returns(p, V, mode, K, 0)
        R1 = ST._pool_ew_drift_returns(p, V, mode, 1, 0)
        k1_gap = float(np.max(np.abs(R1[0] - b1.to_numpy())))
        pl, pe, ri = legacy_proxy(p, mode), IP.exec_time_bench_returns(p, mode), real_index(p, mode)
        S = {"pool_d1": b1, "pool_r5": b5, "proxy_legacy": pl, "proxy_exec": pe, "real_pr": ri}
        for nm, s in S.items():
            assert s.index.equals(b1.index), nm
        for w, (y0, y1) in WINDOWS.items():
            m = ysel(b1.index, y0, y1)
            ann = {nm: geo(s[m]) for nm, s in S.items()}
            row = dict(mode=mode, window=w, n_days=int(m.sum()), **{f"ann_{k}": v for k, v in ann.items()})
            row.update(
                d_pool_r5_minus_d1=ann["pool_r5"] - ann["pool_d1"],
                d_proxy_exec_minus_legacy=ann["proxy_exec"] - ann["proxy_legacy"],
                d_pool_d1_minus_proxy_legacy=ann["pool_d1"] - ann["proxy_legacy"],      # 旧口径的"池等权比代理高"
                d_pool_d1_minus_proxy_exec=ann["pool_d1"] - ann["proxy_exec"],
                d_pool_r5_minus_proxy_exec=ann["pool_r5"] - ann["proxy_exec"],          # 两个修订后基准之差
                arith_pool_r5_minus_d1=float((b5[m] - b1[m]).mean() * TD),
                t_nw_pool_r5_minus_d1=nw_t((b5[m] - b1[m]).to_numpy(), 10),
                arith_proxy_exec_minus_legacy=float((pe[m] - pl[m]).mean() * TD),
                t_nw_proxy_exec_minus_legacy=nw_t((pe[m] - pl[m]).dropna().to_numpy(), 10),
                corr_pool_d1_proxy_legacy=float(b1[m].corr(pl[m])), corr_pool_r5_proxy_exec=float(b5[m].corr(pe[m])),
                ac1_pool_d1=float(b1[m].autocorr(1)), ac1_pool_r5=float(b5[m].autocorr(1)),
                ac1_proxy_exec=float(pe[m].autocorr(1)),
                vr20_pool_d1=ST.lr_variance_ratio(b1[m].to_numpy()), vr20_pool_r5=ST.lr_variance_ratio(b5[m].to_numpy()),
                k1_general_vs_legacy_maxabs=k1_gap)
            idx = b1.index
            for ph in range(K):
                row[f"ann_pool_r5_phase{ph}"] = geo(pd.Series(R5[ph], index=idx)[m])
            rows.append(row)
        print(mode, "k=1 通用路径 vs 旧路径 max|diff|", k1_gap, f"[{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "fixa_bench_compare.csv", index=False)
    show = df[df.window.isin(["2021-2025", "2021-2023", "2024-2025"])]
    print((show.set_index(["mode", "window"])[[c for c in df.columns if c.startswith(("ann_pool", "ann_proxy", "d_"))
                                                and "phase" not in c]] * 100).round(3).to_string())
    fullpool_checks(p)
    print(f"done bench {time.time()-t0:.0f}s")


def _resid_rows(tag: str, mode: str, k: int, res: dict, benches: dict, windows: dict, note: str) -> list:
    rs = res["rets"]
    cost = res["record"]["daily"]["cost_total"].reindex(rs.index)
    out = []
    for bn, b in benches.items():
        prof = {kk: float(rs.corr(b.reindex(rs.index).shift(kk))) for kk in (-2, -1, 0, 1, 2)}
        pk = max(prof, key=prof.get)
        for w, (y0, y1) in windows.items():
            m = ysel(rs.index, y0, y1)
            if m.sum() < 30:
                continue
            s_, b_, c_ = rs[m], b.reindex(rs.index)[m], cost[m]
            ex_g, ex_a, c_a = geo(s_) - geo(b_), float((s_ - b_).mean() * TD), float(c_.mean() * TD)
            out.append(dict(path=tag, mode=mode, rebalance=k, bench=bn, window=w, n_days=int(m.sum()),
                            excess_geo_pp=ex_g * 100, excess_arith_pp=ex_a * 100, cost_pp=c_a * 100,
                            resid_geo_pp=(ex_g + c_a) * 100, resid_arith_pp=(ex_a + c_a) * 100,
                            t_nw_resid=nw_t((s_ - b_ + c_).to_numpy(), 5),
                            exposure_max=float(res["record"]["daily_by_phase"]["exposure"].max()),
                            align_peak_k=pk, note=note))
    return out


def fullpool_checks(p: Panel):
    """纪律 4:全池等权(常数打分、旧成本、1000 万、V1000)vs 池等权基准,超额 + 成本 ≈ 0。

    path=continuous:2020-01-02 起一条连续路径(无 mask),报告逐年 / 窗口切片 —— 不在窗口起点冷启动;
    path=audit_mask2021:与审计探针相同,2021-2025 掩码(2021-01-04 冷启动,相位 k 前 k 天空仓)。
    bench=d1 是修订前的 k=1 基准;bench=same 是同频基准 benchmark(rebalance=k)(k=1 时两者相同)。"""
    ones = pd.DataFrame(1.0, index=p.dates, columns=p.codes, dtype=np.float32)
    rows = []
    yrs = p.dates.year
    for mode in MODES:
        b1 = ST.benchmark(p, V, mode)["rets"]
        for k in (1, K):
            bk = ST.benchmark(p, V, mode, rebalance=k)["rets"]
            res = ST.run_strategy(ones, p, V, mode, scheme="exclude", exclude_frac=0.0, rebalance=k, record=True)
            rows += _resid_rows("continuous", mode, k, res, {"d1": b1, "same": bk}, WINDOWS,
                                "常数打分全池等权,旧成本,连续路径切片")
            m = np.asarray((yrs >= 2021) & (yrs <= 2025))
            resm = ST.run_strategy(ones, p, V, mode, scheme="exclude", exclude_frac=0.0, rebalance=k,
                                   date_mask=m, record=True)
            b1m = ST.benchmark(p, V, mode, date_mask=m)["rets"]
            bkm = ST.benchmark(p, V, mode, date_mask=m, rebalance=k)["rets"]
            rows += _resid_rows("audit_mask2021", mode, k, resm, {"d1": b1m, "same": bkm},
                                {"2021-2025": (2021, 2025), "2021": (2021, 2021)},
                                "审计探针口径:2021-2025 掩码,2021-01-04 冷启动")
            print(mode, k, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "fixa_fullpool_checks.csv", index=False)
    print(df[df.window.isin(["2021-2025"])].round(3).to_string())


# ====================================================================== downstream
SPECIAL = (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-07"))     # 与 scripts/run_index_enhance.py 相同
IE_START = pd.Timestamp("2021-04-20")


def ie_rows(p: Panel) -> list:
    """IE 的两条池等权判据(open10 / vwap1):5 日调仓、常数打分、Y=1 成本、1000 万、2021-04-20 起(冷启动)。
    读缓存的日收益,不重跑;逐年 (超额 + 成本) 的算术口径与 run_index_enhance.report 完全同式。"""
    rows = []
    for nm, mode in (("CHK_pool_ew", "open10"), ("CHK_pool_ew_vwap1", "vwap1")):
        d = pd.read_parquet(TMP / f"ie_{nm}_daily.parquet")
        s, c = d["ret"], d["cost_total"]
        mask = np.asarray((p.dates >= IE_START) & (p.dates <= pd.Timestamp("2025-12-31")))
        benches = {"d1": ST.benchmark(p, V, mode)["rets"],
                   "same": ST.benchmark(p, V, mode, date_mask=mask, rebalance=K)["rets"]}
        for bn, b in benches.items():
            pb = b.reindex(s.index)
            prof = {kk: float(s.corr(b.shift(kk).reindex(s.index))) for kk in (-2, -1, 0, 1, 2)}
            sels = [(str(y), s.index.year == y) for y in range(2021, 2026)]
            sels += [("2021-04-20~2025", np.ones(len(s), bool)),
                     ("2021(剔除前5个信号日冷启动)", (s.index.year == 2021) & (np.arange(len(s)) >= K)),
                     ("2024(剔除01-02~02-07)", (s.index.year == 2024) & ~((s.index >= SPECIAL[0]) & (s.index <= SPECIAL[1])))]
            for w, sel in sels:
                ex = float((s[sel] - pb[sel]).mean() * TD); cc = float(c[sel].mean() * TD)
                rows.append(dict(report="INDEX_ENHANCEMENT_CSI1000 / INDEX_ENHANCEMENT_PEAD", object=nm, mode=mode,
                                 bench=bn, window=w, excess_arith_pp=ex * 100, cost_pp=cc * 100,
                                 resid_pp=(ex + cc) * 100, passed_1p2pp=bool(abs(ex + cc) < 0.012),
                                 align_peak_k=max(prof, key=prof.get),
                                 note="5 日调仓、Y=1、1000 万、2021-04-20 冷启动;判据 |超额+成本| < 1.2pp(IE 原判据)"))
    return rows


def pead_rows(p: Panel) -> list:
    """PEAD 引擎门槛的全池等权(合格池、5 日调仓、1000 万、无整手无最低佣金)重跑,对同频合格池等权基准。
    与 scripts/pead_engine_gate.py 逐行同法(批次、相位、run_account 参数);基准换成
    ST.benchmark(..., rebalance=5, universe_override=合格池)。先核对旧基准 pool_benchmark 与
    ST.benchmark(universe_override=合格池) 逐位相同。"""
    from afactor import earnings_events as EE, pead_engine as PE
    dates, codes = p.dates, p.codes
    Q = EE.qualified_pool(p)
    Qn = Q.to_numpy(bool)
    px, _ = ST.exec_prices(p, "open10")
    g = px.shift(-1) / px - 1.0
    old = g.where(Q.shift(1).fillna(False).astype(bool)).mean(axis=1).shift(-1).iloc[:len(dates) - 2].fillna(0.0)
    b1 = ST.benchmark(p, V, "open10", universe_override=Q)["rets"]
    gold = float(np.max(np.abs(old.to_numpy() - b1.to_numpy())))
    b5 = ST.benchmark(p, V, "open10", rebalance=K, universe_override=Q)["rets"]
    star = PE.star_mask(codes)
    dummy = np.ones((len(dates), len(codes)))
    rets, costs = [], []
    for ph in range(K):
        batches = {}
        for te in range(1, len(dates)):
            if te % 5 != ph:
                continue
            idx = np.flatnonzero(Qn[te - 1])
            if len(idx):
                batches[te] = (idx, np.full(len(idx), 1.0 / len(idx)))
        out = PE.run_account(p, batches, ph, 1e7, dummy, dummy, star, lots=False, min_commission=0.0)
        rets.append(out["daily"]["ret"].rename(ph)); costs.append(out["daily"]["cost_total"].rename(ph))
        print("pead phase", ph, flush=True)
    R = pd.concat(rets, axis=1).mean(axis=1)
    C = pd.concat(costs, axis=1).mean(axis=1)
    rows = []
    for bn, b in (("d1", b1), ("same", b5)):
        bb = b.reindex(R.index)
        prof = {kk: float(R.corr(b.shift(kk).reindex(R.index))) for kk in (-2, -1, 0, 1, 2)}
        for y in range(2020, 2026):
            sel = R.index.year == y
            ex = float((R[sel] - bb[sel]).mean() * TD); cc = float(C[sel].mean() * TD)
            rows.append(dict(report="PEAD_REPORT §2.2", object="pead_gate_fullpool", mode="open10", bench=bn,
                             window=str(y), excess_arith_pp=ex * 100, cost_pp=cc * 100, resid_pp=(ex + cc) * 100,
                             passed_1p2pp=bool(abs(ex + cc) < 0.012), align_peak_k=max(prof, key=prof.get),
                             note=f"合格池全体等权、5 日调仓、pead_engine;旧基准复现 max|diff|={gold:.1e}"))
    return rows


def stress_rows(p: Panel) -> list:
    """STRESS 的全池等权检验:打分 = 决策池指示(vw 方案 = 池内等权)、5 日调仓、open10、旧成本、全样本。
    run_stress_test.run_rec 已证明与 ST.run_strategy 逐位相同,这里直接用后者。"""
    u = ST.universe(p, V)
    ones = u.astype(np.float32).where(u)
    res = ST.run_strategy(ones, p, V, "open10", scheme="vw", rebalance=K, record=True)
    r = res["rets"]; cost = res["record"]["daily"]["cost_total"].reindex(r.index)
    rows = []
    for bn, b in (("d1", ST.benchmark(p, V, "open10")["rets"]),
                  ("same", ST.benchmark(p, V, "open10", rebalance=K)["rets"])):
        bb = b.reindex(r.index)
        for w, sel in (("2020-2025(全样本,原检验口径)", np.ones(len(r), bool)),
                       ("2021-2025", r.index.year >= 2021)):
            ex = geo(r[sel]) - geo(bb[sel]); cc = float(cost[sel].mean() * TD)
            rows.append(dict(report="STRESS_TEST §10", object="FULLPOOL_EW", mode="open10", bench=bn, window=w,
                             excess_geo_pp=ex * 100, excess_arith_pp=float((r[sel] - bb[sel]).mean() * TD * 100),
                             cost_pp=cc * 100, resid_pp=(ex + cc) * 100, passed_1p5pp=bool(-0.005 <= ex + cc <= 0.015),
                             note="vw(池内等权)、5 日调仓、旧成本、1000 万;原检验 = 几何超额 + 算术成本"))
    return rows


STRESS_WINDOWS = {"W1": ("2024-01-02", "2024-02-07"), "W1b": ("2024-02-08", "2024-02-29"),
                  "W2": ("2024-04-12", "2024-04-30"), "W3": ("2024-09-24", "2024-10-08"),
                  "W4": ("2025-04-03", "2025-04-10"), "W5": ("2022-03-01", "2022-04-26")}   # STRESS_TEST.md §0.1,按成交日 te


def stress_window_rows(p: Panel) -> list:
    """STRESS 的窗口超额都对每日重置的池基准(POOL_EW);这里只量化换成同频基准时池基准本身在各窗口的变化
    (窗口按成交日 te 取信号日 t = te − 1,与 run_stress_test 同一口径;复利累计)。策略侧不重算。"""
    b1 = ST.benchmark(p, V, "open10")["rets"]
    b5 = ST.benchmark(p, V, "open10", rebalance=K)["rets"]
    te = pd.Series(p.dates[1:len(b1) + 1], index=b1.index)          # 信号日 t -> 成交日 te = t+1
    rows = []
    for w, (a, b) in STRESS_WINDOWS.items():
        m = ((te >= pd.Timestamp(a)) & (te <= pd.Timestamp(b))).to_numpy()
        c1, c5 = float((1 + b1[m]).prod() - 1), float((1 + b5[m]).prod() - 1)
        rows.append(dict(report="STRESS_TEST 窗口", object="POOL_EW 每日重置 vs 同频", mode="open10", bench="same-d1",
                         window=f"{w} {a}~{b}", excess_geo_pp=(c5 - c1) * 100, n_days=int(m.sum()),
                         note=f"窗口复利:每日重置 {c1*100:+.2f}%,同频 {c5*100:+.2f}%;策略不变,所以各组合对池等权的窗口超额变化 = −(同频 − 每日重置)"))
    return rows


def eng_demo_rows(p: Panel) -> pd.DataFrame:
    """ENGINE §7:eng_cost_demo.csv 的策略年化不变(各窗口掩码、冷启动),基准换成同频(掩码下 = 连续路径切片)。
    先核对旧 bench_ann 与 benchmark(mask) 一致。"""
    demo = pd.read_csv(LEDGER / "eng_cost_demo.csv")
    yrs = p.dates.year
    win = {"2021-2025": (2021, 2025), "2021-2023": (2021, 2023), "2024-2025": (2024, 2025)}
    bann = {}
    for mode in MODES:
        for w, (y0, y1) in win.items():
            m = np.asarray((yrs >= y0) & (yrs <= y1))
            bann[(mode, w)] = (ST.benchmark(p, V, mode, date_mask=m)["ann_return"],
                               ST.benchmark(p, V, mode, date_mask=m, rebalance=K)["ann_return"])
    out = demo[["mode", "window", "plan", "cost_model", "ann_return", "bench_ann", "excess_ann"]].copy()
    out["bench_ann_d1_recomputed"] = [bann[(a, b)][0] for a, b in zip(out["mode"], out["window"])]
    out["bench_ann_same"] = [bann[(a, b)][1] for a, b in zip(out["mode"], out["window"])]
    out["excess_ann_same"] = out["ann_return"] - out["bench_ann_same"]
    out["d_excess_pp"] = (out["excess_ann_same"] - out["excess_ann"]) * 100
    out["bench_check_maxabs"] = (out["bench_ann_d1_recomputed"] - out["bench_ann"]).abs()
    out["note"] = "策略年化取 eng_cost_demo.csv 原值(窗口掩码、冷启动);bench_ann_same = benchmark(mask, rebalance=5)"
    return out


def eng_ir_rows(p: Panel) -> pd.DataFrame:
    """ENGINE §3.1 表的"超额 TE / IR"行(COMP_v5 标准排雷、旧成本、2021-2025 掩码)对同频基准重算。"""
    score = pd.read_parquet(TMP / "comp_v5_score_open10.parquet")
    yrs = p.dates.year
    m = np.asarray((yrs >= 2021) & (yrs <= 2025))
    rows = []
    for mode in ("vwap1", "open10"):      # tail1450 列要 TailPanel(原表由 eng_review_fixes 另算),本轮只重算两个主口径
        r = ST.run_strategy(score, p, V, mode, scheme="exclude", buffer=(0.20, 0.35), rebalance=K, date_mask=m)
        for bn, b in (("d1", ST.benchmark(p, V, mode, date_mask=m)["rets"]),
                      ("same", ST.benchmark(p, V, mode, date_mask=m, rebalance=K)["rets"])):
            es = ST.excess_stats(r["rets"], b)
            rows.append(dict(mode=mode, bench=bn, window="2021-2025", ann_return=r["ann_return"],
                             bench_ann=geo(b), excess_geo=r["ann_return"] - geo(b), **es))
        print("eng ir", mode, flush=True)
    return pd.DataFrame(rows)


def proxy_rows(p: Panel) -> pd.DataFrame:
    """CSI1000_PROXY §7:代理基准对真实指数(收盘到收盘,信号日对齐,价格指数)的跟踪统计,
    与 scripts/run_index_proxy.pair_stats 同式;以及池等权 − 代理的年化差(open10)。"""
    out = []
    ri = IP.index_returns_on_signal_days(p, "open10", "000852")
    b1 = ST.benchmark(p, V, "open10")["rets"]
    b5 = ST.benchmark(p, V, "open10", rebalance=K)["rets"]
    pl, pe = legacy_proxy(p, "open10"), IP.exec_time_bench_returns(p, "open10")
    win = {"2021-2025": ("2021-01-01", "2025-12-31"), "dev_2021-2023": ("2021-01-01", "2023-12-31"),
           "test_2024-2025": ("2024-01-01", "2025-12-31")}

    def pair(a: pd.Series, b: pd.Series) -> dict:
        idx = a.dropna().index.intersection(b.dropna().index)
        a, b = a.reindex(idx), b.reindex(idx)
        d = a - b
        return dict(n_days=len(idx), ann_a=geo(a), ann_b=geo(b), gap_ann_geom=geo(a) - geo(b),
                    gap_ann_arith=float(d.mean() * TD), te_ann=float(d.std() * np.sqrt(TD)),
                    corr_daily=float(a.corr(b)), t_nw_gap=nw_t(d.to_numpy(), 5))

    for w, (s, e) in win.items():
        for nm, a, b in (("proxy_legacy_vs_index", pl, ri), ("proxy_exec_vs_index", pe, ri),
                         ("pool_d1_vs_proxy_legacy", b1, pl), ("pool_d1_vs_proxy_exec", b1, pe),
                         ("pool_r5_vs_proxy_exec", b5, pe)):
            m = (a.index >= s) & (a.index <= e)
            out.append(dict(pair=nm, period=w, **pair(a[m], b[m])))
    # 索引与对齐(引擎断言):推荐基准与 benchmark() 同索引、对真实指数与池基准的相关峰值
    for nm, ref in (("index", ri), ("pool_d1", b1), ("pool_r5", b5)):
        cs = {k: float(pe.corr(ref.shift(k))) for k in (-2, -1, 0, 1, 2)}
        out.append(dict(pair=f"align_proxy_exec_vs_{nm}", period="全样本", corr_daily=cs[0],
                        note=" ".join(f"k{k:+d}:{v:.4f}" for k, v in cs.items()) + f";峰值 k={max(cs, key=cs.get)}",
                        index_equal=bool(pe.index.equals(b1.index))))
    return pd.DataFrame(out)


def cmd_downstream():
    t0 = time.time()
    p = Panel()
    rows = ie_rows(p)
    print(f"ie done [{time.time()-t0:.0f}s]", flush=True)
    rows += stress_rows(p)
    rows += stress_window_rows(p)
    print(f"stress done [{time.time()-t0:.0f}s]", flush=True)
    rows += pead_rows(p)
    print(f"pead done [{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "fixa_downstream_checks.csv", index=False)
    print(df.round(3).to_string())
    ed = eng_demo_rows(p)
    ed.to_csv(LEDGER / "fixa_eng_cost_demo.csv", index=False)
    print(ed.round(4).to_string())
    ei = eng_ir_rows(p)
    ei.to_csv(LEDGER / "fixa_eng_excess_ir.csv", index=False)
    print(ei.round(4).to_string())
    pr = proxy_rows(p)
    pr.to_csv(LEDGER / "fixa_proxy_checks.csv", index=False)
    print(pr.round(4).to_string())
    print(f"done downstream {time.time()-t0:.0f}s")


# ====================================================================== capdiff:容量报告修订前后对照
CAP_BEFORE = TMP / "fixa_cap_before"          # 修订前账本的快照(重算 summarize 之前复制)


def _money(a, flag=None) -> str:
    if a is None or not np.isfinite(a):
        return {"gt_max": ">100亿", "lt_min": "<100万", "ref_nonpositive": "—(1000万已≤0)"}.get(flag, "—")
    return f"{a/1e8:.1f}亿" if a >= 1e8 else f"{a/1e4:.0f}万"


def cmd_capdiff():
    """容量报告(CAPACITY_REPORT.md)修订前后的关键数字 -> research/ledger/fixa_capacity_before_after.csv。
    before = 修订前账本快照 cache/_tmp/fixa_cap_before/capacity_*.csv(= 修订后账本的 *_pool_d1 /
    *_csi1000p_legacy 列,逐位核对);after = 修订后账本。只换了基准,策略日收益与成本逐位不变(核对写进同一张表)。"""
    old = {k: pd.read_csv(CAP_BEFORE / f"capacity_{k}.csv") for k in ("grid", "summary", "scaling")}
    new = {k: pd.read_csv(LEDGER / f"capacity_{k}.csv") for k in ("grid", "summary", "scaling")}
    rows = []

    def one(df, **kw):
        m = np.ones(len(df), bool)
        for k, v in kw.items():
            m &= (df[k] == v).to_numpy()
        r = df[m]
        assert len(r) == 1, (kw, len(r))
        return r.iloc[0]

    def add(group, metric, before, after, src):
        rows.append(dict(group=group, metric=metric, before=before, after=after, source=src))

    pp = lambda x: f"{x*100:+.2f}"
    for m in ("open10", "vwap1"):
        a = one(old["grid"], strategy="COMP_v5_exclude", mode=m, variant="Y1.0", sizing="pov", window_role="full", aum=1e7)
        b = one(new["grid"], strategy="COMP_v5_exclude", mode=m, variant="Y1.0", sizing="pov", window_role="full", aum=1e7)
        add("基准年化 2021-2025", f"{m} 池等权(k=1 → 同频 5 日)", f"{a.bench_pool_ann*100:.2f}%", f"{b.bench_pool_ann*100:.2f}%",
            "capacity_grid.csv bench_pool_ann")
        add("基准年化 2021-2025", f"{m} 中证1000 代理(收盘权重 → 成交时点权重)", f"{a.bench_csi1000p_ann*100:.2f}%",
            f"{b.bench_csi1000p_ann*100:.2f}%", "capacity_grid.csv bench_csi1000p_ann")
    for m in ("open10", "vwap1"):
        for sz in ("pov", "exante"):
            for aum, lab in ((1e7, "1000万"), (1e10, "100亿")):
                a = one(old["grid"], strategy="COMP_v5_exclude", mode=m, variant="Y1.0", sizing=sz, window_role="full", aum=aum)
                b = one(new["grid"], strategy="COMP_v5_exclude", mode=m, variant="Y1.0", sizing=sz, window_role="full", aum=aum)
                add("排雷 Y=1 全期净超额(pp)", f"{m} {sz} {lab}", pp(a.excess_pool), pp(b.excess_pool),
                    "capacity_grid.csv excess_pool")
            for v in ("Y1.0", "Y1.5"):
                for role in ("full", "dev", "test"):
                    a = one(old["summary"], strategy="COMP_v5_exclude", mode=m, variant=v, sizing=sz, window_role=role)
                    b = one(new["summary"], strategy="COMP_v5_exclude", mode=m, variant=v, sizing=sz, window_role=role)
                    add(f"排雷 {v} 半衰 / 盈亏平衡", f"{m} {sz} {a.window}",
                        f"{_money(a.cap_half_pool, a.cap_half_pool_flag)} / {_money(a.cap_be_pool, a.cap_be_pool_flag)}",
                        f"{_money(b.cap_half_pool, b.cap_half_pool_flag)} / {_money(b.cap_be_pool, b.cap_be_pool_flag)}",
                        "capacity_summary.csv cap_half_pool / cap_be_pool")
    for v in ("legacy", "Y0.5", "Y1.0"):
        for s in ("COMP_v5_top20", "COMP_v5_top50", "B_gbt_top20"):
            for m in ("open10", "vwap1"):
                a = one(old["summary"], strategy=s, mode=m, variant=v, sizing="pov", window_role="full")
                b = one(new["summary"], strategy=s, mode=m, variant=v, sizing="pov", window_role="full")
                add(f"Top-N {v} 全期(POV)1000万净超额 / 盈亏平衡", f"{s} {m}",
                    f"{pp(a.excess_pool_1e7)} / {_money(a.cap_be_pool, a.cap_be_pool_flag)}",
                    f"{pp(b.excess_pool_1e7)} / {_money(b.cap_be_pool, b.cap_be_pool_flag)}",
                    "capacity_summary.csv excess_pool_1e7 / cap_be_pool")
    for s in ("COMP_v5_exclude", "COMP_v5_top20", "COMP_v5_top50", "B_gbt_top20"):
        for m in ("open10", "vwap1"):
            a = one(old["grid"], strategy=s, mode=m, variant="Y1.0", sizing="pov", window_role="full", aum=1e7)
            b = one(new["grid"], strategy=s, mode=m, variant="Y1.0", sizing="pov", window_role="full", aum=1e7)
            sa = one(old["summary"], strategy=s, mode=m, variant="Y1.0", sizing="pov", window_role="full")
            sb = one(new["summary"], strategy=s, mode=m, variant="Y1.0", sizing="pov", window_role="full")
            add("相对中证1000 代理(Y=1、1000万、POV)超额 / NW t / 盈亏平衡", f"{s} {m}",
                f"{pp(a.excess_csi1000p)} / {a.t_nw_csi1000p:.2f} / {_money(sa.cap_be_csi1000p, sa.cap_be_csi1000p_flag)}",
                f"{pp(b.excess_csi1000p)} / {b.t_nw_csi1000p:.2f} / {_money(sb.cap_be_csi1000p, sb.cap_be_csi1000p_flag)}",
                "capacity_grid.csv excess_csi1000p / t_nw_csi1000p;capacity_summary.csv cap_be_csi1000p")
    for s in ("COMP_v5_exclude", "B_gbt_top20"):
        for m in ("open10", "vwap1"):
            for w in ((["2021-2025", "2024-2025"]) if s == "COMP_v5_exclude" else ["2023-2025"]):
                a = one(old["scaling"], strategy=s, mode=m, variant="Y1.0", sizing="pov", window=w)
                b = one(new["scaling"], strategy=s, mode=m, variant="Y1.0", sizing="pov", window=w)
                add("量纲外推(非预注册)每元毛超额 / 毛利 / 外推盈亏平衡", f"{s} {m} {w}",
                    f"{a.alpha_per_traded_bp:.1f}bp / {a.margin_bp:.1f}bp / {_money(a.x_be_sqrt_extrap)}",
                    f"{b.alpha_per_traded_bp:.1f}bp / {b.margin_bp:.1f}bp / {_money(b.x_be_sqrt_extrap)}",
                    "capacity_scaling.csv alpha_per_traded_bp / margin_bp / x_be_sqrt_extrap")
    # 核对:只换了基准
    g_o, g_n = old["grid"], new["grid"]
    k = ["strategy", "mode", "variant", "sizing", "aum", "window"]
    mm = g_o.merge(g_n, on=k, suffixes=("_o", "_n"))
    chk = {"策略年化 ann_return": float((mm.ann_return_o - mm.ann_return_n).abs().max()),
           "成本 cost_total_bp": float((mm.cost_total_bp_o - mm.cost_total_bp_n).abs().max()),
           "修订前 excess_pool vs 修订后 excess_pool_d1": float((mm.excess_pool_o - mm.excess_pool_d1).abs().max()),
           "修订前 excess_csi1000p vs 修订后 excess_csi1000p_legacy":
               float((mm.excess_csi1000p_o - mm.excess_csi1000p_legacy).abs().max())}
    for kk, v in chk.items():
        add("核对(最大绝对差)", kk, "—", f"{v:.1e}", "capacity_grid.csv 修订前快照 vs 修订后")
    # 损失分解:相对 1000 万格的差,基准在差里抵消;只有"现金拖累 = 现金 × 池基准收益"一列随基准变
    do, dn = pd.read_csv(CAP_BEFORE / "capacity_decomp.csv"), pd.read_csv(LEDGER / "capacity_decomp.csv")
    kd = ["strategy", "mode", "variant", "sizing", "window", "window_role", "aum"]
    md = do.merge(dn, on=kd, suffixes=("_o", "_n"))
    for c in ("d_net_pp", "d_gross_pp", "d_cost_pp"):
        add("核对(最大绝对差)", f"损失分解 {c}", "—", f"{float((md[c + '_o'] - md[c + '_n']).abs().max()):.1e}",
            "capacity_decomp.csv 修订前快照 vs 修订后")
    for m in ("open10", "vwap1"):
        x = md[md["mode"] == m]
        add("核对(最大绝对差)", f"损失分解 d_cash_drag_pp(现金 × 池基准收益){m}", "—",
            f"{float((x.d_cash_drag_pp_o - x.d_cash_drag_pp_n).abs().max()):.2f}pp",
            "capacity_decomp.csv 修订前快照 vs 修订后")
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "fixa_capacity_before_after.csv", index=False)
    print(df.to_string())


# ====================================================================== review2:第二轮修订的复核审查(1 条 P1 + 4 条 P2)
WARM_START = pd.Timestamp("2020-07-01")          # 与 scripts/cap_capacity_curve.py 的 2020H2 预热同一起点
R2_WINDOWS = {"2021-2025": (2021, 2025), "2021-2023": (2021, 2023), "2024-2025": (2024, 2025)}
EXCL = dict(scheme="exclude", buffer=(0.20, 0.35), rebalance=K)
PLANS = {"exclude_std": EXCL, "top20_r5": dict(scheme="top", n=20, rebalance=K)}


def _demo_models(p: Panel) -> dict:
    """与 scripts/eng_cost_model_tables.cost_demo 同一组成本模型(同构造参数)。"""
    from afactor import impact as IM
    return {"legacy(1bp+印花税)": None,
            "Y0.5_tick": IM.CostModel(p, impact_y=0.5), "Y1.0_tick": IM.CostModel(p, impact_y=1.0),
            "Y1.5_tick": IM.CostModel(p, impact_y=1.5),
            "Y1.0_2tick": IM.CostModel(p, impact_y=1.0, spread_mult=2.0),
            "Y1.0_arwindow": IM.CostModel(p, impact_y=1.0, spread_method="ar_window")}


def _win_row(rs: pd.Series, daily: pd.DataFrame | None, b_same: pd.Series, b_d1: pd.Series, y0: int, y1: int) -> dict:
    """一条窗口切片的统计:几何年化(与 run_strategy 的 ann_return 同式)、对同频 / 每日重置基准的超额与 TE / IR。"""
    s = rs[(rs.index.year >= y0) & (rs.index.year <= y1)]
    pre = rs[rs.index < s.index[0]]
    bs, bd = b_same.reindex(s.index), b_d1.reindex(s.index)
    es, ed = ST.excess_stats(s, bs), ST.excess_stats(s, bd)
    row = dict(n_days=len(s), nav_at_window_start=float((1.0 + pre).prod()),     # 冷启动 = 1;预热路径进入窗口时的净值
               ann_return=geo(s), bench_same_ann=geo(bs), bench_d1_ann=geo(bd),
               excess_same=geo(s) - geo(bs), excess_d1=geo(s) - geo(bd),
               **{f"{k}_same": es.get(k, np.nan) for k in ("excess_arith", "te", "ir", "te_adj", "ir_adj", "t_nw")},
               **{f"{k}_d1": ed.get(k, np.nan) for k in ("excess_arith", "te", "ir", "te_adj", "ir_adj", "t_nw")})
    if daily is not None:
        d = daily.reindex(s.index)
        for c in ("cost_total", "cost_spread", "cost_impact"):
            row[f"{c}_ann_pp"] = float(d[c].mean() * TD * 100)
        row["exposure_mean"] = float(d["exposure"].mean())
    return row


def warm_rows(p: Panel) -> pd.DataFrame:
    """复核 P1:ENGINE §3.1 / §7 的"对同频基准超额"在窗口掩码下冷启动(相位 k 前 k 天空仓),而同频基准从面板起点
    预热、始终满仓,两侧不对称。这里给 2020H2 预热的连续路径切片(与容量报告同法):
      start=warm_2020H2  date_mask 从 2020-07-01 起到 2025-12-31,一条连续路径,切 2021-2025 / 2021-2023 / 2024-2025;
      start=cold_mask    原 eng_cost_demo / eng_vwap_autocorr 的窗口掩码(每个窗口首日冷启动),只对旧成本排雷重跑
                         (§3.1 的 IR 与 §7 的冷启动对照),ann_return 必须与 eng_cost_demo.csv 逐位相同。
    组合与 eng_cost_demo 相同:open10 两个方案 × 6 个成本模型,vwap1 两个方案 × (旧成本, Y=1 一跳);另加 tail1450
    (与 eng_review_fixes.part_autocorr 同法:研究期 Panel()、comp_v5_score_open10 打分,只做旧成本排雷)。
    成本模型的冲击按 capital × 当前净值(引擎默认,与 eng_cost_demo 相同):预热路径进入窗口时净值已不是 1,
    冲击会略高于窗口内从 1 起算 —— cost_*_ann_pp 列给出,对比 eng_cost_demo 的同名列即得这一项。
    同频基准:benchmark(date_mask, rebalance=5),相位平均后与锚点无关,冷 / 热两种掩码下同一天的值相同。"""
    score = pd.read_parquet(TMP / "comp_v5_score_open10.parquet")
    demo = pd.read_csv(LEDGER / "eng_cost_demo.csv")
    yrs = p.dates.year
    mw = np.asarray((p.dates >= WARM_START) & (yrs <= 2025))
    models = _demo_models(p)
    rows = []
    t0 = time.time()
    for mode in ("open10", "vwap1", "tail1450"):
        b_d1 = ST.benchmark(p, V, mode)["rets"]
        b_same = ST.benchmark(p, V, mode, date_mask=mw, rebalance=K)["rets"]
        for pname, plan in PLANS.items():
            for mname, cm in models.items():
                if mode == "vwap1" and mname not in ("legacy(1bp+印花税)", "Y1.0_tick"):
                    continue
                if mode == "tail1450" and (pname != "exclude_std" or cm is not None):
                    continue
                r = ST.run_strategy(score, p, V, mode, date_mask=mw, cost_model=cm, record=True, **plan)
                daily = r["record"]["daily"]
                if pname == "exclude_std" and cm is None:
                    warm_excl = r["rets"]                 # 冷启动对照用(同一组合的预热路径)
                for w, (y0, y1) in R2_WINDOWS.items():
                    row = dict(mode=mode, plan=pname, cost_model=mname, window=w, start="warm_2020H2",
                               **_win_row(r["rets"], daily, b_same, b_d1, y0, y1))
                    ref = demo.query("mode==@mode and window==@w and plan==@pname and cost_model==@mname")
                    if len(ref):
                        row.update(cold_ann_return_eng_cost_demo=float(ref.ann_return.iloc[0]),
                                   cold_cost_ann_pp_eng_cost_demo=float(ref.cost_ann_pp.iloc[0]),
                                   cold_impact_ann_pp_eng_cost_demo=float(ref.impact_ann_pp.iloc[0]))
                    rows.append(row)
                del r, daily
                print(f"warm {mode} {pname} {mname} [{time.time()-t0:.0f}s]", flush=True)
        # 冷启动对照(旧成本排雷,窗口掩码):§3.1 的 IR 与审查探针的冷启动数字
        for w, (y0, y1) in R2_WINDOWS.items():
            m = np.asarray((yrs >= y0) & (yrs <= y1))
            r = ST.run_strategy(score, p, V, mode, date_mask=m, **EXCL)
            bs = ST.benchmark(p, V, mode, date_mask=m, rebalance=K)["rets"]
            row = dict(mode=mode, plan="exclude_std", cost_model="legacy(1bp+印花税)", window=w, start="cold_mask",
                       **_win_row(r["rets"], None, bs, b_d1, y0, y1))
            ref = demo.query("mode==@mode and window==@w and plan=='exclude_std' and cost_model=='legacy(1bp+印花税)'")
            if len(ref):
                row["cold_ann_return_eng_cost_demo"] = float(ref.ann_return.iloc[0])
                row["check_cold_vs_eng_cost_demo_absdiff"] = abs(float(r["ann_return"]) - float(ref.ann_return.iloc[0]))
            row["check_bench_same_cold_vs_warm_maxabs"] = float(np.max(np.abs(
                bs.to_numpy() - b_same.reindex(bs.index).to_numpy())))
            # 伪超额出在哪几天:冷启动 − 预热 的日收益差,窗口首 5 个信号日(相位空仓期)的累计 vs 全窗口累计(算术和)
            dd = r["rets"] - warm_excl.reindex(r["rets"].index)
            row.update(cold_minus_warm_cum_first5_pp=float(dd.iloc[:K].sum() * 100),
                       cold_minus_warm_cum_total_pp=float(dd.sum() * 100),
                       bench_same_cum_first5_pct=float(((1.0 + bs.iloc[:K]).prod() - 1.0) * 100))
            rows.append(row)
        print(f"cold {mode} [{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    # 冷启动伪超额 = 冷启动超额 − 预热超额(同一窗口、同频基准、旧成本排雷)
    key = ["mode", "plan", "cost_model", "window"]
    w_ = df[df.start == "warm_2020H2"].set_index(key)
    c_ = df[df.start == "cold_mask"].set_index(key)
    infl = (c_["excess_same"] - w_["excess_same"].reindex(c_.index)).rename("cold_minus_warm_excess_same")
    df = df.merge(infl.reset_index(), on=key, how="left")
    # §7 其他组合的冷启动对照:策略冷启动年化取 eng_cost_demo 原值,基准同频(冷 / 热同值)
    m7 = df["cold_ann_return_eng_cost_demo"].notna() & (df.start == "warm_2020H2")
    df.loc[m7, "cold_excess_same_eng_cost_demo"] = df.loc[m7, "cold_ann_return_eng_cost_demo"] - df.loc[m7, "bench_same_ann"]
    df["note"] = ("COMP_v5 信号样本内;V1000、1000 万、5 日调仓 5 相位平均;warm = 2020-07-01 起连续路径切片,"
                  "cold = 窗口首日冷启动(相位 k 前 k 天空仓);基准同频池等权(rebalance=5)与每日重置池等权(d1)")
    return df


def mech_rows() -> pd.DataFrame:
    """复核 P2(机制)与 P2(tail1450 量级):
    (a) 同频(k=5)池基准 − 每日重置池基准的年化差(几何 / 算术 / NW t),open10 / vwap1 / tail1450,
        研究期 Panel()(与 ENGINE §3.1 表、审查探针同法)与 TailPanel()(尾盘研究脚本实际用的面板,决策池用 T-1 信息)。
    (b) 机制:每日重置等权与 k 日漂移等权的市场暴露相同,二者之差取决于**个股相对池均值收益**的序列相关,
        不是池等权序列本身的自相关。一阶展开:漂移组合第 s 天(s=0..k-1)相对等权的超额
        ≈ Σ_{j<s} cov_cs(g_j, g_s),相位平均后每天 ≈ (1/k) Σ_{l=1}^{k-1} (k−l)·γ_l,γ_l = 截面相对收益的
        滞后 l 期截面协方差(信号日 t 的决策池、t 与 t+l 都有收益的股票,逐日截面去均值后取均值,再对时间平均)。
        pred_gap_arith = 242 × 上式,与实际的算术年化差 arith_r5_minus_d1 对照;xs_corr_l* 是同一组的截面相关(均值)。"""
    from afactor.panel import TailPanel
    rows = []
    wins = {"2020-2025": (2020, 2025), **R2_WINDOWS}
    for ptag, pcls, modes in (("Panel", Panel, ("open10", "vwap1", "tail1450")), ("TailPanel", TailPanel, ("tail1450",))):
        p = pcls()
        yrs = np.asarray(p.dates.year)
        U = ST.universe(p, V).reindex(index=p.dates, columns=p.codes).fillna(False).to_numpy(bool)
        for mode in modes:
            b1 = ST.benchmark(p, V, mode)["rets"]
            b5 = ST.benchmark(p, V, mode, rebalance=K)["rets"]
            px, _ = ST.exec_prices(p, mode)
            lag = ST.EXEC_MODES[mode]["lag"]
            P = px.reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
            n = len(p.dates) - 1 - lag
            with np.errstate(invalid="ignore", divide="ignore"):
                G = P[1 + lag:1 + lag + n] / P[lag:lag + n] - 1.0
            del P
            fin = np.isfinite(G)
            cov = np.full((K, n), np.nan); cor = np.full((K, n), np.nan); var0 = np.full(n, np.nan)
            for t in range(n):
                for L in range(0, K):
                    if t + L >= n:
                        break
                    m = U[t] & fin[t] & fin[t + L]
                    if m.sum() < 100:
                        continue
                    a = G[t][m] - G[t][m].mean(); b = G[t + L][m] - G[t + L][m].mean()
                    c = float((a * b).mean()); va, vb = float((a * a).mean()), float((b * b).mean())
                    cov[L, t] = c
                    cor[L, t] = c / np.sqrt(va * vb) if va > 0 and vb > 0 else np.nan
            ty = yrs[:n]
            for w, (y0, y1) in wins.items():
                sel = (ty >= y0) & (ty <= y1)
                mb = (b1.index.year >= y0) & (b1.index.year <= y1)
                g = {f"xs_cov_l{L}": float(np.nanmean(cov[L, sel])) for L in range(K)}
                pred = sum((K - L) * g[f"xs_cov_l{L}"] for L in range(1, K)) / K * TD
                d = (b5[mb] - b1[mb]).to_numpy()
                rows.append(dict(panel=ptag, mode=mode, window=w, n_days=int(mb.sum()),
                                 pool_ann_d1=geo(b1[mb]), pool_ann_r5=geo(b5[mb]),
                                 d_r5_minus_d1_geo=geo(b5[mb]) - geo(b1[mb]), arith_r5_minus_d1=float(d.mean() * TD),
                                 t_nw_r5_minus_d1=nw_t(d, 10), pool_ac1_d1=float(b1[mb].autocorr(1)),
                                 **{f"xs_corr_l{L}": float(np.nanmean(cor[L, sel])) for L in range(1, K)},
                                 **g, xs_std_daily=float(np.sqrt(g["xs_cov_l0"])),
                                 pred_gap_arith=pred))
            print(ptag, mode, "done", flush=True)
        del p, U
    df = pd.DataFrame(rows)
    df["pred_over_actual"] = df["pred_gap_arith"] / df["arith_r5_minus_d1"]
    df["note"] = ("pred_gap_arith = 242 × (1/5) Σ_{l=1..4} (5−l)·xs_cov_l(一阶展开);"
                  "xs_corr_l1 = 截面相对收益滞后 1 期相关(逐日截面相关的均值);pool_ac1_d1 = 每日重置池等权序列的一阶自相关(旁证)")
    return df


def barra_delta_rows() -> pd.DataFrame:
    """复核 P2(传播):BARRA_ATTRIBUTION 的时间序列回归 y = 策略 − 池基准 ~ 1 + 纯因子收益(scripts/run_barra_attribution.py)。
    OLS 对 y 线性,所以把基准从每日重置换成同频时,各项的变化 = 把 Δy = −(b5 − b1) 对同一组回归元回归的系数,
    不需要重跑策略。回归元直接读 research/ledger/barra_factor_returns.csv(该脚本落盘的 Fi,行 = 回归样本日)。
    open10 用 Panel(),tail1450 用 TailPanel()(与 run_barra_attribution 一致)。new = old(barra_attribution_summary.csv)+ Δ。
    alpha 的 t 值需要原残差,这里不重算。"""
    from afactor.panel import TailPanel
    from afactor.barra import nw_t as _nw_t_lags          # 与 run_barra_attribution 的 alpha t 同一函数
    F_all = pd.read_csv(LEDGER / "barra_factor_returns.csv", header=[0, 1], index_col=0, parse_dates=True)
    old = pd.read_csv(LEDGER / "barra_attribution_summary.csv")
    rows = []
    for mode, pcls in (("open10", Panel), ("tail1450", TailPanel)):
        p = pcls()
        b1 = ST.benchmark(p, V, mode)["rets"]
        b5 = ST.benchmark(p, V, mode, rebalance=K)["rets"]
        Fi = F_all[mode].dropna(how="any")
        idx = Fi.index.intersection(b1.index)
        assert len(idx) == len(Fi), (mode, len(idx), len(Fi))
        dy = -(b5.reindex(idx) - b1.reindex(idx)).astype(float).to_numpy()
        styles = [c for c in Fi.columns if not c.startswith("行业:") and c != "Country"]
        indus = [c for c in Fi.columns if c.startswith("行业:")]
        A = np.column_stack([np.ones(len(idx)), Fi.loc[idx, styles + indus].to_numpy(np.float64)])
        bcoef, *_ = np.linalg.lstsq(A, dy, rcond=None)
        terms = styles + indus
        dcon = {t: bcoef[i + 1] * float(Fi.loc[idx, t].mean()) * TD for i, t in enumerate(terms)}
        o = old[old["mode"] == mode].set_index("term")["contrib_ann"]
        dtot, dalpha = float(dy.mean() * TD), float(bcoef[0] * TD)
        dsty = sum(dcon[t] for t in styles); dind = sum(dcon[t] for t in indus)
        liq = [t for t in styles if t.startswith("Liquidity")][0]
        for term, d in (("__总超额", dtot), ("__alpha", dalpha), ("__风格合计", dsty), ("__行业合计", dind), (liq, dcon[liq])):
            rows.append(dict(mode=mode, term=term, old_pp=float(o[term]) * 100, delta_pp=d * 100,
                             new_pp=(float(o[term]) + d) * 100, n_days=len(idx)))
        rows.append(dict(mode=mode, term="流动性占总超额(old → new)", old_pp=float(o[liq] / o["__总超额"]) * 100,
                         delta_pp=np.nan, new_pp=float((o[liq] + dcon[liq]) / (o["__总超额"] + dtot)) * 100, n_days=len(idx)))
        if mode == "open10":
            # alpha 的 t 需要原残差:open10 的 COMP_v5 打分有缓存(cache/_tmp/comp_v5_score_open10.parquet),
            # 先用它复现原回归(总超额 / alpha / t 必须与账本一致),再对同频基准重算 t。tail1450 的打分没有缓存,不重算。
            score = pd.read_parquet(TMP / "comp_v5_score_open10.parquet")
            strat = ST.run_strategy(score, p, V, mode, **EXCL)["rets"].reindex(idx).astype(float)
            Xf = Fi.loc[idx, styles + indus].to_numpy(np.float64)
            for bn, bb in (("d1", b1), ("same", b5)):
                y = (strat - bb.reindex(idx)).to_numpy()
                A2 = np.column_stack([np.ones(len(idx)), Xf])
                c2, *_ = np.linalg.lstsq(A2, y, rcond=None)
                res2 = y - A2 @ c2
                rows.append(dict(mode=mode, term=f"复算 bench={bn}:总超额 / alpha", old_pp=float(y.mean() * TD * 100),
                                 delta_pp=np.nan, new_pp=float(c2[0] * TD * 100), n_days=len(idx)))
                for qq in (0, 5, 10, 20):
                    tt = _nw_t_lags(res2 + c2[0], qq)
                    rows.append(dict(mode=mode, term=f"复算 bench={bn}:alpha t(NW q{qq})",
                                     old_pp=float(o[f"__alpha_t_q{qq}"]) if bn == "d1" else np.nan,
                                     delta_pp=np.nan, new_pp=tt, n_days=len(idx)))
        print(mode, "barra delta done", flush=True)
        del p
    df = pd.DataFrame(rows)
    df["note"] = ("y=策略−池基准 的时间序列 OLS 对 y 线性:Δ = Δy=−(同频−每日重置) 对同一回归元的系数;"
                  "new = barra_attribution_summary.csv 原值 + Δ;占比一行单位为 %;alpha t 未重算")
    return df


def cmd_review2():
    """python3 scripts/fixa_benchmark_fix.py review2 [warm|mech|barra]  (默认三项都跑)"""
    which = sys.argv[2:] or ["mech", "barra", "warm"]
    t0 = time.time()
    if "mech" in which:
        df = mech_rows()
        df.to_csv(LEDGER / "fixa_r2_mech.csv", index=False)
        print(df.drop(columns="note").round(5).to_string(), flush=True)
    if "barra" in which:
        df = barra_delta_rows()
        df.to_csv(LEDGER / "fixa_r2_barra_delta.csv", index=False)
        print(df.drop(columns="note").round(3).to_string(), flush=True)
    if "warm" in which:
        p = Panel()
        df = warm_rows(p)
        df.to_csv(LEDGER / "fixa_r2_eng_warm.csv", index=False)
        show = ["mode", "plan", "cost_model", "window", "start", "excess_same", "excess_d1", "ir_same", "ir_adj_same",
                "cold_minus_warm_excess_same", "cold_excess_same_eng_cost_demo", "cost_total_ann_pp"]
        print(df[show].round(4).to_string(), flush=True)
    print(f"done review2 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    {"bench": cmd_bench, "downstream": cmd_downstream, "capdiff": cmd_capdiff, "review2": cmd_review2}[cmd]()
