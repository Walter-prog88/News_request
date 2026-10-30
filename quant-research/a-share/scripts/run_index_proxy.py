"""中证 300 / 500 / 1000 PIT 代理基准:构建、用真实指数收益校验、给下游落缓存。

做什么(口径见 src/afactor/index_proxy.py 文档):
  1. 按中证公开编制方法的简化版重建三只代理(PIT,2020-06 起每半年调整一次);
  2. 校验 2021-01 ~ 2025-12:代理收盘日收益 vs 真实 000300/000905/000852 收盘收益
     —— 相关、年化跟踪误差、逐年收益差、收益差的 NW t;分开发窗 2021-2023 / 检验窗 2024-2025;
  3. 收益差来源的拆解:
       a. 价格指数 vs 全收益(kind=pr / tr);
       b. 权重近似(circ / total / banded / equal);
       c. 规则简化(去掉缓冲区 / 去掉 10% 上限 / 去掉成交额剔除 / 换初始期);
       d. 幸存者偏差(把 delisted_history.csv 里仅有的 10 只退市股加回股票池,看代理收益变多少);
       e. 收益差对 指数收益 与 大小盘价差 的 HAC 回归(暴露差 vs 常数项);
  4. 下游接口落缓存并做引擎断言:proxy_weights、engine_consistent_bench_returns
     (open10 与 tail1450),金标准 = 池等权权重经同一函数必须逐位复现 strategy.benchmark。

封存:所有输入截到 2025-12-31(见 index_proxy._sealed_newdata);指数收盘截断;
      2026-08-21 的成分快照只读"它包含哪些指数代码"这一条元数据。
产出:research/ledger/proxy_validation.csv、proxy_membership_stats.csv、
      proxy_gap_attribution.csv、proxy_engine_checks.csv、proxy_survivor_check.csv、
      reports/proxy_vs_index.png、cache/_tmp/proxy_*.parquet
"""
from __future__ import annotations
import os, sys, json, time, warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, REPORTS, ROOT
from afactor.panel import Panel, TailPanel
from afactor import index_proxy as IP, strategy as ST

TD = 242
YEARS = list(range(2021, 2026))
WINDOWS = {"2021-2025": ("2021-01-01", "2025-12-31"),
           "dev_2021-2023": ("2021-01-01", "2023-12-31"),
           "test_2024-2025": ("2024-01-01", "2025-12-31")}
DELISTED_10 = ROOT / "data_8:21" / "delisted_history.csv"
SNAPSHOT = ROOT / "data_8:21" / "index_cons" / "csindex_cons_snapshot_20260821.csv"


# ---------------- 统计 ----------------
def nw_t(x: pd.Series, lag: int) -> float:
    """Newey-West 均值 t(Bartlett 核,自协方差按 n 归一)。与 scripts/run_ir_analysis.py::nw_t 同式。"""
    x = pd.Series(x).dropna().to_numpy(float)
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    s = float((e ** 2).sum() / n)
    for L in range(1, lag + 1):
        s += 2.0 * (1.0 - L / (lag + 1.0)) * float((e[L:] * e[:-L]).sum() / n)
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def pair_stats(rp: pd.Series, ri: pd.Series) -> dict:
    """代理 rp 与真实指数 ri(同索引日收益)的跟踪统计。"""
    idx = rp.dropna().index.intersection(ri.dropna().index)
    a, b = rp.reindex(idx), ri.reindex(idx)
    d = a - b
    n = len(idx)
    geo = lambda r: float((1 + r).prod() ** (TD / len(r)) - 1) if len(r) else np.nan
    # 非重叠 5 日收益相关(日频收益可能因收盘价噪声低估/高估相关)
    g5 = np.arange(n) // 5
    a5 = (1 + a).groupby(g5).prod() - 1
    b5 = (1 + b).groupby(g5).prod() - 1
    return dict(n_days=n,
                ret_proxy_period=float((1 + a).prod() - 1), ret_index_period=float((1 + b).prod() - 1),
                ann_proxy=geo(a), ann_index=geo(b), gap_ann_geom=geo(a) - geo(b),
                gap_ann_arith=float(d.mean() * TD), te_ann=float(d.std() * np.sqrt(TD)),
                corr_daily=float(a.corr(b)), corr_5d=float(a5.corr(b5)),
                beta_proxy_on_index=float(np.cov(a, b)[0, 1] / np.var(b, ddof=1)),
                vol_proxy=float(a.std() * np.sqrt(TD)), vol_index=float(b.std() * np.sqrt(TD)),
                t_nw_gap=nw_t(d, 5))


def stats_rows(rp: pd.Series, ri: pd.Series, **tags) -> list[dict]:
    rows = []
    for w, (s, e) in WINDOWS.items():
        m = (rp.index >= s) & (rp.index <= e)
        rows.append(dict(**tags, period=w, **pair_stats(rp[m], ri[m])))
    for y in YEARS:
        m = rp.index.year == y
        rows.append(dict(**tags, period=str(y), **pair_stats(rp[m], ri[m])))
    return rows


def index_close_rets(p) -> dict:
    out = {}
    for c in IP.CSI_RULES:
        px = IP.index_close(c).reindex(p.dates)
        out[c] = px / px.shift(1) - 1.0                # 打在实现日 t(收盘到收盘)
    return out


# ---------------- 幸存者偏差:扩展股票池 ----------------
def extend_with_delisted(p, inp: dict, R: dict):
    """把 delisted_history.csv 中仅有的 10 只退市股(BaoStock 未复权日线)加回股票池。

    mv   = 收盘 × PIT 总股本(财报,先截断);circ = 收盘 × 成交量 / 换手率(反推流通股本);
    收益 = close/pre_close - 1(pre_close 已除权,近似全收益;这 10 只多为困境股,分红可忽略,
           价格口径与全收益口径用同一序列);ST 用文件自带 is_st(日频,PIT)。
    这只是 206 只缺失退市股中的 10 只,结果是**下界量级**,不能外推成完整修正。
    """
    d = pd.read_csv(DELISTED_10)
    d = d[d["trade_date"] <= 20251231].copy()                       # 封存:先截
    d["trade_date"] = pd.to_datetime(d["trade_date"].astype(str), format="%Y%m%d")
    extra = sorted(set(d["ts_code"]) - set(p.codes))
    codes = p.codes.append(pd.Index(extra))
    dates = p.dates
    piv = lambda col: d.pivot(index="trade_date", columns="ts_code", values=col).reindex(
        index=dates, columns=extra).astype(float)
    close, pre, amt, vol, turn = (piv(c) for c in ("close", "pre_close", "amount", "vol", "turnover_rate"))
    status, st_flag = piv("trade_status"), piv("is_st")
    valid_x = (status == 1) & (amt > 0) & close.notna()
    with IP._sealed_newdata() as (ND, FU):
        rep = FU.build_report_level()
        sh = FU._pit_daily(rep, "total_share", dates, pd.Index(extra)).replace(0, np.nan)
    mv_x = (close * sh).where(valid_x).ffill()
    with np.errstate(divide="ignore", invalid="ignore"):
        fs = vol / (turn / 100.0)
    circ_x = (close * fs).where(valid_x & (turn > 0)).ffill().fillna(mv_x)
    r_x = (close / pre - 1.0).where(valid_x).fillna(0.0)

    ext = {}
    ext["mv"] = pd.concat([inp["mv"], mv_x.astype(np.float32)], axis=1)
    ext["circ"] = pd.concat([inp["circ"], circ_x.astype(np.float32)], axis=1)
    ext["st"] = pd.concat([inp["st"], (st_flag.fillna(0) > 0)], axis=1)
    ext["mvsrc"] = pd.concat([inp["mvsrc"], pd.DataFrame(3, index=dates, columns=extra, dtype=np.int8)], axis=1)
    ext["raw"] = None
    valid = pd.concat([p["valid"], valid_x], axis=1)
    amount = pd.concat([p.raw("amount"), amt.where(valid_x).astype(np.float32)], axis=1)
    Rx = {k: np.hstack([R[k], r_x.to_numpy(np.float64)]) for k in R}
    return ext, valid, amount, Rx, extra


# ---------------- 主流程 ----------------
def main():
    t0 = time.time()
    p = Panel()
    inp = IP.load_inputs(p)
    R = {k: IP.daily_returns(p, inp, k) for k in ("pr", "tr")}
    ri = index_close_rets(p)
    print(f"inputs ready {time.time() - t0:.0f}s")

    val_rows, mem_rows = [], []

    # ---- 1. 主口径(价格指数口径 pr 与全收益 tr)----
    main_res = IP.build_proxy(p, IP.ProxySpec(), kinds=("pr", "tr"), inp=inp, R=R, keep_weights=False)
    for code in IP.CSI_RULES:
        for kind in ("pr", "tr"):
            val_rows += stats_rows(main_res["ret"][(code, kind)], ri[code], index=code,
                                   name=IP.INDEX_NAMES[code], variant="main", kind=kind,
                                   timing="close_to_close")

    # ---- 2. 规则/权重敏感性(全部报告,不挑)----
    variants = {
        "w_total": IP.ProxySpec(weight="total"),
        "w_banded": IP.ProxySpec(weight="banded"),
        "w_equal": IP.ProxySpec(weight="equal"),
        "no_buffer_no_cap": IP.ProxySpec(buffer=False),
        "no_cap": IP.ProxySpec(cap=False),
        "no_liquidity": IP.ProxySpec(liquidity=False),
        "init_2020H2": IP.ProxySpec(first_review="2020H2"),
        "init_2021H1": IP.ProxySpec(first_review="2021H1"),
    }
    var_res = {}
    for vn, sp in variants.items():
        res = IP.build_proxy(p, sp, kinds=("pr",), inp=inp, R=R, keep_weights=False)
        var_res[vn] = res
        for code in IP.CSI_RULES:
            val_rows += stats_rows(res["ret"][(code, "pr")], ri[code], index=code,
                                   name=IP.INDEX_NAMES[code], variant=vn, kind="pr",
                                   timing="close_to_close")
        print(f"variant {vn} {time.time() - t0:.0f}s")

    # 成分重合:各变体 vs 主口径(同期名单的 Jaccard 之外,用"主口径成分中也在变体里的比例")
    overlap_rows = []
    for vn, res in var_res.items():
        for rv in main_res["members"]:
            if rv not in res["members"]:
                continue
            for code in IP.CSI_RULES:
                a, b = set(main_res["members"][rv][code]), set(res["members"][rv][code])
                overlap_rows.append(dict(variant=vn, review=rv, index=code,
                                         overlap_with_main=len(a & b) / len(a)))
    ov = pd.DataFrame(overlap_rows)

    # ---- 3. 幸存者偏差:把 10 只有日线的退市股加回 ----
    ext, valid_x, amount_x, Rx, extra = extend_with_delisted(p, inp, R)
    sres = IP.build_proxy(p, IP.ProxySpec(), kinds=("pr",), inp=ext, valid=valid_x,
                          amount=amount_x, R=Rx, keep_weights=False)
    surv_rows = []
    for code in IP.CSI_RULES:
        val_rows += stats_rows(sres["ret"][(code, "pr")], ri[code], index=code,
                               name=IP.INDEX_NAMES[code], variant="main_plus_10_delisted",
                               kind="pr", timing="close_to_close")
        for rv, mem in sres["members"].items():
            ins = [c for c in mem[code] if c in extra]
            if ins:
                k = list(sres["sched"]["review"]).index(rv)
                w0 = sres["base_w"][code][k]
                cidx = valid_x.columns
                surv_rows.append(dict(index=code, review=rv, n_delisted_in=len(ins),
                                      codes=" ".join(ins),
                                      base_weight_sum=float(sum(w0[cidx.get_loc(c)] for c in ins))))
        dm = (sres["ret"][(code, "pr")] - main_res["ret"][(code, "pr")])
        m = (dm.index >= "2021-01-01")
        surv_rows.append(dict(index=code, review="2021-2025", n_delisted_in=np.nan, codes="",
                              base_weight_sum=np.nan,
                              delta_ann_arith_vs_main=float(dm[m].mean() * TD),
                              delta_te_vs_index=float((sres["ret"][(code, "pr")] - ri[code])[m].std() * np.sqrt(TD)
                                                      - (main_res["ret"][(code, "pr")] - ri[code])[m].std() * np.sqrt(TD))))
    # 缺失退市股的计数(stock_basic 元数据)
    sb = pd.read_csv(ROOT / "data_8:21" / "stock_basic.csv")
    sb["dd"] = pd.to_datetime(sb["delist_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    miss = sb[(sb["dd"] >= "2021-01-01") & (sb["dd"] <= "2025-12-31") & ~sb["ts_code"].isin(p.codes)
              & ~sb["ts_code"].str.endswith(".BJ")]
    distress = miss["name"].astype(str).str.contains("退|ST")
    for y, g in miss.groupby(miss["dd"].dt.year):
        surv_rows.append(dict(index="all", review=f"delisted_{y}", n_delisted_in=len(g),
                              codes=f"distress_name={int(distress[g.index].sum())};other(merger/privatization等)={int((~distress[g.index]).sum())}",
                              base_weight_sum=np.nan))
    pd.DataFrame(surv_rows).to_csv(LEDGER / "proxy_survivor_check.csv", index=False)
    print(f"survivor {time.time() - t0:.0f}s")

    # ---- 4. 收益差的暴露回归(HAC)----
    import statsmodels.api as sm
    spread = ri["000852"] - ri["000300"]
    gap_rows = []
    for code in IP.CSI_RULES:
        for w, (s, e) in WINDOWS.items():
            rp = main_res["ret"][(code, "pr")]
            m = (rp.index >= s) & (rp.index <= e)
            dd = (rp - ri[code])[m].dropna()
            X = sm.add_constant(pd.concat([ri[code], spread], axis=1).reindex(dd.index))
            X.columns = ["const", "index_ret", "size_spread_1000_minus_300"]
            f = sm.OLS(dd, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            gap_rows.append(dict(index=code, period=w, n_days=len(dd),
                                 alpha_ann=float(f.params["const"] * TD), t_alpha=float(f.tvalues["const"]),
                                 b_index=float(f.params["index_ret"]), t_b_index=float(f.tvalues["index_ret"]),
                                 b_size=float(f.params["size_spread_1000_minus_300"]),
                                 t_b_size=float(f.tvalues["size_spread_1000_minus_300"]),
                                 r2=float(f.rsquared),
                                 gap_ann_arith=float(dd.mean() * TD),
                                 gap_from_exposure_ann=float((f.fittedvalues - f.params["const"]).mean() * TD)))
    pd.DataFrame(gap_rows).to_csv(LEDGER / "proxy_gap_attribution.csv", index=False)

    # ---- 5. 下游缓存 + 引擎断言 ----
    W = IP.proxy_weights(p, "000852", refresh=True)        # 同时写 300/500 与 tail 版
    print(f"weights cached {time.time() - t0:.0f}s")
    rs = W.sum(axis=1)
    eng_rows = []
    first_eff = rs[rs > 0].index[0]
    eng_rows.append(dict(check="weights_row_sum_max", value=float(rs.max()), ok=bool(rs.max() <= 1 + 1e-5)))
    eng_rows.append(dict(check="weights_row_sum_min_after_first_eff", value=float(rs[rs.index >= first_eff].min()),
                         ok=bool(rs[rs.index >= first_eff].min() >= 1 - 1e-5)))
    eng_rows.append(dict(check="weights_min", value=float(np.nanmin(W.to_numpy())), ok=bool(np.nanmin(W.to_numpy()) >= 0)))
    eng_rows.append(dict(check="first_effective_signal_day", value=str(first_eff.date()), ok=True))
    nz = (W > 0).sum(axis=1)
    eng_rows.append(dict(check="n_members_per_day_after_first_eff_min_max",
                         value=f"{int(nz[nz.index >= first_eff].min())}/{int(nz.max())}", ok=True))
    # 与收盘口径代理收益一致:Σ W[t-1]·R_tr[t] == 主口径 tr 收益
    Wv = W.to_numpy(np.float64)
    r_chk = np.r_[np.nan, (Wv[:-1] * R["tr"][1:]).sum(axis=1)]
    r_ref = main_res["ret"][("000852", "tr")].to_numpy()
    mm = np.isfinite(r_ref) & (Wv.sum(1) > 0)[np.r_[0, np.arange(len(Wv) - 1)]]
    dmax = float(np.nanmax(np.abs(r_chk[mm] - r_ref[mm])))
    eng_rows.append(dict(check="cached_weights_reproduce_close_proxy_tr_maxabs", value=dmax, ok=bool(dmax < 1e-6)))
    del Wv

    for mode, panel in (("open10", p), ("tail1450", TailPanel())):
        bench = ST.benchmark(panel, "V1000", mode)["rets"]
        U = ST.universe(panel, "V1000")
        Weq = U.astype(np.float64).div(U.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        r_eq = IP.weighted_exec_returns(panel, mode, Weq, "renorm").fillna(0.0)
        gold = float((r_eq - bench).abs().max())
        same_idx = bool(r_eq.index.equals(bench.index))
        eng_rows.append(dict(check=f"golden_{mode}_poolEW_vs_ST.benchmark_maxabs", value=gold,
                             ok=bool(gold < 1e-6 and same_idx)))
        for code in IP.CSI_RULES:
            rb = IP.engine_consistent_bench_returns(panel, mode, code)
            rix = IP.index_returns_on_signal_days(panel, mode, code)
            eng_rows.append(dict(check=f"index_equal_{mode}_{code}_vs_benchmark_index",
                                 value=bool(rb.index.equals(bench.index) and rix.index.equals(bench.index)),
                                 ok=bool(rb.index.equals(bench.index) and rix.index.equals(bench.index))))
            for ref_name, ref in (("index", rix), ("pool_ew_bench", bench)):
                cs = {k: float(rb.corr(ref.shift(k))) for k in (-2, -1, 0, 1, 2)}
                peak = max(cs, key=cs.get)
                eng_rows.append(dict(check=f"align_{mode}_{code}_vs_{ref_name}",
                                     value=" ".join(f"k{k:+d}:{v:.4f}" for k, v in cs.items()),
                                     ok=bool(peak == 0)))
            val_rows += stats_rows(rb, rix, index=code, name=IP.INDEX_NAMES[code],
                                   variant="main", kind="tr_engine", timing=f"engine_{mode}_signal_day")
            # 缓存给下游
            rb.to_frame().to_parquet(IP.TMP / f"proxy_bench_{code}_{mode}.parquet")
            # 参照:项目原有的池等权基准(V1000 池)相对同一真实指数 —— 让下游知道换基准差多少
            val_rows += stats_rows(bench, rix, index=code, name=IP.INDEX_NAMES[code],
                                   variant="pool_ew_V1000_benchmark", kind="tr_engine",
                                   timing=f"engine_{mode}_signal_day")
            val_rows += stats_rows(bench, rb, index=code, name=IP.INDEX_NAMES[code],
                                   variant="pool_ew_V1000_vs_engine_proxy", kind="tr_engine",
                                   timing=f"engine_{mode}_signal_day")
        print(f"engine {mode} {time.time() - t0:.0f}s")
    eng = pd.DataFrame(eng_rows)
    eng.to_csv(LEDGER / "proxy_engine_checks.csv", index=False)
    print(eng.to_string())
    assert eng["ok"].all(), "引擎断言失败,拒绝出数"

    # ---- 6. 成分统计 ----
    diag = main_res["diag"].copy()
    sched = main_res["sched"]
    for code in IP.CSI_RULES:
        w0s = main_res["base_w"][code]
        D, _ = IP.drift_weights(w0s, sched["base_idx"].tolist(), R["tr"])
        for k, r in enumerate(sched.itertuples()):
            w0 = w0s[k]
            nzw = np.sort(w0[w0 > 0])[::-1]
            old = D[r.base_idx] if k > 0 else None
            row = dict(review=r.review, index=code,
                       w_max=float(nzw[0]), w_top10_sum=float(nzw[:10].sum()),
                       eff_n=float(1.0 / (nzw ** 2).sum()),
                       rebalance_turnover_oneway=float(0.5 * np.abs(w0 - old).sum()) if old is not None else np.nan)
            mem_rows.append(row)
        del D
    ws = pd.DataFrame(mem_rows)
    diag = diag.merge(ws, on=["review", "index"], how="left")
    ovp = ov.pivot_table(index=["review", "index"], columns="variant", values="overlap_with_main").add_prefix("overlap_main_vs_")
    diag = diag.merge(ovp.reset_index(), on=["review", "index"], how="left")
    # 快照诊断:只读"快照含哪些指数代码"(名单元数据,不进入任何权重/收益计算)
    snap_codes = sorted(pd.read_csv(SNAPSHOT, usecols=["index_code"], dtype=str)["index_code"].dropna().unique())
    snap_codes = [c for c in snap_codes if c != "index_code"]
    diag["snapshot_20260821_has_this_index"] = diag["index"].isin(snap_codes)
    diag["snapshot_20260821_index_codes"] = ";".join(snap_codes)
    diag.to_csv(LEDGER / "proxy_membership_stats.csv", index=False)

    val = pd.DataFrame(val_rows)
    val.to_csv(LEDGER / "proxy_validation.csv", index=False)
    print(f"ledger written {time.time() - t0:.0f}s")

    # ---- 7. 图 ----
    plot(main_res, ri)
    show = val[(val.period.isin(["2021-2025", "dev_2021-2023", "test_2024-2025"]))]
    print(show[["index", "variant", "kind", "timing", "period", "ann_proxy", "ann_index", "gap_ann_geom",
                "te_ann", "corr_daily", "t_nw_gap"]].round(4).to_string())


def plot(res: dict, ri: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    c_proxy, c_index, c_tr, ink, grid = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1]))
    for j, code in enumerate(IP.CSI_RULES):
        rp = res["ret"][(code, "pr")]
        rt = res["ret"][(code, "tr")]
        m = rp.index >= "2021-01-01"
        a, b, t = rp[m].fillna(0), ri[code][m].fillna(0), rt[m].fillna(0)
        na, nb = (1 + a).cumprod(), (1 + b).cumprod()
        ax = axes[0, j]
        ax.plot(na.index, na, color=c_proxy, lw=2, label="代理(价格口径)")
        ax.plot(nb.index, nb, color=c_index, lw=2, label=f"真实 {code}")
        ax.set_title(f"{IP.INDEX_NAMES[code]}:代理 vs 真实指数(2021-01 = 1)", fontsize=11, color="#0b0b0b")
        te = (a - b).std() * np.sqrt(TD)
        ax.text(0.02, 0.04, f"日收益相关 {a.corr(b):.4f}\n年化跟踪误差 {te:.2%}",
                transform=ax.transAxes, fontsize=9, color=ink)
        ax.grid(color=grid, lw=0.8); ax.tick_params(colors=ink, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if j == 0:
            ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax2 = axes[1, j]
        gap_pr = np.log(na / nb)
        gap_tr = np.log((1 + t).cumprod() / nb)
        ax2.plot(gap_pr.index, gap_pr * 100, color=c_proxy, lw=2, label="代理(价格口径) − 指数")
        ax2.plot(gap_tr.index, gap_tr * 100, color=c_tr, lw=2, label="代理(全收益) − 指数")
        ax2.axhline(0, color=ink, lw=0.8)
        ax2.set_ylabel("累计对数收益差 %", fontsize=9, color=ink)
        ax2.grid(color=grid, lw=0.8); ax2.tick_params(colors=ink, labelsize=8)
        for s in ("top", "right"):
            ax2.spines[s].set_visible(False)
        if j == 0:
            ax2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("中证指数 PIT 代理重建校验(收盘到收盘,2021-01 ~ 2025-12;代理不含退市股)",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout()
    out = REPORTS / "proxy_vs_index.png"
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
