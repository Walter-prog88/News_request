"""INC-T6 执行脚本(预注册 research/INC_T6_PREREG.md §12)。

阶段(每阶段日志 logs/inc_t6_<stage>.log,逐臂落盘 cache/_tmp/inc_t6_arms/,可断点续跑):
  cells    场地掩码 + 覆盖统计(inc_t6_cells_coverage.csv;不含任何收益)
  vpscore  纯量价版 COMP_v5(去掉 3 个基本面交互成员)+ 24 员配方对缓存的复现核对
  arms     主口径 13 场地 × {Bench, U0, U1, U2, C1, C2} + 2×2 六臂与两个基准(只落逐日序列,不打印任何统计)
  stageA   开发窗门控 -> inc_t6_stageA_selection.csv(检验窗统计的前置条件,脚本强制)
  checks   §10 前置引擎断言 -> inc_t6_engine_checks.csv
  stageB   场地结果(inc_t6_cell_results.csv)、H1 / H2 判定数字
  overlay  2×2、变体、交互、成本 / 换手增量、财报季、压力窗(inc_t6_overlay_2x2.csv)
  sens     Q / 入围场地 / 2×2 的成本、打分、选优比例、排雷缓冲敏感性(臂)
  grid     五分位 / D1 定义 / 生效日 +1 的全场地敏感性(臂)
  gridtab  网格全部结果 -> inc_t6_grid.csv
  longhist §6.3 长历史 D1 叠加(inc_t6_longhist.csv)
  attrib   §6.2 A10 场地拆分 + Barra 辅助归因(inc_t6_attribution.csv)
  report   汇总判定 inc_t6_verdict.json、图 reports/inc_t6_*.png
  diagU0b  (审查修复)前置断言残差机理重做:与引擎同规则的无摩擦同频基准 -> inc_t6_engine_diag_enginelike.csv
  h1profit (审查修复,事后)H1′ 重叠统计 + 盈利股内部重做 -> inc_t6_h1prime_overlap.csv、inc_t6_h1_profit*.csv
  survivor (审查修复,事后)幸存者偏差同窗口对比 -> inc_t6_longhist_samewin.csv

用法:PYTHONPATH=src python3 scripts/inc_t6_run.py <stage> [--part i/n]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import inc_t6_common as C  # noqa: E402
from inc_t6_common import LEDGER, TMP, REPORTS, PFX, TD  # noqa: E402

ARGS = None


def _part(items: list) -> list:
    if not ARGS or not ARGS.part:
        return items
    i, n = (int(x) for x in ARGS.part.split("/"))
    return [x for j, x in enumerate(items) if j % n == i]


# ====================================================================== cells
def stage_cells():
    log = C.Logger("cells")
    p = C.panel()
    U = C.pool(p)
    Uv = U.to_numpy(bool)
    fv = C.fund_vars(p, 0)
    codes = C.build_cell_codes(p, U, fv)
    codes["D1fc"] = C.forecast_loss_code(p, U)
    fv1 = C.fund_vars(p, 1)
    c1 = C.build_cell_codes(p, U, fv1)
    for k in ("D1", "D2", "D3", "D4"):
        codes[f"{k}_l1"] = c1[k]
    C.save_codes(p, codes)
    log("场地编码落盘", C.CELLS_FILE)

    score = C.comp_score(p).to_numpy(np.float32)
    sfin = np.isfinite(score)
    yrs = p.dates.year
    rows = []
    # §3.4 核对数字(池规模、字段覆盖率、亏损股占比)
    for y in range(2020, 2026):
        r = yrs == y
        n = Uv[r].sum(1)
        cov = {k: float((Uv[r] & np.isfinite(fv[k].to_numpy()[r])).sum() / Uv[r].sum())
               for k in ("NI", "OPY", "MV", "ROE", "EP")}
        loss = float(((codes["D1"][r] == 0)).sum() / (codes["D1"][r] >= 0).sum())
        fcv = float((codes["D1fc"][r] >= 0).sum() / Uv[r].sum())
        rows.append(dict(kind="pool", cell="V1000", year=y, n_mean=float(n.mean()), share_of_pool=1.0,
                         cov_NI=cov["NI"], cov_OPY=cov["OPY"], cov_MV=cov["MV"], cov_ROE=cov["ROE"],
                         cov_EP=cov["EP"], loss_share=loss, cov_forecast_state=fcv))
    sets = {"main": C.cell_masks_from_codes(codes, "main"), "q5": C.cell_masks_from_codes(codes, "q5"),
            "d1q": C.cell_masks_from_codes(codes, "d1q"), "d1fc": C.cell_masks_from_codes(codes, "d1fc"),
            "lag1": C.cell_masks_from_codes({k: codes[f"{k}_l1"] for k in ("D1", "D2", "D3", "D4")}, "main")}
    size_rule = []
    for cs, masks in sets.items():
        for cell, M in masks.items():
            M = M & Uv
            ns = (M & sfin).sum(1)
            nn = M.sum(1)
            prev = np.vstack([np.zeros((1, M.shape[1]), bool), M[:-1]])
            exits = (prev & ~M & Uv).sum(1)
            for y in range(2020, 2026):
                r = yrs == y
                rows.append(dict(kind=cs, cell=cell, year=y, n_mean=float(nn[r].mean()),
                                 n_scored_mean=float(ns[r].mean()), n_scored_min=int(ns[r].min()),
                                 share_of_pool=float(nn[r].sum() / Uv[r].sum()),
                                 exit_rate_ann=float(exits[r].sum() / max(nn[r].mean(), 1e-9) * TD / r.sum())))
            for w, (a, b) in (("dev", C.WINDOWS["dev"]), ("test", C.WINDOWS["test"])):
                r = np.asarray((p.dates >= pd.Timestamp(a)) & (p.dates <= pd.Timestamp(b)))
                frac = float((ns[r] >= C.MIN_SCORED).mean())
                size_rule.append(dict(cellset=cs, cell=cell, window=w, frac_days_scored_ge300=frac,
                                      n_scored_median=float(np.median(ns[r])), ok=frac >= C.MIN_SCORED_FRAC))
    cov = pd.DataFrame(rows)
    sr = pd.DataFrame(size_rule)
    ok = sr.groupby(["cellset", "cell"])["ok"].all().rename("size_ok").reset_index()
    cov = cov.merge(ok, left_on=["kind", "cell"], right_on=["cellset", "cell"], how="left").drop(columns="cellset")
    cov.to_csv(LEDGER / f"{PFX}_cells_coverage.csv", index=False)
    sr.to_csv(LEDGER / f"{PFX}_cells_sizerule.csv", index=False)
    log(cov[cov.kind == "pool"].round(3).to_string())
    log(cov[cov.kind == "main"].pivot_table(index="cell", columns="year", values="n_scored_mean").round(0).to_string())
    log("最小规模规则不过的场地:", sr[~sr.ok][["cellset", "cell", "window", "frac_days_scored_ge300"]].to_string())


# ====================================================================== vpscore
def stage_vpscore():
    log = C.Logger("vpscore")
    from afactor import factors as FA, ops
    p = C.panel()
    FA.build_all()
    man = json.load(open(LEDGER / "comp_v5_members.json"))
    signs = man["signs"]
    fu3 = ["fx_fu_rev_yoy__P__ivol_20", "fx_fu_roe__P__ivol_20", "fx_fu_roe__P__id_ret_intraday_m60"]
    allm = [n for n in man["members"] if n in FA.REGISTRY]
    missing = [n for n in man["members"] if n not in FA.REGISTRY]
    vp = [n for n in allm if n not in fu3]
    log(f"COMP_v5 清单 {len(man['members'])} 员,REGISTRY 可建 {len(allm)} 员(缺 {missing});纯量价版 {len(vp)} 员")
    univ = p["universe"]
    T, N = len(p.dates), len(p.codes)
    acc = {"vp": [np.zeros((T, N), np.float64), np.zeros((T, N), np.int16)],
           "all": [np.zeros((T, N), np.float64), np.zeros((T, N), np.int16)]}
    for n in allm:
        t0 = time.time()
        f = FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes).replace([np.inf, -np.inf], np.nan)
        z = (ops.standardize(f, univ) * float(np.sign(signs.get(n, 1.0)))).to_numpy(np.float32)
        fin = np.isfinite(z)
        for key in (["vp", "all"] if n in vp else ["all"]):
            acc[key][0] += np.where(fin, z, 0.0)
            acc[key][1] += fin
        del f, z
        log(f"  {n:45s} {time.time() - t0:5.1f}s")
    out = {}
    for key, nm in (("vp", vp), ("all", allm)):
        s, c = acc[key]
        with np.errstate(invalid="ignore", divide="ignore"):
            v = (s / c).astype(np.float32)
        v[c < max(1, len(nm) // 2)] = np.nan
        out[key] = pd.DataFrame(v, index=p.dates, columns=p.codes)
    out["vp"].to_parquet(C.vp_score_path())
    ref = C.comp_score(p)
    a, b = out["all"].to_numpy(np.float64), ref.to_numpy(np.float64)
    both = np.isfinite(a) & np.isfinite(b)
    chk = dict(n_members_registry=len(allm), n_members_vp=len(vp), missing=";".join(missing),
               nan_agree=float((np.isfinite(a) == np.isfinite(b)).mean()),
               max_abs_diff=float(np.abs(a[both] - b[both]).max()),
               corr=float(np.corrcoef(a[both], b[both])[0, 1]))
    vv = out["vp"].to_numpy(np.float64)
    both2 = np.isfinite(vv) & np.isfinite(b)
    chk["corr_vp_vs_comp"] = float(np.corrcoef(vv[both2], b[both2])[0, 1])
    pd.DataFrame([chk]).to_csv(LEDGER / f"{PFX}_vpscore_check.csv", index=False)
    log("配方复现核对", chk)


# ====================================================================== arms
def _scores(p, which=("comp", "C1", "C2")) -> dict:
    out = {}
    if "comp" in which:
        out["comp"] = C.comp_score(p).to_numpy(np.float32)
    if "vp" in which:
        out["vp"] = pd.read_parquet(C.vp_score_path()).reindex(index=p.dates, columns=p.codes).to_numpy(np.float32)
    if "C1" in which or "C2" in which:
        cl = C.clone_scores(p)
        for k in ("C1", "C2"):
            if k in which:
                out[k] = cl[k].reindex(index=p.dates, columns=p.codes).to_numpy(np.float32)
    return out


HOLD_CELLS = {"D1_loss", "D1_profit", "Q"}


def run_cell_set(p, log, cellset: str, masks: dict, Uv, S: dict, arms=("U0", "U1", "U2", "C1", "C2"),
                 cost="Y1", score_key="comp", tag_sfx="", hold=frozenset(), top_buf=C.TOP_BUF,
                 excl_buf=C.EXCL_BUF):
    for cell in _part(list(masks)):
        M = masks[cell] & Uv
        C.run_bench(p, f"{cellset}__{cell}", M)
        for arm in arms:
            tag = f"{cellset}__{cell}__{arm}__{cost}{tag_sfx}"
            if C.arm_path(tag).exists():
                continue
            if arm == "U0":
                W = C.w_const(p, M)
            elif arm == "U1":
                W = C.w_exclude(S[score_key], M, excl_buf)
            elif arm == "U2":
                W = C.w_top(S[score_key], M, top_buf)
            elif arm in ("C1", "C2"):
                W = C.w_top(S[arm], M, top_buf)
            else:
                raise ValueError(arm)
            C.run_arm(p, tag, W, cost=cost, keep_holdings=(cell in hold and arm in ("U1", "U2")), log=log)
            del W


def overlay_weights(p, Uv, loss: np.ndarray, S: np.ndarray, arm: str, excl_buf=C.EXCL_BUF) -> np.ndarray:
    nonloss = Uv & ~loss
    if arm == "A00":
        return C.w_const(p, Uv)
    if arm == "A10":
        return C.w_exclude(S, Uv, excl_buf)
    if arm == "A01":
        return C.w_const(p, nonloss)
    if arm == "A11":
        return C.w_exclude(S, nonloss, excl_buf)
    if arm == "A11n":
        return C.w_merge_groups([C.w_exclude(S, Uv & loss, C.EXCL_BUF), C.w_exclude(S, nonloss, C.EXCL_BUF)])
    if arm == "A11d":
        return C.w_merge_groups([C.w_exclude(S, Uv & loss, C.EXCL_BUF_D), C.w_exclude(S, nonloss, C.EXCL_BUF)])
    raise ValueError(arm)


def run_overlay(p, log, loss: np.ndarray, Uv, S: dict, prefix="ovl", arms=("A00", "A10", "A01", "A11", "A11n", "A11d"),
                cost="Y1", score_key="comp", capital=C.AUM, tag_sfx="", excl_buf=C.EXCL_BUF, hold=frozenset()):
    C.run_bench(p, f"{prefix}__all", Uv)
    C.run_bench(p, f"{prefix}__nonloss", Uv & ~loss)
    for arm in _part(list(arms)):
        tag = f"{prefix}__{arm}__{cost}{tag_sfx}"
        if C.arm_path(tag).exists():
            continue
        W = overlay_weights(p, Uv, loss, S[score_key], arm, excl_buf)
        C.run_arm(p, tag, W, cost=cost, capital=capital, keep_holdings=arm in hold, log=log)
        del W


def stage_arms():
    log = C.Logger("arms")
    log("只落逐日序列,不打印任何收益统计(阶段 A 前不得计算检验窗统计)")
    p = C.panel()
    Uv = C.pool(p).to_numpy(bool)
    codes = C.load_codes(p)
    masks = C.cell_masks_from_codes(codes, "main")
    S = _scores(p)
    C.cost_model(p, "Y1")
    run_overlay(p, log, masks["D1_loss"], Uv, S, hold={"A10"})
    run_cell_set(p, log, "main", masks, Uv, S, hold=HOLD_CELLS)
    log("arms 完成")


# ====================================================================== precheck(只用开发窗)
def stage_precheck():
    """阶段 A 之前的引擎健康检查,只用开发窗(2021-2023),不触碰检验窗:
    U0 / A00 / A01 相对各自同频基准 ≈ −成本、对齐峰值、exposure、A10 复现 capacity_grid 开发窗。
    正式的 §10 前置断言(2021-2025)在 checks 阶段、阶段 A 落盘之后做。"""
    log = C.Logger("precheck")
    rows = []

    def u0(item, arm, bench):
        d = C.load_arm(arm); b = C.load_bench(bench)
        st = C.arm_window_stats(d, b, "dev")
        a_, b_ = d["ret"].align(b, join="inner")
        k, cs = C.align_peak(C.wslice(a_, "dev"), C.wslice(b_, "dev"))
        rows.append(dict(item=item, window="dev", ex_ann=st["ex_ann"], cost_ann=st["cost_ann"],
                         resid_pp=(st["ex_ann"] + st["cost_ann"]) * 100, align_k=k, exposure_max=st["exposure_max"]))
    for cell in C.CELLS:
        u0(cell, f"main__{cell}__U0__Y1", f"main__{cell}")
    u0("A00", "ovl__A00__Y1", "ovl__all")
    u0("A01 vs 剔亏损池", "ovl__A01__Y1", "ovl__nonloss")
    d = C.load_arm("ovl__A10__Y1"); b = C.load_bench("ovl__all")
    st = C.arm_window_stats(d, b, "dev")
    rows.append(dict(item="A10 复现 capacity_grid(dev)", window="dev", ex_geo=st["ex_geo"], ref=C.CAP_REF["dev"],
                     dev_pp=(st["ex_geo"] - C.CAP_REF["dev"]) * 100, cost_ann=st["cost_ann"],
                     turnover_daily=st["turnover_daily"]))
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / f"{PFX}_precheck_dev.csv", index=False)
    log(df.round(5).to_string())


# ====================================================================== stageA
def stage_stageA():
    log = C.Logger("stageA")
    if C.stageA_done():
        log("阶段 A 名单已存在,不重写:", C.STAGEA_FILE)
        return
    rows = []
    for cell in C.STAGEA_CELLS:
        b = C.load_bench(f"main__{cell}")
        d = {a: C.load_arm(f"main__{cell}__{a}__Y1") for a in ("U1", "U2", "C1", "C2")}
        e = {a: C.arm_window_stats(d[a], b, "dev")["ex_ann"] for a in d}
        delta = C.diff_stats(d["U2"]["ret"] - d["U1"]["ret"], "dev")
        r = dict(cell=cell, e_U1=e["U1"], e_U2=e["U2"], e_C1=e["C1"], e_C2=e["C2"],
                 delta=delta["ann"], delta_t=delta["t_nw"], delta_lo95=delta["lo95"], delta_hi95=delta["hi95"],
                 u2_minus_c1=C.diff_stats(d["U2"]["ret"] - d["C1"]["ret"], "dev")["ann"],
                 u2_minus_c2=C.diff_stats(d["U2"]["ret"] - d["C2"]["ret"], "dev")["ann"])
        r["pass_rule"] = bool(r["delta"] > 0 and r["delta_t"] > 2 and r["e_U2"] > max(r["e_C1"], r["e_C2"]))
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("delta_t", ascending=False)
    passed = df[df.pass_rule].head(2)
    df["selected"] = df.cell.isin(passed.cell)
    df["window"] = "dev 2021-01-04~2023-12-31"
    df["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S JST")
    df["rule"] = "Δ_c>0 且 NW t>2(bw10) 且 e_U2>max(e_C1,e_C2);按 t 取最高两个"
    df.to_csv(C.STAGEA_FILE, index=False)
    log(df.round(4).to_string())
    log("入围:", list(passed.cell) if len(passed) else "无")


def selected_cells() -> list:
    C.require_stageA("selected_cells")
    df = pd.read_csv(C.STAGEA_FILE)
    return list(df[df.selected].cell)


# ====================================================================== checks
def stage_checks():
    log = C.Logger("checks")
    C.require_stageA("checks")
    rows = []

    def u0_row(item, arm, bench):
        d = C.load_arm(arm); b = C.load_bench(bench)
        st = C.arm_window_stats(d, b, "full")
        resid = st["ex_ann"] + st["cost_ann"]
        k, cs = C.align_peak(d["ret"], b)
        same_idx = bool(d.index.equals(b.index))
        rows.append(dict(check="U0_vs_bench", item=item, ex_ann=st["ex_ann"], cost_ann=st["cost_ann"],
                         resid_pp=resid * 100, pass_resid=abs(resid) <= 0.003, align_k=k, pass_align=k == 0,
                         corr_k0=cs[0], same_index=same_idx, exposure_max=st["exposure_max"],
                         pass_exposure=st["exposure_max"] <= 1 + 1e-6))
    for cell in C.CELLS:
        u0_row(cell, f"main__{cell}__U0__Y1", f"main__{cell}")
    u0_row("A00(全池)", "ovl__A00__Y1", "ovl__all")
    u0_row("A01(剔亏损池)vs 自身池", "ovl__A01__Y1", "ovl__nonloss")
    # exposure:全部主口径臂
    expmax = []
    for f in sorted(C.ARMS_DIR.glob(f"{PFX}_arm_*.parquet")):
        d = pd.read_parquet(f, columns=["exposure_max_phase"])
        expmax.append(float(d["exposure_max_phase"].max()))
    rows.append(dict(check="exposure_all_arms", item=f"{len(expmax)} 个臂", exposure_max=max(expmax),
                     pass_exposure=max(expmax) <= 1 + 1e-6))
    # A10 复现 capacity_grid
    d = C.load_arm("ovl__A10__Y1"); b = C.load_bench("ovl__all")
    for w in ("full", "dev", "test"):
        st = C.arm_window_stats(d, b, w)
        rows.append(dict(check="A10_reproduce_capacity_grid", item=w, ex_geo=st["ex_geo"], ref=C.CAP_REF[w],
                         ref_ride=C.CAP_REF_RIDE[w], dev_pp=(st["ex_geo"] - C.CAP_REF[w]) * 100,
                         pass_resid=(abs(st["ex_geo"] - C.CAP_REF[w]) <= 0.003) if w == "full" else np.nan,
                         turnover_daily=st["turnover_daily"], cost_ann=st["cost_ann"]))
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / f"{PFX}_engine_checks.csv", index=False)
    req = df[df.check.isin(["U0_vs_bench", "exposure_all_arms"]) | ((df.check == "A10_reproduce_capacity_grid") & (df.item == "full"))]
    ok = True
    for c in ("pass_resid", "pass_align", "pass_exposure"):
        v = req[c].dropna()
        ok &= bool(v.astype(bool).all())
    log(df.round(5).to_string())
    log("引擎前置断言:", "全部通过" if ok else "未通过 -> 引擎未过,不判定")
    json.dump(dict(engine_ok=ok), open(TMP / f"{PFX}_engine_ok.json", "w"))


# ====================================================================== diagU0(前置断言不过时的诊断)
def stage_diagU0():
    """U0 对场地同频基准的残差(超额 + 成本)在差场地 > 0.3pp 的来源诊断(不改任何判定规则)。
    假说:基准不可投资——调仓日按成交日 te 的一字板价"买入"(一字跌停的股票被基准按跌停价补回等权),
    引擎不能成交一字板(dead = one_word_up | one_word_down)。诊断基准 = 调仓日剔除 te 日一字板的同频等权
    (universe_override[t] = 场地[t] ∩ ~dead[t+1];只用于诊断,用到 te 当天信息,不得作为基准)。
    若 U0 毛收益 − 诊断基准 ≈ 0,则残差来自基准可成交性,不是引擎缺陷。
    ⚠ 审查后更正:此处'剔池'诊断的机理假说(一字跌停补仓)已被 diagU0b 证伪——剔池等于把已持仓一并卖光,不是引擎行为;
    残差来自一字涨停与停牌。本阶段保留仅为追溯,判断以 diagU0b 为准。"""
    log = C.Logger("diagU0")
    C.require_stageA("diagU0")
    from afactor import strategy as ST
    p = C.panel()
    Uv = C.pool(p).to_numpy(bool)
    codes = C.load_codes(p)
    masks = C.cell_masks_from_codes(codes, "main")
    dead = (p["one_word_up"] | p["one_word_down"]).reindex(index=p.dates, columns=p.codes).fillna(False).to_numpy(bool)
    dn = p["one_word_down"].reindex(index=p.dates, columns=p.codes).fillna(False).to_numpy(bool)
    up = p["one_word_up"].reindex(index=p.dates, columns=p.codes).fillna(False).to_numpy(bool)
    dead_te = np.vstack([dead[1:], np.zeros((1, dead.shape[1]), bool)])
    dn_te = np.vstack([dn[1:], np.zeros((1, dn.shape[1]), bool)])
    up_te = np.vstack([up[1:], np.zeros((1, up.shape[1]), bool)])
    rows = []
    items = [(c, f"main__{c}__U0__Y1", f"main__{c}", masks[c] & Uv) for c in C.CELLS]
    items += [("A00(全池)", "ovl__A00__Y1", "ovl__all", Uv),
              ("A01(剔亏损池)", "ovl__A01__Y1", "ovl__nonloss", Uv & ~masks["D1_loss"])]
    for item, arm, bench, M in items:
        d = C.load_arm(arm); b = C.load_bench(bench)
        out = dict(item=item)
        for nm, excl in (("no_dead", dead_te), ("no_down", dn_te), ("no_up", up_te)):
            bd = ST.benchmark(p, C.VERSION, C.MODE, rebalance=C.REBALANCE,
                              universe_override=C.to_frame(p, M & ~excl))["rets"]
            for w in ("dev", "test", "full"):
                g = C.wslice(d["gross_ret"], w)
                bb = C.wslice(b.reindex(d.index), w); bdd = C.wslice(bd.reindex(d.index), w)
                out[f"{w}_gross_vs_bench_pp"] = float((g - bb).mean() * TD * 100)
                out[f"{w}_gross_vs_bench_{nm}_pp"] = float((g - bdd).mean() * TD * 100)
                out[f"{w}_bench_minus_bench_{nm}_pp"] = float((bb - bdd).mean() * TD * 100)
        r = np.asarray((p.dates >= pd.Timestamp("2021-01-04")))
        out["dead_share_in_cell_full"] = float((M[r] & dead_te[r]).sum() / max(M[r].sum(), 1))
        out["down_share_in_cell_full"] = float((M[r] & dn_te[r]).sum() / max(M[r].sum(), 1))
        rows.append(out)
        log(item, {k: round(v, 3) for k, v in out.items() if k.startswith("full") or k.endswith("full")})
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / f"{PFX}_engine_diag.csv", index=False)
    log(df.round(3).to_string())


ENGINE_LIKE_VARIANTS = [   # (名称, 说明);不可交易集合 = 并集,再加 te 无价
    ("E_fin", "仅 te 无价"),
    ("E_dn", "一字跌停"),
    ("E_up", "一字涨停"),
    ("E_zero", "窗口成交额=0/缺(停牌,前向填充价)"),
    ("E_upzero", "一字涨停 ∪ 成交额=0"),
    ("E_all", "一字板(涨停∪跌停) ∪ 成交额=0 = 引擎 can 规则"),
]


def stage_diagU0b():
    """审查 P1 修复:前置断言残差的机理诊断重做。
    T6 原诊断 diagU0 把"te 日一字板"股票从基准**池**里剔掉(no_dead / no_down / no_up),这改变了基准的成分,
    不是引擎的行为:引擎对不可成交的持仓保留漂移权重(stuck)、对不可成交的非持仓不买。
    本阶段用与引擎同口径的无摩擦同频基准(common.bench_engine_like):调仓日不可交易的股票保留漂移权重,其余预算
    在可交易股票上等权,与官方基准同日历同漂移。核对:notrade=∅ 时与官方基准逐日最大差应为 0。
    对 15 项(13 场地 U0 + A00 + A01)× 6 种不可交易集合 × 3 窗口报告 U0 毛收益 − 基准(pp/年)。"""
    log = C.Logger("diagU0b")
    C.require_stageA("diagU0b")
    from afactor import strategy as ST
    p = C.panel()
    Uv = C.pool(p).to_numpy(bool)
    codes = C.load_codes(p)
    masks = C.cell_masks_from_codes(codes, "main")
    px, amt = ST.exec_prices(p, C.MODE)
    P = px.reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
    A = amt.reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
    dn = p["one_word_down"].reindex(index=p.dates, columns=p.codes).fillna(False).to_numpy(bool)
    up = p["one_word_up"].reindex(index=p.dates, columns=p.codes).fillna(False).to_numpy(bool)
    with np.errstate(invalid="ignore"):
        zero = ~(A > 0)
    sets = {"E_fin": np.zeros_like(dn), "E_dn": dn, "E_up": up, "E_zero": zero, "E_upzero": up | zero,
            "E_all": up | dn | zero}
    T = len(p.dates)
    idx = p.dates[:T - 2]
    items = [(c, f"main__{c}__U0__Y1", f"main__{c}", masks[c] & Uv) for c in C.CELLS]
    items += [("A00(全池)", "ovl__A00__Y1", "ovl__all", Uv),
              ("A01(剔亏损池)", "ovl__A01__Y1", "ovl__nonloss", Uv & ~masks["D1_loss"])]
    rows, cov = [], []
    r0 = np.asarray(p.dates >= pd.Timestamp("2021-01-04"))
    for item, arm, bench, M in items:
        d = C.load_arm(arm); b = C.load_bench(bench)
        rep = pd.Series(C.bench_engine_like(P, M, None, engine_like=False), index=idx)
        maxdiff = float((rep.reindex(b.index) - b).abs().max())
        res = {}
        for nm, ns in sets.items():
            res[nm] = pd.Series(C.bench_engine_like(P, M, ns, engine_like=True), index=idx)
        for w in ("dev", "test", "full"):
            g = C.wslice(d["gross_ret"], w)
            bb = C.wslice(b.reindex(d.index), w)
            rows.append(dict(item=item, window=w, variant="官方基准", resid_pp=float((g - bb).mean() * TD * 100),
                             bench_minus_official_pp=0.0, replica_maxdiff=maxdiff))
            for nm, s in res.items():
                bs = C.wslice(s.reindex(d.index), w)
                rows.append(dict(item=item, window=w, variant=nm, resid_pp=float((g - bs).mean() * TD * 100),
                                 bench_minus_official_pp=float((bs - bb).mean() * TD * 100), replica_maxdiff=maxdiff))
        # 场地内股票-日占比:信号日 t 的场地成员在成交日 te=t+1 属于各类不可交易集合的比例(2021-2025)
        base = max(int(M[r0].sum()), 1)
        row = dict(item=item)
        with np.errstate(invalid="ignore", divide="ignore"):
            Gte = P[2:] / P[1:T - 1] - 1.0                      # 行 t = te(=t+1) -> te+1 的持有收益
        Mt = M[:T - 2] & r0[:T - 2, None] & np.isfinite(Gte)
        row["mean_G_cell_bp"] = float(Gte[Mt].mean() * 1e4)
        for nm, ns in sets.items():
            if nm == "E_fin":
                continue
            fl = ns[1:T - 1]                                    # 行 t = te 的标志
            row[f"share_{nm}"] = float((M & np.vstack([ns[1:], np.zeros((1, ns.shape[1]), bool)]))[r0].sum()) / base
            row[f"meanG_flag_{nm}_bp"] = float(Gte[Mt & fl].mean() * 1e4) if (Mt & fl).any() else np.nan
        cov.append(row)
        log(item, f"replica_maxdiff={maxdiff:.2e}",
            {nm: round(float(((C.wslice(d['gross_ret'], 'full')) - C.wslice(s.reindex(d.index), 'full')).mean() * TD * 100), 3)
             for nm, s in res.items()})
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / f"{PFX}_engine_diag_enginelike.csv", index=False)
    pd.DataFrame(cov).to_csv(LEDGER / f"{PFX}_engine_diag_coverage.csv", index=False)
    pv = df[df.window == "full"].pivot(index="item", columns="variant", values="resid_pp")
    log(pv.round(3).to_string())


# ====================================================================== stageB
WINS_ALL =["dev", "test", "full", "y2021", "y2022", "y2023", "y2024", "y2025", "stress"]


def cell_result_rows(cellset: str, cell: str, arms, cost="Y1", tag_sfx="", wins=WINS_ALL, extra=None) -> list:
    rows = []
    b = C.load_bench(f"{cellset}__{cell}")
    for arm in arms:
        tag = f"{cellset}__{cell}__{arm}__{cost}{tag_sfx}"
        if not C.arm_path(tag).exists():
            continue
        d = C.load_arm(tag)
        for w in wins:
            st = C.arm_window_stats(d, b, w)
            rows.append(dict(cellset=cellset, cell=cell, arm=arm, cost=cost, variant=tag_sfx.strip("_") or "main",
                             window=w, **(extra or {}), **st))
    return rows


def paired_rows(cellset, cell, a_tag, b_tag, label, wins=WINS_ALL, extra=None) -> list:
    rows = []
    a, b = C.load_arm(a_tag)["ret"], C.load_arm(b_tag)["ret"]
    for w in wins:
        st = C.diff_stats(a - b, w)
        rows.append(dict(cellset=cellset, cell=cell, comparison=label, window=w, **(extra or {}), **st))
    return rows


# ====================================================================== h1robust(偏离登记:事后稳健性,不改原规则判定)
def _engine_like_all_bench(p, M: np.ndarray) -> pd.Series:
    """与引擎同口径的无摩擦同频基准(调仓日不可交易 = 一字板 ∪ 成交额=0 ∪ te 无价的股票保留漂移权重),见 common.bench_engine_like。"""
    from afactor import strategy as ST
    px, amt = ST.exec_prices(p, C.MODE)
    P = px.reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
    A = amt.reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
    dead = (p["one_word_up"] | p["one_word_down"]).reindex(index=p.dates, columns=p.codes).fillna(False).to_numpy(bool)
    with np.errstate(invalid="ignore"):
        nt = dead | ~(A > 0)
    return pd.Series(C.bench_engine_like(P, M, nt, engine_like=True), index=p.dates[:len(p.dates) - 2])


def load_enginelike_bench(p, tag: str, M: np.ndarray) -> pd.Series:
    f = C.bench_path(f"diag_enginelike__{tag}")
    if f.exists():
        return pd.read_parquet(f)["ret"]
    b = _engine_like_all_bench(p, M)
    C._atomic_parquet(b.rename("ret").to_frame(), f)
    return b


def stage_h1robust():
    """前置断言在差场地不过后,H1 / H1′ 的去摩擦读法(事后,不改原规则判定):
      adjA :e_U1 改对"可交易全场地等权" = r_U1(净) − r_U0(毛)。U0 毛收益与基准之差是纯摩擦,从每个场地的排雷超额里扣掉它。
      adjB :e_U1 改对与引擎同口径的无摩擦同频基准(审查 P1 修复后的定义):调仓日不可交易(一字板 ∪ 成交额=0 ∪ 无价)
            的股票保留漂移权重,不被重置回等权;U0 毛收益对它的残差 ≈ 0(diagU0b)。用到 te 当天信息,只作诊断。
      adjB0:修复前的 adjB(调仓日把 te 一字跌停股从基准池里剔掉)。机理有误(见 diagU0b),保留仅为追溯,不用于任何判断。
    原规则结果(对 Bench_c)照列于 inc_t6_cell_paired.csv,本表只是附加读法。"""
    log = C.Logger("h1robust")
    C.require_stageA("h1robust")
    from afactor import strategy as ST
    p = C.panel()
    Uv = C.pool(p).to_numpy(bool)
    codes = C.load_codes(p)
    masks = C.cell_masks_from_codes(codes, "main")
    dn = p["one_word_down"].reindex(index=p.dates, columns=p.codes).fillna(False).to_numpy(bool)
    dn_te = np.vstack([dn[1:], np.zeros((1, dn.shape[1]), bool)])
    cells = sorted({c for a, b, _ in C.H1_PAIRS for c in (a, b)})
    E = {}
    for c in cells:
        u1 = C.load_arm(f"main__{c}__U1__Y1")["ret"]
        u0 = C.load_arm(f"main__{c}__U0__Y1")
        b = C.load_bench(f"main__{c}")
        f = C.bench_path(f"diag_nodown__{c}")
        if f.exists():
            bd = pd.read_parquet(f)["ret"]
        else:
            bd = ST.benchmark(p, C.VERSION, C.MODE, rebalance=C.REBALANCE,
                              universe_override=C.to_frame(p, masks[c] & Uv & ~dn_te))["rets"]
            C._atomic_parquet(bd.rename("ret").to_frame(), f)
        be = load_enginelike_bench(p, c, masks[c] & Uv)
        E[c] = {"orig": u1 - b.reindex(u1.index), "adjA": u1 - u0["gross_ret"].reindex(u1.index),
                "adjB": u1 - be.reindex(u1.index), "adjB0": u1 - bd.reindex(u1.index)}
    rows = []
    for a, bb, lab in C.H1_PAIRS:
        for k in ("orig", "adjA", "adjB", "adjB0"):
            for w in WINS_ALL:
                rows.append(dict(pair=f"{a}−{bb}", label=lab, reading=k, window=w,
                                 **C.diff_stats(E[a][k] - E[bb][k], w)))
    for c in cells:
        for k in ("orig", "adjA", "adjB", "adjB0"):
            for w in ("dev", "test", "full"):
                rows.append(dict(pair=c, label="e_U1 单场地", reading=k, window=w, **C.diff_stats(E[c][k], w)))
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / f"{PFX}_h1_robust.csv", index=False)
    log(df[df.window.isin(["dev", "test"])][["pair", "label", "reading", "window", "ann", "t_nw", "lo95", "hi95"]]
        .round(4).to_string())


PROFIT_PAIRS = [("PD2_low", "PD2_high", "下滑−增长(盈利股内)"), ("PD3_low", "PD3_high", "低ROE−高ROE(盈利股内)"),
                ("PQm", "PQ", "Q⁻−Q(盈利股内)"), ("PD4_low", "PD4_high", "贵−便宜(盈利股内)")]


def stage_h1profit():
    """审查 P2 修复(事后,只报告):H1′ 四对与 D1 亏损场地高度重叠(低 ROE / 贵 / Q⁻ 场地里 63~100% 是亏损股,对照端 0%)。
    (a) h1prime_overlap:各场地的亏损股占比(场地成员中 TTM 归母净利 < 0 的比例,2021-2025 逐日平均);
    (b) 在盈利股(TTM 归母净利 >= 0)内部重做:ROE / 营业利润同比 / EP 的三分位改为在"池 ∩ 盈利"内计算,
        场地 = 低/高三分位及 Q = ROE高∩增长、Q⁻ = ROE低∩下滑;各场地只跑 U0、U1(5 日相位、Y=1、1000 万,同主口径),
        对场地官方同频基准与引擎同口径无摩擦基准各算一遍。"""
    log = C.Logger("h1profit")
    C.require_stageA("h1profit")
    p = C.panel()
    Uv = C.pool(p).to_numpy(bool)
    codes = C.load_codes(p)
    masks = C.cell_masks_from_codes(codes, "main")
    r0 = np.asarray(p.dates >= pd.Timestamp("2021-01-04"))
    d1 = codes["D1"]
    # (a) 重叠
    rows = []
    for c in ["D1_loss"] + sorted({x for a, b, _ in C.H1_PAIRS[1:] for x in (a, b)}):
        M = masks[c] & Uv
        n = M[r0].sum(1)
        loss_n = (M & (d1 == 0))[r0].sum(1)
        ok = n > 0
        rows.append(dict(cell=c, n_mean=float(n.mean()), loss_share_mean=float((loss_n[ok] / n[ok]).mean()),
                         loss_share_pooled=float(loss_n.sum() / n.sum()),
                         share_of_all_loss_stocks=float(loss_n.sum() / max((((d1 == 0) & Uv)[r0]).sum(), 1))))
    ov = pd.DataFrame(rows)
    ov.to_csv(LEDGER / f"{PFX}_h1prime_overlap.csv", index=False)
    log(ov.round(4).to_string())
    # (b) 盈利股内部
    fv = C.fund_vars(p, 0)
    ni = fv["NI"].to_numpy(np.float64)
    prof = Uv & np.isfinite(ni) & (ni >= 0)
    Upf = C.to_frame(p, prof)
    c2, c3, c4 = (C.tercile_code(fv[k], Upf) for k in ("OPY", "ROE", "EP"))
    pm = {"PD2_low": c2 == 1, "PD2_high": c2 == 3, "PD3_low": c3 == 1, "PD3_high": c3 == 3,
          "PD4_low": c4 == 1, "PD4_high": c4 == 3, "PQm": (c3 == 1) & (c2 == 1), "PQ": (c3 == 3) & (c2 == 3)}
    S = _scores(p, which=("comp",))
    C.cost_model(p, "Y1")
    run_cell_set(p, log, "prof", pm, Uv, S, arms=("U0", "U1"))
    E, resid, size = {}, [], []
    for c, M in pm.items():
        M = M & Uv
        n = M[r0].sum(1)
        size.append(dict(cell=c, n_mean=float(n.mean()), n_min=int(n.min())))
        u0 = C.load_arm(f"prof__{c}__U0__Y1")
        u1 = C.load_arm(f"prof__{c}__U1__Y1")["ret"]
        b = C.load_bench(f"prof__{c}")
        be = load_enginelike_bench(p, f"prof__{c}", M)
        E[c] = {"orig": u1 - b.reindex(u1.index), "adjB": u1 - be.reindex(u1.index)}
        for w in ("dev", "test", "full"):
            g = C.wslice(u0["gross_ret"], w)
            resid.append(dict(cell=c, window=w, U0_gross_minus_bench_pp=float((g - C.wslice(b.reindex(u0.index), w)).mean() * TD * 100),
                              U0_gross_minus_enginelike_pp=float((g - C.wslice(be.reindex(u0.index), w)).mean() * TD * 100)))
    rows = []
    for a, bb, lab in PROFIT_PAIRS:
        for k in ("orig", "adjB"):
            for w in WINS_ALL:
                rows.append(dict(pair=f"{a}−{bb}", label=lab, reading=k, window=w, **C.diff_stats(E[a][k] - E[bb][k], w)))
    for c in pm:
        for k in ("orig", "adjB"):
            for w in ("dev", "test", "full"):
                rows.append(dict(pair=c, label="e_U1 单场地", reading=k, window=w, **C.diff_stats(E[c][k], w)))
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / f"{PFX}_h1_profit.csv", index=False)
    pd.DataFrame(size).merge(pd.DataFrame(resid).pivot(index="cell", columns="window").pipe(
        lambda x: x.set_axis([f"{a}_{b}" for a, b in x.columns], axis=1)).reset_index(), on="cell").to_csv(
        LEDGER / f"{PFX}_h1_profit_cells.csv", index=False)
    log(pd.DataFrame(size).round(1).to_string())
    log(pd.DataFrame(resid).round(3).to_string())
    log(df[df.window.isin(["dev", "test", "full"]) & ~df.label.eq("e_U1 单场地")][
        ["pair", "reading", "window", "ann", "t_nw", "lo95", "hi95"]].round(4).to_string())


SURV_WINS = {"y2021_2025": ("2021-01-04", "2025-12-31"), "y2020_2025": ("2020-01-02", "2025-12-31")}


def stage_survivor():
    """审查 P2 修复(事后,只报告):幸存者偏差量级用同窗口对比。长历史(含退市股)与"只保留研究面板股票"两种池,
    同一窗口(2021-2025 与 2020-2025)、同一执行价(LongPanel 日开盘价)、同一 q92 池与换手率版排雷,
    比较"剔亏损池基准 − 全池基准"与 2×2(A00 / A10 / A01 / A11);并列研究期面板(V1000,open10)的同窗口数字。"""
    log = C.Logger("survivor")
    C.require_stageA("survivor")
    from afactor import lh_engine as LE, fundamental as FU, strategy as ST
    X = LE.load_inputs()
    dates, codes = X.dates, X.codes
    rep = FU.build_report_level()
    U = LE.pool(X, "q92")
    Uv = U.to_numpy(bool)
    ni = FU._pit_daily(rep, "n_income_attr_p_ttm", dates, codes, lag_days=0).to_numpy(np.float64)
    loss = Uv & np.isfinite(ni) & (ni < 0)
    pcodes = set(C.panel().codes)
    surv_col = np.array([c in pcodes for c in codes])
    log(f"长历史股票 {len(codes)} 只,其中研究面板内 {int(surv_col.sum())} 只;池内股票日中退市/不在研究面板的占比 "
        f"{float((Uv & ~surv_col[None, :]).sum() / Uv.sum()):.4f}")
    ad = LE.adapter(X)
    S = X.df("score_S1")
    shape = LE._Shape(dates, codes)
    rows = []

    def bw(score, Um, excl):
        sc = score if score is not None else pd.DataFrame(1.0, index=dates, columns=codes)
        return ST.build_weights(sc, shape, "V1000", scheme="exclude", exclude_frac=excl,
                                buffer=(C.EXCL_BUF if score is not None else None),
                                universe_override=pd.DataFrame(Um, index=dates, columns=codes))

    def st_(d, a, b):
        x = d.dropna(); x = x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))]
        se = C.nw_se_mean(x.to_numpy())
        return dict(n_days=len(x), ann=float(x.mean() * TD), t_nw=float(x.mean() / se) if se else np.nan,
                    lo95=float((x.mean() - C.Z95 * se) * TD), hi95=float((x.mean() + C.Z95 * se) * TD))
    for pool_name, col_ok in (("含退市股(全部)", np.ones(len(codes), bool)), ("只保留研究面板股票", surv_col)):
        Ua = Uv & col_ok[None, :]
        La = loss & col_ok[None, :]
        nonloss = Ua & ~La
        b = LE.bench(ad, pd.DataFrame(Ua, index=dates, columns=codes), 5)
        bn = LE.bench(ad, pd.DataFrame(nonloss, index=dates, columns=codes), 5)
        R = {}
        for name, Um, sc in (("A00", Ua, None), ("A10", Ua, S), ("A01", nonloss, None), ("A11", nonloss, S)):
            f = TMP / f"{PFX}_lh_surv_{'all' if col_ok.all() else 'surv'}_{name}.parquet"
            if f.exists():
                R[name] = pd.read_parquet(f)["ret"]
            else:
                t0 = time.time()
                res = LE.run(bw(sc, Um, 0.0 if sc is None else 0.3), ad, rebalance=5, cost="Y1", record=False)
                R[name] = res["rets"]
                pd.DataFrame({"ret": R[name]}).to_parquet(f)
                log(f"  {pool_name} {name} {time.time() - t0:.0f}s")
        items = {"剔亏损池基准−全池基准": bn - b, "筛子=A01−A00": R["A01"] - R["A00"], "d=A11−A10": R["A11"] - R["A10"],
                 "A10−B": R["A10"] - b, "A11−B": R["A11"] - b}
        for w, (a, bb) in {**SURV_WINS, "lh_2016_2019": ("2016-07-01", "2019-12-31")}.items():
            for k, sr in items.items():
                rows.append(dict(table="longhist_samewin", pool=pool_name, window=w, item=k, **st_(sr, a, bb)))
    # 研究期面板(V1000,open10)的同窗口数字:基准层面与 2×2
    for w, (a, bb) in SURV_WINS.items():
        bn_, b_ = C.load_bench("ovl__nonloss"), C.load_bench("ovl__all")
        pairs = {"剔亏损池基准−全池基准": bn_ - b_,
                 "筛子=A01−A00": C.load_arm("ovl__A01__Y1")["ret"] - C.load_arm("ovl__A00__Y1")["ret"],
                 "d=A11−A10": C.load_arm("ovl__A11__Y1")["ret"] - C.load_arm("ovl__A10__Y1")["ret"]}
        for k, sr in pairs.items():
            rows.append(dict(table="research_panel", pool="研究面板 V1000 open10", window=w, item=k, **st_(sr, a, bb)))
    out = pd.DataFrame(rows)
    out.to_csv(LEDGER / f"{PFX}_longhist_samewin.csv", index=False)
    log(out.assign(ann_pp=out.ann * 100).round(3)[["table", "pool", "window", "item", "ann_pp", "t_nw", "n_days"]].to_string())


def stage_stageB():
    log = C.Logger("stageB")
    C.require_stageA("stageB")
    sel = selected_cells()
    rows, prs = [], []
    for cell in C.CELLS:
        rows += cell_result_rows("main", cell, ["U0", "U1", "U2", "C1", "C2"])
        pre = f"main__{cell}__"
        prs += paired_rows("main", cell, pre + "U2__Y1", pre + "U1__Y1", "Δ=U2−U1")
        prs += paired_rows("main", cell, pre + "U2__Y1", pre + "C1__Y1", "U2−C1")
        prs += paired_rows("main", cell, pre + "U2__Y1", pre + "C2__Y1", "U2−C2")
    res = pd.DataFrame(rows)
    res.to_csv(LEDGER / f"{PFX}_cell_results.csv", index=False)
    pr = pd.DataFrame(prs)
    # H1 / H1′:两条超额序列之差
    h1 = []
    for a, bb, lab in C.H1_PAIRS:
        ea = C.load_arm(f"main__{a}__U1__Y1")["ret"] - C.load_bench(f"main__{a}")
        eb = C.load_arm(f"main__{bb}__U1__Y1")["ret"] - C.load_bench(f"main__{bb}")
        for w in WINS_ALL:
            h1.append(dict(cellset="main", cell=f"{a}−{bb}", comparison=f"e_U1 差 {lab}", window=w,
                           **C.diff_stats(ea - eb, w)))
    pr = pd.concat([pr, pd.DataFrame(h1)], ignore_index=True)
    pr.to_csv(LEDGER / f"{PFX}_cell_paired.csv", index=False)
    show = res[(res.window.isin(["dev", "test"]))].pivot_table(index=["cell", "arm"], columns="window",
                                                               values="ex_ann").round(4)
    log("场地 × 臂 净超额(算术年化,对场地同频基准)\n" + show.to_string())
    log("H1 / H1′\n" + pd.DataFrame(h1)[lambda x: x.window.isin(["dev", "test"])][
        ["cell", "comparison", "window", "ann", "t_nw", "lo95", "hi95"]].round(4).to_string())
    log("入围场地:", sel)


# ====================================================================== overlay
def stage_overlay():
    log = C.Logger("overlay")
    C.require_stageA("overlay")
    rows = []
    ball, bnl = C.load_bench("ovl__all"), C.load_bench("ovl__nonloss")
    arms = ["A00", "A10", "A01", "A11", "A11n", "A11d"]
    D = {a: C.load_arm(f"ovl__{a}__Y1") for a in arms}
    for a in arms:
        for w in WINS_ALL:
            rows.append(dict(table="arm_vs_bench_all", item=a, window=w, **C.arm_window_stats(D[a], ball, w)))
    for w in WINS_ALL:
        rows.append(dict(table="arm_vs_own_pool", item="A11 vs 剔亏损池同频基准", window=w,
                         **C.arm_window_stats(D["A11"], bnl, w)))
        rows.append(dict(table="arm_vs_own_pool", item="A01 vs 剔亏损池同频基准", window=w,
                         **C.arm_window_stats(D["A01"], bnl, w)))
        rows.append(dict(table="bench", item="剔亏损池同频基准 − 全池同频基准", window=w,
                         **{f"ex_{k}": v for k, v in C.diff_stats(bnl - ball, w).items()}))
    r = {a: D[a]["ret"] for a in arms}
    g = {a: D[a]["gross_ret"] for a in arms}
    pairs = {"d=A11−A10(净)": r["A11"] - r["A10"], "d=A11−A10(毛)": g["A11"] - g["A10"],
             "筛子=A01−A00(净)": r["A01"] - r["A00"], "筛子=A01−A00(毛)": g["A01"] - g["A00"],
             "交互 I(净)": (r["A11"] - r["A10"]) - (r["A01"] - r["A00"]),
             "交互 I(毛)": (g["A11"] - g["A10"]) - (g["A01"] - g["A00"]),
             "A11n−A10(净)": r["A11n"] - r["A10"], "A11d−A10(净)": r["A11d"] - r["A10"],
             "排雷=A10−A00(净)": r["A10"] - r["A00"]}
    for k, s in pairs.items():
        for w in WINS_ALL:
            rows.append(dict(table="paired", item=k, window=w, **{f"ex_{kk}": v for kk, v in C.diff_stats(s, w).items()}))
    # 换手 / 成本增量:逐年 + 财报季(4/8/10 月)
    for a in ("A10", "A11", "A11n", "A11d"):
        dd = D[a]
        for y in range(2021, 2026):
            for lab, months in (("全年", None), ("财报季4/8/10月", (4, 8, 10)), ("其余月份", "rest")):
                m = dd.index.year == y
                if months == "rest":
                    m &= ~dd.index.month.isin([4, 8, 10])
                elif months:
                    m &= dd.index.month.isin(months)
                x = dd[m]
                rows.append(dict(table="turnover_cost", item=a, window=f"y{y}", sub=lab,
                                 turnover_daily=float(x["turnover"].mean()), cost_ann=float(x["cost_total"].mean() * TD),
                                 holdings=float(x["n_hold"].mean())))
    out = pd.DataFrame(rows)
    out.to_csv(LEDGER / f"{PFX}_overlay_2x2.csv", index=False)
    sh = out[(out.table == "paired") & out.window.isin(["dev", "test", "full"])][
        ["item", "window", "ex_ann", "ex_t_nw", "ex_lo95", "ex_hi95", "ex_t_quarter"]]
    log(sh.round(4).to_string())
    sh2 = out[(out.table == "arm_vs_bench_all") & out.window.isin(["dev", "test", "full"])][
        ["item", "window", "ex_ann", "ex_t_nw", "ex_geo", "cost_ann", "turnover_daily", "holdings"]]
    log(sh2.round(4).to_string())


# ====================================================================== sens
def stage_sens():
    log = C.Logger("sens")
    C.require_stageA("sens")
    p = C.panel()
    Uv = C.pool(p).to_numpy(bool)
    codes = C.load_codes(p)
    masks = C.cell_masks_from_codes(codes, "main")
    sel = selected_cells()
    focus = {c: masks[c] for c in [C.CONFIRM_CELL] + sel}
    S = _scores(p, ("comp", "vp"))
    # 成本敏感性:U1 / U2(Δ_c);2×2:A10 / A11(d),外加 A00 / A01 以便交互项
    for cost in ("legacy", "Y0.5", "Y1.5"):
        run_cell_set(p, log, "main", focus, Uv, S, arms=("U1", "U2"), cost=cost)
        run_overlay(p, log, masks["D1_loss"], Uv, S, arms=("A00", "A10", "A01", "A11"), cost=cost)
    # 纯量价版打分
    run_cell_set(p, log, "main", focus, Uv, S, arms=("U1", "U2"), score_key="vp", tag_sfx="_vp")
    run_overlay(p, log, masks["D1_loss"], Uv, S, arms=("A10", "A11"), score_key="vp", tag_sfx="_vp")
    # 选优 20% / 缓冲 (0.20, 0.25)
    run_cell_set(p, log, "main", focus, Uv, S, arms=("U2",), top_buf=C.TOP_BUF20, tag_sfx="_top20")
    # 排雷无缓冲剔 30%
    run_cell_set(p, log, "main", focus, Uv, S, arms=("U1",), excl_buf=None, tag_sfx="_nobuf")
    run_overlay(p, log, masks["D1_loss"], Uv, S, arms=("A10", "A11"), excl_buf=None, tag_sfx="_nobuf")
    # 1 亿(仅 2×2)
    run_overlay(p, log, masks["D1_loss"], Uv, S, arms=("A10", "A11"), capital=1e8, tag_sfx="_1e8")
    # 入围场地的 Barra 归因需要持仓
    for c in sel:
        for arm, W in (("U1", C.w_exclude(S["comp"], masks[c] & Uv)), ("U2", C.w_top(S["comp"], masks[c] & Uv))):
            if not (C.ARMS_DIR / f"{PFX}_hold_main__{c}__{arm}__Y1.parquet").exists():
                C.run_arm(p, f"main__{c}__{arm}__Y1", W, keep_holdings=True, log=log, force=True)
    log("sens 完成")


# ====================================================================== grid
def stage_grid():
    log = C.Logger("grid")
    C.require_stageA("grid")
    p = C.panel()
    Uv = C.pool(p).to_numpy(bool)
    codes = C.load_codes(p)
    S = _scores(p)
    which = ARGS.which.split(",") if ARGS.which else ["q5", "d1q", "d1fc", "lag1"]
    for cs in which:
        if cs == "lag1":
            masks = C.cell_masks_from_codes({k: codes[f"{k}_l1"] for k in ("D1", "D2", "D3", "D4")}, "main")
        else:
            masks = C.cell_masks_from_codes(codes, cs)
        run_cell_set(p, log, cs, masks, Uv, S, arms=("U1", "U2", "C1", "C2"))
        if cs in ("d1q", "d1fc", "lag1"):
            loss = masks["D1_loss" if cs == "lag1" else f"{'D1q' if cs == 'd1q' else 'D1fc'}_loss"]
            run_overlay(p, log, loss, Uv, S, prefix=f"ovl_{cs}", arms=("A01", "A11"))
    log("grid 完成")


# ====================================================================== gridtab
def stage_gridtab():
    log = C.Logger("gridtab")
    C.require_stageA("gridtab")
    sel = selected_cells()
    rows = []
    W3 = ["dev", "test", "full"]

    def addcell(cellset, cell, arm_list, cost, sfx, grid_dim, grid_val):
        b = C.load_bench(f"{cellset}__{cell}")
        for arm in arm_list:
            tag = f"{cellset}__{cell}__{arm}__{cost}{sfx}"
            if not C.arm_path(tag).exists():
                rows.append(dict(grid=grid_dim, value=grid_val, cellset=cellset, cell=cell, item=arm, window="—",
                                 status="未跑"))
                continue
            d = C.load_arm(tag)
            for w in W3:
                st = C.arm_window_stats(d, b, w)
                rows.append(dict(grid=grid_dim, value=grid_val, cellset=cellset, cell=cell, item=arm, window=w,
                                 status="ok", ann=st["ex_ann"], t_nw=st["ex_t_nw"], lo95=st["ex_lo95"],
                                 hi95=st["ex_hi95"], turnover_daily=st["turnover_daily"], cost_ann=st["cost_ann"],
                                 holdings=st["holdings"]))

    def addpair(grid_dim, grid_val, cellset, cell, a_tag, b_tag, item):
        if not (C.arm_path(a_tag).exists() and C.arm_path(b_tag).exists()):
            rows.append(dict(grid=grid_dim, value=grid_val, cellset=cellset, cell=cell, item=item, window="—",
                             status="未跑"))
            return
        a, b = C.load_arm(a_tag)["ret"], C.load_arm(b_tag)["ret"]
        for w in W3:
            st = C.diff_stats(a - b, w)
            rows.append(dict(grid=grid_dim, value=grid_val, cellset=cellset, cell=cell, item=item, window=w,
                             status="ok", ann=st["ann"], t_nw=st["t_nw"], lo95=st["lo95"], hi95=st["hi95"],
                             t_quarter=st["t_quarter"]))

    focus = [C.CONFIRM_CELL] + sel
    for cost in ("Y1", "legacy", "Y0.5", "Y1.5"):
        for c in focus:
            addcell("main", c, ["U1", "U2"], cost, "", "成本", cost)
            addpair("成本", cost, "main", c, f"main__{c}__U2__{cost}", f"main__{c}__U1__{cost}", "Δ=U2−U1")
        for a in ("A00", "A10", "A01", "A11"):
            tag = f"ovl__{a}__{cost}"
            if C.arm_path(tag).exists():
                d = C.load_arm(tag); b = C.load_bench("ovl__all")
                for w in W3:
                    st = C.arm_window_stats(d, b, w)
                    rows.append(dict(grid="成本", value=cost, cellset="ovl", cell="全池", item=a, window=w, status="ok",
                                     ann=st["ex_ann"], t_nw=st["ex_t_nw"], lo95=st["ex_lo95"], hi95=st["ex_hi95"],
                                     turnover_daily=st["turnover_daily"], cost_ann=st["cost_ann"],
                                     holdings=st["holdings"]))
        addpair("成本", cost, "ovl", "全池", f"ovl__A11__{cost}", f"ovl__A10__{cost}", "d=A11−A10")
    for c in focus:
        addcell("main", c, ["U1", "U2"], "Y1", "_vp", "打分", "纯量价版")
        addpair("打分", "纯量价版", "main", c, f"main__{c}__U2__Y1_vp", f"main__{c}__U1__Y1_vp", "Δ=U2−U1")
        addcell("main", c, ["U2"], "Y1", "_top20", "选优", "保留20%缓冲(0.20,0.25)")
        addpair("选优", "保留20%缓冲(0.20,0.25)", "main", c, f"main__{c}__U2__Y1_top20", f"main__{c}__U1__Y1",
                "Δ=U2(20%)−U1")
        addcell("main", c, ["U1"], "Y1", "_nobuf", "排雷", "无缓冲剔30%")
        addpair("排雷", "无缓冲剔30%", "main", c, f"main__{c}__U2__Y1", f"main__{c}__U1__Y1_nobuf",
                "Δ=U2−U1(无缓冲)")
    for sfx, dim, val in (("_vp", "打分", "纯量价版"), ("_nobuf", "排雷", "无缓冲剔30%"), ("_1e8", "资金", "1亿")):
        b = C.load_bench("ovl__all")
        for a in ("A10", "A11"):
            tag = f"ovl__{a}__Y1{sfx}"
            if C.arm_path(tag).exists():
                d = C.load_arm(tag)
                for w in W3:
                    st = C.arm_window_stats(d, b, w)
                    rows.append(dict(grid=dim, value=val, cellset="ovl", cell="全池", item=a, window=w, status="ok",
                                     ann=st["ex_ann"], t_nw=st["ex_t_nw"], lo95=st["ex_lo95"], hi95=st["ex_hi95"],
                                     turnover_daily=st["turnover_daily"], cost_ann=st["cost_ann"],
                                     holdings=st["holdings"]))
        addpair(dim, val, "ovl", "全池", f"ovl__A11__Y1{sfx}", f"ovl__A10__Y1{sfx}", "d=A11−A10")
    # 全场地网格:五分位 / D1 定义 / 生效日
    for cs, dim in (("q5", "分档=五分位"), ("d1q", "D1=最新单季归母净利<0"), ("d1fc", "D1″=预告首亏/续亏/增亏(探索)"),
                    ("lag1", "生效日=可见日+1交易日")):
        cells = sorted({f.stem.split("__")[1] for f in C.ARMS_DIR.glob(f"{PFX}_bench_{cs}__*.parquet")})
        for c in cells:
            addcell(cs, c, ["U1", "U2", "C1", "C2"], "Y1", "", dim, cs)
            addpair(dim, cs, cs, c, f"{cs}__{c}__U2__Y1", f"{cs}__{c}__U1__Y1", "Δ=U2−U1")
        if cs != "q5":
            pre = f"ovl_{cs}"
            ball = C.load_bench(f"{pre}__all") if C.bench_path(f"{pre}__all").exists() else None
            for a in ("A01", "A11"):
                tag = f"{pre}__{a}__Y1"
                if ball is not None and C.arm_path(tag).exists():
                    d = C.load_arm(tag)
                    for w in W3:
                        st = C.arm_window_stats(d, ball, w)
                        rows.append(dict(grid=dim, value=cs, cellset=pre, cell="全池", item=a, window=w, status="ok",
                                         ann=st["ex_ann"], t_nw=st["ex_t_nw"], lo95=st["ex_lo95"], hi95=st["ex_hi95"],
                                         turnover_daily=st["turnover_daily"], cost_ann=st["cost_ann"],
                                         holdings=st["holdings"]))
            addpair(dim, cs, pre, "全池", f"{pre}__A11__Y1", "ovl__A10__Y1", "d=A11−A10")
            # H1 在替代 D1 定义下
            lc, pc = (("D1_loss", "D1_profit") if cs == "lag1" else
                      (f"{'D1q' if cs == 'd1q' else 'D1fc'}_loss", f"{'D1q' if cs == 'd1q' else 'D1fc'}_profit"))
            if C.arm_path(f"{cs}__{lc}__U1__Y1").exists():
                ea = C.load_arm(f"{cs}__{lc}__U1__Y1")["ret"] - C.load_bench(f"{cs}__{lc}")
                eb = C.load_arm(f"{cs}__{pc}__U1__Y1")["ret"] - C.load_bench(f"{cs}__{pc}")
                for w in W3:
                    st = C.diff_stats(ea - eb, w)
                    rows.append(dict(grid=dim, value=cs, cellset=cs, cell=f"{lc}−{pc}", item="H1 e_U1 差", window=w,
                                     status="ok", ann=st["ann"], t_nw=st["t_nw"], lo95=st["lo95"], hi95=st["hi95"]))
    df = pd.DataFrame(rows)
    df = df[~((df.cellset == "ovl") & (df.cell == "") & (df.status == "未跑"))]
    df.to_csv(LEDGER / f"{PFX}_grid.csv", index=False)
    log(f"网格 {len(df)} 行;未跑 {int((df.status == '未跑').sum())} 行")


# ====================================================================== longhist
def stage_longhist():
    log = C.Logger("longhist")
    C.require_stageA("longhist")
    from afactor import lh_engine as LE, fundamental as FU, strategy as ST
    X = LE.load_inputs()
    dates, codes = X.dates, X.codes
    rep = FU.build_report_level()
    fin_codes = set(rep["stock_code"].unique())
    U = LE.pool(X, "q92")
    Uv = U.to_numpy(bool)
    rows = []
    # 代码映射覆盖:2016-07-01 起池内出现过的股票有多少能在财报表里找到
    for a, b in (("2016-07-01", "2019-12-31"), ("2020-01-02", "2025-12-31")):
        r = np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))
        inpool = codes[Uv[r].any(axis=0)]
        cov = float(np.mean([c in fin_codes for c in inpool]))
        rows.append(dict(table="code_map", window=f"{a}~{b}", n_pool_codes=len(inpool), map_cov=cov))
        log(f"代码映射覆盖 {a}~{b}: {cov:.4f}({len(inpool)} 只)")
    if min(r_["map_cov"] for r_ in rows) < 0.99:
        pd.DataFrame(rows).to_csv(LEDGER / f"{PFX}_longhist.csv", index=False)
        log("映射覆盖 < 99%:按预注册停在覆盖报告,本节记'未做'")
        return
    ni = FU._pit_daily(rep, "n_income_attr_p_ttm", dates, codes, lag_days=0).to_numpy(np.float64)
    loss = Uv & np.isfinite(ni) & (ni < 0)
    yrs = dates.year
    for y in range(2015, 2026):
        r = yrs == y
        rows.append(dict(table="coverage", window=str(y), pool_n=float(Uv[r].sum(1).mean()),
                         ni_cov=float((Uv[r] & np.isfinite(ni[r])).sum() / Uv[r].sum()),
                         loss_share=float(loss[r].sum() / max((Uv[r] & np.isfinite(ni[r])).sum(), 1))))
    ad = LE.adapter(X)
    S = X.df("score_S1")
    nonloss = Uv & ~loss
    shape = LE._Shape(dates, codes)

    def bw(score, Um, excl):
        sc = score if score is not None else pd.DataFrame(1.0, index=dates, columns=codes)
        return ST.build_weights(sc, shape, "V1000", scheme="exclude", exclude_frac=excl,
                                buffer=(C.EXCL_BUF if score is not None else None),
                                universe_override=pd.DataFrame(Um, index=dates, columns=codes))
    R = {}
    for name, Um, sc in (("A00", Uv, None), ("A10", Uv, S), ("A01", nonloss, None), ("A11", nonloss, S)):
        f = TMP / f"{PFX}_lh_{name}.parquet"
        if f.exists():
            R[name] = pd.read_parquet(f)
        else:
            t0 = time.time()
            W = bw(sc, Um, 0.0 if sc is None else 0.3)
            res = LE.run(W, ad, rebalance=5, cost="Y1", record=False)
            R[name] = pd.DataFrame({"ret": res["rets"]})
            R[name].to_parquet(f)
            log(f"  lh {name} turn {res['turnover']:.4f} cost {res['cost_bps']:.2f}bp/d {time.time() - t0:.0f}s")
    fb = TMP / f"{PFX}_lh_bench.parquet"
    if fb.exists():
        b = pd.read_parquet(fb)["ret"]
    else:
        b = LE.bench(ad, U, 5)
        b.rename("ret").to_frame().to_parquet(fb)
    fbn = TMP / f"{PFX}_lh_bench_nonloss.parquet"
    if fbn.exists():
        bn = pd.read_parquet(fbn)["ret"]
    else:
        bn = LE.bench(ad, pd.DataFrame(nonloss, index=dates, columns=codes), 5)
        bn.rename("ret").to_frame().to_parquet(fbn)
    r = {k: v["ret"] for k, v in R.items()}
    wins = {"lh_2016_2019": ("2016-07-01", "2019-12-31"), "lh_2020_2025": ("2020-01-02", "2025-12-31")}

    def st_(d, a, bb):
        x = d.dropna(); x = x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(bb))]
        se = C.nw_se_mean(x.to_numpy())
        return dict(n_days=len(x), ann=float(x.mean() * TD), t_nw=float(x.mean() / se) if se else np.nan,
                    lo95=float((x.mean() - C.Z95 * se) * TD), hi95=float((x.mean() + C.Z95 * se) * TD))
    items = {"A00−B": r["A00"] - b, "A10−B": r["A10"] - b, "A01−B": r["A01"] - b, "A11−B": r["A11"] - b,
             "d=A11−A10": r["A11"] - r["A10"], "筛子=A01−A00": r["A01"] - r["A00"],
             "交互 I": (r["A11"] - r["A10"]) - (r["A01"] - r["A00"]), "A11−剔亏损池基准": r["A11"] - bn,
             "剔亏损池基准−B": bn - b}
    for w, (a, bb) in wins.items():
        for k, s in items.items():
            rows.append(dict(table="overlay_lh", window=w, item=k, **st_(s, a, bb)))
    out = pd.DataFrame(rows)
    d1 = out[(out.table == "overlay_lh") & (out["item"] == "d=A11−A10")].set_index("window")
    lab = ("长历史支持" if (d1.loc["lh_2016_2019", "ann"] > 0 and d1.loc["lh_2020_2025", "ann"] > 0
                          and d1.loc["lh_2016_2019", "t_nw"] > 1.5) else "仅研究期")
    out = pd.concat([out, pd.DataFrame([dict(table="label", item=lab)])], ignore_index=True)
    out.to_csv(LEDGER / f"{PFX}_longhist.csv", index=False)
    log(out[out.table == "overlay_lh"].round(4).to_string())
    log("长历史标签:", lab)


# ====================================================================== attrib
def stage_attrib():
    log = C.Logger("attrib")
    C.require_stageA("attrib")
    p = C.panel()
    from afactor import barra as BR, strategy as ST
    Uv = C.pool(p).to_numpy(bool)
    codes = C.load_codes(p)
    px, _ = ST.exec_prices(p, C.MODE)
    G = BR.fwd_returns(p, C.MODE)                                    # 信号日索引的持有收益
    Gv = G.reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
    rows = []
    # §6.2 A10 场地拆分(参照 = 当日全池等权;Brinson:配置 + 场地内选择)
    H = pd.read_parquet(C.ARMS_DIR / f"{PFX}_hold_ovl__A10__Y1.parquet").reindex(index=p.dates, columns=p.codes).fillna(0.0)
    Hv = H.to_numpy(np.float64)
    Rz = np.nan_to_num(Gv)
    n = Uv.sum(1, keepdims=True)
    B = np.divide(Uv.astype(np.float64), n, out=np.zeros(Uv.shape), where=n > 0)
    tot_ex = ((Hv - B) * Rz).sum(1)
    for dim in ("D1", "D2", "D3", "D4"):
        cd = codes[dim]
        labels = sorted(set(np.unique(cd)) - {-2})
        alloc, selc = {}, {}
        RB = (B * Rz).sum(1)
        for k in labels:
            m = (cd == k)
            wp = (Hv * m).sum(1); wb = (B * m).sum(1)
            rbc = np.divide((B * m * Rz).sum(1), wb, out=np.zeros(len(wb)), where=wb > 0)
            rpc = np.divide((Hv * m * Rz).sum(1), wp, out=np.zeros(len(wp)), where=wp > 0)
            alloc[k] = (wp - wb) * (rbc - RB)
            selc[k] = wp * (rpc - rbc)
        idx = p.dates
        for k in labels:
            for w in ("dev", "test", "full"):
                a_, b_ = C.WINDOWS[w]
                r = np.asarray((idx >= pd.Timestamp(a_)) & (idx <= pd.Timestamp(b_)))[:len(alloc[k])]
                name = {("D1", -1): "NaN", ("D1", 0): "亏损", ("D1", 1): "盈利"}.get((dim, int(k)),
                                                                                   "NaN" if k == -1 else ["", "低", "中", "高"][int(k)])
                rows.append(dict(table="A10_cell_decomp", dim=dim, cell=name, window=w,
                                 alloc_pp=float(alloc[k][r].mean() * TD * 100),
                                 select_pp=float(selc[k][r].mean() * TD * 100),
                                 w_port=float((Hv * (cd == k)).sum(1)[r].mean()),
                                 w_bench=float((B * (cd == k)).sum(1)[r].mean())))
        for w in ("dev", "test", "full"):
            a_, b_ = C.WINDOWS[w]
            r = np.asarray((p.dates >= pd.Timestamp(a_)) & (p.dates <= pd.Timestamp(b_)))
            rows.append(dict(table="A10_cell_decomp", dim=dim, cell="合计(毛,vs 当日全池等权)", window=w,
                             alloc_pp=float(tot_ex[r].mean() * TD * 100)))
    del H, Hv
    # Barra 辅助归因(只描述):U1 / U2 相对场地当日等权的主动权重
    rm = BR.RiskModel.load(C.MODE)
    masks = C.cell_masks_from_codes(codes, "main")
    sel = selected_cells()
    for cell in sorted(set(["D1_loss", "D1_profit", "Q"] + sel)):
        M = masks[cell] & Uv
        nm = M.sum(1, keepdims=True)
        Bc = np.divide(M.astype(np.float64), nm, out=np.zeros(M.shape), where=nm > 0)
        for arm in ("U1", "U2"):
            f = C.ARMS_DIR / f"{PFX}_hold_main__{cell}__{arm}__Y1.parquet"
            if not f.exists():
                rows.append(dict(table="barra", cell=cell, arm=arm, note="无持仓记录,未做"))
                continue
            Ha = pd.read_parquet(f).reindex(index=p.dates, columns=p.codes).fillna(0.0)
            A = Ha - Bc
            A = A[(A.index >= pd.Timestamp("2021-01-04"))]
            att = rm.attribute(A, G)
            for w in ("dev", "test", "full"):
                x = C.wslice(att, w)
                for comp in ("total", "style", "industry", "specific"):
                    s = x[comp].to_numpy()
                    rows.append(dict(table="barra", cell=cell, arm=arm, window=w, component=comp,
                                     ann_pp=float(s.mean() * TD * 100),
                                     se_q5_pp=float(C.nw_se_mean(s, 5) * TD * 100),
                                     se_q10_pp=float(C.nw_se_mean(s, 10) * TD * 100)))
            log(f"  barra {cell} {arm} 完成")
            del Ha, A
    out = pd.DataFrame(rows)
    out.to_csv(LEDGER / f"{PFX}_attribution.csv", index=False)
    log(out.round(3).to_string())


# ====================================================================== report(判定汇总 + 图)
def stage_report():
    """按预注册 §10 机械地汇总判定(原规则 + 登记偏离后的条件读法),写 inc_t6_verdict.json,画两张图。"""
    log = C.Logger("report")
    C.require_stageA("report")
    eng = pd.read_csv(LEDGER / f"{PFX}_engine_checks.csv")
    req = eng[eng.check.isin(["U0_vs_bench", "exposure_all_arms"]) |
              ((eng.check == "A10_reproduce_capacity_grid") & (eng["item"] == "full"))]
    engine_ok = all(bool(req[c].dropna().astype(bool).all()) for c in ("pass_resid", "pass_align", "pass_exposure"))
    failed = list(req[(req.check == "U0_vs_bench") & (~req.pass_resid.astype(bool))]["item"])
    pr = pd.read_csv(LEDGER / f"{PFX}_cell_paired.csv")
    res = pd.read_csv(LEDGER / f"{PFX}_cell_results.csv")
    ov = pd.read_csv(LEDGER / f"{PFX}_overlay_2x2.csv")
    grid = pd.read_csv(LEDGER / f"{PFX}_grid.csv") if (LEDGER / f"{PFX}_grid.csv").exists() else None

    def pget(df, **kw):
        m = np.ones(len(df), bool)
        for k, v in kw.items():
            m &= (df[k] == v).to_numpy()
        x = df[m]
        assert len(x) == 1, (kw, len(x))
        return x.iloc[0]

    # ---- H1
    h1 = {w: pget(pr, comparison="e_U1 差 H1(验证性)", window=w) for w in ("dev", "test")}
    if h1["dev"].ann > 0 and h1["test"].ann > 0 and h1["test"].t_nw > 2:
        h1_v = "支持"
    elif (h1["dev"].ann <= 0 and h1["test"].ann <= 0) or h1["test"].hi95 <= 0:
        h1_v = "不支持"
    else:
        h1_v = "证据不足"
    h1r = pd.read_csv(LEDGER / f"{PFX}_h1_robust.csv") if (LEDGER / f"{PFX}_h1_robust.csv").exists() else None
    h1_adj = {}
    if h1r is not None:
        for rd in ("adjA", "adjB"):
            d_ = {w: pget(h1r, pair="D1_loss−D1_profit", reading=rd, window=w) for w in ("dev", "test")}
            if d_["dev"].ann > 0 and d_["test"].ann > 0 and d_["test"].t_nw > 2:
                h1_adj[rd] = "支持"
            elif (d_["dev"].ann <= 0 and d_["test"].ann <= 0) or d_["test"].hi95 <= 0:
                h1_adj[rd] = "不支持"
            else:
                h1_adj[rd] = "证据不足"

    # ---- H2(Q)与入围场地
    def h2_verdict(cell):
        e = {(a, w): pget(res, cell=cell, arm=a, window=w, cost="Y1").ex_ann
             for a in ("U1", "U2", "C1", "C2") for w in ("dev", "test")}
        dl = {w: pget(pr, cell=cell, comparison="Δ=U2−U1", window=w) for w in ("dev", "test")}
        c2 = dl["dev"].ann > 0 and e[("U2", "dev")] > 0 and e[("U2", "dev")] > max(e[("C1", "dev")], e[("C2", "dev")])
        c3 = dl["test"].lo95 > 0 and e[("U2", "test")] > 0 and e[("U2", "test")] > max(e[("C1", "test")], e[("C2", "test")])
        c4 = c5 = None
        if grid is not None:
            g = grid[(grid.cell == cell) & (grid.window == "test")]
            cs = g[(g.grid == "成本") & (g["item"] == "Δ=U2−U1") & g.value.isin(["legacy", "Y0.5", "Y1.5"])]
            c4 = bool(len(cs) == 3 and (cs.ann > 0).all())
            vp = g[(g.grid == "打分") & (g["item"] == "Δ=U2−U1")]
            c5 = bool(len(vp) == 1 and (vp.ann > 0).all())
        if c2 and c3 and c4 and c5:
            v = "有增量"
        elif dl["test"].hi95 <= 0 or e[("U2", "test")] <= max(e[("C1", "test")], e[("C2", "test")]):
            v = "无增量"
        else:
            v = "证据不足"
        return dict(verdict=v, cond_dev=bool(c2), cond_test=bool(c3), cond_cost=c4, cond_vp=c5,
                    delta_dev=float(dl["dev"].ann), delta_test=float(dl["test"].ann),
                    delta_test_lo95=float(dl["test"].lo95), delta_test_hi95=float(dl["test"].hi95),
                    e_U2_test=e[("U2", "test")], e_C1_test=e[("C1", "test")], e_C2_test=e[("C2", "test")])
    sel = selected_cells()
    h2 = {c: h2_verdict(c) for c in [C.CONFIRM_CELL] + sel}

    # ---- H3
    def ovp(item, w, table="paired"):
        return pget(ov, table=table, item=item, window=w)
    d = {w: ovp("d=A11−A10(净)", w) for w in ("dev", "test", "full")}
    a11 = {w: ovp("A11", w, "arm_vs_bench_all") for w in ("dev", "test")}
    a10 = {w: ovp("A10", w, "arm_vs_bench_all") for w in ("dev", "test")}
    c4 = None
    if grid is not None:
        g = grid[(grid.cellset == "ovl") & (grid["item"] == "d=A11−A10") & (grid.window == "test") & (grid.grid == "成本")]
        g = g[g.value.isin(["legacy", "Y1.5"])]
        c4 = bool(len(g) == 2 and (g.ann > 0).all())
    conds = dict(c1=bool(d["dev"].ex_ann > 0), c2=bool(d["test"].ex_lo95 > 0),
                 c3=bool(all(a11[w].ex_ann > a10[w].ex_ann for w in ("dev", "test"))), c4=c4,
                 c5=bool(d["test"].ex_t_quarter > 1.5))
    if all(v is True for v in conds.values()):
        h3_v = "有增量"
    elif d["test"].ex_hi95 <= 0 or (d["dev"].ex_ann <= 0 and d["test"].ex_ann <= 0):
        h3_v = "无增量"
    else:
        h3_v = "证据不足"
    I = {w: ovp("交互 I(净)", w).ex_ann for w in ("dev", "test")}
    tag_I = "含交互" if all(v > 0 for v in I.values()) else "增量来自基本面筛子本身"
    lh_tag = "未做"
    if (LEDGER / f"{PFX}_longhist.csv").exists():
        lh = pd.read_csv(LEDGER / f"{PFX}_longhist.csv")
        if "label" in set(lh.table):
            lh_tag = str(lh[lh.table == "label"]["item"].iloc[0])

    # ---- 轨道判定
    def track(h3v, h2s):
        if h3v == "有增量" or any(x["verdict"] == "有增量" for x in h2s.values()):
            return "有增量"
        if h3v == "无增量" and all(x["verdict"] == "无增量" for x in h2s.values()):
            return "无增量"
        return "证据不足"
    tv = track(h3_v, h2)
    el = pd.read_csv(LEDGER / f"{PFX}_engine_diag_enginelike.csv")
    el_all = el[(el.window == "full") & (el.variant == "E_all")].set_index("item").resid_pp
    el_dn = el[(el.window == "full") & (el.variant == "E_dn")].set_index("item").resid_pp
    el_uz = el[(el.window == "full") & (el.variant == "E_upzero")].set_index("item").resid_pp
    el_off = el[(el.window == "full") & (el.variant == "官方基准")].set_index("item").resid_pp
    out = dict(engine_ok=engine_ok, engine_failed_cells=failed,
               verdict_original_rule=("引擎未过,不判定" if not engine_ok else tv),
               verdict_conditional=tv,
               verdict_text=(("预注册:引擎未过,不判定;条件判定(偏离表第 10 行,读结果后登记):" + tv)
                             if not engine_ok else tv),
               conditional_note="登记偏离(机理经审查更正,偏离表第 14 行):前置残差超标来自官方同频基准在调仓日把引擎无法成交的股票"
                                "(一字涨停、停牌即窗口成交额为 0)重置回等权,而引擎对这些持仓保留漂移权重;一字跌停几乎没有贡献。"
                                "改用与引擎同口径的无摩擦基准(inc_t6_engine_diag_enginelike.csv),15 项残差全部 |x| <= "
                                f"{float(el_all.abs().max()):.3f}pp。轨道判定所用的 2×2 臂与 Q 场地的断言(官方基准)本来就通过",
               engine_diag_pp_full=dict(
                   D1_loss=dict(official=float(el_off["D1_loss"]), only_down=float(el_dn["D1_loss"]),
                                up_and_zero=float(el_uz["D1_loss"]), engine_rule=float(el_all["D1_loss"])),
                   A00=dict(official=float(el_off["A00(全池)"]), only_down=float(el_dn["A00(全池)"]),
                            up_and_zero=float(el_uz["A00(全池)"]), engine_rule=float(el_all["A00(全池)"])),
                   max_abs_engine_rule=float(el_all.abs().max())),
               H1=dict(verdict=h1_v, dev=float(h1["dev"].ann), test=float(h1["test"].ann),
                       test_t=float(h1["test"].t_nw), test_lo95=float(h1["test"].lo95), adj=h1_adj),
               H2=h2, stageA_selected=sel,
               H3=dict(verdict=h3_v, conds=conds, d_dev=float(d["dev"].ex_ann), d_test=float(d["test"].ex_ann),
                       d_test_lo95=float(d["test"].ex_lo95), d_test_hi95=float(d["test"].ex_hi95),
                       d_full=float(d["full"].ex_ann), d_full_lo95=float(d["full"].ex_lo95),
                       d_full_hi95=float(d["full"].ex_hi95), interaction=I, tag_interaction=tag_I,
                       tag_longhist=lh_tag))
    json.dump(out, open(LEDGER / f"{PFX}_verdict.json", "w"), ensure_ascii=False, indent=1, default=float)
    log(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    _figures(res, ov)


def _figures(res, ov):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fn in ("PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS", "Songti SC"):
        if any(fn in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fn
            break
    plt.rcParams["axes.unicode_minus"] = False
    S1, S2, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    # 图 1:各场地 U1 排雷对场地同频基准的净超额(开发窗 / 检验窗)
    x = res[(res.arm == "U1") & (res.cost == "Y1") & res.window.isin(["dev", "test"])]
    piv = x.pivot_table(index="cell", columns="window", values="ex_ann").reindex(C.CELLS[::-1]) * 100
    lab = [f"{c}  {C.CELL_DESC[c]}".replace("⁻", "-") for c in piv.index]   # 字体无上标负号
    fig, ax = plt.subplots(figsize=(9, 6.2), dpi=150)
    y = np.arange(len(piv))
    h = 0.36
    ax.barh(y + h / 2, piv["dev"], height=h - 0.04, color=S1, label="开发窗 2021-2023")
    ax.barh(y - h / 2, piv["test"], height=h - 0.04, color=S2, label="检验窗 2024-2025")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8.5, color=INK)
    ax.set_xlabel("COMP_v5 排雷(20/35)对场地 5 日同频等权基准的净超额,pp/年(Y=1,1000 万)", fontsize=9, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    ax.set_title("排雷超额按基本面场地拆分(出处 research/ledger/inc_t6_cell_results.csv)", fontsize=10.5, color=INK,
                 loc="left", pad=24)
    fig.tight_layout(); fig.savefig(REPORTS / f"{PFX}_cell_exclude_excess.png"); plt.close(fig)
    # 图 2:2×2 配对差累计(全样本 2021-2025 连续路径)
    arms = ["A00", "A10", "A01", "A11", "A11n"]
    D = {a: C.load_arm(f"ovl__{a}__Y1")["ret"] for a in arms}
    s0 = pd.Timestamp("2021-01-04")
    series = {"叠加 d = A11 − A10(剔亏损后排雷 − 全池排雷)": (D["A11"] - D["A10"], S1),
              "筛子 = A01 − A00(剔亏损等权 − 全池等权)": (D["A01"] - D["A00"], S2),
              "变体 A11n − A10(场地中性排雷,只报告)": (D["A11n"] - D["A10"], "#1baf7a")}
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    ends = []
    for nm, (s, col) in series.items():
        s = s[s.index >= s0]
        cum = s.cumsum() * 100
        ax.plot(cum.index, cum.values, color=col, lw=2, label=nm)
        ends.append([cum.index[-1], float(cum.iloc[-1])])
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):                      # 末端标签互相避让(≥ 0.6pp)
        if ends[i][1] - ends[i - 1][1] < 0.6:
            ends[i - 1][1] -= 0.3; ends[i][1] += 0.3
    for (x_, y_), (_, v_) in zip(ends, sorted([(0, float((series[k][0][series[k][0].index >= s0]).sum() * 100))
                                             for k in series], key=lambda e: e[1])):
        ax.annotate(f"{v_:+.1f}", (x_, y_), xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                    color=INK, va="center")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(pd.Timestamp("2024-01-02"), color=MUTED, lw=0.8, ls="--")
    ax.text(pd.Timestamp("2024-01-10"), ax.get_ylim()[1] * 0.92, "检验窗起点", fontsize=8.5, color=MUTED)
    ax.set_ylabel("配对日差累计(算术),pp", fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title("2×2 叠加检验:净收益配对差累计(出处 research/ledger/inc_t6_overlay_2x2.csv)", fontsize=10.5,
                 color=INK, loc="left")
    fig.tight_layout(); fig.savefig(REPORTS / f"{PFX}_overlay_paths.png"); plt.close(fig)


# ====================================================================== main
STAGES = dict(cells=stage_cells, vpscore=stage_vpscore, arms=stage_arms, precheck=stage_precheck,
              stageA=stage_stageA, checks=stage_checks, diagU0=stage_diagU0, diagU0b=stage_diagU0b, h1profit=stage_h1profit, survivor=stage_survivor,
              h1robust=stage_h1robust, report=stage_report,
              stageB=stage_stageB, overlay=stage_overlay, sens=stage_sens, grid=stage_grid, gridtab=stage_gridtab,
              longhist=stage_longhist, attrib=stage_attrib)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--part", default=None)
    ap.add_argument("--which", default=None)
    ARGS = ap.parse_args()
    C.check_prereg()
    STAGES[ARGS.stage]()
