"""INC-T4 方法 (a)(b) 与引擎断言(预注册 §3.4、§3.5、§6.1、§6.2、§6.4)。

口径(与 capacity_grid.csv 的 COMP_v5_exclude|open10|Y1.0|1e7|pov 逐项相同):
  V1000 决策池、open10、1000 万恒定 AUM(cap_capacity_curve.ConstAUMCost)、Y=1 一跳半价差 + 平方根冲击、
  参与率按 te 实际窗口成交额(pov)、5 日调仓 5 相位平均、2020-07-01 起连续路径(预热),切 2021-2025 / 2021-2023 /
  2024-2025;基准 = 同频池等权 ST.benchmark(p,"V1000","open10",rebalance=5)。判定统计量 = 变体 − 基线 日收益差
  的算术年化 Δ 与 NW t(带宽 10)。

子命令(按预注册 §11 顺序):
  checks  引擎断言 ①~④(基线复现门槛不过即停)
  a       方法 (a) 网格 + COMP_v5 自身平滑对照 -> 按 §6.1 规则(只看开发窗换手)定 HL*,写入 inc_t4_checks.csv
  b       方法 (b) 网格 + 深剔对照 + H2 分解(需要 HL*)
  sens    三个主配置之 (a)(b) 的成本敏感性(legacy / Y=1.5)与 D′;D_B 主配置;λ 敏感性
  summary 汇总 -> inc_t4_grid_a.csv / inc_t4_grid_b.csv / inc_t4_paired.csv(a、b 部分)/ inc_t4_yearly.csv
每格的日频明细落 cache/_tmp/inc_t4_cell_<id>.parquet(已存在则跳过,便于断点续跑)。
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

sys.path.insert(0, os.path.dirname(__file__))
import inc_t4_common as C  # noqa: E402
import cap_capacity_curve as CAP  # noqa: E402
from afactor.config import LEDGER, TMP, RESEARCH_END  # noqa: E402
from afactor.panel import Panel  # noqa: E402
from afactor import strategy as ST, ops  # noqa: E402
from afactor.backtest import weights_exclude_buffer  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

KEEP = ["ret", "gross_ret", "cost_total", "cost_commission", "cost_stamp", "cost_spread", "cost_impact",
        "exposure", "cash", "turnover", "n_hold"]
CHECKS_CSV = LEDGER / f"{C.PFX}_checks.csv"
_S: dict = {}


# ====================================================================== 载入
def P() -> Panel:
    if "p" not in _S:
        _S["p"] = Panel()
    return _S["p"]


def sim_mask() -> np.ndarray:
    p = P()
    return np.asarray((p.dates >= pd.Timestamp(C.SIM_START)) & (p.dates <= pd.Timestamp(RESEARCH_END)))


def U_dec() -> np.ndarray:
    if "U" not in _S:
        p = P()
        _S["U"] = ST.universe(p, C.VERSION).to_numpy(bool) & sim_mask()[:, None]
    return _S["U"]


def comp_raw() -> pd.DataFrame:
    if "comp" not in _S:
        p = P()
        _S["comp"] = pd.read_parquet(TMP / "comp_v5_score_open10.parquet").reindex(index=p.dates, columns=p.codes)
    return _S["comp"]


def zscore_u(x: pd.DataFrame) -> pd.DataFrame:
    """§3.3:p["universe"] 内截面标准化(去极值 + z)。"""
    p = P()
    return ops.standardize(x.reindex(index=p.dates, columns=p.codes), p["universe"]).astype(np.float32)


def zC(hl: int = 0) -> pd.DataFrame:
    k = ("zC", hl)
    if k not in _S:
        _S[k] = zscore_u(C.ewm_smooth(comp_raw().astype(np.float32), hl))
    return _S[k]


def D_hl(s: str, hl: int) -> pd.DataFrame:
    """D 面板的 EWM 平滑版本(HL=0 = 原面板),缓存 cache/_tmp/inc_t4_Ds_{s}_hl{HL}.parquet。"""
    if hl == 0:
        return pd.read_parquet(C.d_path(s))
    f = C.ds_path(s, hl)
    if f.exists():
        return pd.read_parquet(f)
    Ds = C.ewm_smooth(pd.read_parquet(C.d_path(s)), hl)
    Ds.to_parquet(f)
    return Ds


def zD(s: str, hl: int) -> pd.DataFrame:
    k = ("zD", s, hl)
    if k not in _S:
        if len([x for x in _S if isinstance(x, tuple) and x[0] == "zD"]) > 6:      # 控制内存
            for x in [x for x in _S if isinstance(x, tuple) and x[0] == "zD"]:
                del _S[x]
        _S[k] = zscore_u(D_hl(s, hl))
    return _S[k]


def models() -> dict:
    if "models" not in _S:
        _S["models"] = {}
    return _S["models"]


def bench() -> pd.DataFrame:
    if "bench" not in _S:
        p = P()
        b5 = ST.benchmark(p, C.VERSION, C.MODE, rebalance=C.REBALANCE)["rets"].rename("pool")
        b1 = ST.benchmark(p, C.VERSION, C.MODE)["rets"].rename("pool_d1")
        _S["bench"] = pd.concat([b5, b1], axis=1)
    return _S["bench"]


# ====================================================================== 权重
def W_excl(score: pd.DataFrame, buf=C.BASE_BUF) -> np.ndarray:
    S = score.reindex(index=P().dates, columns=P().codes).to_numpy(np.float32)
    return weights_exclude_buffer(S, U_dec(), 1, *buf)


def W_baseline() -> np.ndarray:
    """基线 = capacity 同法:ST.build_weights(comp 原始打分, scheme=exclude, buffer=20/35, date_mask=sim)。"""
    if "Wb" not in _S:
        p = P()
        _S["Wb"] = ST.build_weights(comp_raw(), p, C.VERSION, scheme="exclude", buffer=C.BASE_BUF, date_mask=sim_mask())
    return _S["Wb"]


def W_two_layer(zd: pd.DataFrame, e2: float, x2: float) -> tuple[np.ndarray, np.ndarray]:
    """§6.2:两个独立滞回过滤器,持仓 = (W1>0)&(W2>0) 等权;同时返回"额外剔除"集合 E=(W1>0)&(W2=0) 的等权权重。"""
    return C.two_layer(W_baseline(), W_excl(zd, (e2, x2)))


# ====================================================================== 一格
def cell_file(cid: str):
    return TMP / f"{C.PFX}_cell_{cid}.parquet"


def run_cell(cid: str, W: np.ndarray, variant: str = "Y1.0", meta: dict | None = None, force=False) -> pd.DataFrame:
    f = cell_file(cid)
    if f.exists() and not force:
        return pd.read_parquet(f)
    p = P()
    t0 = time.time()
    cm, pon = CAP.cost_setup(p, variant, models())
    res = ST.run_weights(W, p, C.MODE, C.AUM, rebalance=C.REBALANCE, date_mask=sim_mask(), cost_model=cm,
                         record=True, participation_on_nav=pon)
    rec = res["record"]
    d = rec["daily"][KEEP].copy()
    by = rec["daily_by_phase"]
    d["exp_max_phase"] = by.groupby(level=0)["exposure"].max()
    dret = float(np.nanmax(np.abs(d["ret"].to_numpy() - res["rets"].reindex(d.index).to_numpy())))
    assert dret < 1e-12, f"{cid}: record.ret 与 rets 不一致 {dret}"
    d["ret"] = res["rets"].reindex(d.index)
    prof = CAP.align_profile(res["rets"], bench()["pool"])
    pk = max(prof, key=prof.get)
    em = float(d["exp_max_phase"].max())
    if pk != 0:
        raise AssertionError(f"{cid}: 对齐峰值 k={pk} {prof}")
    if em > 1 + 1e-9:
        raise AssertionError(f"{cid}: exposure>1 {em}")
    d.attrs = {}
    d["cell"] = cid
    d["variant"] = variant
    d["align_peak_k"] = pk
    for k_, v_ in (meta or {}).items():
        d[k_] = v_
    d.index.name = "date"
    d.to_parquet(f)
    print(f"  {cid:48s} ann {res['ann_return']*100:7.2f}% turn {res['turnover']:.4f} hold {res['holdings']:.0f} "
          f"exp_max {em:.4f} k={pk} {time.time()-t0:.1f}s", flush=True)
    return d


def window_stats(d: pd.DataFrame, w) -> dict:
    B = bench()
    dw = C.win_slice(d, w)
    rs = dw["ret"]
    rb = B["pool"].reindex(rs.index)
    out = dict(n_days=len(rs), ann_return=C.geo_ann(rs), bench_ann=C.geo_ann(rb))
    out["excess_geo"] = out["ann_return"] - out["bench_ann"]
    out["excess_arith"] = float((rs - rb).mean() * C.TD)
    out["t_nw"] = C.nw_t((rs - rb).to_numpy())
    es = ST.excess_stats(rs, rb, bandwidth=20, t_bandwidth=C.NW_BW)
    out.update(te_adj=es.get("te_adj", np.nan), ir_adj=es.get("ir_adj", np.nan))
    out["excess_geo_pool_d1"] = out["ann_return"] - C.geo_ann(B["pool_d1"].reindex(rs.index))
    out["gross_excess_arith"] = float((dw["gross_ret"] - rb).mean() * C.TD)
    for c in ("commission", "stamp", "spread", "impact", "total"):
        out[f"cost_{c}_bp"] = float(dw[f"cost_{c}"].mean() * C.TD * 1e4)
    out["turnover_daily"] = float(dw["turnover"].mean())
    out["holdings"] = float(dw["n_hold"].mean())
    nav = (1 + rs).cumprod()
    out["max_dd"] = float((nav / nav.cummax() - 1).min())
    out["exposure_max"] = float(dw["exp_max_phase"].max())
    out["exposure_mean"] = float(dw["exposure"].mean())
    return out


def write_check(rows: list[dict]):
    df = pd.DataFrame(rows)
    df["written_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    if CHECKS_CSV.exists():
        old = pd.read_csv(CHECKS_CSV)
        old = old[~old["check"].isin(df["check"].unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(CHECKS_CSV, index=False)


def read_hl_star() -> int:
    ck = pd.read_csv(CHECKS_CSV)
    r = ck[ck["check"] == "hl_star"]
    if not len(r):
        raise RuntimeError("HL* 尚未确定:先跑 `a`")
    return int(r["value"].iloc[0])


# ====================================================================== 子命令
def cmd_checks():
    C.check_prereg()
    p = P()
    rows = []
    # ④ 基线复现
    d = run_cell("base_Y1.0", W_baseline(), "Y1.0", dict(method="base", set="", hl=0, lam=0.0))
    ok_all = True
    for w, ref in C.BASE_REF.items():
        st = window_stats(d, w)
        diff = st["excess_geo"] - ref
        ok = abs(diff) <= 0.0002
        ok_all &= ok
        rows.append(dict(check="baseline_reproduce", item=f"excess_geo {w}", value=st["excess_geo"], reference=ref,
                         diff=diff, passed=ok, note="capacity_grid.csv COMP_v5_exclude|open10|Y1.0|1e7|pov;门槛 0.02pp"))
    st = window_stats(d, "full")
    dt = st["turnover_daily"] - C.BASE_TURN_REF
    ok_all &= abs(dt) <= 0.0005
    rows.append(dict(check="baseline_reproduce", item="turnover_daily full", value=st["turnover_daily"],
                     reference=C.BASE_TURN_REF, diff=dt, passed=abs(dt) <= 0.0005, note="门槛 0.0005"))
    # ① ② 基线
    rows.append(dict(check="exposure_le_1", item="base_Y1.0", value=float(d["exp_max_phase"].max()), reference=1.0,
                     diff=float(d["exp_max_phase"].max()) - 1, passed=float(d["exp_max_phase"].max()) <= 1 + 1e-9, note=""))
    prof = CAP.align_profile(d["ret"], bench()["pool"])
    rows.append(dict(check="align_peak_k0", item="base_Y1.0", value=max(prof, key=prof.get), reference=0, diff=0,
                     passed=max(prof, key=prof.get) == 0, note=json.dumps({str(k): round(v, 4) for k, v in prof.items()})))
    # ③ 全池等权常数打分 5 日调仓 Y=1 连续路径 vs 同频基准:|超额 + 成本| <= 0.5pp
    ones = pd.DataFrame(1.0, index=p.dates, columns=p.codes, dtype=np.float32)
    Wf = ST.build_weights(ones, p, C.VERSION, scheme="exclude", exclude_frac=0.0, date_mask=sim_mask())
    df_ = run_cell("fullpool_Y1.0", Wf, "Y1.0", dict(method="check", set="", hl=0, lam=0.0))
    st = window_stats(df_, "full")
    cost_pp = st["cost_total_bp"] / 100
    resid_g = st["excess_geo"] * 100 + cost_pp
    resid_a = st["excess_arith"] * 100 + cost_pp
    rows.append(dict(check="fullpool_resid_pp", item="Y1.0 1e7 2021-2025 rebalance=5 vs 同频基准",
                     value=resid_g, reference=0.0, diff=resid_g, passed=abs(resid_g) <= 0.5,
                     note=f"excess_geo={st['excess_geo']*100:.3f}pp excess_arith={st['excess_arith']*100:.3f}pp "
                          f"cost={cost_pp:.3f}pp resid_arith={resid_a:.3f}pp;capacity_checks 同项 0.245pp"))
    write_check(rows)
    print(pd.DataFrame(rows).to_string())
    if not ok_all or not all(r["passed"] for r in rows):
        raise SystemExit("停止规则 §11.3:引擎断言 / 基线复现未通过")


def cmd_a():
    """§6.1 网格:HL ∈ {0,2,5,10,20,40} × {D 单独, S_1} × {D_A, D_main};COMP_v5 自身平滑对照;D_B 等 HL* 后在 sens 跑。"""
    C.check_prereg()
    for s in ("A", "main"):
        for hl in C.HL_GRID:
            zd = zD(s, hl)
            run_cell(f"a_D_{s}_hl{hl}", W_excl(zd), "Y1.0", dict(method="a", kind="D_only", set=s, hl=hl, lam=np.nan))
            run_cell(f"a_S1_{s}_hl{hl}", W_excl(zC(0) + zd), "Y1.0", dict(method="a", kind="S1", set=s, hl=hl, lam=1.0))
    for hl in C.HL_GRID:
        if hl == 0:
            continue
        run_cell(f"a_Csm_hl{hl}", W_excl(zC(hl)), "Y1.0", dict(method="a", kind="C_smooth", set="", hl=hl, lam=0.0))
    # HL* 规则:D_A 开发窗 S_1 排雷日换手 <= 1.15 × COMP_v5 排雷开发窗日换手 的最小 HL;都不满足取 40
    base = pd.read_parquet(cell_file("base_Y1.0"))
    tb = window_stats(base, "dev")["turnover_daily"]
    thr = C.TURN_RULE_MULT * tb
    rows, star = [], None
    for hl in C.HL_GRID:
        d = pd.read_parquet(cell_file(f"a_S1_A_hl{hl}"))
        tv = window_stats(d, "dev")["turnover_daily"]
        rows.append(dict(check="hl_rule_turnover", item=f"S1_A_hl{hl} dev turnover", value=tv, reference=thr,
                         diff=tv - thr, passed=tv <= thr, note=f"基线开发窗日换手 {tb:.5f} × {C.TURN_RULE_MULT}"))
        if star is None and tv <= thr:
            star = hl
    if star is None:
        star = 40
    rows.append(dict(check="hl_star", item="HL*(§6.1 换手规则,只看开发窗换手)", value=star, reference=np.nan,
                     diff=np.nan, passed=True, note=f"阈值 {thr:.5f};无满足者时取 40"))
    write_check(rows)
    print(pd.DataFrame(rows).to_string())
    print(f"HL* = {star}", flush=True)


def cmd_b():
    C.check_prereg()
    hs = read_hl_star()
    p = P()
    for s in ("A", "main"):
        for hl in sorted({0, hs}):
            zd = zD(s, hl)
            for e2, x2 in C.B_GRID:
                Wh, We = W_two_layer(zd, e2, x2)
                tag = f"{s}_hl{hl}_{int(e2*100)}_{int(x2*100)}"
                run_cell(f"b_{tag}", Wh, "Y1.0", dict(method="b", kind="two_layer", set=s, hl=hl, lam=np.nan,
                                                      e2=e2, x2=x2))
                if (e2, x2) == C.B_MAIN and hl == hs:
                    # H2:被第二层额外剔除的股票等权组合(旧成本,看毛收益)
                    run_cell(f"bE_{tag}", We, "legacy", dict(method="b_extra", kind="extra_excluded", set=s, hl=hl,
                                                             lam=np.nan, e2=e2, x2=x2))
    for e, x in C.DEEP_GRID:
        Wd = ST.build_weights(comp_raw(), p, C.VERSION, scheme="exclude", buffer=(e, x), date_mask=sim_mask())
        run_cell(f"deep_{int(e*100)}_{int(x*100)}", Wd, "Y1.0", dict(method="deep", kind="comp_deep", set="", hl=0,
                                                                     lam=0.0, e2=e, x2=x))


def cmd_sens():
    """主配置的成本敏感性(legacy / Y1.5)、D′、D_B 主配置、λ 敏感性(只在 HL* 上)。"""
    C.check_prereg()
    hs = read_hl_star()
    for v in ("legacy", "Y1.5"):
        run_cell(f"base_{v}", W_baseline(), v, dict(method="base", set="", hl=0, lam=0.0))
    for s in ("A", "main", "B", "A_nopead", "main_nopead", "B_nopead"):
        zd = zD(s, hs)
        meta = dict(method="a", kind="S1", set=s, hl=hs, lam=1.0)
        run_cell(f"a_S1_{s}_hl{hs}", W_excl(zC(0) + zd), "Y1.0", meta)
        if s == "B":
            run_cell(f"a_D_B_hl{hs}", W_excl(zd), "Y1.0", dict(method="a", kind="D_only", set="B", hl=hs, lam=np.nan))
            z0 = zD("B", 0)
            run_cell("a_D_B_hl0", W_excl(z0), "Y1.0", dict(method="a", kind="D_only", set="B", hl=0, lam=np.nan))
            run_cell("a_S1_B_hl0", W_excl(zC(0) + z0), "Y1.0", dict(method="a", kind="S1", set="B", hl=0, lam=1.0))
        Wh, _ = W_two_layer(zd, *C.B_MAIN)
        tag = f"{s}_hl{hs}_{int(C.B_MAIN[0]*100)}_{int(C.B_MAIN[1]*100)}"
        bmeta = dict(method="b", kind="two_layer", set=s, hl=hs, lam=np.nan, e2=C.B_MAIN[0], x2=C.B_MAIN[1])
        run_cell(f"b_{tag}", Wh, "Y1.0", bmeta)
        if s in ("A", "main"):
            for v in ("legacy", "Y1.5"):
                run_cell(f"a_S1_{s}_hl{hs}_{v}", W_excl(zC(0) + zd), v, meta)
                run_cell(f"b_{tag}_{v}", Wh, v, bmeta)
        if s == "A":
            for lam in (0.5, 2.0):
                run_cell(f"a_S{lam:g}_A_hl{hs}", W_excl(zC(0) + lam * zd), "Y1.0",
                         dict(method="a", kind=f"S{lam:g}", set="A", hl=hs, lam=lam))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["checks", "a", "b", "sens"])
    a = ap.parse_args()
    t0 = time.time()
    {"checks": cmd_checks, "a": cmd_a, "b": cmd_b, "sens": cmd_sens}[a.cmd]()
    print(f"done {a.cmd} {time.time()-t0:.0f}s", flush=True)
