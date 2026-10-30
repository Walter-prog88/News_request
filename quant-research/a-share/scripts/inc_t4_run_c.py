"""INC-T4 方法 (c):指增优化器里两个 α 一起进目标(预注册 §6.3)。

全部沿用 scripts/run_index_enhance.py 的 V1 设定(OptConfig() 默认:TE 5%、行业 ±2%、Size/NonlinSize ±0.1、
其余风格 ±0.5、个股 +1%、成本进目标 κ=1、λ_risk=1;open10、1000 万、V1000 决策池、CostModel(impact_y=1)、
5 日调仓、2021-04-20 起),只把 α 打分换成 S_λ = z(COMP_v5) + λ·z(D_HL)(两项都是 p["universe"] 内截面 z),
IC 预测按原规则对 S_λ 重算:rank_ic_series(S, p.fwd_ret(5,"open"), V1000 池) -> ic_for_decision(250, lag 6, min 20, floor 0)。
判定:相位 0 对 cache/_tmp/ie_V1_ph0_daily.parquet(V1_ph0)的日收益配对差;主配置过相位 0 门槛后补 1~4 相位与 V1 五相位配对。

用法:
  python3 scripts/inc_t4_run_c.py prep                 # 打分 + IC 缓存(cache/_tmp/inc_t4_ie_ic_<kind>.parquet)
  python3 scripts/inc_t4_run_c.py run --job JOB --phase 0
  python3 scripts/inc_t4_run_c.py drive --procs 3 [--only J1 J2 ...]
作业名见 jobs();日频明细落 cache/_tmp/inc_t4_ie_<job>_ph<k>_{daily,solver}.parquet 与 _summary.json。
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

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t4_common as C  # noqa: E402
from afactor.config import TMP, LEDGER  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Solution may be inaccurate")

START = pd.Timestamp("2021-04-20")
END = pd.Timestamp("2025-12-31")
_C: dict = {}


def P():
    if "p" not in _C:
        from afactor.panel import Panel
        _C["p"] = Panel()
    return _C["p"]


def hl_star() -> int:
    ck = pd.read_csv(LEDGER / f"{C.PFX}_checks.csv")
    return int(ck[ck["check"] == "hl_star"]["value"].iloc[0])


def jobs() -> dict:
    """作业表(§6.3 网格 + §6.4 敏感性 + 复现 V1_ph0)。score: (set, hl, lam_C, lam_D)。"""
    hs = hl_star()
    J = {
        "V1rep": dict(score=None, cost="Y1.0", desc="复现 V1_ph0(打分 = COMP_v5 原始缓存,ie_ic_comp 同法重算)"),
        f"S1_A_hl{hs}": dict(score=("A", hs, 1.0, 1.0), cost="Y1.0", desc="主配置 λ=1、HL*"),
        f"S0.5_A_hl{hs}": dict(score=("A", hs, 1.0, 0.5), cost="Y1.0", desc="λ=0.5、HL*"),
        "S1_A_hl0": dict(score=("A", 0, 1.0, 1.0), cost="Y1.0", desc="λ=1、HL=0"),
        f"D_A_hl{hs}": dict(score=("A", hs, 0.0, 1.0), cost="Y1.0", desc="D_HL* 单独(COMP_v5 权重 0)"),
        f"S1_main_hl{hs}": dict(score=("main", hs, 1.0, 1.0), cost="Y1.0", desc="D_main 主配置"),
        f"S1_B_hl{hs}": dict(score=("B", hs, 1.0, 1.0), cost="Y1.0", desc="D_B 主配置(§8 条件 3 需要)"),
        f"S1_A_nopead_hl{hs}": dict(score=("A_nopead", hs, 1.0, 1.0), cost="Y1.0", desc="D′ 主配置"),
        f"S1_main_nopead_hl{hs}": dict(score=("main_nopead", hs, 1.0, 1.0), cost="Y1.0", desc="D_main′ 主配置"),
        f"S1_A_hl{hs}_Y1.5": dict(score=("A", hs, 1.0, 1.0), cost="Y1.5", desc="主配置 Y=1.5(引擎与优化器同)"),
        "V1_Y1.5": dict(score=None, cost="Y1.5", desc="V1_ph0 在 Y=1.5 下重跑(配对基线)"),
        f"S1_A_hl{hs}_legacy": dict(score=("A", hs, 1.0, 1.0), cost="legacy", desc="主配置旧成本(引擎与优化器同)"),
        "V1_legacy": dict(score=None, cost="legacy", desc="V1_ph0 在旧成本下重跑(配对基线)"),
    }
    return J


def score_kind(spec) -> str:
    if spec is None:
        return "comp"
    s, hl, lc, ld = spec
    return f"{s}_hl{hl}_c{lc:g}_d{ld:g}"


def load_score(spec) -> pd.DataFrame:
    from afactor import ops
    p = P()
    comp = pd.read_parquet(TMP / "comp_v5_score_open10.parquet").reindex(index=p.dates, columns=p.codes)
    if spec is None:
        return comp
    s, hl, lc, ld = spec
    u = p["universe"]
    zd = ops.standardize(C.ewm_smooth(pd.read_parquet(C.d_path(s)), hl), u)
    if lc == 0:
        return zd.astype(np.float32)
    zc = ops.standardize(comp, u)
    return (lc * zc + ld * zd).astype(np.float32)


def ic_dec(spec) -> pd.Series:
    from afactor import optimizer as OPT, strategy as ST
    kind = score_kind(spec)
    f = TMP / f"{C.PFX}_ie_ic_{kind}.parquet"
    if f.exists():
        return pd.read_parquet(f)["ic_dec"]
    p = P()
    U = ST.universe(p, "V1000")
    ic = OPT.rank_ic_series(load_score(spec), p.fwd_ret(5, "open"), U)
    cfg = OPT.OptConfig()
    icd = OPT.ic_for_decision(ic, cfg.ic_window, cfg.ic_lag, cfg.ic_min_obs, 0.0)
    pd.DataFrame({"ic": ic, "ic_dec": icd}).to_parquet(f)
    return icd


def run_job(name: str, phase: int) -> None:
    from afactor import strategy as ST, index_proxy as IP, optimizer as OPT
    from afactor.barra import RiskModel
    from afactor.impact import CostModel
    t0 = time.time()
    sp = jobs()[name]
    p = P()
    rm = RiskModel.load("open10")
    W = IP.proxy_weights(p, "000852")
    U = ST.universe(p, "V1000")
    score = load_score(sp["score"])
    icd = ic_dec(sp["score"])
    cfg = OPT.OptConfig()
    cm = None if sp["cost"] == "legacy" else CostModel(p, impact_y=float(sp["cost"][1:]))
    enh = OPT.IndexEnhancer(p, rm, W, score, icd, U, cfg, tag=name)
    d = p.dates
    mask = np.asarray((d >= START) & (d <= END))
    print(f"[{name} ph{phase}] cost={sp['cost']} score={score_kind(sp['score'])} 载入 {time.time()-t0:.0f}s", flush=True)
    res = ST.simulate(enh, p, "open10", 1e7, rebalance=5, date_mask=mask, phase=phase,
                      cost_model=cm, record=True, participation_on_nav=False)
    prefix = f"{C.PFX}_ie_{name}_ph{phase}"
    rec = ST.record_to_frames(res)
    rec["daily"].to_parquet(TMP / f"{prefix}_daily.parquet")
    H = rec["holdings"]
    H.loc[:, (H != 0).any(axis=0)].to_parquet(TMP / f"{prefix}_holdings.parquet")
    lg = enh.log_frame()
    lg.to_parquet(TMP / f"{prefix}_solver.parquet")
    summ = {k: res[k] for k in ("ann_return", "sharpe", "max_dd", "turnover", "cost_bps", "holdings",
                                 "exposure", "n_days", "vol")}
    summ.update(job=name, phase=phase, n_solves=len(lg), n_failed=int((lg["status"] == "failed").sum()),
                exp_max=float(rec["daily"]["exposure"].max()), elapsed_s=time.time() - t0,
                cost=sp["cost"], score=score_kind(sp["score"]), desc=sp["desc"])
    json.dump(summ, open(TMP / f"{prefix}_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(f"[{name} ph{phase}] 年化 {res['ann_return']:.2%} 换手 {res['turnover']:.4f} 求解 {len(lg)} "
          f"失败 {summ['n_failed']} 用时 {time.time()-t0:.0f}s", flush=True)


def done(name, ph) -> bool:
    return (TMP / f"{C.PFX}_ie_{name}_ph{ph}_summary.json").exists()


def drive(procs: int, only=None, phases=(0,)):
    J = jobs()
    for nm, sp in J.items():                 # IC 缓存先在主进程建好
        ic_dec(sp["score"])
    tasks = [(nm, ph) for nm in J for ph in phases if (not only or nm in only)]
    todo = [x for x in tasks if not done(*x)]
    print(f"作业 {len(tasks)},待跑 {len(todo)},并行 {procs}", flush=True)
    running = []
    script = os.path.abspath(__file__)
    while todo or running:
        while todo and len(running) < procs:
            nm, ph = todo.pop(0)
            lf = C.LOGS / f"{C.PFX}_ie_{nm}_ph{ph}.log"
            fh = open(lf, "w")
            pr = subprocess.Popen([sys.executable, script, "run", "--job", nm, "--phase", str(ph)],
                                  stdout=fh, stderr=subprocess.STDOUT)
            running.append((nm, ph, pr, fh, time.time()))
            print(f"  启动 {nm} ph{ph}", flush=True)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run", "drive"])
    ap.add_argument("--job")
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--phases", nargs="*", type=int, default=[0])
    a = ap.parse_args()
    C.check_prereg()
    if a.cmd == "prep":
        for nm, sp in jobs().items():
            ic_dec(sp["score"])
            print("ic", nm, flush=True)
    elif a.cmd == "run":
        run_job(a.job, a.phase)
    else:
        drive(a.procs, a.only, tuple(a.phases))
