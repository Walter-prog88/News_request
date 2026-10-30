"""PEAD 市值修复重跑的审查补充核对(只读研究期数据,不读任何收益,不改任何已有账本或缓存)。

问题:原运行的 B / C / B_F 对照是按旧口径市值匹配的。用修正后的真实市值给这些"旧对照"重新算平衡性,
log 市值的 SMD 有多大?如果很小,说明检验窗 E−B 从 +1.51 变为 +0.82pp/年主要来自对照重新抽取的噪声,
而不是修正了一个市值偏差。

做法:
  - 新口径 log 市值:与 scripts/pead_build_batches.py 完全相同(sealed_daily_basic + sealed_total_mv,
    财报 avail <= 2025-12-31;缺失时用 log 流通市值补),直接调用该脚本的函数;
  - 旧对照 = cache/_tmp/pre_mvfix_backup/pead_batches.parquet,新对照 = cache/_tmp/pead_batches.parquet;
    处理组(E、F)与对照都在新口径市值上取值,窗口按 t_signal,分组与 SMD 公式同 pead_build_batches.balance_table;
  - 自检:新对照在新口径上的 SMD 必须与 research/ledger/pead_balance.csv 一致;
    旧对照在其落盘匹配变量(旧口径)上的 SMD 必须与 research/ledger/_pre_mvfix/pead_balance.csv 一致。

产出:research/ledger/pead_mvfix_oldctrl_truesize.csv(新文件)
日志:logs/pead_mvfix_review_checks.log
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from afactor.config import TMP, LEDGER
from afactor.panel import Panel
from afactor import newdata as ND, earnings_events as EE
import pead_build_batches as PB

PAIRS = (("E", "B"), ("E", "C"), ("F", "B_F"))


def smd(xt: pd.Series, xc: pd.Series) -> float:
    xt, xc = xt.dropna(), xc.dropna()
    s = np.sqrt((xt.var() + xc.var()) / 2.0)
    return float((xt.mean() - xc.mean()) / s) if s > 0 else np.nan


def table(V: pd.DataFrame, col: str) -> dict:
    out = {}
    for arm_t, arm_c in PAIRS:
        trt = V[V["arm"] == arm_t]
        ctl = V[(V["arm"] == arm_c) & V["ok"]]
        for wn, (a, b) in PB.WINDOWS.items():
            x = trt[(trt["t_signal"] >= a) & (trt["t_signal"] <= b)]
            y = ctl[(ctl["t_signal"] >= a) & (ctl["t_signal"] <= b)]
            out[(arm_c, arm_t, wn)] = (smd(x[col], y[col]), len(x), len(y))
    return out


def main():
    t0 = time.time()
    p = Panel()
    dates, codes = p.dates, p.codes
    db = PB.sealed_daily_basic()
    assert db["trade_date"].max() <= pd.Timestamp(EE.CUTOFF), "daily_basic 未截断"
    print(f"[{time.time()-t0:.0f}s] daily_basic 截断后最大日期 {db['trade_date'].max().date()},{len(db)} 行", flush=True)
    del db
    mv = PB.sealed_total_mv(dates, codes)
    circ = ND.circ_mv(dates, codes)
    L = np.log(mv.where(mv > 0)).fillna(np.log(circ.where(circ > 0))).to_numpy(np.float64)
    del mv, circ
    print(f"[{time.time()-t0:.0f}s] 新口径 log 市值就绪 {L.shape}", flush=True)

    rows = []
    rec_new = pd.read_csv(LEDGER / "pead_balance.csv")
    rec_old = pd.read_csv(LEDGER / "_pre_mvfix" / "pead_balance.csv")
    for tag, bpath, vpath, rec in (
            ("原运行对照", TMP / "pre_mvfix_backup" / "pead_batches.parquet",
             TMP / "pre_mvfix_backup" / "pead_match_vars.parquet", rec_old),
            ("新口径对照", TMP / "pead_batches.parquet", TMP / "pead_match_vars.parquet", rec_new)):
        B = pd.read_parquet(bpath, columns=["arm", "rep", "T_pos", "t_pos", "slot", "j", "t_signal"])
        V = pd.read_parquet(vpath, columns=["arm", "rep", "T_pos", "slot", "t_signal", "ok", "logmv"])
        assert len(B) == len(V)
        for c in ("arm", "rep", "T_pos", "slot"):
            assert (B[c].to_numpy() == V[c].to_numpy()).all(), f"{tag}: 批次表与匹配变量行序不一致({c})"
        ok = (B["j"] >= 0).to_numpy()
        jj = B["j"].clip(lower=0).to_numpy(); ii = B["t_pos"].to_numpy()
        V["logmv_true"] = np.where(ok, L[ii, jj], np.nan)
        stored = table(V, "logmv")       # 落盘的匹配变量(各自当时的口径)
        true = table(V, "logmv_true")    # 新口径真实市值
        for key, (s_true, nt, nc) in true.items():
            arm_c, arm_t, wn = key
            r = rec[(rec["control"] == arm_c) & (rec["treated"] == arm_t) & (rec["window"] == wn)]
            s_rec = float(r["smd_logmv"].iloc[0])
            rows.append(dict(controls=tag, control=arm_c, treated=arm_t, window=wn,
                             n_treated_slots=nt, n_control_slots_ok=nc,
                             smd_logmv_true=s_true, smd_logmv_stored=stored[key][0],
                             smd_logmv_ledger=s_rec,
                             absdiff_stored_vs_ledger=abs(stored[key][0] - s_rec)))
        print(f"[{time.time()-t0:.0f}s] {tag} 完成", flush=True)

    out = pd.DataFrame(rows)
    # 自检:落盘匹配变量重算的 SMD 与各自的 pead_balance 一致;新对照的落盘值 = 新口径真实值
    assert out["absdiff_stored_vs_ledger"].max() < 1e-12, out
    nw = out[out["controls"] == "新口径对照"]
    assert (nw["smd_logmv_true"] - nw["smd_logmv_stored"]).abs().max() < 1e-12, nw
    out.to_csv(LEDGER / "pead_mvfix_oldctrl_truesize.csv", index=False)
    print(out.to_string())
    print(f"完成 [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
