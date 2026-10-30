"""
用账本里已有的头部指标重读全部因子(不重跑任何因子)。

背景:SELECTION_ABILITY_AUDIT.md §4 A.2。评价流水线的"极端组"门槛实际是 IC 符号检查
(fast_eval.py:143 `is_top_long = long_k >= n_q-1 or short_k <= 0`,原值不做方向对齐),
头部指标 long_t / long_excess_ann 只写账本不进门槛。这里做的是:
  1. 读全部 research/ledger/trials_*.jsonl,按 (factor, mode) 取 ts 最新一行,只保留有 q_excess 的行;
  2. 方向对齐:head_group = 10 if ic_mean>0 else 1,tail_group = 另一端;
  3. 复现 extreme_long 的符号 bug;
  4. 全库统计 + 头部因子列表 + 头部因子按 h1 IC 序列去重。

只读账本与 research/ic_series 下的 IC 序列,不碰面板、不碰封存样本。

局限(必须随结果一起读):
  * q_excess 是 h1、毛收益(不扣成本)、vs 池等权 的年化超额;long_t 是该组日超额的普通 t(只对 argmax 组有)。
  * 选优真正该看的是 h5/h10 持有期,但 open/tail 两个 mode 的分层只在 h1 算过。
    账本另有 mode=open_h5(1,673 行)在 h5 上做过分层,但其 q_excess 是 5 日重叠收益 ×252
    (年化虚高约 5 倍)、long_t 是重叠样本普通 t(虚高约 √5);本脚本把它作为补充口径单独输出,
    并做粗略缩放(q_excess/5,long_t/√5),不与 open/tail 混在一起统计。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "research" / "ledger"
IC_DIR = ROOT / "research" / "ic_series"
OUT_CSV = LEDGER / "head_side_screen.csv"
OUT_JSON = LEDGER / "head_side_summary.json"
OUT_CLU = LEDGER / "head_side_clusters.csv"
OUT_TOP = LEDGER / "head_side_top40.csv"

N_Q = 10
IC_THRESHOLD = 0.025          # 原 pass_ic 门槛 (|h1 IC| >= 0.025)
HEAD_T, HEAD_EXC = 2.0, 0.03  # 头部有信息的基本判据
CLUSTER_RHO = 0.7


# ----------------------------------------------------------------------------
# 1. 读账本
# ----------------------------------------------------------------------------
def load_ledger() -> pd.DataFrame:
    rows, bad = [], 0
    for f in sorted(LEDGER.glob("trials_*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                rows.append(d)
    df = pd.DataFrame(rows)
    print(f"[ledger] files={len(list(LEDGER.glob('trials_*.jsonl')))} rows={len(df)} bad_json={bad}")
    df = df[df["q_excess"].notna() & df["mode"].notna() & df["ic_mean"].notna()].copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.sort_values("ts").drop_duplicates(["factor", "mode"], keep="last").reset_index(drop=True)
    print(f"[ledger] after dedup (factor,mode) with q_excess: rows={len(df)} "
          f"factors={df['factor'].nunique()} modes={df['mode'].value_counts().to_dict()}")
    return df


# ----------------------------------------------------------------------------
# 2. 方向对齐
# ----------------------------------------------------------------------------
def align(df: pd.DataFrame, comp_members: set[str]) -> pd.DataFrame:
    q = np.array([np.asarray(x, dtype=float) for x in df["q_excess"]])
    assert q.shape[1] == N_Q, q.shape
    ic = df["ic_mean"].to_numpy(float)
    pos = ic > 0
    head_group = np.where(pos, N_Q, 1)
    tail_group = np.where(pos, 1, N_Q)
    head_exc = q[np.arange(len(df)), head_group - 1]
    tail_exc = q[np.arange(len(df)), tail_group - 1]
    mid = q[:, 2:8].mean(axis=1)                      # 第 3..8 组(中间 6 组)
    head_minus_mid = head_exc - mid
    long_group = df["long_group"].to_numpy(int)
    short_group = df["short_group"].to_numpy(int)
    head_is_best = long_group == head_group
    tail_is_worst = short_group == tail_group
    long_t = pd.to_numeric(df["long_t"], errors="coerce").to_numpy(float)
    head_t = np.where(head_is_best, long_t, np.nan)   # long_t 只对 argmax 组存在
    # 头部第二名:头部不是最好组时,最好组落在哪
    out = df[["factor", "family", "mode", "ts", "ic_mean", "ic_t_nw", "h5_ic_mean", "h10_ic_mean",
              "mono_spearman", "mono_mid_spearman", "long_group", "short_group", "extreme_long",
              "long_excess_ann", "short_excess_ann", "long_t", "spread_ann", "tail_concentration",
              "long_turnover", "pass_ic"]].copy()
    # open_h5 行的 h5 IC 在 ic_mean 里、h1 在 h1_ic_mean 里
    if "h1_ic_mean" in df.columns:
        out["h1_ic_mean"] = df["h1_ic_mean"]
    out["ic_sign"] = np.where(pos, 1, -1)
    out["head_group"] = head_group
    out["tail_group"] = tail_group
    out["head_excess"] = head_exc
    out["tail_excess"] = tail_exc
    out["abs_tail_excess"] = np.abs(tail_exc)
    out["mid6_excess"] = mid
    out["head_minus_mid"] = head_minus_mid
    out["head_is_best"] = head_is_best
    out["tail_is_worst"] = tail_is_worst
    out["head_t"] = head_t
    out["tail_dominates"] = np.abs(tail_exc) > head_exc
    for k in range(N_Q):
        out[f"q{k+1}"] = q[:, k]
    out["pass_ic_recomputed"] = np.abs(ic) >= IC_THRESHOLD
    out["in_comp_v5"] = out["factor"].isin(comp_members)
    out["head_pass"] = head_is_best & (head_t > HEAD_T) & (head_exc > HEAD_EXC)
    out["head_pass_t3"] = head_is_best & (head_t > 3.0) & (head_exc > HEAD_EXC)
    out["head_pass_e5"] = head_is_best & (head_t > HEAD_T) & (head_exc > 0.05)
    out["head_pass_t3_e5"] = head_is_best & (head_t > 3.0) & (head_exc > 0.05)
    return out


# ----------------------------------------------------------------------------
# 3/4. 统计
# ----------------------------------------------------------------------------
def qtiles(s: pd.Series) -> dict:
    s = pd.Series(s).dropna()
    return {k: float(s.quantile(v)) for k, v in
            {"p10": .1, "p25": .25, "p50": .5, "p75": .75, "p90": .9, "p95": .95, "p99": .99}.items()}


def stats_for(d: pd.DataFrame) -> dict:
    n = len(d)
    s = {"n_factors": int(n)}
    # 3. extreme_long bug 复现
    el = d["extreme_long"].astype(bool)
    s["bug_extreme_long"] = {
        "pos_ic_n": int((d.ic_sign > 0).sum()),
        "pos_ic_extreme_long_true_share": float(el[d.ic_sign > 0].mean()),
        "neg_ic_n": int((d.ic_sign < 0).sum()),
        "neg_ic_extreme_long_true_share": float(el[d.ic_sign < 0].mean()),
        "neg_ic_head_is_best_share": float(d.head_is_best[d.ic_sign < 0].mean()),
        "pos_ic_head_is_best_share": float(d.head_is_best[d.ic_sign > 0].mean()),
        "rule": "extreme_long = (argmax==10) or (argmin==1); 正 IC 因子几乎总满足 argmin==1,负 IC 因子两条都难满足",
    }
    s["head_is_best_share"] = float(d.head_is_best.mean())
    s["tail_is_worst_share"] = float(d.tail_is_worst.mean())
    s["tail_dominates_share"] = float(d.tail_dominates.mean())
    s["head_excess_quantiles"] = qtiles(d.head_excess)
    s["abs_tail_excess_quantiles"] = qtiles(d.abs_tail_excess)
    s["head_minus_mid_quantiles"] = qtiles(d.head_minus_mid)
    s["median_head_excess"] = float(d.head_excess.median())
    s["median_abs_tail_excess"] = float(d.abs_tail_excess.median())
    s["median_ratio_abs_tail_over_head"] = float((d.abs_tail_excess / d.head_excess.clip(lower=1e-4)).median())
    # long_group 分布(方向对齐后:head 侧编号统一成 10)
    lg_aligned = np.where(d.ic_sign > 0, d.long_group, N_Q + 1 - d.long_group)
    s["long_group_aligned_dist"] = {int(k): int(v) for k, v in
                                   pd.Series(lg_aligned).value_counts().sort_index().items()}
    # 头部有信息
    s["n_head_is_best"] = int(d.head_is_best.sum())
    s["n_head_t2_e3"] = int(d.head_pass.sum())
    s["n_head_t3_e3"] = int(d.head_pass_t3.sum())
    s["n_head_t2_e5"] = int(d.head_pass_e5.sum())
    s["n_head_t3_e5"] = int(d.head_pass_t3_e5.sum())
    s["n_head_t2_e3_and_tail_worst"] = int((d.head_pass & d.tail_is_worst).sum())
    s["n_head_t2_e3_head_dominates"] = int((d.head_pass & ~d.tail_dominates).sum())
    # 与 pass_ic 交叉
    hp = d[d.head_pass]
    s["cross_pass_ic"] = {
        "head_pass_total": int(len(hp)),
        "head_pass_and_pass_ic": int(hp.pass_ic_recomputed.sum()),
        "head_pass_not_pass_ic": int((~hp.pass_ic_recomputed).sum()),
        "pass_ic_total": int(d.pass_ic_recomputed.sum()),
        "pass_ic_and_head_pass": int((d.pass_ic_recomputed & d.head_pass).sum()),
        "pass_ic_share_among_head_pass": float(hp.pass_ic_recomputed.mean()) if len(hp) else None,
        "head_pass_share_among_pass_ic": float(d.head_pass[d.pass_ic_recomputed].mean()),
        "head_pass_share_among_not_pass_ic": float(d.head_pass[~d.pass_ic_recomputed].mean()),
    }
    # 按 family
    fam = d.groupby("family").agg(
        n=("factor", "size"),
        head_is_best=("head_is_best", "sum"),
        tail_is_worst=("tail_is_worst", "sum"),
        head_pass=("head_pass", "sum"),
        head_pass_t3_e5=("head_pass_t3_e5", "sum"),
        pass_ic=("pass_ic_recomputed", "sum"),
        median_head_excess=("head_excess", "median"),
        median_abs_tail_excess=("abs_tail_excess", "median"),
    )
    fam["head_pass_rate"] = fam.head_pass / fam.n
    fam["pass_ic_rate"] = fam.pass_ic / fam.n
    s["by_family"] = fam.sort_values("n", ascending=False).round(4).to_dict(orient="index")
    # COMP_v5 成员
    cm = d[d.in_comp_v5]
    if len(cm):
        s["comp_v5"] = {
            "n": int(len(cm)),
            "head_is_best": int(cm.head_is_best.sum()),
            "head_pass": int(cm.head_pass.sum()),
            "tail_is_worst": int(cm.tail_is_worst.sum()),
            "tail_dominates": int(cm.tail_dominates.sum()),
            "median_head_excess": float(cm.head_excess.median()),
            "median_abs_tail_excess": float(cm.abs_tail_excess.median()),
        }
    return s


# ----------------------------------------------------------------------------
# 6. 去重
# ----------------------------------------------------------------------------
def cluster(d: pd.DataFrame, mode: str, ic_col: str = "h1") -> tuple[pd.DataFrame, dict]:
    cand = d[d.head_pass].sort_values("head_t", ascending=False)
    series, missing = {}, []
    for name in cand.factor:
        f = IC_DIR / f"{name}__{mode}.parquet"
        if not f.exists():
            missing.append(name)
            continue
        s = pd.read_parquet(f)[ic_col]
        series[name] = s
    if not series:
        return pd.DataFrame(), {"n_candidates": int(len(cand)), "n_with_ic": 0, "n_clusters": 0,
                                "missing": missing}
    M = pd.DataFrame(series)
    # 负 IC 因子翻转符号,使相关性反映"同一信息"而非符号约定
    sign = cand.set_index("factor").ic_sign.reindex(M.columns).to_numpy()
    M = M * sign
    C = M.corr(min_periods=200)
    order = [n for n in cand.factor if n in C.index]     # 已按 head_t 降序
    rep_of, clusters = {}, []
    for n in order:
        if n in rep_of:
            continue
        rep_of[n] = n
        members = [n]
        for m in order:
            if m in rep_of:
                continue
            r = C.loc[n, m]
            if np.isfinite(r) and abs(r) >= CLUSTER_RHO:
                rep_of[m] = n
                members.append(m)
        clusters.append((n, members))
    rows = []
    for cid, (rep, members) in enumerate(clusters, 1):
        for m in members:
            r = cand[cand.factor == m].iloc[0]
            rows.append({"cluster": cid, "representative": rep, "is_rep": m == rep, "factor": m,
                         "family": r.family, "rho_to_rep": float(C.loc[rep, m]),
                         "ic_mean": r.ic_mean, "h5_ic_mean": r.h5_ic_mean,
                         "head_excess": r.head_excess, "tail_excess": r.tail_excess,
                         "head_t": r.head_t, "head_minus_mid": r.head_minus_mid,
                         "mono_spearman": r.mono_spearman, "in_comp_v5": bool(r.in_comp_v5),
                         "pass_ic": bool(r.pass_ic_recomputed)})
    cl = pd.DataFrame(rows)
    reps = cl[cl.is_rep].copy()
    reps["cluster_size"] = reps.cluster.map(cl.cluster.value_counts())
    # 簇代表之间的最大相关(应 <0.7)
    rep_names = reps.factor.tolist()
    Cr = C.loc[rep_names, rep_names].abs().to_numpy()
    np.fill_diagonal(Cr, np.nan)
    info = {"n_candidates": int(len(cand)), "n_with_ic": int(len(series)), "n_clusters": int(len(clusters)),
            "missing_ic_series": missing,
            "max_abs_rho_between_reps": float(np.nanmax(Cr)) if len(rep_names) > 1 else None,
            "cluster_sizes": reps.set_index("factor").cluster_size.to_dict(),
            "representatives": reps[["cluster", "factor", "family", "cluster_size", "head_t", "head_excess",
                                     "tail_excess", "ic_mean", "h5_ic_mean", "in_comp_v5", "pass_ic"]]
                                .round(4).to_dict(orient="records")}
    return cl, info


# ----------------------------------------------------------------------------
def main():
    comp = json.load(open(LEDGER / "comp_v5_members.json"))
    comp_members = set(comp["members"])
    df = load_ledger()
    out = align(df, comp_members)

    # open_h5 补充口径:粗缩放(见模块 docstring)
    m5 = out["mode"] == "open_h5"
    if m5.any():
        for c in ["head_excess", "tail_excess", "abs_tail_excess", "mid6_excess", "head_minus_mid",
                  "long_excess_ann", "short_excess_ann", "spread_ann"] + [f"q{k+1}" for k in range(N_Q)]:
            out.loc[m5, c] = out.loc[m5, c] / 5.0
        for c in ["head_t", "long_t"]:
            out.loc[m5, c] = out.loc[m5, c] / np.sqrt(5.0)
        out.loc[m5, "tail_dominates"] = out.loc[m5, "abs_tail_excess"] > out.loc[m5, "head_excess"]
        hb = out.loc[m5, "head_is_best"]
        out.loc[m5, "head_pass"] = hb & (out.loc[m5, "head_t"] > HEAD_T) & (out.loc[m5, "head_excess"] > HEAD_EXC)
        out.loc[m5, "head_pass_t3"] = hb & (out.loc[m5, "head_t"] > 3) & (out.loc[m5, "head_excess"] > HEAD_EXC)
        out.loc[m5, "head_pass_e5"] = hb & (out.loc[m5, "head_t"] > HEAD_T) & (out.loc[m5, "head_excess"] > 0.05)
        out.loc[m5, "head_pass_t3_e5"] = hb & (out.loc[m5, "head_t"] > 3) & (out.loc[m5, "head_excess"] > 0.05)
        out.loc[m5, "scaled_from_h5_overlap"] = True
    if "scaled_from_h5_overlap" not in out.columns:
        out["scaled_from_h5_overlap"] = False
    out["scaled_from_h5_overlap"] = out["scaled_from_h5_overlap"].eq(True)

    out.to_csv(OUT_CSV, index=False)

    summary = {
        "caveats": [
            "q_excess 为 h1 持有期、毛收益(未扣成本)、相对池等权 的年化超额;head_excess/tail_excess 直接取自它。",
            "long_t 只对 argmax 组存在,故 head_t 仅在 head_is_best=True 时有值;head_is_best=False 的因子头部 t 未知(不是 0)。",
            "选优真正应看 h5/h10 持有期,但 open/tail 两口径分层只在 h1 算过;open_h5 口径的分层是 5 日重叠收益,"
            "本脚本按 q_excess/5、t/sqrt(5) 粗缩放,仅作补充参考。",
            "head_t 是日超额普通 t,无 Newey-West;h1 不重叠,基本可用。",
            "去重相关用 h1 IC 序列(负 IC 因子翻转符号后)计算,|rho|>=0.7 同簇,按 head_t 降序贪心保留。",
        ],
        "thresholds": {"head_t": HEAD_T, "head_excess": HEAD_EXC, "ic_threshold": IC_THRESHOLD,
                       "cluster_rho": CLUSTER_RHO},
        "modes": {},
    }
    top_tables = []
    cluster_tables = []
    for mode in ["open", "tail", "open_h5"]:
        d = out[out["mode"] == mode].copy()
        if d.empty:
            continue
        s = stats_for(d)
        cols = ["factor", "family", "ic_mean", "h5_ic_mean", "h10_ic_mean", "head_excess", "tail_excess",
                "head_minus_mid", "head_t", "mono_spearman", "head_is_best", "tail_is_worst", "in_comp_v5",
                "pass_ic_recomputed"]
        top_e = d[d.head_is_best].sort_values("head_excess", ascending=False).head(40)[cols]
        top_t = d.sort_values("head_t", ascending=False).head(40)[cols]
        top_e = top_e.assign(mode=mode, ranking="head_excess")
        top_t = top_t.assign(mode=mode, ranking="head_t")
        top_tables += [top_e, top_t]
        s["top40_by_head_excess"] = top_e.round(4).to_dict(orient="records")
        s["top40_by_head_t"] = top_t.round(4).to_dict(orient="records")
        cl, info = cluster(d, mode)
        s["clusters"] = info
        if len(cl):
            cluster_tables.append(cl.assign(mode=mode))
        # COMP_v5 成员逐个
        cm = d[d.in_comp_v5][["factor", "ic_mean", "head_excess", "tail_excess", "head_t", "head_is_best",
                              "tail_is_worst", "long_group", "short_group", "head_group", "mono_spearman"]]
        s["comp_v5_members"] = cm.round(4).to_dict(orient="records")
        summary["modes"][mode] = s
        print(f"\n===== mode={mode} n={s['n_factors']} =====")
        print(f"bug: pos-IC extreme_long True share={s['bug_extreme_long']['pos_ic_extreme_long_true_share']:.3f} "
              f"(n={s['bug_extreme_long']['pos_ic_n']}), neg-IC True share="
              f"{s['bug_extreme_long']['neg_ic_extreme_long_true_share']:.3f} (n={s['bug_extreme_long']['neg_ic_n']})")
        print(f"head_is_best={s['head_is_best_share']:.3f} tail_is_worst={s['tail_is_worst_share']:.3f} "
              f"tail_dominates={s['tail_dominates_share']:.3f}")
        print(f"median head_excess={s['median_head_excess']:.4f} median |tail_excess|={s['median_abs_tail_excess']:.4f}")
        print(f"head_pass t>2&e>3%: {s['n_head_t2_e3']}  t>3&e>3%: {s['n_head_t3_e3']}  "
              f"t>2&e>5%: {s['n_head_t2_e5']}  t>3&e>5%: {s['n_head_t3_e5']}")
        print("cross pass_ic:", s["cross_pass_ic"])
        print("clusters:", {k: v for k, v in info.items() if k not in ("representatives", "cluster_sizes")})
        print(pd.DataFrame(s["by_family"]).T[["n", "head_pass", "head_pass_rate", "pass_ic_rate"]])

    pd.concat(top_tables).to_csv(OUT_TOP, index=False)
    if cluster_tables:
        pd.concat(cluster_tables).to_csv(OUT_CLU, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=lambda o: None if pd.isna(o) else o)
    print("\nwrote", OUT_CSV, OUT_JSON, OUT_CLU, OUT_TOP)


if __name__ == "__main__":
    sys.exit(main())
