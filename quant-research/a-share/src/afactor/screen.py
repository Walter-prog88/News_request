"""筛选漏斗:账本汇总 -> 统计门槛 -> 方向一致性 -> IC 序列去重 -> 分层形态 -> 残差 IC。"""
from __future__ import annotations
import glob, json
import numpy as np
import pandas as pd

from afactor.config import LEDGER, IC_THRESHOLD
from afactor.fast_eval import IC_DIR


def load_ledger(pattern: str = "trials*.jsonl") -> pd.DataFrame:
    rows = []
    # 必须按**写入时间**排序,不能按文件名字符串排序:trials_10086(08-22 重评)
    # 字符串序排在 trials_4829(08-20 初评)之前,配合 keep="last" 会保留**旧**记录。
    # 实测 19 个因子有多份 open 记录且 IC 极差 >0.005,其中 5 个因此被错误地判为过关
    # (lit_vola_overnight_vol_share_w5/w20/w60 真实 IC 只有 -0.003~-0.005)。
    for f in sorted(LEDGER.glob(pattern), key=lambda p: (p.stat().st_mtime, p.name)):
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["factor", "mode"], keep="last")
    return df.reset_index(drop=True)


def load_ic_matrix(names: list[str], mode: str = "open", col: str = "h1") -> pd.DataFrame:
    out = {}
    for n in names:
        f = IC_DIR / f"{n}__{mode}.parquet"
        if f.exists():
            s = pd.read_parquet(f)[col]
            out[n] = s
    return pd.DataFrame(out)


# ---------------- 漏斗各关 ----------------
def gate_stats(df: pd.DataFrame, ic_thr: float = IC_THRESHOLD, t_thr: float = 3.0) -> pd.DataFrame:
    d = df[df["error"].isna()] if "error" in df.columns else df
    d = d[d["ic_mean"].notna()].copy()
    d["abs_ic"] = d["ic_mean"].abs()
    d["abs_t"] = d["ic_t"].abs()
    return d[(d["abs_ic"] >= ic_thr) & (d["abs_t"] >= t_thr)].copy()


def gate_consistency(d: pd.DataFrame, year_req: float = 1.0, win_req: float = 1.0,
                     min_worst_year: float = 0.0) -> pd.DataFrame:
    m = ((d["year_sign_consistency"] >= year_req) &
         (d["window_sign_consistency"] >= win_req) &
         (d["worst_year_abs_ic"] >= min_worst_year))
    return d[m].copy()


def dedupe_by_ic_corr(d: pd.DataFrame, mode: str = "open", thr: float = 0.7,
                      rank_col: str = "worst_window_abs_ic") -> tuple[pd.DataFrame, dict]:
    """按 IC 序列相关度贪心去重:优先保留 rank_col 最大的因子。"""
    names = d.sort_values(rank_col, ascending=False)["factor"].tolist()
    M = load_ic_matrix(names, mode)
    # 方向对齐后再算相关(反向因子与正向因子承载同一信息)
    sign = d.set_index("factor")["ic_mean"].reindex(M.columns).apply(np.sign).fillna(1.0)
    Ms = M.mul(sign, axis=1)
    C = Ms.corr(min_periods=200)
    keep, dropped = [], {}
    for n in names:
        if n not in C.columns:
            continue
        dup = next((k for k in keep if abs(C.loc[n, k]) >= thr), None)
        if dup is None:
            keep.append(n)
        else:
            dropped[n] = dup
    return d[d["factor"].isin(keep)].copy(), dropped


def gate_layers(d: pd.DataFrame, min_mono: float = 0.5, require_extreme: bool = True) -> pd.DataFrame:
    m = d["mono_spearman"].abs() >= min_mono
    if require_extreme:
        m &= d["extreme_long"].fillna(False)
    return d[m].copy()


def summary_table(d: pd.DataFrame) -> pd.DataFrame:
    cols = ["factor", "family", "ic_mean", "ic_t", "ic_ir", "ic_pos",
            "year_sign_consistency", "window_sign_consistency", "worst_year_abs_ic",
            "worst_window_abs_ic", "mono_spearman", "long_group", "long_excess_ann",
            "spread_ann", "long_turnover", "tail_concentration", "coverage",
            "h5_ic_mean", "h10_ic_mean", "note"]
    cols = [c for c in cols if c in d.columns]
    out = d[cols].copy().sort_values("worst_window_abs_ic", ascending=False)
    return out


def funnel(mode: str = "open", verbose: bool = True) -> dict:
    df = load_ledger()
    df = df[df.get("mode", mode) == mode] if "mode" in df.columns else df
    stages = {"0_全部试验": len(df)}
    if "error" in df.columns:
        stages["0b_成功计算"] = int(df["error"].isna().sum())
    g1 = gate_stats(df); stages["1_IC与t值达标"] = len(g1)
    g2 = gate_consistency(g1); stages["2_分年与滚动窗口方向一致"] = len(g2)
    g3, dropped = dedupe_by_ic_corr(g2, mode); stages["3_IC序列去重"] = len(g3)
    g4 = gate_layers(g3); stages["4_分层单调且极端组为多头"] = len(g4)
    if verbose:
        for k, v in stages.items():
            print(f"  {k}: {v}")
    return {"stages": stages, "all": df, "g1": g1, "g2": g2, "g3": g3, "g4": g4,
            "dropped": dropped}
