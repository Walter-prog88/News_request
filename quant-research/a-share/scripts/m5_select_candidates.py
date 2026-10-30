"""5 分钟近似候选的选定(前缀 m5):只用 2020 年选,2021-2025 只报告。

选定准则(minute5 模块文档预先写定):
  twap_open10 / vwap   V1000 决策池股票日 |log(候选/1 分钟真值)| 中位数最小,并列看 99% 分位
  amt_open10           |log(候选/真值)| 中位数最小
  amt_openauc_share    成员因子 id_amt_openauc_share_m5(5 日均值)的逐日截面秩相关均值最大
2021-2025 的同口径数字一并写入账本,但**不参与选择**。
产出:research/ledger/m5_candidate_selection.csv
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, DAILY
from afactor.panel import Panel
from afactor import strategy as ST, ops, minute5 as M5
from afactor.build_daily_5m import CANDIDATE_GROUPS

WINDOWS = {"2020(选定)": (2020, 2020), "2021-2025(仅报告)": (2021, 2025)}


def cs_spearman(a: pd.DataFrame, b: pd.DataFrame, mask: pd.DataFrame, min_n: int = 200) -> pd.Series:
    """逐日截面 Spearman(限定 mask 内两者都有限的股票)。"""
    A = a.where(mask).rank(axis=1); B = b.where(mask).rank(axis=1)
    ok = A.notna() & B.notna()
    A = A.where(ok); B = B.where(ok)
    A = A.sub(A.mean(axis=1), axis=0); B = B.sub(B.mean(axis=1), axis=0)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A ** 2).sum(axis=1) * (B ** 2).sum(axis=1))
    out = num / den.replace(0, np.nan)
    return out.where(ok.sum(axis=1) >= min_n)


def main():
    p = Panel()
    U = ST.universe(p, "V1000")
    yrs = p.dates.year
    rows = []
    for fld in ("twap_open10", "amt_open10", "vwap"):
        truth = p.raw(fld)
        for cand in _cands(fld):
            c = pd.read_parquet(M5.M5_DIR / f"{fld}__{cand}.parquet").astype(np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                e = np.log(c / truth) * 1e4                       # bp
            for wn, (y0, y1) in WINDOWS.items():
                m = U.mul(pd.Series((yrs >= y0) & (yrs <= y1), index=p.dates), axis=0).astype(bool)
                v = e.where(m & (truth > 0) & (c > 0)).to_numpy().ravel()
                v = v[np.isfinite(v)]
                # 收益误差:log(c[t+1]/c[t]) - log(truth[t+1]/truth[t])(回测只用相邻两日之比)
                er = (e.shift(-1) - e).where(m).to_numpy().ravel(); er = er[np.isfinite(er)]
                rows.append(dict(field=fld, candidate=cand, window=wn, n=v.size,
                                 abs_median_bp=float(np.median(np.abs(v))), abs_p99_bp=float(np.percentile(np.abs(v), 99)),
                                 mean_bp=float(v.mean()), median_bp=float(np.median(v)), std_bp=float(v.std()),
                                 p01_bp=float(np.percentile(v, 1)), p99_bp=float(np.percentile(v, 99)),
                                 ret_err_std_bp=float(er.std()) if er.size else np.nan,
                                 ret_err_abs_median_bp=float(np.median(np.abs(er))) if er.size else np.nan))
        print(f"{fld} 完成", flush=True)

    # 占比类:按成员因子(5 日均值)的截面秩相关选
    truth = ops.ts_mean(p.raw("amt_openauc_share"), 5)
    for cand in _cands("amt_openauc_share"):
        c = ops.ts_mean(pd.read_parquet(M5.M5_DIR / f"amt_openauc_share__{cand}.parquet").astype(np.float32), 5)
        raw_c = pd.read_parquet(M5.M5_DIR / f"amt_openauc_share__{cand}.parquet").astype(np.float32)
        rc_f = cs_spearman(c, truth, U)
        rc_r = cs_spearman(raw_c, p.raw("amt_openauc_share"), U)
        for wn, (y0, y1) in WINDOWS.items():
            sel = (yrs >= y0) & (yrs <= y1)
            rows.append(dict(field="amt_openauc_share", candidate=cand, window=wn,
                             n=int(rc_f[sel].notna().sum()),
                             factor_m5_cs_spearman_mean=float(rc_f[sel].mean()),
                             raw_cs_spearman_mean=float(rc_r[sel].mean())))
    df = pd.DataFrame(rows)

    # 选定(只看 2020)
    chosen = {}
    sel = df[df.window == "2020(选定)"]
    for fld in ("twap_open10", "vwap", "amt_open10"):
        d = sel[sel.field == fld].sort_values(["abs_median_bp", "abs_p99_bp"])
        chosen[fld] = d.iloc[0]["candidate"]
    d = sel[sel.field == "amt_openauc_share"].sort_values("factor_m5_cs_spearman_mean", ascending=False)
    chosen["amt_openauc_share"] = d.iloc[0]["candidate"]
    df["chosen"] = [chosen.get(f) == c for f, c in zip(df.field, df.candidate)]
    df.to_csv(LEDGER / "m5_candidate_selection.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n选定:", chosen)
    print("当前 minute5 常量:", {f: getattr(M5, CANDIDATE_GROUPS[f][0]) for f in CANDIDATE_GROUPS})


def _cands(fld):
    import re
    return sorted({re.sub(rf"^{fld}__", "", f.stem) for f in M5.M5_DIR.glob(f"{fld}__*.parquet")})


if __name__ == "__main__":
    main()
