"""财报 PIT 的滚动原点检验(前缀 fixb,2026-10-10 第二轮审查后)。

问题:报告表(fundamental.build_report_level)在时点 t 给出的取值,是否等于"只用 t 时已公告的数据"重建的取值?
做法:对每个原点 t,用 config.set_data_cutoff(t)(收紧,任何代码都可用)重建报告表 = 真值 rep_t;与默认截止日
(2025-12-31)的报告表 rep_full 在 t 日按 _pit_daily 的选行规则逐股比较。差异分类:
  full_newer_period   默认构造用了比真值更新的报告期            -> 疑似前视
  same_period_final   同一报告期、默认用最终快照行但取值不同      -> 疑似前视(用了 t 之后的版本)
  same_period_early   同一报告期、默认用 as-of 快照行但取值不同   -> 版本处理依赖未来(第二轮审查 P2)
  full_older          默认构造用了比真值更旧的报告期            -> 偏陈旧(保守)
  only_full / only_tr 只有一边有值(only_tr = 默认构造缺值,偏陈旧)
已知映射错误的列(sue_bt / sue_ta / earn_streak / earn_vol_q / d_ni_q,§4.1)单独标注,它们的 full_newer 是既有前视。

产出:research/ledger/fixb_rolling_origin.csv(原点 x 列)与 fixb_rolling_origin_summary.csv(按列汇总)。
纪律:只读 <= 2025-12-31 的数据(截止日只收紧),不算任何收益 / IC。

    python3 scripts/fixb_rolling_origin.py          # 约 5 分钟
"""
from __future__ import annotations
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

from afactor import config as C, fundamental as FU   # noqa: E402

COLS = ["total_share", "total_assets", "total_hldr_eqy_exc_min_int",
        "n_income_attr_p_q", "revenue_q",
        "n_income_attr_p_ttm", "revenue_ttm", "n_cashflow_act_ttm",
        "total_hldr_eqy_exc_min_int_avg", "total_assets_avg",
        "revenue_yoy", "n_income_attr_p_yoy", "total_assets_yoy", "inventories_yoy",
        "sue_bt", "sue_ta", "earn_streak", "earn_vol_q", "d_ni_q"]
MISMAPPED = {"sue_bt", "sue_ta", "earn_streak", "earn_vol_q", "d_ni_q"}       # §4.1 既有问题
ORIGINS = [f"{y}-{md}" for y in range(2021, 2026) for md in ("03-31", "04-30", "08-31", "10-31")] + ["2025-12-31"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sel(rep: pd.DataFrame, col: str, t: pd.Timestamp) -> pd.DataFrame:
    """复刻 _pit_daily 的选行规则,返回 t 日每只股票所用行的 (报告期, 可见日, 快照序号, 值)。"""
    av = FU._avail_col(col)
    vint = ["pit_vintage"] if "pit_vintage" in rep.columns else []
    d = rep[["stock_code", "end_date", av, col] + vint].dropna(subset=[col, av]).rename(columns={av: "eff"})
    d = d.sort_values(["stock_code", "eff", "end_date"] + vint)
    d["frontier"] = d.groupby("stock_code", sort=False)["end_date"].cummax()
    d = d[d["end_date"] >= d["frontier"]]
    d = d.drop_duplicates(["stock_code", "eff"], keep="last")
    d = d[d["eff"] <= t]
    d = d.groupby("stock_code", sort=False).tail(1).set_index("stock_code")
    if not vint:
        d["pit_vintage"] = FU.PIT_FINAL
    return d[["end_date", "eff", "pit_vintage", col]].rename(columns={col: "val"})


def main() -> None:
    t0 = time.time()
    rep_full = FU.build_report_level()
    snaps = FU._snapshot_dates([FU._load(n, []) for n in FU.REPORT_TABLES])
    multi = set(snaps["stock_code"]) if snaps is not None else set()        # 有"被替代版本"的股票
    log(f"默认报告表 {rep_full.shape},as-of 快照行 {int((rep_full.pit_vintage != FU.PIT_FINAL).sum())};"
        f"多版本股票 {len(multi)} 只,快照日 {0 if snaps is None else len(snaps)} 个")
    # 复刻正确性:sel 与 _pit_daily 在一个原点逐格一致
    t_chk = pd.Timestamp("2023-08-31")
    a = sel(rep_full, "total_assets", t_chk)["val"]
    b = FU._pit_daily(rep_full, "total_assets", pd.DatetimeIndex([t_chk]), a.index).iloc[0]
    assert np.allclose(a.to_numpy(np.float32), b.to_numpy(np.float32), equal_nan=True), "sel 与 _pit_daily 不一致"
    rows = []
    for t in map(pd.Timestamp, ORIGINS):
        with C.set_data_cutoff(t, reason="fixb 滚动原点 PIT 检验(收紧截止日,不算收益)"):
            rep_t = FU.build_report_level()
        for col in COLS:
            j = sel(rep_full, col, t).join(sel(rep_t, col, t), how="outer", lsuffix="_full", rsuffix="_tr")
            va, vb = j["val_full"].to_numpy(np.float64), j["val_tr"].to_numpy(np.float64)
            same = np.isclose(va, vb, rtol=1e-9, atol=0) | (np.isnan(va) & np.isnan(vb))
            dif = j[~same]
            both = dif["val_full"].notna() & dif["val_tr"].notna()
            sp = dif[both & (dif["end_date_full"] == dif["end_date_tr"])]
            mv = dif.index.isin(list(multi))
            rows.append(dict(
                origin=str(t.date()), column=col, mismapped=col in MISMAPPED, n=len(j), n_diff=len(dif),
                full_newer_period=int((both & (dif["end_date_full"] > dif["end_date_tr"])).sum()),
                same_period_final=int((sp["pit_vintage_full"] == FU.PIT_FINAL).sum()),
                same_period_early=int((sp["pit_vintage_full"] != FU.PIT_FINAL).sum()),
                full_older=int((both & (dif["end_date_full"] < dif["end_date_tr"])).sum()),
                only_full=int((dif["val_full"].notna() & dif["val_tr"].isna()).sum()),
                only_tr=int((dif["val_full"].isna() & dif["val_tr"].notna()).sum()),
                diff_in_multiversion_stocks=int(mv.sum())))
        log(f"原点 {t.date()} 完成")
    out = pd.DataFrame(rows)
    out.to_csv(C.LEDGER / "fixb_rolling_origin.csv", index=False)
    g = out.groupby("column", sort=False)
    summ = g[["n", "n_diff", "full_newer_period", "same_period_final", "same_period_early", "full_older",
              "only_full", "only_tr", "diff_in_multiversion_stocks"]].sum()
    summ.insert(0, "mismapped", g["mismapped"].first())
    summ["lookahead_suspect"] = summ["full_newer_period"] + summ["same_period_final"]
    summ["stale"] = summ["full_older"] + summ["only_tr"]
    summ["stale_share"] = summ["stale"] / summ["n"]
    summ["lookahead_share"] = summ["lookahead_suspect"] / summ["n"]
    summ.reset_index().to_csv(C.LEDGER / "fixb_rolling_origin_summary.csv", index=False)
    log(f"写出 fixb_rolling_origin.csv / fixb_rolling_origin_summary.csv;{len(ORIGINS)} 个原点,"
        f"总耗时 {time.time() - t0:.0f}s")
    print(summ.to_string())


if __name__ == "__main__":
    main()
