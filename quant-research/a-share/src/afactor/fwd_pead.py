"""前向检验 T4(PEAD E 臂与 B 臂)的适配器(前缀 fwd;预注册 FORWARD_TEST_PREREG §2 T4、§3.4、§5.3)。

只放宽日期守卫、只换数据源,公式一律沿用冻结代码:
  - 事件表:earnings_events.build_formal_events / map_v / event_pool 原样调用;适配器只把模块常量 CUTOFF
    放宽到 2026-10-09、交易日历换成"研究期日历 ∪ 前向面板 2026 交易日"(T_pos 与研究期同一编号,2020-01-02 = 0,
    相位 φ = T_pos % 5 沿用研究期日历往后接),利润表换成 load_income_fwd(见下);主事件判据与 build_events 逐字相同。
  - 利润表 load_income_fwd(splice_em):
      splice_em=False(**预注册原规则**,fwd_decisions:两源一致率未全部达 99% -> 沿用最后已知报告期):
          只用 data_8:21 tushare 利润表(源到 2026-08-20),与 earnings_events.load_income 同一截断与字段处理;
      splice_em=True(**偏离预注册 D1,补充读数,不是主读数**):tushare 没有的 (股票, 报告期) 用东财 2026H1 / Q3
          快照补:operate_profit <- OPERATE_PROFIT、n_income_attr_p <- PARENT_NETPROFIT(两项一致率 99.94% / 99.92%,
          半年报重叠 100% / 100%),revenue 置空(东财没有一般企业的营业收入),ann_date = f_ann_date = NOTICE_DATE。
  - 合格池:earnings_events.qualified_pool 原样调用,只把 st_daily_pit(读 data_8:21 ST 名单,源到 07-03)换成前向面板的
    is_st(<= 06-30 同一名单,07-01 起 baostock isST)。上市天数沿用 listed_trading_days(stock_basic 上市日;
    08-18 以后的新股不在 stock_basic 里 -> 面板首个有成交日,与冻结代码的回退规则相同)。
  - 选股与匹配:与 scripts/pead_build_batches.py main() 同一循环(只跑 E、B、B@tol05 三臂;C / F / E_rand 按方案门控不跑),
    pool_quantile / match_arm 直接 import 冻结函数;随机种子 [20261010, arm, φ, T_pos, r] 用研究期编号的 T_pos。
只有前缀 fwd 的前向检验代码可以 import 本模块。
"""
from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd

from afactor import earnings_events as EE

FWD_CUTOFF = pd.Timestamp("2026-10-09")
RAW_EM = EE.ROOT / "cache" / "forward" / "raw" / "ak_em"


def ext_calendar(fwd_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """研究期交易日历(cache/daily 行索引)∪ 前向面板 2026 年交易日。"""
    probe = pd.read_parquet(EE.DAILY / "close_px.parquet", columns=["000001.SZ"])
    cal = pd.DatetimeIndex(probe.index).sort_values()
    add = fwd_dates[fwd_dates > cal.max()]
    return cal.append(add)


@contextlib.contextmanager
def lifted_cutoff(cal: pd.DatetimeIndex, cutoff=FWD_CUTOFF):
    """earnings_events 的 CUTOFF 与 trading_calendar 换成前向版本,退出时恢复。"""
    saved = (EE.CUTOFF, EE.trading_calendar)
    EE.CUTOFF = pd.Timestamp(cutoff)
    EE.trading_calendar = lambda: cal
    try:
        yield
    finally:
        EE.CUTOFF, EE.trading_calendar = saved


def _em_income(period: str) -> pd.DataFrame:
    import glob
    f = sorted(glob.glob(str(RAW_EM / f"raw_lrb_{period}_fetched*.csv")))[-1]
    d = pd.read_csv(f, dtype={"SECURITY_CODE": str})
    out = pd.DataFrame({"stock_code": d["SECUCODE"].astype(str),
                        "ann_date": pd.to_datetime(d["NOTICE_DATE"]).dt.normalize(),
                        "end_date": pd.to_datetime(d["REPORT_DATE"]).dt.normalize(),
                        "operate_profit": pd.to_numeric(d["OPERATE_PROFIT"], errors="coerce"),
                        "n_income_attr_p": pd.to_numeric(d["PARENT_NETPROFIT"], errors="coerce"),
                        "revenue": np.nan})
    out = out[out["stock_code"].str.endswith((".SH", ".SZ"))]
    return out.drop_duplicates(["stock_code", "end_date"], keep="last")


def load_income_fwd(splice_em: bool, items=("operate_profit", "n_income_attr_p", "revenue")) -> pd.DataFrame:
    """earnings_events.load_income 的前向版(截断日 = 当前 EE.CUTOFF);splice_em 见模块文档。"""
    d = EE.load_income(items)                                    # 冻结函数:按 EE.CUTOFF 截断
    if not splice_em:
        return d
    add = pd.concat([_em_income("20260630"), _em_income("20260930")], ignore_index=True)
    have = set(zip(d["stock_code"], d["end_date"]))
    add = add[[(a, b) not in have for a, b in zip(add["stock_code"], add["end_date"])]]
    add = add[add["ann_date"] <= EE.CUTOFF].copy()
    ct = d.sort_values("end_date").drop_duplicates("stock_code", keep="last").set_index("stock_code")["comp_type"]
    add["comp_type"] = add["stock_code"].map(ct)
    add["f_ann_date"] = add["ann_date"]
    add["avail"] = add["ann_date"]
    add["sealed_value"] = False
    add["qi"] = (add["end_date"].dt.year * 4 + (add["end_date"].dt.quarter - 1)).astype(int)
    add["quarter"] = add["end_date"].dt.quarter.astype(int)
    out = pd.concat([d, add[d.columns]], ignore_index=True)
    assert not out.duplicated(["stock_code", "end_date"]).any()
    out.attrs["n_em_rows"] = len(add)
    return out.reset_index(drop=True)


def main_events(inc: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """定期报告主事件(与 earnings_events.build_events 的 is_main 判据逐字相同;只构造定期报告部分,
    预告 / 快报只影响 freshness 诊断列,不影响 is_main)。须在 lifted_cutoff 内调用(或研究期默认)。"""
    formal = EE.build_formal_events(inc, cal)
    formal["V"], formal["V_status"] = EE.map_v(formal["P"], cal)
    formal["is_main"] = ((formal["source"] == "定期报告") & (formal["indicator"] == EE.MAIN_INDICATOR)
                         & (formal["version"] == 1) & (formal["value_status"] == "ok")
                         & formal["is_frontier"].eq(True) & formal["V"].notna())
    out = formal[formal["is_main"]].copy()
    # freshness(预告在先 / 快报在先 ...)只是事件层诊断列,不进入选股、匹配或任何读数;前向不计算,置缺失
    out["freshness"] = pd.NA
    return out


@contextlib.contextmanager
def panel_st(p):
    """earnings_events.st_daily_pit -> 前向面板 is_st(数据源替换)。"""
    saved = EE.st_daily_pit
    EE.st_daily_pit = lambda dates, codes: p["is_st"].reindex(index=dates, columns=codes).fillna(False)
    try:
        yield
    finally:
        EE.st_daily_pit = saved


def build_batches(pool: pd.DataFrame, Qn: np.ndarray, qv: np.ndarray, ind_lab: np.ndarray, dates: pd.DatetimeIndex,
                  codes: pd.Index, pos_offset: int = 0, arms=("B",), tols=(("tol10", 0.10), ("tol05", 0.05)),
                  T_range=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """pead_build_batches.main() 的选股 + 匹配循环(E 与 B / B@tol05;逐行同式)。
    pool 需含 T_pos、j、value、stock_code(面板行号);pos_offset = 面板行号 -> 研究期日历编号的偏移
    (相位与随机种子都用研究期编号 T_ext = T_pos + pos_offset)。T_range=(lo, hi) 只处理 date_T 在其中的批次。"""
    import pead_build_batches as PBB
    rows, meta = [], []
    pool = pool.sort_values(["T_pos", "value", "stock_code"], ascending=[True, False, True])
    for T_pos, g in pool.groupby("T_pos", sort=True):
        if T_range is not None and not (T_range[0] <= dates[T_pos] <= T_range[1]):
            continue
        t = T_pos - 1
        T_ext = int(T_pos + pos_offset)
        phase = int(T_ext % 5)
        n = len(g)
        rec = dict(T_pos=T_pos, T_ext=T_ext, t_pos=t, date_T=dates[T_pos], phase=phase, n_pool=n, n_sel=0)
        if n < 5:
            meta.append(rec); continue
        n_sel = -(-n // 5)
        sel = g.iloc[:n_sel]
        ej = sel["j"].to_numpy(np.int64)
        w = min(1.0 / n_sel, 0.05)
        rec.update(n_sel=n_sel, w=w, tgt_cash=1.0 - n_sel * w)
        for s, jj in enumerate(ej):
            rows.append(("E", 0, T_pos, T_ext, t, phase, s, jj, w, 0, np.nan))
        candB = np.setdiff1d(np.flatnonzero(Qn[t]), ej)
        q_t = qv[t]
        for arm in arms:
            for tol_tag, tol1 in tols:
                ctrl, lvl, dist = PBB.match_arm(ej, candB, q_t, ind_lab, PBB.ARM_CODE[arm], phase, T_ext, tol1)
                arm_name = arm if tol_tag == "tol10" else f"{arm}@tol05"
                for r in range(PBB.K_CTRL):
                    for s in range(n_sel):
                        rows.append((arm_name, r + 1, T_pos, T_ext, t, phase, s, int(ctrl[r, s]), w,
                                     int(lvl[r, s]), dist[r, s]))
        meta.append(rec)
    B = pd.DataFrame(rows, columns=["arm", "rep", "T_pos", "T_ext", "t_pos", "phase", "slot", "j", "w", "level", "dist"])
    if len(B):
        B["date_T"] = dates[B["T_pos"].to_numpy()]
        B["t_signal"] = dates[B["t_pos"].to_numpy()]
        ok = B["j"] >= 0
        B["code"] = pd.Series(np.where(ok, codes.to_numpy()[B["j"].clip(lower=0)], None), index=B.index)
    return B, pd.DataFrame(meta)


def match_inputs(p, Q: pd.DataFrame, mv: pd.DataFrame, circ: pd.DataFrame):
    """匹配变量(与 pead_build_batches.main() 同式):log 市值(缺失用 log 流通市值)、log ADV20、20 日 ret_c2c 波动(>=15),
    换算为同日合格池内分位。返回 (qv (T,N,3), raw_vals (T,N,3), ind_lab)。"""
    import pead_build_batches as PBB
    from afactor import newdata as ND
    ind_lab = ND.sw_industry_labels(p.dates[:1], p.codes).iloc[0].to_numpy(np.int64)
    logmv = np.log(mv.where(mv > 0)).fillna(np.log(circ.where(circ > 0)))
    logadv = np.log(p["adv20"].where(p["adv20"] > 0))
    vol = p["ret_c2c"].rolling(20, min_periods=15).std()
    qv = np.stack([PBB.pool_quantile(logmv, Q), PBB.pool_quantile(logadv, Q), PBB.pool_quantile(vol, Q)], axis=2)
    raw_vals = np.stack([logmv.to_numpy(np.float64), logadv.to_numpy(np.float64), vol.to_numpy(np.float64)], axis=2)
    return qv, raw_vals, ind_lab
