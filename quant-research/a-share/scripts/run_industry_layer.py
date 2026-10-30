"""行业配置层 + 状态判断 实验(2026-08-23,只用 2020-2025,不碰封存期)。

动机(见 README §10.7):行业层面 6-12 月动量在 2020-21 / 2025-26 为正、2022-24 为负,
是**随状态翻面**的效应;个股动量每年都是负的。考试版模型(训练 2023-25)学不到它。
用户在 ETF 项目里能选出 AI/芯片,正是行业层面的这个效应。

结构(对应"小碎步交易法"的四层):
  状态层  S_t = 过去 12 个月"行业 6 月动量 → 次月行业收益"的已实现 IC 均值(walk-forward)
           S_t > 0 趋势市:配置到动量前 K 个行业;S_t <= 0 轮动市:行业等权(不反着做)
  行业层  行业 = 证监会大类(成分 >= 15 只),信号 = 过去 120 日行业等权收益
  个股层  行业内按 B×GBT 打分(考试同款,滚动 OOS 存盘面板)取前 n
  规则层  "小碎步"入场条件:60 日上涨 + 距 20 日高点回踩 >= 3% + 近 5 日出现"低开收红"

对照:池等权基准、纯 Top20(B×GBT,相位平均)、恒定趋势/恒定等权(证明状态判断的增量)。

已知局限:行业标签用 2026-08 快照(证监会大类极少变更,轻微前视,记录在案);
没有最高价字段,"突破前高"用收盘价 20 日高点近似。
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel
from afactor import strategy as ST

pd.set_option("display.width", 250)
OOS_YEARS = [2023, 2024, 2025]
MIN_MEMBERS = 15
K_IND, N_PER = 5, 4          # 5 个行业 × 4 只 = 20 只,与 Top20 可比
MOM_WIN = 120                # 行业动量回看(≈6 个月)


def industry_labels(codes) -> pd.Series:
    raw = pd.read_csv("data_8:21/source_raw/baostock_stock_industry_raw.csv")
    raw["code"] = raw["code"].str.upper().str.replace(r"^(SH|SZ)\.(\d+)$", r"\2.\1", regex=True)
    m = dict(zip(raw["code"], raw["industry"]))
    return pd.Series({c: m.get(c) for c in codes}).replace("", np.nan)


def regime_state(ind_ret: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """走到每一天时,只用**已结束**的过去 12 个月算行业动量的已实现 IC 均值。"""
    m = ind_ret.add(1).resample("ME").prod() - 1
    mom = (m.add(1).rolling(6).apply(np.prod, raw=True) - 1).shift(1)    # 过去 6 月,滞后 1 月
    ic = pd.Series(index=m.index, dtype=float)
    for d in m.index:
        a, b = mom.loc[d], m.loc[d]; ok = a.notna() & b.notna()
        if ok.sum() >= 10:
            ic[d] = a[ok].rank().corr(b[ok].rank())
    s = pd.Series(np.nan, index=dates)
    for d in dates:
        past = ic[ic.index < d].dropna().iloc[-12:]
        if len(past) >= 6:
            s[d] = past.mean()
    return s


def main():
    t0 = time.time()
    p = Panel()
    dates, codes = p.dates, p.codes
    years = np.asarray(dates.year)
    u = ST.universe(p, "V1000")
    close = p.raw("close_px"); r = close.pct_change().where(u)
    lab = industry_labels(codes).dropna()
    cnt = lab[lab.index.isin(u.columns[u.mean() > 0.2])].value_counts()
    lab = lab[lab.map(cnt) >= MIN_MEMBERS]
    ind_ret = r[lab.index].T.groupby(lab).mean().T
    print(f"行业 {ind_ret.shape[1]} 个(成分>={MIN_MEMBERS}),池内股票 {len(lab)} 只", flush=True)

    # ---- 状态层 ----
    S = regime_state(ind_ret, dates)
    for y in OOS_YEARS:
        sy = S[years == y]
        print(f"  状态 S {y}: 均值 {sy.mean():+.3f}  趋势市占比 {(sy > 0).mean()*100:.0f}%")

    # ---- 行业层信号:过去 120 日行业收益(T 日收盘可得)----
    ind_mom = ind_ret.add(1).rolling(MOM_WIN, min_periods=60).apply(np.prod, raw=True) - 1
    ind_rank = ind_mom.rank(axis=1, ascending=False)            # 1 = 最强
    stock_ind_rank = pd.DataFrame({c: ind_rank[lab[c]] for c in lab.index})   # 每只股票所属行业的名次

    # ---- 个股层:B×GBT 滚动 OOS 打分 ----
    score = pd.DataFrame(np.nan, index=dates, columns=codes, dtype=np.float32)
    for y in OOS_YEARS:
        s = pd.read_parquet(LEDGER / f"scores/B_gbt_open_h5_{y}.parquet")
        score.loc[s.index, s.columns] = s.to_numpy()
    score = score.where(u)
    # 行业内名次(1 = 该行业内打分最高)
    within = pd.DataFrame(np.nan, index=dates, columns=lab.index, dtype=np.float32)
    for ind, members in lab.groupby(lab).groups.items():
        sub = score[list(members)]
        within[list(members)] = sub.rank(axis=1, ascending=False, method="first").to_numpy()

    # ---- 规则层:小碎步入场条件 ----
    ret60 = close / close.shift(60) - 1
    hi20 = close.rolling(20).max()
    pullback = (close / hi20 - 1) <= -0.03
    lowopen_red = (p["overnight"] < -0.003) & (p.raw("ret_intraday") > 0)
    recent_red = lowopen_red.rolling(5).max() > 0
    rule_ok = (ret60 > 0) & pullback & recent_red

    # ---- 构造各变体的"掩码打分"(只在选中股票上有值,Top-N 引擎取全部)----
    def masked(select: pd.DataFrame, rank_by: pd.DataFrame, n_total: int):
        sc = rank_by.where(select)
        return -sc          # 引擎取 score 最大 -> 名次最小
    in_top_ind = stock_ind_rank <= K_IND
    in_top_ind = in_top_ind.reindex(columns=codes).fillna(False)
    within_full = within.reindex(columns=codes)
    trend_sel = in_top_ind & (within_full <= N_PER)
    S_b = S.reindex(dates)
    trend_day = (S_b > 0).to_numpy()[:, None]
    trend_df = pd.DataFrame(np.broadcast_to(trend_day, (len(dates), len(codes))), index=dates, columns=codes)
    score_rank = score.rank(axis=1, ascending=False, method="first")
    # 趋势市用"行业内名次",轮动市用"全池名次"作为排序依据
    rank_switch = within_full.where(trend_df, score_rank)
    # 状态切换:趋势市用行业配置,轮动市退回纯 Top20
    ind_neutral_sel = score.rank(axis=1, ascending=False, method="first") <= K_IND * N_PER
    switch_sel = pd.DataFrame(np.where(trend_day, trend_sel.to_numpy(), ind_neutral_sel.to_numpy()),
                              index=dates, columns=codes)
    rule_sel = in_top_ind & rule_ok.reindex(columns=codes).fillna(False)
    rule_rank = within_full.where(rule_sel)
    rule_sel_n = rule_sel & (rule_rank.rank(axis=1, method="first") <= K_IND * N_PER)
    rule_switch = pd.DataFrame(np.where(trend_day, rule_sel_n.to_numpy(), ind_neutral_sel.to_numpy()),
                               index=dates, columns=codes)
    slots5_sel = in_top_ind & (within_full <= 1)                 # 5 行业 × 1 只 = 5 槽
    slots5_switch = pd.DataFrame(np.where(trend_day, slots5_sel.to_numpy(), (score_rank <= 5).to_numpy()),
                                 index=dates, columns=codes)

    variants = {
        "纯Top20(对照)":            (score, 20),
        "恒定趋势:5行业×4只":        (masked(trend_sel, within_full, 20), 20),
        "状态切换:趋势市行业配置/轮动市Top20": (masked(switch_sel, rank_switch, 20), 20),
        "状态切换+小碎步入场规则":    (masked(rule_switch, rank_switch, 20), 20),
        "状态切换:5槽(5行业×1只)":   (masked(slots5_switch, rank_switch, 5), 5),
        "纯Top5(对照)":             (score, 5),
    }
    rows = []
    for name, (sc, n) in variants.items():
        for y in OOS_YEARS:
            dm = years == y
            bm = ST.benchmark(p, "V1000", "open10", date_mask=dm)
            r_ = ST.run_strategy(sc, p, "V1000", "open10", scheme="top", n=n, rebalance=5, date_mask=dm)
            if not np.isfinite(r_.get("ann_return", np.nan)):
                continue
            pa = r_.get("phase_ann") or [r_["ann_return"]]
            rows.append(dict(变体=name, 年=y, 基准=bm["ann_return"], 年化=r_["ann_return"],
                             超额=r_["ann_return"] - bm["ann_return"], 夏普=r_["sharpe"],
                             回撤=r_["max_dd"], 换手=r_["turnover"], 持仓=r_["holdings"],
                             仓位=r_.get("exposure"), 相位极差=max(pa) - min(pa)))
        print(f"  {name} 完成 {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "industry_layer_open_h5.csv", index=False)
    print("\n=== 逐年(V1000,主口径,相位平均,含成本)===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== 三年平均 ===")
    print(df.groupby("变体", sort=False).agg(超额=("超额", "mean"), 夏普=("夏普", "mean"), 回撤=("回撤", "mean"),
                                           最差年超额=("超额", "min"), 换手=("换手", "mean"),
                                           持仓=("持仓", "mean"), 相位极差=("相位极差", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
