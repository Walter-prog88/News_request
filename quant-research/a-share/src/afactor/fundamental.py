"""基本面数据层:严格 point-in-time 的日度面板 + 近似市值 + ST 标记。

数据源(data_8:21/,tushare 格式):利润表/现金流(**年内累计**口径)、资产负债表(时点)、每日 ST 名单。

防前视(2026-08-22 审查后重写,修复 4 个 P0):
  1. 可见日 avail = max(ann_date, f_ann_date);
  2. 三表按 (股票, 报告期) 合并,avail 取三表最晚者 —— 此前按 avail 做 outer merge
     会产生幽灵行,使 shift(4) 按"行"而非按"季"错位(影响 35% 股票);
  3. 所有 shift(4)/rolling(4) 前先 reindex 到**连续季度网格**,缺季不再被跳过;
  4. 每个衍生量带自己的**生效可见日**:TTM/均值用窗口内 avail 的 rolling max,
     同比用 max(本期, 去年同期) —— 保证构成它的每一期都已公告(此前晚报/补报股
     的单季还原用到了当时尚未公开的上期数字,中位提前 182 天)。

已知未修复的泄漏(需外部数据):见 build_market_cap 的说明。

数据截止与版本 PIT(2026-10-10,前缀 fixb;见 research/DATA_CUTOFF_DISCIPLINE.md):
  5. 三张财报表与 ST 名单**读入后立即**按 config.data_cutoff()(默认 = RESEARCH_END)截断:
     只保留可见日 avail = max(ann_date, f_ann_date) <= 截止日的版本,然后才做版本选择、合并、衍生。
     此前是整表读入后"同一 (股票, 报告期) 保留最晚公告版本":2026 年的更正公告会把截止日前已公告的
     版本挤掉,使该报告期(及依赖它的单季/TTM/同比)在研究期内被推迟到 2026 才可见 —— 选择效应。
  6. 版本选择改为 PIT:时点 t 用 t 时已公告的最新版本。数据源每个 (股票, 报告期) 一般只存一行
     (最新版),只有资产负债表有多版本(截止日前 377 个报告期、240 只股票)。实现见 build_report_level:
     最终快照(每期最新版)之外,再为有多版本的股票在每个"有被替代版本生效"的公告日 tau 构造 as-of 快照
     (每个报告期取 avail <= tau 的最新版),只把与最终快照不同的行以 pit_vintage < PIT_FINAL 追加;
     _pit_daily 按可见日取最新,同日并列时取较晚的快照。t 时可选的候选行只由 avail <= t 的版本决定,
     与 t 之后是否还有修订无关(第二轮审查 P2 后改写;此前按"第 k 版"拼快照,交错修订时取值依赖未来是否修订)。
     股本 <= 0 的占位行一律视为未披露(与版本无关)。

  取值不前视,但**会偏陈旧**(scripts/fixb_rolling_origin.py:21 个原点、截止日=t 重建 vs 默认构造,映射正确的 14 列
  前视 0 格,t 日有 0.24%~2.07% 的股票-日比"数据截至 t"的重建更旧,全部是保守方向;
  research/ledger/fixb_rolling_origin_summary.csv,DATA_CUTOFF_DISCIPLINE.md §4 第 2 条)。三条机制:
    (a) 数据源用更正版**覆盖**了原行(利润表 / 现金流量表每期只存一版),原版本的公告日已不可知,
        整期被推迟到更正日,依赖它的单季 / TTM / 同比随之推迟;
    (b) 三表取最晚:一张表被更正,整期连同另外两张表一起推迟到更正日;此时下一期往往已经公告,
        该期的行被 _pit_daily 的 frontier(只取最新报告期)永久丢弃;
    (c) Q1 的单季值等于累计值、并不依赖上一期,但 avail_q 仍取 max(本期, 上期)。
  (c) 可以单独放宽,会改变研究期数值,另行立项。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from afactor.config import (ROOT, DAILY, PRICE_ADJ_BASE_END, data_cutoff, assert_within_cutoff,
                            record_source_end, source_end, check_source_coverage)

DATA = ROOT / "data_8:21"
REPORT_TABLES = ("stock_income_2015_now.pkl", "stock_cashflow_2015_now.pkl", "stock_balancesheet_2015_now.pkl")
PIT_FINAL = 99          # 报告表 pit_vintage 列:99 = 最终快照(每期最新版);0,1,.. = as-of 快照(按快照日先后)
# 诊断开关:False 时只截断(不追加 as-of 快照,也不做股本 0 占位处理),复现"仅截断"口径
# (仅供 scripts/fixb_cutoff_impact.py 把影响拆成"截断"与"版本 PIT"两部分)。研究与前向检验一律用默认 True。
PIT_EARLY_VERSIONS = True

CUMULATIVE_ITEMS = {
    "total_revenue", "revenue", "oper_cost", "operate_profit", "total_profit",
    "n_income", "n_income_attr_p", "sell_exp", "admin_exp", "rd_exp",
    "n_cashflow_act", "free_cashflow", "c_fr_sale_sg",
}
INCOME_ITEMS = ["total_revenue", "revenue", "oper_cost", "operate_profit", "total_profit",
                "n_income", "n_income_attr_p", "sell_exp", "admin_exp", "rd_exp"]
BALANCE_ITEMS = ["total_share", "total_assets", "total_liab", "total_hldr_eqy_exc_min_int",
                 "money_cap", "inventories", "accounts_receiv", "total_cur_assets",
                 "total_cur_liab", "goodwill", "fix_assets"]
CASHFLOW_ITEMS = ["n_cashflow_act", "free_cashflow", "c_fr_sale_sg"]

# 报告表缓存。正常路径键为 ("rep", 截止日),set_data_cutoff() 切换后自动换键;
# _load 被外部包装时(index_proxy._sealed_newdata、pead 脚本的封存截断)沿用旧键 "rep",
# 由包装方按原约定 pop/clear,包装期间的结果不会落进正常路径的键里。
_REP_CACHE: dict[object, pd.DataFrame] = {}


def _load(name: str, items: list[str], cutoff=None) -> pd.DataFrame:
    """读一张财报表,**读入后立即**按截止日截断,返回截止日前已公告的**全部版本**。

    avail = max(ann_date, f_ann_date) 是该版本的公告日(tushare:f_ann_date 为实际公告日,
    修正过的以修正日为准)。只保留 avail <= 截止日(默认 data_cutoff())的版本。
    返回按 (股票, 报告期, avail) 排序;**不在这里去重** —— 版本选择交给 build_report_level。
    签名 (name, items) 与旧版兼容:index_proxy / pead 脚本把它包一层再截到 2025-12-31,行为不变。
    """
    cut = data_cutoff() if cutoff is None else pd.Timestamp(cutoff)
    d = pd.read_pickle(DATA / name)
    d = d[["stock_code", "ann_date", "f_ann_date", "end_date", "comp_type"] + items].copy()
    for c in ("ann_date", "f_ann_date", "end_date"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d["avail"] = d[["ann_date", "f_ann_date"]].max(axis=1)
    record_source_end(name, d["avail"].max())        # 原始文件最晚可见日(截断前;只供源覆盖守卫)
    d = d[d["avail"].notna() & d["end_date"].notna() & (d["avail"] <= cut)]     # 先截断
    # 与旧版完全相同的排序(其后 keep="last" 选出的最新版与旧版逐行一致)
    return d.sort_values(["stock_code", "end_date", "avail"])


_LOAD_IMPL = _load      # 未被包装的原始实现(build_report_level 据此判断 _load 是否被外部替换)


def _pick_version(d: pd.DataFrame, k: int | None) -> pd.DataFrame:
    """每个 (股票, 报告期) 取一个版本。k=None:最新版(与旧口径 keep="last" 逐行相同);
    k=0,1,..:第 min(k, 版本数-1) 个版本(按 avail 先后;as-of 快照改写后只剩诊断用途)。"""
    if k is None:
        return d.drop_duplicates(["stock_code", "end_date"], keep="last")
    g = d.groupby(["stock_code", "end_date"], sort=False)
    ver = g.cumcount().to_numpy()
    nver = g["avail"].transform("size").to_numpy()
    return d[ver == np.minimum(k, nver - 1)]


def _zero_share_as_missing(df: pd.DataFrame) -> pd.DataFrame:
    """股本 <= 0 是占位(截止日前最终版 598 行、425 只,多为上市前的招股书期间;早期版本另有 19 行),
    视为未披露:_pit_daily 会跳过它、沿用上一期股本。对所有版本一视同仁 —— 此前只在"之后会被修订"
    的早期版本上置 NaN,t 时用哪种处理取决于 t 之后有没有修订(第二轮审查 P2)。"""
    if "total_share" in df.columns:
        df.loc[df["total_share"] <= 0, "total_share"] = np.nan
    return df


def _to_quarter_grid(d: pd.DataFrame) -> pd.DataFrame:
    """按股票 reindex 到连续季度网格,缺季补 NaN 行,使 shift(4)/rolling(4) 严格对齐季度。"""
    d = d.copy()
    d["qi"] = d["end_date"].dt.year * 4 + (d["end_date"].dt.quarter - 1)
    out = []
    for code, g in d.groupby("stock_code", sort=False):
        g = g.sort_values("qi").set_index("qi")
        full = range(int(g.index.min()), int(g.index.max()) + 1)
        g = g.reindex(full)
        g["stock_code"] = code
        g["qi"] = g.index
        out.append(g.reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


def build_report_level() -> pd.DataFrame:
    """报告期维度的财报表(PIT,按当前截止日)。

    行 = 最终快照(每个报告期在截止日前的最新版,pit_vintage = PIT_FINAL,与旧版逐行相同)
       + 早期版本快照里与最终快照不同的行(pit_vintage = 0, 1, ..)。
    "按 (股票, 报告期) 一行"的消费方(build_event_features / comp_type_panel / build_market_cap)
    先筛 pit_vintage == PIT_FINAL;_pit_daily 直接用全部行。
    """
    cut = data_cutoff()
    if _load is _LOAD_IMPL:
        key = ("rep", cut, bool(PIT_EARLY_VERSIONS))
        for k in [k for k in _REP_CACHE if isinstance(k, tuple) and k[0] == "rep" and k != key]:
            del _REP_CACHE[k]
    else:
        key = "rep"
    if key in _REP_CACHE:
        return _REP_CACHE[key]
    tabs = [_load(REPORT_TABLES[0], INCOME_ITEMS),
            _load(REPORT_TABLES[1], CASHFLOW_ITEMS),
            _load(REPORT_TABLES[2], BALANCE_ITEMS)]
    final = _derive(_merge_tables(*[_pick_version(t, None) for t in tabs]))
    if PIT_EARLY_VERSIONS:
        final = _zero_share_as_missing(final)
    final["pit_vintage"] = np.int16(PIT_FINAL)
    early = _asof_snapshots(tabs, final) if PIT_EARLY_VERSIONS else None
    df = final if early is None else pd.concat([final, early], ignore_index=True)
    _REP_CACHE[key] = df
    return df


def _snapshot_dates(tabs: list[pd.DataFrame]) -> pd.DataFrame | None:
    """需要单独构造 as-of 快照的 (股票, 快照日 tau)。

    某个 (股票, 报告期, 表) 的版本 v_k 在 [avail(v_k), avail(v_k+1)) 区间是"当时最新、后来被替代"的版本。
    只有在这种区间里,t 时的最新版本组合才可能不同于最终快照;区间外 t 时的最新版本都是最终版,
    最终快照的行就是精确值。tau 取该股票所有表、所有版本的公告日中落在任一这种区间里的那些
    (t 时适用的快照 = t 之前最后一个公告日的快照,它必然落在同一区间里)。
    同日重发(两版 avail 相同)的前一版从未"当时最新",不产生区间。
    """
    iv, av = [], []
    for t in tabs:
        nxt = t.groupby(["stock_code", "end_date"], sort=False)["avail"].shift(-1)
        m = (nxt > t["avail"]).to_numpy()
        iv.append(pd.DataFrame({"stock_code": t["stock_code"].to_numpy()[m],
                                "lo": t["avail"].to_numpy()[m], "hi": nxt.to_numpy()[m]}))
        av.append(t[["stock_code", "avail"]])
    iv = pd.concat(iv, ignore_index=True)
    if iv.empty:
        return None
    av = pd.concat(av, ignore_index=True).drop_duplicates()
    av = av[av["stock_code"].isin(set(iv["stock_code"]))]
    x = av.merge(iv, on="stock_code")
    x = x[(x["avail"] >= x["lo"]) & (x["avail"] < x["hi"])]
    s = (x[["stock_code", "avail"]].drop_duplicates().rename(columns={"avail": "tau"})
         .sort_values(["stock_code", "tau"]).reset_index(drop=True))
    s["sid"] = np.arange(len(s))
    s["rank"] = s.groupby("stock_code", sort=False).cumcount()
    return s


def _pick_asof(t: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """每个 (快照, 报告期) 取 avail <= tau 的最新版本(并列时与最终快照 keep="last" 同序);
    tau 时该表尚无任何版本的报告期取首版 —— 其可见日晚于 tau,行的可见日随之推迟,
    与最终快照同样遵守"三表取最晚",不会产生只有部分表的行。"""
    t = t[t["stock_code"].isin(set(snaps["stock_code"]))].copy()
    t["_r"] = np.arange(len(t))                                 # _load 的排序(最终快照据此取 last)
    m = t.merge(snaps[["stock_code", "sid", "tau"]], on="stock_code").sort_values(["sid", "end_date", "_r"])
    key = ["sid", "end_date"]
    last_ok = m[(m["avail"] <= m["tau"]).to_numpy()].drop_duplicates(key, keep="last")
    first = m.drop_duplicates(key, keep="first")
    have = pd.MultiIndex.from_frame(last_ok[key])
    fb = first[~pd.MultiIndex.from_frame(first[key]).isin(have)]
    out = pd.concat([last_ok, fb], ignore_index=True).drop(columns=["_r", "tau"])
    out["stock_code"] = out["stock_code"] + "@" + out["sid"].astype(str)   # 每个快照当作一只"伪股票"派生
    return out.drop(columns=["sid"])


def _asof_snapshots(tabs: list[pd.DataFrame], final: pd.DataFrame) -> pd.DataFrame | None:
    """as-of 快照:对每个 (股票, tau),每个 (报告期, 表) 取 tau 时已公告的最新版本,整体派生一遍,
    只保留与最终快照不同的行,pit_vintage = 该快照在本股票内的先后序号(0, 1, ..)。

    为什么这样做:时点 t 应当用 t 时已公告的最新版本。快照行的可见日是它用到的全部版本公告日的最大值
    (沿用 _derive 的依赖传递),所以可见日 <= t 的候选行只用到 avail <= t 的版本,不会前视;
    同一报告期的候选行,快照越晚可见日越晚(版本只会被更晚的版本替代),_pit_daily 取可见日最晚、
    并列取 pit_vintage 最大的行,正是"t 之前最后一个公告日的快照"——即 t 时的最新版本组合。
    候选集合只由 avail <= t 的版本决定,与 t 之后是否还有修订无关。
    (旧实现 S_k 按"第 k 版"跨报告期拼快照:两个报告期交错修订时,"旧期新版 + 新期首版"这类组合
    要等未来某期被修订才会出现或消失,第二轮审查 P2。)
    """
    snaps = _snapshot_dates(tabs)
    if snaps is None:
        return None
    assert snaps["rank"].max() < PIT_FINAL, "单只股票的快照数超过 pit_vintage 编码上限"
    dk = _derive(_merge_tables(*[_pick_asof(t, snaps) for t in tabs]))
    dk = _zero_share_as_missing(dk)
    sid = dk["stock_code"].str.split("@").str[1].astype(np.int64).to_numpy()
    dk["stock_code"] = dk["stock_code"].str.split("@").str[0]
    dk["pit_vintage"] = snaps["rank"].to_numpy()[sid].astype(np.int16)
    fin = final.set_index(["stock_code", "qi"])
    a = dk.drop(columns=["pit_vintage"]).set_index(["stock_code", "qi"])
    b = fin.reindex(a.index)
    same = np.ones(len(a), dtype=bool)
    for c in a.columns:
        if c not in b.columns:
            same[:] = False
            break
        x, y = a[c], b[c]
        eq = (x == y).fillna(False).to_numpy(bool) | (x.isna().to_numpy() & y.isna().to_numpy())
        same &= eq
    dk = dk[~same]
    # 不同快照派生出完全相同的行只留一行,留**最晚**的快照:可见日相同而取值不同的候选行并列时,
    # _pit_daily 取 pit_vintage 最大者,它必须是最晚快照的那一行
    dk = dk.sort_values(["stock_code", "qi", "pit_vintage"])
    dk = dk.drop_duplicates([c for c in dk.columns if c != "pit_vintage"], keep="last")
    return dk.reset_index(drop=True) if len(dk) else None


def _merge_tables(inc: pd.DataFrame, cfl: pd.DataFrame, bal: pd.DataFrame) -> pd.DataFrame:
    """三表各取一个版本后按 (股票, 报告期) 合并。"""
    # 按 (股票, 报告期) 合并;avail 取三表最晚者(避免按 avail merge 产生幽灵行)
    key = ["stock_code", "end_date"]
    df = inc[key + ["avail", "comp_type"] + INCOME_ITEMS].merge(
        cfl[key + ["avail"] + CASHFLOW_ITEMS], on=key, how="outer", suffixes=("", "_cf"))
    df = df.merge(bal[key + ["avail"] + BALANCE_ITEMS], on=key, how="outer", suffixes=("", "_bs"))
    df["avail"] = df[["avail", "avail_cf", "avail_bs"]].max(axis=1)
    df = df.drop(columns=[c for c in ("avail_cf", "avail_bs") if c in df.columns])
    assert not df.duplicated(key).any(), "合并后仍有重复 (股票, 报告期)"
    return df


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    """合并后的报告期表 -> 季度网格 -> 单季 / TTM / 同比 / 均值 / SUE 等衍生量(逻辑与旧版逐行相同)。"""
    df = _to_quarter_grid(df)
    df["end_date"] = pd.to_datetime(
        df["qi"].floordiv(4).astype(int).astype(str) + "Q" + (df["qi"].mod(4) + 1).astype(int).astype(str)
    ) + pd.offsets.QuarterEnd(0)
    df["quarter"] = df["qi"].mod(4) + 1
    g = df.groupby("stock_code", sort=False)

    # --- 单季还原(累计 -> 单季),生效可见日 = max(本期, 上期) ---
    all_cum = [c for c in INCOME_ITEMS + CASHFLOW_ITEMS if c in CUMULATIVE_ITEMS]
    # datetime 不支持 rolling,统一转成 int64 纳秒运算,最后转回
    df["_av"] = df["avail"].astype("int64").where(df["avail"].notna())
    g = df.groupby("stock_code", sort=False)
    df["_av_q"] = np.fmax(df["_av"], g["_av"].shift(1))
    for c in all_cum:
        prev = g[c].shift(1)
        df[c + "_q"] = np.where(df["quarter"] == 1, df[c], df[c] - prev)

    # --- TTM / 同比 / 均值,各自带生效可见日 ---
    qcols = [c + "_q" for c in all_cum]
    df["_av_ttm"] = g["_av_q"].transform(lambda s: s.rolling(4, min_periods=4).max())
    df["_av_yoy"] = np.fmax(df["_av_q"], g["_av_q"].shift(4))
    df["_av_avg"] = g["_av"].transform(lambda s: s.rolling(4, min_periods=4).max())
    for src, dst in [("_av_q", "avail_q"), ("_av_ttm", "avail_ttm"),
                     ("_av_yoy", "avail_yoy"), ("_av_avg", "avail_avg")]:
        df[dst] = pd.to_datetime(df[src], unit="ns", errors="coerce")
    for c in qcols:
        df[c.replace("_q", "_ttm")] = g[c].transform(lambda s: s.rolling(4, min_periods=4).sum())
        prev = g[c].shift(4)
        # 对称增长率:分母用 (|s|+|prev|)/2,扭亏(prev<0, s>0)方向为正,不会被 abs 反转
        denom = (df[c].abs() + prev.abs()) / 2.0
        df[c.replace("_q", "_yoy")] = ((df[c] - prev) / denom.where(denom > 1e-6)).clip(-2, 2)
    for c in ["total_assets", "total_hldr_eqy_exc_min_int"]:
        prev = g[c].shift(4)
        denom = (df[c].abs() + prev.abs()) / 2.0
        df[c + "_yoy"] = ((df[c] - prev) / denom.where(denom > 1e-6)).clip(-2, 2)
        df[c + "_avg"] = g[c].transform(lambda s: s.rolling(4, min_periods=4).mean())
    # --- PEAD 衍生量:盈余意外 SUE(第三轮新增)---
    ni = "n_income_attr_p_q"
    if ni in df.columns:
        g = df.groupby("stock_code", sort=False)
        d_ni = df[ni] - g[ni].shift(4)                       # 单季盈余同比增量
        df["d_ni_q"] = d_ni
        gg = d_ni.groupby(df["stock_code"], sort=False)
        drift = gg.transform(lambda s: s.shift(1).rolling(8, min_periods=6).mean())
        sig = gg.transform(lambda s: s.shift(1).rolling(8, min_periods=6).std())
        df["sue_bt"] = ((d_ni - drift) / sig.where(sig > 1e-6)).clip(-5, 5)
        # sue_bt 的真实依赖可见日(只供诊断 / 将来修复 §4.1 用;_pit_daily 目前按后缀把 sue_bt 映射到 avail):
        # drift/sig 用 d_ni(P-1..P-8),d_ni(P-8) 要 ni_q(P-12),ni_q(P-12) 要累计值 P-13 -> 依赖 P..P-13 共 14 个季度。
        # (此前写成 rolling(12),注释"12 个季度全部已公告"漏了 P-12 / P-13,第二轮审查 P2。)
        # earn_streak = sue_bt(P..P-3) 的滚动计数 -> 依赖 P..P-16 共 17 个季度。
        # min_periods=1:网格空行(缺季)的 avail 为 NaN,取窗口内已有季度的最晚可见日。
        df["_av2"] = df["avail"].astype("int64").where(df["avail"].notna())
        gs = df.groupby("stock_code", sort=False)["_av2"]
        df["_av_sue"] = gs.transform(lambda s: s.rolling(14, min_periods=1).max())
        df["_av_streak"] = gs.transform(lambda s: s.rolling(17, min_periods=1).max())
        df["avail_sue"] = pd.to_datetime(df["_av_sue"], unit="ns", errors="coerce")
        df["avail_streak"] = pd.to_datetime(df["_av_streak"], unit="ns", errors="coerce")
        # 规模标准化版(不需要历史波动估计,次新股也可用)
        df["sue_ta"] = df["d_ni_q"] / df["total_assets_avg"].abs().where(
            df["total_assets_avg"].abs() > 1e-6)
        for c in ["revenue_q", "operate_profit_q", "n_cashflow_act_q"]:
            if c in df.columns:
                df["d_" + c] = df[c] - df.groupby("stock_code", sort=False)[c].shift(4)
        # 盈余稳定性与连续超预期(第三轮文献规格需要)
        gq = df.groupby("stock_code", sort=False)
        df["earn_vol_q"] = gq["n_income_attr_p_q"].transform(
            lambda s: s.rolling(8, min_periods=6).std())
        pos = (df["sue_bt"] > 0).astype(float).where(df["sue_bt"].notna())
        df["earn_streak"] = pos.groupby(df["stock_code"], sort=False).transform(
            lambda s: s.rolling(4, min_periods=4).sum())
        # 资产负债表项的对称同比(存货/应收/固定资产等)
        for c in ["inventories", "accounts_receiv", "fix_assets", "total_cur_assets",
                  "total_cur_liab", "money_cap", "goodwill", "total_liab"]:
            if c in df.columns:
                prev = gq[c].shift(4)
                den = (df[c].abs() + prev.abs()) / 2.0
                df[c + "_yoy"] = ((df[c] - prev) / den.where(den > 1e-6)).clip(-2, 2)

    df = df[df["avail"].notna()].drop(columns=[c for c in df.columns if c.startswith("_av")])
    return df


def _final_rows(rep: pd.DataFrame) -> pd.DataFrame:
    """报告表中每个 (股票, 报告期) 一行的最终快照(旧口径的形状)。"""
    if "pit_vintage" in rep.columns:
        return rep[rep["pit_vintage"] == PIT_FINAL]
    return rep


def build_event_features(dates: pd.DatetimeIndex, codes: pd.Index,
                         ret_c2c: pd.DataFrame, open_auc: pd.DataFrame,
                         close_px: pd.DataFrame) -> pd.DataFrame:
    """财报公告事件量(第三轮新增,PEAD 用)。

    d0 = 首个 >= avail 的交易日。产出:
      ear_car3  公告窗口 [d0-1, d0+1] 的市场调整累计超额收益(EAR)
      ear_gap   公告后首日开盘跳空(市场调整),开源证券"净利润断层"的核心量
      days_since_ann  距最近一次公告的交易日数(用于限定漂移窗口)
    生效可见日:ear_car3 取 d0+2(保证 d0+1 收盘已发生),ear_gap 取 d0+1(保守)。
    """
    assert_within_cutoff(dates, "build_event_features")
    rep = _final_rows(build_report_level())
    _check_report_coverage(dates, "build_event_features")
    mkt = ret_c2c.mean(axis=1)
    abn = ret_c2c.sub(mkt, axis=0)
    gap = (open_auc / close_px.shift(1) - 1.0)
    gap_abn = gap.sub(gap.mean(axis=1), axis=0)
    pos = {d: i for i, d in enumerate(dates)}
    rows = []
    ev = rep[["stock_code", "end_date", "avail"]].dropna()
    ev = ev[ev["stock_code"].isin(codes)]
    di = dates.to_numpy()
    for code, g in ev.groupby("stock_code", sort=False):
        if code not in abn.columns:
            continue
        a = abn[code]; gp = gap_abn[code]
        for _, r in g.iterrows():
            k = int(np.searchsorted(di, np.datetime64(r["avail"])))
            if k >= len(dates) or k < 1:
                continue
            d0 = dates[k]
            lo, hi = max(k - 1, 0), min(k + 1, len(dates) - 1)
            rows.append({"stock_code": code, "end_date": r["end_date"], "d0": d0,
                         "ear_car3": float(a.iloc[lo:hi + 1].sum()),
                         "ear_gap": float(gp.iloc[k]) if np.isfinite(gp.iloc[k]) else np.nan,
                         "eff_car": dates[min(k + 2, len(dates) - 1)],
                         "eff_gap": dates[min(k + 1, len(dates) - 1)]})
    return pd.DataFrame(rows)


def _check_report_coverage(dates, what: str):
    """三张财报表的源覆盖守卫:请求日期晚于任一表原始文件最晚可见日时默认报 SourceCoverageError
    (放宽截止日后,_pit_daily 会把源文件最后一期 ffill 到覆盖之外的日子)。研究期恒不触发。"""
    for n in REPORT_TABLES:
        if source_end(n) is None:              # 本进程尚未读过(报告表来自别处):登记一次
            _LOAD_IMPL(n, [])
    return check_source_coverage(list(REPORT_TABLES), dates, what)


# 每个列使用哪个生效可见日
def _avail_col(col: str) -> str:
    if col.endswith("_ttm"):
        return "avail_ttm"
    if col.endswith("_yoy"):
        return "avail_yoy"
    if col.endswith("_avg"):
        return "avail_avg"
    if col.endswith("_q"):
        return "avail_q"
    return "avail"


def _pit_daily(df: pd.DataFrame, col: str, dates: pd.DatetimeIndex,
               codes: pd.Index, lag_days: int = 0) -> pd.DataFrame:
    """报告期维度 -> 日频:t 时点取"生效可见日 <= t"中报告期最新的那条。

    lag_days>0 时整体再推迟若干日(尾盘口径用:T 日公告的财报 T+1 起生效)。
    报告表含 as-of 快照行(pit_vintage < PIT_FINAL)时:可见日相同的并列行取较晚的快照。
    源覆盖:dates 晚于三张财报表原始文件最晚可见日时默认报 SourceCoverageError;在
    config.accept_source_gap(REPORT_TABLES, reason) 内接受后沿用最后已知的报告期(发 RuntimeWarning)。
    """
    assert_within_cutoff(dates, "_pit_daily")
    _check_report_coverage(dates, "_pit_daily")
    av = _avail_col(col)
    vint = ["pit_vintage"] if "pit_vintage" in df.columns and col != "pit_vintage" else []
    d = df[["stock_code", "end_date", av, col] + vint].dropna(subset=[col, av])
    if d.empty:
        return pd.DataFrame(np.nan, index=dates, columns=codes, dtype=np.float32)
    d = d.rename(columns={av: "eff"})
    if lag_days:
        d = d.assign(eff=d["eff"] + pd.Timedelta(days=lag_days))
    d = d.sort_values(["stock_code", "eff", "end_date"] + vint)
    d["frontier"] = d.groupby("stock_code", sort=False)["end_date"].cummax()
    d = d[d["end_date"] >= d["frontier"]]
    d = d.drop_duplicates(["stock_code", "eff"], keep="last")
    piv = d.pivot_table(index="eff", columns="stock_code", values=col, aggfunc="last")
    piv = piv.reindex(columns=codes)
    return piv.reindex(piv.index.union(dates)).ffill().reindex(dates).astype(np.float32)


def comp_type_panel(dates: pd.DatetimeIndex, codes: pd.Index) -> pd.Series:
    """每只股票的公司类型(1一般 2银行 3保险 4证券)。金融股的毛利率/周转率等无意义。"""
    rep = _final_rows(build_report_level())
    ct = rep.dropna(subset=["comp_type"]).drop_duplicates("stock_code", keep="last")
    return ct.set_index("stock_code")["comp_type"].reindex(codes).astype("object")


def nonfinancial_mask(dates: pd.DatetimeIndex, codes: pd.Index) -> pd.DataFrame:
    ct = comp_type_panel(dates, codes)
    ok = (ct.isna() | (ct.astype(str) == "1")).to_numpy()
    return pd.DataFrame(np.repeat(ok[None, :], len(dates), 0), index=dates, columns=codes)


def _read_st() -> pd.DataFrame:
    """每日风险警示名单 st_info_latest.csv,读入后立即丢掉 trade_date > 截止日的行。"""
    st = pd.read_csv(DATA / "st_info_latest.csv")
    st["trade_date"] = pd.to_datetime(st["trade_date"])
    record_source_end("st_info_latest.csv", st["trade_date"].max())   # 截断前登记原始最晚日期
    return st[st["trade_date"] <= data_cutoff()]


def build_st_panel(dates: pd.DatetimeIndex, codes: pd.Index) -> pd.DataFrame:
    """每日是否处于风险警示板。名单文件有日期缺口(研究期 1455 天缺 14 天,
    13 个缺口块中 12 个仅 1 天),按前值填充;实测多标 1 个、漏标 3 个股票-日。
    读入后立即丢掉 trade_date > 截止日的行(先截断再建面板)。

    源覆盖(第二轮审查 P1):ST 名单原始文件只到 2026-07-03。请求日期晚于它时默认报 SourceCoverageError ——
    否则最后一天的 ST 状态会被 ffill 到覆盖之外,引擎(panel._limit_frame)据此判定 5% 涨跌停 / 一字板 /
    不可成交,前向窗口的可成交性会被悄悄算错。前向检验应自建 ST 面板(新下载的名单)放进 p._derived["is_st"];
    确需沿用旧状态时在 config.accept_source_gap("st_info_latest.csv", reason) 内调用(发 RuntimeWarning)。
    守卫按读入时登记的原始最晚日期判断:若改为包装 _read_st、把新下载的名单拼进来,包装里要用
    config.record_source_end("st_info_latest.csv", 新名单最晚日期) 更新登记,否则仍按 07-03 报错。"""
    assert_within_cutoff(dates, "build_st_panel")
    st = _read_st()
    check_source_coverage("st_info_latest.csv", dates, "build_st_panel")
    st = st[st["stock_code"].isin(codes)]
    m = pd.crosstab(st["trade_date"], st["stock_code"]) > 0
    m = m.reindex(columns=codes, fill_value=False)
    first = m.index.min()
    m = m.reindex(m.index.union(dates)).astype(float).ffill().reindex(dates)
    m.loc[dates < first] = 0.0
    return (m.fillna(0.0) > 0.5)


# 送转比例识别:每 10 股送/转 n 股 -> 股本比值 (10+n)/10
def _songzhuan_factor(shares_q: pd.DataFrame) -> pd.DataFrame:
    """从 PIT 股本序列识别送转跳变,返回"从该期到样本末的累计送转倍数"。

    送转使股本按 (10+n)/10 的"整十分之一"比例跳增,同时前复权价已同比例下调;
    增发/回购/激励的比例是任意值,前复权价**不**调整。故只对送转部分做还原。
    """
    r = shares_q / shares_q.shift(1)
    x = r * 10.0
    clean = (r >= 1.05) & ((x - x.round()).abs() < 0.02) & (r <= 11.0)
    step = r.where(clean, 1.0).fillna(1.0)
    # 从 t 到末端的累计送转倍数 = 之后所有送转步长之积
    cum_fwd = step[::-1].cumprod()[::-1] / step
    return cum_fwd


MARKET_CAP_IS_APPROXIMATE = True   # 供上层判断是否该采信市值相关结论


def _adjbase_quarterly_shares() -> pd.DataFrame:
    """[qi x 股票] 季度总股本,截到前复权基准 PRICE_ADJ_BASE_END(**不按 DATA_CUTOFF**),
    每个报告期取最新版。只供 build_market_cap 识别送转(理由见其文档)。

    与旧版 rep.pivot_table(index="qi", values="total_share", aggfunc="last") 逐格相同:
    旧版报告表的 total_share 就是资产负债表每期最新版。用原始实现 _LOAD_IMPL,
    不受外部对 _load 的包装影响。
    """
    ck = ("adjbase_shares", pd.Timestamp(PRICE_ADJ_BASE_END))
    if ck not in _REP_CACHE:
        b = _pick_version(_LOAD_IMPL("stock_balancesheet_2015_now.pkl", ["total_share"],
                                     cutoff=PRICE_ADJ_BASE_END), None)
        b = b.assign(qi=(b["end_date"].dt.year * 4 + (b["end_date"].dt.quarter - 1)).astype(np.int64))
        _REP_CACHE[ck] = b.pivot_table(index="qi", columns="stock_code", values="total_share",
                                       aggfunc="last")
    return _REP_CACHE[ck]


def build_market_cap(dates: pd.DatetimeIndex, codes: pd.Index,
                     close_px: pd.DataFrame) -> pd.DataFrame:
    """⚠ **近似**市值,含无法消除的分红泄漏 —— 依赖它的结论一律标注为"待真实市值确认"。 = 前复权价 × PIT 股本 × 从当期到样本末的累计送转倍数。

    为什么这样算:前复权只抵消送转与分红,不抵消增发/回购。
      - 送转:前复权价已按比例下调,故乘"到末端的累计送转倍数"还原(2026-08-22 修正,
        此前一律乘"最新股本",把增发也当成送转,导致增发股历史市值被高估最多 11 倍);
      - 增发/回购:用 PIT 股本直接反映,不做额外调整。

    **仍存在的泄漏(无法用现有数据修复)**:前复权价还扣除了历年现金分红,
    使历史市值系统性偏低,且偏低幅度与"未来累计分红率"成正比(实测 600519 −17.5%、
    601398 −24.3%、000651 −33.2%;该分红率单独作为因子 IC 达 +0.0178)。
    因此本函数产出的是**近似市值**,所有依赖它的结论(fu_size、EP/BP 等分母、
    市值中性化)都带这一泄漏,须标注。彻底修复需要 tushare `daily_basic.total_mv`
    或 `adj_factor`(未复权价)。

    数据截止(2026-10-10):PIT 股本 sh 与"t 时可见哪一期"用截止日截断后的报告表;但**送转倍数**
    用截到价格前复权基准(PRICE_ADJ_BASE_END)的股本序列识别 —— 这是全项目唯一越过 DATA_CUTOFF
    读股本的地方。原因:close_px 的前复权基准在 2026-08,研究期价格已经按 2026 年的送转下调过;
    还原它必须用同一基准的送转记录。若也截到 2025-12-31,2026 年有送转的股票在整个研究期的市值会
    按送转比例系统性偏低(且"2026 年会送转"本身与 2025 年表现相关),反而引入与未来相关的误差。
    这些股本只进入送转倍数,不进入任何其它字段。前向检验请用 newdata.total_mv(真实市值),不要用本函数。
    **例外并不完整**(第二轮审查 P2):资产负债表原始文件最晚报告期是 2026Q1(可见日到 2026-06-13),
    只能识别 2026Q1 股本里已经反映的送转;2026-04 ~ 08 除权的送转(年报送转多在 5~7 月除权)
    已经压低了前复权价,却不在股本序列里,这些股票整个研究期的近似市值仍按送转比例偏低。
    量级见 DATA_CUTOFF_DISCIPLINE.md §4(用 2021-2025 年同期的送转频率估计)。这是既有问题,新旧代码相同。
    """
    rep = build_report_level()
    sh = _pit_daily(rep, "total_share", dates, codes).replace(0, np.nan)
    # 季度频率的股本序列(用于识别送转;截到前复权基准,见上),再映射回日频
    q = _adjbase_quarterly_shares()
    q = q.reindex(columns=codes).replace(0, np.nan)
    cum = _songzhuan_factor(q)
    cum_long = cum.stack().rename("sz_cum").reset_index()
    cum_long.columns = ["qi", "stock_code", "sz_cum"]
    cum_daily = _pit_daily(rep.merge(cum_long, on=["qi", "stock_code"], how="left"),
                           "sz_cum", dates, codes)
    cap = close_px.reindex(index=dates, columns=codes) * sh * cum_daily.fillna(1.0)
    return cap.where(sh.notna()).astype(np.float32)      # 无 PIT 股本者不给市值
