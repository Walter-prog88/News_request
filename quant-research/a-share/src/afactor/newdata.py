"""第四轮新数据层:真实市值/换手率、真实行业、盈余意外三层、龙虎榜、指数状态。

数据源(data_8:21/,用户 2026-08-22 补充):
  daily_basic.csv   新浪未复权日线 + 流通股本;含 close(未复权)、turnover_rate、float_share(万股)
  stock_basic.csv   BaoStock 证监会行业 + 上市/退市日(覆盖我们池子 93%)
  forecast.csv      业绩预告 11.1 万条(2018-12 起)
  express.csv       业绩快报 1.6 万条(2019-04 起)
  top_list.csv      龙虎榜 12.2 万条(2020-01 起)
  consensus_eps.csv 一致预期(仅 FY2026-28,99.8% 集中在 2024 年后,仅作探索)

关键:**真实市值 = 未复权价 × PIT 总股本**,彻底消除前复权的分红泄漏(第三轮 P0-1)。
`adj_factor.csv` 弃用 —— 新浪后复权与本项目数据源的前复权不同源,还原偏差 2.5%~7.9%。

数据截止(2026-10-10,前缀 fixb;见 research/DATA_CUTOFF_DISCIPLINE.md):
  这些文件都含 2026 行。每个加载器读入后**立即**按 config.data_cutoff()(默认 = RESEARCH_END)截断:
    daily_basic / top_list 按 trade_date;forecast / express 按 ann_date(公告日);
    stock_basic 丢掉 list_date > 截止日的行(截止日时尚未上市),delist_date > 截止日置空(尚未发生的退市不可知);
    宽基指数快照按日期。
  截断之后才做去重、透视、ffill。进程内缓存按截止日分键,set_data_cutoff() 切换后不会串用。
  stock_sw_industry.csv 与 stock_basic 的行业字段是**当前快照**、没有日期,无法截断(已知局限,见各函数说明)。

源覆盖守卫(2026-10-10 第二轮审查 P1):放宽截止日不等于有数据。各源读入时登记原始文件最晚日期
(config.record_source_end),龙虎榜 / 预告快报面板的请求日期晚于它时默认报 config.SourceCoverageError;
在 config.accept_source_gap(...) 内显式接受后:龙虎榜(脉冲型)覆盖之外的行置 NaN,不再 fillna(0) 冒充"未上榜";
预告 / 快报(状态型)沿用最后已知状态,距上次公告天数置 NaN。daily_basic 覆盖之外本来就是 NaN,仍只发 RuntimeWarning。
研究期所有源都覆盖到 2026 年,守卫不触发,数值不变。
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

from afactor import config as _C
from afactor.config import (ROOT, data_cutoff, assert_within_cutoff, record_source_end,
                            check_source_coverage)

DATA = ROOT / "data_8:21"
# 进程内缓存。默认截止日(DATA_CUTOFF)下键就是旧名字("db" / "ind" / "idx"),其它截止日下键为
# (名字, 截止日):set_data_cutoff() 切换后自动换键,不会串用另一个截止日的结果;同名只保留一份
# (daily_basic 一份就 ~1GB)。
# 默认键沿用旧名字是有意的(第二轮审查 P2):index_proxy._sealed_newdata、scripts/pead_build_batches、
# scripts/run_index_enhance_pead 按旧约定往 _CACHE["db"] 注入截到 2025-12-31 的 daily_basic,
# 并断言 _CACHE["db"] 的最大日期 —— 默认截止日下加载器读的正是这个键,注入与断言因此真正生效,
# 也不会出现"注入一份 + 加载器自己再读一份"的双份内存。注入的表若含截止日之后的行,_load_daily_basic 会再截一次。
_CACHE: dict[object, object] = {}


def _ymd(s):
    return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


def _cut_int(cut: pd.Timestamp) -> int:
    """截止日的 YYYYMMDD 整数(源文件日期列是整数,读入后立即按整数比较截断,先于任何解析)。"""
    return int(cut.strftime("%Y%m%d"))


def _int_date(v):
    return pd.to_datetime(str(int(v)), format="%Y%m%d") if pd.notna(v) else pd.NaT


def _read_dated_csv(name: str, date_col: str, **kw) -> pd.DataFrame:
    """读 data_8:21 下带 YYYYMMDD 整数日期列的 csv,读入后立即丢掉 date_col > 截止日的行。
    (forecast / express 按公告日 ann_date,top_list 按上榜日 trade_date。)
    截断前登记原始文件最晚日期(只是一个日期标量,供源覆盖守卫判断"源文件本身只到哪天")。"""
    d = pd.read_csv(DATA / name, **kw)
    record_source_end(name, _int_date(d[date_col].max()) if len(d) else None)
    return d[d[date_col] <= _cut_int(data_cutoff())].copy()


def _cache_key(name: str, cut: pd.Timestamp):
    """默认截止日下用旧键 name(兼容旧脚本的注入与断言),其它截止日下用 (name, 截止日)。"""
    return name if cut == pd.Timestamp(_C.DATA_CUTOFF) else (name, cut)


def _cached(name: str, build):
    """按当前截止日取缓存;未命中则丢弃同名的其它截止日版本后重建。"""
    cut = data_cutoff()
    key = _cache_key(name, cut)
    if key not in _CACHE:
        for k in [k for k in _CACHE if k == name or (isinstance(k, tuple) and k[0] == name)]:
            del _CACHE[k]
        _CACHE[key] = build(cut)
    return _CACHE[key]


# ---------------- 未复权价 / 真实市值 / 真实换手 ----------------
def _load_daily_basic() -> pd.DataFrame:
    """daily_basic.csv,读入后立即丢掉 trade_date > 截止日的行。

    默认截止日下缓存键是旧键 "db":旧脚本注入的截断版在这里被真正使用;若注入的表含截止日之后的行,
    这里再截一次并写回(不会把截止日之后的行交给调用方)。"""
    def build(cut):
        d = pd.read_csv(DATA / "daily_basic.csv",
                        usecols=["ts_code", "trade_date", "close", "turnover_rate",
                                 "turnover_rate_f", "float_share", "circ_mv"])
        record_source_end("daily_basic.csv", _int_date(d["trade_date"].max()) if len(d) else None)
        d = d[d["trade_date"] <= _cut_int(cut)].copy()          # 先截断
        d["trade_date"] = _ymd(d["trade_date"])
        return d
    d = _cached("db", build)
    cut = data_cutoff()
    if len(d) and d["trade_date"].max() > cut:                  # 外部注入的表没截干净
        d = d[d["trade_date"] <= cut].copy()
        _CACHE[_cache_key("db", cut)] = d
    return d


def _piv(col, dates, codes):
    assert_within_cutoff(dates, f"newdata.{col}")
    d = _load_daily_basic()
    if len(dates) and len(d) and pd.Timestamp(max(dates)) > d["trade_date"].max():
        # 放宽截止日 != 有数据:源文件本身只到某天(data_8:21 的 daily_basic 到 2026-08-21),之后全是 NaN
        import warnings
        warnings.warn(f"daily_basic 只覆盖到 {d['trade_date'].max().date()},请求到 {pd.Timestamp(max(dates)).date()}:"
                      f"之后的 {col} 为 NaN(前向检验需另行补数)", RuntimeWarning, stacklevel=2)
    return (d.pivot(index="trade_date", columns="ts_code", values=col)
            .reindex(index=dates, columns=codes).astype(np.float32))


def raw_close(dates, codes):
    """未复权收盘价(与本项目分钟数据的非除权日收益一致,实测差 1e-6)。"""
    return _piv("close", dates, codes)


def turnover_rate(dates, codes):
    """真实换手率 %(成交量/流通股本)。与项目原有的成交额代理秩相关仅 0.35,是独立维度。"""
    return _piv("turnover_rate", dates, codes)


def float_share(dates, codes):
    """流通股本。源文件单位为**万股**,此处换算为股。"""
    return _piv("float_share", dates, codes) * 1e4


def circ_mv(dates, codes):
    """流通市值(源文件单位万元 -> 元)。"""
    return _piv("circ_mv", dates, codes) * 1e4


def total_mv(dates, codes):
    """**真实总市值** = 未复权价 × 日频总股本(无分红泄漏)。股本口径见 total_share。"""
    return (raw_close(dates, codes) * total_share(dates, codes)).astype(np.float32)


# ---------------- 日频总股本:两次财报之间只外推送转(2026-10-10,前缀 mvfix)----------------
# 见 research/TOTAL_MV_FIX.md。阈值在 2020-2025 研究期上标定,验证标签是新浪复权因子(adj_factor.csv,
# 独立来源)的除权跳变 —— 只在 scripts/mvfix_total_mv.py 里当标签用,不进入本函数。
SZ_MIN_STEP = 1.05        # 最小送转比例(每 10 股送转 0.5 股)
SZ_PX_TOL = 0.21          # 按送转除权后的隐含日收益 |log(价比) + log r| 上限(覆盖 20% 涨跌幅板)
SZ_CLEAN_BELOW = 1.2      # r 低于它时还要求接近名义送转比例(0.05 的整数倍):小比例解禁碰上大跌太常见
SZ_CLEAN_DEFICIT = 0.05   # 回购专户股份不参与送转,流通股本比例略低于名义比例:允许差名义增量的 5%
SZ_CLEAN_EXCESS = 0.002
REFLECT_TOL = 0.005       # 新财报股本与旧外推值(含期末之后的送转)相差 < 0.5% -> 视为该期股本已含这些送转
IPO_MAX_LAG = 10          # 上市日(stock_basic)到 daily_basic 首个有效流通股本的最大交易日数,超过则不做新股加法
# 诊断开关:"legacy" 复现 2026-08-22 ~ 2026-10-10 的旧口径,只供新旧对照与回归测试;研究一律用默认值
TOTAL_SHARE_METHOD = "songzhuan"


def _ffill_shift(a: np.ndarray) -> np.ndarray:
    """每列在 t 之前(不含 t)最后一个有效值。"""
    return pd.DataFrame(a).ffill().shift(1).to_numpy(np.float64)


def _songzhuan_steps(fs: np.ndarray, px: np.ndarray) -> np.ndarray:
    """逐日识别送转,返回 [T x N] 步长:送转日 = 流通股本跳升比例 r,其余 = 1。t 日只用 t 及之前的数据。

    判据(三条同时满足):
      1. 流通股本相对上一个有效观测跳升 r >= SZ_MIN_STEP;
      2. 未复权价在 t 或 t-1 有对应的除权下跌:|log(价比) + log r| <= SZ_PX_TOL,且价格至少跌了 log r 的一半。
         沪市红股上市日(流通股本更新)常比除权日晚一个交易日,所以看两天;
      3. r < SZ_CLEAN_BELOW 时比例还要接近 0.05 的整数倍(名义送转比例)。
    限售解禁、定增股份上市让流通股本跳升但价格不除权,不算送转 —— 解禁不改变总股本,定增在下一期财报里反映。
    """
    fs = np.asarray(fs, np.float64)
    px = np.asarray(px, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = fs / _ffill_shift(fs)
        lr = np.log(r)
        lp0 = np.log(px / _ffill_shift(px))
    lp1 = np.vstack([np.full((1, px.shape[1]), np.nan), lp0[:-1]])       # t-1 当天的价比
    with np.errstate(invalid="ignore"):
        ex = np.zeros(r.shape, bool)
        for lp in (lp0, lp1):
            ex |= (np.abs(lp + lr) <= SZ_PX_TOL) & (lp <= -0.5 * lr)
        m = np.ceil((r - SZ_CLEAN_EXCESS - 1.0) / 0.05 - 1e-9) * 0.05 + 1.0    # 不低于 r - 容差的最小名义比例
        clean = (m >= SZ_MIN_STEP - 1e-9) & (r >= m - SZ_CLEAN_DEFICIT * (m - 1.0)) & (r <= m + SZ_CLEAN_EXCESS)
        is_sz = (r >= SZ_MIN_STEP) & ex & ((r >= SZ_CLEAN_BELOW) | clean)
    return np.where(is_sz, r, 1.0)


def _quarter_end(qi) -> np.ndarray:
    """季度序号 qi = 年 × 4 + (季 - 1) -> 报告期末(datetime64[D])。"""
    q = np.asarray(qi).astype(np.int64)
    first_next = (q // 4 - 1970) * 12 + (q % 4 + 1) * 3                  # 下一季首月(自 1970-01 起的月数)
    return first_next.astype("datetime64[M]").astype("datetime64[D]") - np.timedelta64(1, "D")


def total_share(dates, codes) -> pd.DataFrame:
    """**日频总股本**(股)= max((PIT 总股本 [+ 新股发行]) × 该期报告期末以来识别出的送转倍数, 流通股本)。

    为什么要外推:财报股本是季频的、按报告期末计。送转除权当天价格立刻下跌,不外推的话市值在除权日掉到 1/r
    (2026-08-22 审查 H-3:1375 个送转除权日 100% 未跟随)。`float_share` 是日频准确的,但它的跳升多数是
    **限售解禁**,解禁不改变总股本。

    旧口径(TOTAL_SHARE_METHOD="legacy",2026-08-22 ~ 2026-10-10)把流通股本相对"PIT 总股本最近一次数值变化时"
    的全部增长乘到总股本上,只在股本数值变化时重新锚定:总股本多年不变、流通股本因解禁增长时倍数一直累积
    (研究期 13.6% 的格子高估 1.5 倍以上,600339.SH 全程 3.60 倍;DATA_CUTOFF_DISCIPLINE.md §4 第 8 条)。

    新口径(2026-10-10,前缀 mvfix;research/TOTAL_MV_FIX.md):
      1. 每份 PIT 财报都重新锚定,锚点是它的**报告期末** E(资产负债表股本是期末数:研究期"期末之后、
         公告之前"除权的送转 86% 没有反映在该期股本里);
      2. 只外推 E 之后识别出的送转(_songzhuan_steps:流通股本跳升 + 同步除权下跌 + 名义比例);
         解禁、定增上市、可转债转股都不外推,留给下一期财报;
      3. 新财报生效当天,若它的股本与旧外推值(含 E 之后的送转)相差不到 REFLECT_TOL,视为该期股本已含这些
         送转,锚点改为生效日(防止重复计入;研究期约 6% 的期后送转属于这种);
      4. 新股:报告期末早于上市日的财报是发行前股本(上市当天生效的这类财报 1,593 份里只有 6 份已含发行股份),
         上市后加上首个交易日的流通股本(= 新发行的可流通股份);若之后公告的"上市前报告期"财报股本已比上市时
         增加了至少一半的这部分股份,视为已含发行股份,不再加(研究期这类财报 731 份中 87 份已含;mvfix_ipo_reports.csv);
      5. 流通股本为下界:全流通股票的增发 / 换股合并 / 重整转增让流通股本超过旧总股本时,由下界接住。
    严格 PIT:t 日只用 avail <= t 的财报与 <= t 的流通股本、未复权价;锚点不晚于 t。
    局限:流通股本从 daily_basic 首日(2020-01-02)开始,此前的期后送转看不到(按 1 处理,与旧口径相同);
    部分流通股票的配股 / 换股合并等加法型变化与解禁无法区分,等下一期财报;科创 / 创业板新股首日流通股本不含
    战略配售与网下限售部分,新股加法仍略偏低。stock_basic 缺上市日的股票不做新股加法。无 PIT 股本者退回流通股本。
    """
    assert_within_cutoff(dates, "newdata.total_share")
    if TOTAL_SHARE_METHOD == "legacy":
        return _total_share_legacy(dates, codes)
    from afactor import fundamental as FU
    dates = pd.DatetimeIndex(dates)
    codes = pd.Index(codes)
    # 在 daily_basic 的交易日历上算(送转可能发生在请求区间之前、报告期末之后),只到最后一个请求日:
    # 之后的日子用不到,也不该去触发财报源覆盖守卫
    tds = pd.DatetimeIndex(_load_daily_basic()["trade_date"].unique())
    cal = (tds[tds <= dates.max()] if len(dates) else tds[:0]).union(dates)
    rep = FU.build_report_level()
    sh = FU._pit_daily(rep, "total_share", cal, codes).replace(0, np.nan).to_numpy(np.float64)
    # 同一行的报告期:_pit_daily 对两列选出同一组候选行(qi 没有缺失)
    qrep = rep.assign(sh_qi=rep["qi"].where(rep["total_share"].notna()).astype(np.float64))
    qi = FU._pit_daily(qrep, "sh_qi", cal, codes).to_numpy(np.float64)
    fs = float_share(cal, codes).to_numpy(np.float64)
    C = np.cumsum(np.log(_songzhuan_steps(fs, raw_close(cal, codes).to_numpy())), axis=0)  # 第 0..t 日送转的对数累计
    calv = cal.values.astype("datetime64[D]")
    T, N = sh.shape
    rows = np.arange(T)
    # 新股:上市日(stock_basic)与 daily_basic 首个有效流通股本(上市首日的可流通股份)
    lst = listing_info(codes)["list_date"].to_numpy("datetime64[D]")
    has_fs = np.isfinite(fs)
    first = np.where(has_fs.any(axis=0), has_fs.argmax(axis=0), -1)
    out = np.full((T, N), np.nan)
    for j in range(N):
        s, q, c = sh[:, j], qi[:, j], C[:, j]
        ok = np.isfinite(s) & np.isfinite(q)
        if not ok.any():
            continue
        L, f0 = lst[j], int(first[j])
        ipo = (not np.isnat(L) and f0 >= 0 and calv[0] < L <= calv[f0]
               and f0 - int(np.searchsorted(calv, L)) <= IPO_MAX_LAG)
        base = s.copy()                                                # 财报股本(+ 新股加法)
        okq = np.where(ok, q, np.nan)
        starts = np.nonzero(ok & ~(np.r_[np.nan, okq[:-1]] == okq))[0]     # 新报告期开始生效的行
        anchor = np.zeros(T, np.int64)
        prev = None                                                    # (上一段最后一行, 上一段锚点)
        for i0, i1 in zip(starts, np.r_[starts[1:], T]):
            qe = _quarter_end(q[i0])
            if ipo and qe < L and i1 > f0:                             # 上市前报告期的财报,在上市后生效的行
                A, sL = fs[f0, j], s[f0]
                seg = np.arange(max(i0, f0), i1)
                done = np.isfinite(sL) & (s[seg] - sL >= 0.5 * A)      # 该期股本已含发行股份
                base[seg] = np.where(done, s[seg], s[seg] + A)
            a = min(int(np.searchsorted(calv, qe, side="right")) - 1, i0)
            ca = c[a] if a >= 0 else 0.0
            if prev is not None and c[i0] - ca > 1e-12:               # E 之后、生效日之前有送转
                pi, pa = prev
                old_est = base[pi] * np.exp(c[i0] - (c[pa] if pa >= 0 else 0.0))
                if abs(np.log(base[i0] / old_est)) < REFLECT_TOL:
                    a = i0                                             # 该期股本已含这些送转
            anchor[i0:i1] = a
            prev = (i1 - 1, a)
        a = np.minimum(anchor, rows)                                   # 锚点不晚于 t
        ca = np.where(a >= 0, c[np.maximum(a, 0)], 0.0)
        out[:, j] = np.where(ok, base * np.exp(c - ca), np.nan)
    out = np.fmax(out, fs)                                             # 流通股本为下界;无 PIT 股本者退回流通股本
    return pd.DataFrame(out, index=cal, columns=codes).reindex(dates)


def _total_share_legacy(dates, codes) -> pd.DataFrame:
    """旧口径(2026-08-22 ~ 2026-10-10),**只供新旧对照与回归测试**:流通股本相对"PIT 总股本最近一次数值变化时"
    的累计倍数(clip 到 [1, 20])乘到 PIT 总股本上,流通股本为下界。会把限售解禁当成股本增加。"""
    from afactor import fundamental as FU
    rep = FU.build_report_level()
    sh = FU._pit_daily(rep, "total_share", dates, codes).replace(0, np.nan)
    fs = float_share(dates, codes)
    # PIT 总股本的"生效日"= 它发生跳变的日子;在两次跳变之间用流通股本的相对变化外推
    changed = sh.diff().abs() > 0
    changed.iloc[0] = True
    fs_at_report = fs.where(changed).ffill()          # 每次财报股本生效时的流通股本
    mult = (fs / fs_at_report).clip(lower=1.0, upper=20.0)   # 只向上外推(送转/增发)
    sh_adj = sh * mult.fillna(1.0)
    sh_adj = np.fmax(sh_adj.to_numpy(np.float64), fs.to_numpy(np.float64))  # 流通股本为下界
    return pd.DataFrame(sh_adj, index=dates, columns=codes)


def data_coverage() -> pd.DataFrame:
    """当前截止日下每个 data_8:21 源实际覆盖到哪天(放宽截止日不等于有数据,前向检验先看这张表)。"""
    from afactor import fundamental as FU
    rows = [("daily_basic.csv", "trade_date", _load_daily_basic()["trade_date"].max())]
    for name, col in [("forecast.csv", "ann_date"), ("express.csv", "ann_date"), ("top_list.csv", "trade_date")]:
        v = _read_dated_csv(name, col, usecols=[col])[col].max()
        rows.append((name, col, _int_date(v)))
    sb = _stock_basic()
    rows.append(("stock_basic.csv", "list_date", pd.to_datetime(str(int(sb["list_date"].max())), format="%Y%m%d")))
    rows.append(("st_info_latest.csv", "trade_date", FU._read_st()["trade_date"].max()))
    for f in FU.REPORT_TABLES:
        rows.append((f, "max(ann_date,f_ann_date)", FU._load(f, [])["avail"].max()))
    idx = index_daily()
    rows.append(("宽基指数快照", "date", idx.index.max() if len(idx) else pd.NaT))
    out = pd.DataFrame(rows, columns=["source", "date_key", "max_date"])
    out["data_cutoff"] = data_cutoff()
    # 源文件原始最晚日期(与截止日无关;源覆盖守卫按它判断)。daily_basic 被外部注入时未登记,为 NaT
    out["source_end"] = [_C.source_end(s) if _C.source_end(s) is not None else pd.NaT for s in out["source"]]
    return out


# ---------------- 真实行业 ----------------
def industry_labels(dates, codes) -> pd.DataFrame:
    """证监会行业(BaoStock),整数编码的日频面板;-1 表示无行业。

    局限:是**当前快照**而非历史 PIT 成员关系(行业变更未记录),覆盖我们池子 93%。
    行业变化缓慢,业内在无历史成员数据时通行此做法,但需标注。
    截止日只能去掉"截止日时尚未上市"的股票(list_date > 截止日),行业标签本身无法回到截止日。
    (实测去掉的 48 只 2026 年上市股里没有独有的行业,整数编码与截断前一致。)
    """
    def build(cut):
        sb = _stock_basic(cut)
        m = sb.dropna(subset=["industry"]).drop_duplicates("ts_code").set_index("ts_code")["industry"]
        cats = {v: i for i, v in enumerate(sorted(m.unique()))}
        return m.map(cats)
    lab = _cached("ind", build).reindex(codes).fillna(-1).astype(np.int16)
    return pd.DataFrame(np.repeat(lab.to_numpy()[None, :], len(dates), 0),
                        index=dates, columns=codes)


def sw_industry_labels(dates, codes, level: str = "sw") -> pd.DataFrame:
    """申万行业(用户 2026-08-23 提供的 stock_sw_industry.csv),整数编码日频面板;-1 = 无行业。

    level="sw"   -> sw_industry 申万一级(31 类)
    level="fine" -> fine_industry 细分行业(111 类)

    ⚠ 与 industry_labels() 同样是**当前快照**,没有 in_date/out_date,用它给 2020 年
    的股票贴标签有轻微前视(期间发生过行业变更的个股会用 2026 的标签)。行业一级
    变更率低,但结论须标注,并与统计行业(statind)交叉验证。
    """
    ck = f"sw_ind_{level}"
    if ck not in _CACHE:
        col = {"sw": "sw_industry", "fine": "fine_industry"}[level]
        sw = pd.read_csv(ROOT / "stock_sw_industry.csv", encoding="utf-8-sig")
        m = sw.dropna(subset=[col]).drop_duplicates("ts_code").set_index("ts_code")[col]
        cats = {v: i for i, v in enumerate(sorted(m.unique()))}
        _CACHE[ck] = (m.map(cats), {i: v for v, i in cats.items()})
    lab, _ = _CACHE[ck]
    v = lab.reindex(codes).fillna(-1).astype(np.int16)
    return pd.DataFrame(np.repeat(v.to_numpy()[None, :], len(dates), 0),
                        index=dates, columns=codes)


def sw_industry_names(level: str = "sw") -> dict:
    """整数编码 -> 行业名。调用前需先调用过 sw_industry_labels。"""
    sw_industry_labels(pd.DatetimeIndex([]), pd.Index([]), level)
    return _CACHE[f"sw_ind_{level}"][1]


def _stock_basic(cut: pd.Timestamp | None = None) -> pd.DataFrame:
    """stock_basic.csv,读入后立即按截止日处理日期字段。

    list_date > 截止日 -> 整行丢弃(截止日时这只股票还不存在);
    delist_date > 截止日 -> 置空(截止日时尚未发生的退市不可知)。
    name / industry 是下载时的快照,无法回到截止日(name 不被任何加载器使用)。
    """
    cut = data_cutoff() if cut is None else cut
    ci = _cut_int(cut)
    sb = pd.read_csv(DATA / "stock_basic.csv")
    sb = sb[sb["list_date"].isna() | (sb["list_date"] <= ci)].copy()
    sb.loc[sb["delist_date"] > ci, "delist_date"] = np.nan
    return sb


def listing_info(codes) -> pd.DataFrame:
    """上市日 / 退市日(按截止日截断:截止日之后才上市的股票两列都为 NaT,之后的退市日为 NaT)。"""
    sb = _stock_basic().drop_duplicates("ts_code").set_index("ts_code")
    out = pd.DataFrame(index=codes)
    out["list_date"] = _ymd(sb["list_date"]).reindex(codes)
    out["delist_date"] = _ymd(sb["delist_date"]).reindex(codes)
    return out


# ---------------- 盈余意外三层(预告 / 快报 / 正式财报)----------------
def earnings_surprise_events(dates, codes) -> pd.DataFrame:
    """业绩预告与快报的事件级意外,含公告日(可见日)。

    三层意外:
      layer1 预告 vs 去年同期  = p_change 中值(公司自己给的同比预测)
      layer2 快报 vs 预告      = 快报净利 相对 预告区间中值 的偏离
      layer3 正式财报 vs 快报  = 在 fundamental 侧处理(已有 SUE)

    截止:读入后立即丢掉 ann_date > 截止日的行,**再**按 (股票, 报告期) 保留最晚公告的版本。
    顺序不能反:先去重会让 2026 年的修正预告把截止日前的原预告挤掉(实测 4 个报告期)。
    去重后的行在其自身 ann_date 之后才生效(_event_panel),所以修正版不会提前生效;
    被修正的原版本在 [原公告, 修正公告) 区间不可见 —— 是延迟,不是前视(沿用原口径)。
    源覆盖:dates 晚于 forecast / express 原始文件最晚公告日时默认报 SourceCoverageError(见模块说明)。
    """
    assert_within_cutoff(dates, "earnings_surprise_events")
    fc = _read_dated_csv("forecast.csv", "ann_date")           # 先截断
    fc["ann_date"] = _ymd(fc["ann_date"]); fc["end_date"] = _ymd(fc["end_date"])
    fc = fc[fc["ts_code"].isin(codes) & fc["ann_date"].notna()]
    fc["p_change_mid"] = fc[["p_change_min", "p_change_max"]].mean(axis=1)
    fc["np_mid"] = fc[["net_profit_min", "net_profit_max"]].mean(axis=1)
    fc = fc.sort_values(["ts_code", "end_date", "ann_date"]).drop_duplicates(
        ["ts_code", "end_date"], keep="last")

    ex = _read_dated_csv("express.csv", "ann_date")            # 先截断
    ex["ann_date"] = _ymd(ex["ann_date"]); ex["end_date"] = _ymd(ex["end_date"])
    ex = ex[ex["ts_code"].isin(codes) & ex["ann_date"].notna()]
    ex = ex.sort_values(["ts_code", "end_date", "ann_date"]).drop_duplicates(
        ["ts_code", "end_date"], keep="last")
    check_source_coverage(["forecast.csv", "express.csv"], dates, "earnings_surprise_events")

    m = ex.merge(fc[["ts_code", "end_date", "np_mid", "p_change_mid", "ann_date"]],
                 on=["ts_code", "end_date"], how="outer", suffixes=("", "_fc"))
    # 快报相对预告的偏离(仅当两者都有,且快报晚于预告)
    den = m["np_mid"].abs()
    m["ex_vs_fc"] = ((m["n_income"] - m["np_mid"]) / den.where(den > 1e-6)).clip(-2, 2)
    m.loc[m["ann_date"].notna() & m["ann_date_fc"].notna() &
          (m["ann_date"] <= m["ann_date_fc"]), "ex_vs_fc"] = np.nan
    return {"forecast": fc, "express": ex, "combined": m}


def _next_trading_day(ts: pd.Series, dates: pd.DatetimeIndex, lag: int = 1):
    """把公告日映射到"其后第 lag 个交易日"(日历 +1 天可能落在周末,会丢事件)。"""
    di = dates.to_numpy()
    # side="right":公告日**当天**不算,严格取其后的交易日。
    # 财报预告/快报与龙虎榜都在收盘后发布,当天可见即为前视(2026-08-22 修正)。
    pos = np.searchsorted(di, ts.to_numpy(), side="right") + (lag - 1)
    ok = pos < len(dates)
    pos = np.clip(pos, 0, len(dates) - 1)
    return pd.Series(np.where(ok, di[pos], np.datetime64("NaT")), index=ts.index)


def _event_panel(ev: pd.DataFrame, val: str, ann: str, dates, codes,
                 lag_days: int = 1, ffill: bool = True) -> pd.DataFrame:
    """事件 -> 日频面板:公告日之后第 lag 个交易日起生效。

    ffill=True 用于"状态型"事件(财报意外,持续到下一次公告);
    ffill=False 用于"脉冲型"事件(龙虎榜,只在当日有效)。
    """
    d = ev[["ts_code", ann, val]].dropna()
    if d.empty:
        return pd.DataFrame(np.nan, index=dates, columns=codes, dtype=np.float32)
    d = d.rename(columns={ann: "eff"})
    d["eff"] = _next_trading_day(d["eff"], dates, lag_days)
    d = d.dropna(subset=["eff"]).sort_values(["ts_code", "eff"])
    d = d.drop_duplicates(["ts_code", "eff"], keep="last")
    piv = d.pivot_table(index="eff", columns="ts_code", values=val, aggfunc="last")
    piv = piv.reindex(columns=codes)
    piv = piv.reindex(piv.index.union(dates))
    if ffill:
        piv = piv.ffill()
    return piv.reindex(dates).astype(np.float32)


def surprise_panels(dates, codes, lag_days: int = 1) -> dict[str, pd.DataFrame]:
    """盈余意外的日频面板(状态型,持续到下一次公告)+ 距上次公告的交易日数。

    源覆盖:请求日期晚于预告 / 快报原始文件最晚公告日时默认报错;在 accept_source_gap 内接受后,
    状态型面板沿用最后已知值,days_since_surprise 在覆盖之外置 NaN(不知道其间有没有新公告)。"""
    e = earnings_surprise_events(dates, codes)
    out = {}
    out["fc_pchange"] = _event_panel(e["forecast"], "p_change_mid", "ann_date", dates, codes, lag_days)
    out["ex_yoy_np"] = _event_panel(e["express"], "yoy_net_profit", "ann_date", dates, codes, lag_days)
    out["ex_yoy_sales"] = _event_panel(e["express"], "yoy_sales", "ann_date", dates, codes, lag_days)
    out["ex_roe"] = _event_panel(e["express"], "diluted_roe", "ann_date", dates, codes, lag_days)
    out["ex_vs_fc"] = _event_panel(e["combined"], "ex_vs_fc", "ann_date", dates, codes, lag_days)
    # 距最近一次预告/快报的交易日数(限定漂移窗口用)
    allev = pd.concat([e["forecast"][["ts_code", "ann_date"]],
                       e["express"][["ts_code", "ann_date"]]])
    allev["mark"] = 1.0
    pulse = _event_panel(allev, "mark", "ann_date", dates, codes, lag_days, ffill=False)
    idx = np.arange(len(dates), dtype=np.float32)
    stamp = pd.DataFrame(np.where(pulse.notna(), idx[:, None], np.nan),
                         index=dates, columns=codes).ffill()
    grid = pd.DataFrame(np.repeat(idx[:, None], len(codes), 1), index=dates, columns=codes)
    out["days_since_surprise"] = (grid - stamp).astype(np.float32)
    end = check_source_coverage(["forecast.csv", "express.csv"], dates, "surprise_panels")
    if end is not None:                                           # 已显式接受缺口
        out["days_since_surprise"].loc[pd.DatetimeIndex(dates) > end] = np.nan
    return out


# ---------------- 龙虎榜 ----------------
def toplist_panels(dates, codes, lag_days: int = 1) -> dict[str, pd.DataFrame]:
    """龙虎榜事件面板。上榜日 T 的信息在 T 日盘后公布,故 T+1 起可用。

    源覆盖:请求日期晚于 top_list.csv 原始最晚上榜日时默认报 SourceCoverageError —— 否则覆盖之外的日子会被
    fillna(0) 成"全市场未上榜"。在 accept_source_gap 内接受后,生效日晚于"源最晚上榜日之后第 lag 个交易日"
    的行(它们对应的上榜日都在源覆盖之外)一律置 NaN,20 日滚动量也一样。"""
    assert_within_cutoff(dates, "toplist_panels")
    tl = _read_dated_csv("top_list.csv", "trade_date")         # 先截断
    end = check_source_coverage("top_list.csv", dates, "toplist_panels")
    tl["trade_date"] = _ymd(tl["trade_date"])
    tl = tl[tl["ts_code"].isin(codes) & tl["trade_date"].notna()]
    tl["is_up"] = tl["reason"].astype(str).str.contains("涨幅|上涨").astype(float)
    tl["is_down"] = tl["reason"].astype(str).str.contains("跌幅|下跌").astype(float)
    tl["is_turn"] = tl["reason"].astype(str).str.contains("换手").astype(float)
    g = tl.groupby(["ts_code", "trade_date"]).agg(
        net_ratio=("net_amount_ratio", "mean"), net_amt=("net_amount", "sum"),
        deal_ratio=("deal_amount_ratio", "mean"),
        up=("is_up", "max"), down=("is_down", "max"), turn=("is_turn", "max")).reset_index()
    g["on_list"] = 1.0
    out = {}
    for col in ["net_ratio", "net_amt", "deal_ratio", "up", "down", "turn", "on_list"]:
        out["tl_" + col] = _event_panel(g, col, "trade_date", dates, codes,
                                        lag_days, ffill=False).fillna(0.0)
    # 上榜是稀疏脉冲事件(约 2% 股票日),滚动窗口版才是可用的截面变量
    out["tl_count20"] = out["tl_on_list"].rolling(20, min_periods=5).sum()
    out["tl_net20"] = out["tl_net_ratio"].rolling(20, min_periods=5).sum()
    out["tl_updown20"] = (out["tl_up"] - out["tl_down"]).rolling(20, min_periods=5).sum()
    if end is not None:                                           # 已显式接受缺口:脉冲型覆盖之外不可知
        di = pd.DatetimeIndex(dates)
        # 源最晚上榜日 end 的信息在其后第 lag 个交易日生效;更晚的生效日对应的上榜日都在覆盖之外
        k = int(np.searchsorted(di.to_numpy(), np.datetime64(end), side="right")) + (lag_days - 1)
        if k < len(di):
            for v in out.values():
                v.iloc[k + 1:] = np.nan
    return out


# ---------------- 指数状态(时序模型用)----------------
# 宽基指数冻结快照(来自同作者的 ETF 项目)。用 INDEX_DIR 环境变量覆盖;
# 缺失时 timing.market_features(index_code=None) 会退回自建全A等权指数。
INDEX_DIR = Path(os.environ.get(
    "INDEX_DIR",
    ROOT.parent / "ETF-Research" / "index project" / "data" / "raw" /
    "akshare_index_history_frozen"))
INDEX_NAMES = {"000300": "沪深300", "000905": "中证500", "000852": "中证1000",
               "000016": "上证50", "932000": "北证50"}


def index_daily() -> pd.DataFrame:
    """五个宽基指数的日收盘(取每个目录中最新的冻结快照),**按截止日截断**后返回。

    快照文件含 2026 行;每个序列读入后立即丢掉日期 > 截止日的行,再拼表。
    调用方拿到的表不会含截止日之后的日期(index_proxy.index_close 等处原有的二次截断仍保留,无害)。
    """
    def build(cut):
        import glob
        out = {}
        for code in INDEX_NAMES:
            fs = sorted(glob.glob(str(INDEX_DIR / code / "*.parquet")))
            if not fs:
                continue
            d = pd.read_parquet(fs[-1])
            s = d["close"] if "close" in d.columns else d.iloc[:, 0]
            s.index = pd.to_datetime(s.index)
            out[code] = s[s.index <= cut]                       # 先截断
        return pd.DataFrame(out).sort_index()
    return _cached("idx", build)
