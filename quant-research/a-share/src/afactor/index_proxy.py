"""中证 300 / 500 / 1000 历史成分与权重的 point-in-time 代理重建。

为什么需要它
------------
指数增强优化器需要**每个调仓日的基准权重**,但项目没有指数历史成分权重
(data_8:21/index_cons/ 只有一份 2026-08-21 的主题指数快照,且其中根本不含
000300/000905/000852)。本模块按中证指数公司公开编制方法的简化版重建 PIT 代理,
再用真实指数收盘收益校验(见 scripts/run_index_proxy.py 与 research/CSI1000_PROXY.md)。

编制规则(公开方法的简化版;"简化"处逐条在注释里标明)
------------------------------------------------------
样本空间  沪深 A 股(含 CDR,不含北交所);非 ST/*ST(用每日 ST 名单,PIT);公告日前未退市;
          科创板/创业板上市满一年,其他上市满一个季度(除非上市以来日均总市值排前 30)。
沪深300   样本空间按过去一年日均成交额剔除后 50%,再按过去一年日均总市值取前 300;
          缓冲区:排名 240 内的新样本优先进入,360 内的老样本优先保留。
中证500   样本空间剔除 沪深300 成分 与 日均总市值前 300,按成交额剔除后 20%,
          再按日均总市值取前 500;缓冲区 400 / 600。
中证1000  样本空间剔除 中证800(=300+500) 成分 与 日均总市值前 300,按成交额剔除后 20%,
          再按日均总市值取前 1000;缓冲区 800 / 1200。
定期调整  每年 6 月、12 月第二个星期五的下一交易日生效;考察数据截止 4/30、10/31
          (取该日及之前最后一个交易日);单次调整比例不超过 10%。
权重      自由流通市值的近似 = 流通市值 circ_mv(daily_basic,未复权价 × 流通股本)。
          项目没有自由流通股本(free_share 全空),**做不了真正的分级靠档**;
          `weight="banded"` 用 流通/总股本 比例套靠档表,只是敏感性对照,不是真实口径。
          调仓日之间权重随价格漂移(期间股本变动不跟随)。

PIT(只用当时已知的数据)
-----------------------
- 排名用的日均市值/成交额只取 [截止日-1年, 截止日] 窗口;截止日远早于生效日。
- ST 状态取"公告日代理"= 生效日前 10 个交易日(真实公告约在生效前两周)。
- 基准权重取生效日前 **2** 个交易日收盘的流通市值,再按价格漂移一天:
    收盘口径 W_close[t_e-1] 与"用 t_e-1 收盘流通市值算权重"只差当天的股本变动;
    尾盘口径 W_tail[t_e-1] = 生效前 2 日收盘权重,在 t_e-1 的 14:50 已知(无日内前视)。
- 所有 data_8:21 的输入(daily_basic、ST 名单、财报股本)先截到 <= 2025-12-31 再计算;
  指数收盘同样截断。**不读任何 2026 年数据**。

已知偏差(详见报告)
------------------
- 幸存者偏差:面板缺约 206 只 2020-2025 退市股。它们不会进入代理 -> 排名相邻的
  存活股替补进来;退市股在被剔除前的亏损也不在代理里 -> 代理收益**偏高**。
- 价格指数 vs 全收益:真实 000852 是价格指数(除息不调整);面板 close_px 是前复权价
  (含分红再投资)。校验用 `kind="pr"` 剔除现金分红,`kind="tr"` 为全收益口径。
- 临时调整(退市/吸收合并替补、*ST 临时剔除)未实现。
- 300114.SZ(中航电测)于 2025-02-17 更名为 302132.SZ,面板缺 300114 全部历史;
  302132 的上市日按 300114 的 2010-08-27 处理(唯一的代码变更硬编码)。

对下游的基准接口(2026-10-10 第二轮修订,前缀 fixa)
------------------------------------------------
**推荐** exec_time_bench_returns(p, mode):代理收盘权重先漂移到成交时点再乘引擎成交价收益
(= 在成交价上计价的同一只指数)。engine_consistent_bench_returns(收盘权重直接乘成交价收益)
已弃用并发 DeprecationWarning:它每天把权重重置回过时一个隔夜的收盘权重,系统性偏低
(2021-2025 open10 约 0.8pp/年、vwap1 约 2.5pp/年,research/ledger/fixa_bench_compare.csv)。
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd

from afactor.config import RESEARCH_END, TMP

# ---------------- 常量 ----------------
SEAL_END = pd.Timestamp(RESEARCH_END)          # 2025-12-31,一切输入先截到这里
PROXY_VERSION = "proxy-v1"                     # 规则或输入口径变化时改它,缓存自动失效

# (成分数, 成交额剔除比例, 新样本优先进入名次, 老样本优先保留名次)
CSI_RULES = {
    "000300": dict(n=300, liq_cut=0.50, new_in=240, old_keep=360),
    "000905": dict(n=500, liq_cut=0.20, new_in=400, old_keep=600),
    "000852": dict(n=1000, liq_cut=0.20, new_in=800, old_keep=1200),
}
INDEX_NAMES = {"000300": "沪深300", "000905": "中证500", "000852": "中证1000"}
MAX_CHANGE_FRAC = 0.10      # 单次调整比例上限
TOP_EXCLUDE = 300           # 中证500/1000 剔除的"日均总市值前 300 名"
NEW_LISTING_TOPK = 30       # 主板上市不满一季度但日均总市值前 30 的豁免
ANNOUNCE_LAG = 10           # 公告日代理 = 生效日前 10 个交易日(ST 状态取这一天)
BASE_LAG = 2                # 基准权重取生效日前 2 个交易日收盘

# 代码变更:新代码 -> 旧代码上市日(面板缺旧代码历史)
_CODE_CHANGE_LIST_DATE = {"302132.SZ": pd.Timestamp("2010-08-27")}

# 中证分级靠档表:自由流通比例上界 -> 加权比例(<=15% 用实际比例)
_BANDS = [(0.15, None), (0.20, 0.20), (0.30, 0.30), (0.40, 0.40), (0.50, 0.50),
          (0.60, 0.60), (0.70, 0.70), (0.80, 0.80), (1.01, 1.00)]

WEIGHT_CACHE = {"000852": TMP / "proxy_csi1000_weights.parquet",
                "000905": TMP / "proxy_csi500_weights.parquet",
                "000300": TMP / "proxy_csi300_weights.parquet"}
INPUT_CACHE = TMP / "proxy_inputs_{}.parquet"
MEMBER_CACHE = TMP / "proxy_membership.parquet"
META_CACHE = TMP / "proxy_meta.json"


# ======================================================================
# 输入:先截断再计算,且不污染同进程其他代码的缓存
# ======================================================================
@contextmanager
def _sealed_newdata():
    """在截断到 2025-12-31 的 daily_basic / 财报上运行 newdata 与 fundamental。

    为什么要这样做:ND.total_mv 内部读 daily_basic.csv 与三张财报 pkl,两者都含 2026 行。
    虽然 PIT 映射后 2026 行不会落到 <=2025 的日期上,但财报"同一报告期保留最晚公告版本"
    的去重会被 2026 年的更正公告改变(选择效应),封存纪律要求**先截断再算**。
    退出时恢复两个模块的进程内缓存与 FU._load,避免同进程其他代码拿到截断版而数值漂移。
    """
    from afactor import newdata as ND, fundamental as FU
    saved_db = ND._CACHE.pop("db", None)
    saved_rep = FU._REP_CACHE.pop("rep", None)
    orig_load = FU._load

    d = pd.read_csv(ND.DATA / "daily_basic.csv",
                    usecols=["ts_code", "trade_date", "close", "turnover_rate",
                             "turnover_rate_f", "float_share", "circ_mv"])
    d = d[d["trade_date"] <= int(SEAL_END.strftime("%Y%m%d"))].copy()     # 封存:先截
    d["trade_date"] = ND._ymd(d["trade_date"])
    ND._CACHE["db"] = d

    def _load_trunc(name, items):
        x = orig_load(name, items)
        return x[x["avail"] <= SEAL_END]                                 # 封存:先截

    FU._load = _load_trunc
    try:
        yield ND, FU
    finally:
        FU._load = orig_load
        ND._CACHE.pop("db", None)
        FU._REP_CACHE.pop("rep", None)
        if saved_db is not None:
            ND._CACHE["db"] = saved_db
        if saved_rep is not None:
            FU._REP_CACHE["rep"] = saved_rep


def _st_panel_trunc(dates: pd.DatetimeIndex, codes: pd.Index) -> pd.DataFrame:
    """每日 ST 名单(与 FU.build_st_panel 同逻辑,但先把 2026 行截掉再建面板)。"""
    from afactor.fundamental import DATA
    st = pd.read_csv(DATA / "st_info_latest.csv", usecols=["stock_code", "trade_date"])
    st["trade_date"] = pd.to_datetime(st["trade_date"])
    st = st[(st["trade_date"] <= SEAL_END) & st["stock_code"].isin(codes)]
    m = pd.crosstab(st["trade_date"], st["stock_code"]) > 0
    m = m.reindex(columns=codes, fill_value=False)
    first = m.index.min()
    m = m.reindex(m.index.union(dates)).astype(float).ffill().reindex(dates)
    m.loc[dates < first] = 0.0
    return m.fillna(0.0) > 0.5


def _base_panel(p):
    """代理一律用**收盘口径**的原料建;TailPanel 的 close_px/amount 被改写过,不能直接用。"""
    if getattr(p, "_is_tail", False) or getattr(p, "sealed", False):
        if getattr(p, "sealed", False):
            raise PermissionError("index_proxy 不服务封存面板")
        from afactor.panel import Panel
        return Panel()
    return p


def _board(code: str) -> str:
    if code.endswith(".BJ"):
        return "bse"
    if code.startswith(("688", "689")):
        return "star"
    if code.startswith(("300", "301", "302")):
        return "gem"
    return "main"


def load_inputs(p, refresh: bool = False) -> dict:
    """代理需要的全部日频输入(T x N,p 的日期与代码),带磁盘缓存。

    mv     真实总市值(ND.total_mv,未复权价 × 日频外推的 PIT 总股本,无分红泄漏);
           daily_basic 缺失的 21 只(多为 2026 年退市股)回退为 前复权价 × PIT 总股本
    circ   流通市值(daily_basic.circ_mv);缺失者回退为 mv(流通比例按 1 处理)
    raw    未复权收盘价(用于剥离现金分红,构造价格指数口径收益)
    st     每日 ST 状态(PIT 名单,先截断)
    """
    p = _base_panel(p)
    dates, codes = p.dates, p.codes
    assert dates.max() <= SEAL_END, "封存违规:面板含 2026 日期"
    keys = ["mv", "circ", "raw", "st", "mvsrc"]
    files = {k: INPUT_CACHE.with_name(INPUT_CACHE.name.format(k)) for k in keys}
    if not refresh and all(f.exists() for f in files.values()) and _meta_ok("inputs"):
        out = {k: pd.read_parquet(files[k]) for k in keys}
        if out["mv"].index.equals(dates) and out["mv"].columns.equals(codes):
            out["st"] = out["st"].astype(bool)
            return out
    with _sealed_newdata() as (ND, FU):
        mv = ND.total_mv(dates, codes).astype(np.float32)
        circ = ND.circ_mv(dates, codes).astype(np.float32)
        raw = ND.raw_close(dates, codes).astype(np.float32)
        rep = FU.build_report_level()
        sh = FU._pit_daily(rep, "total_share", dates, codes).replace(0, np.nan)
    # 回退:daily_basic 只覆盖"当前上市"股票,2026 年退市的 21 只没有未复权价
    fb = (p.raw("close_px") * sh).astype(np.float32)
    mvsrc = pd.DataFrame(np.where(mv.notna(), 1, np.where(fb.notna(), 2, 0)).astype(np.int8),
                         index=dates, columns=codes)
    mv = mv.fillna(fb)
    circ = circ.fillna(mv)
    st = _st_panel_trunc(dates, codes)
    out = dict(mv=mv, circ=circ, raw=raw, st=st, mvsrc=mvsrc)
    for k in keys:
        v = out[k].astype(np.int8) if k == "st" else out[k]
        v.to_parquet(files[k])
    _meta_write("inputs")
    return out


def _meta_ok(key: str) -> bool:
    try:
        return json.loads(META_CACHE.read_text()).get(key) == PROXY_VERSION
    except Exception:
        return False


def _meta_write(key: str) -> None:
    try:
        m = json.loads(META_CACHE.read_text())
    except Exception:
        m = {}
    m[key] = PROXY_VERSION
    META_CACHE.write_text(json.dumps(m, ensure_ascii=False, indent=1))


def listing_dates(codes: pd.Index, valid: pd.DataFrame) -> pd.Series:
    """上市日:stock_basic 优先;缺失者(新代码/北交所)用面板首个有效交易日。

    面板首个有效日 = 面板起点时视为"早已上市"(返回 1990-01-01)。
    """
    from afactor import newdata as ND
    li = ND.listing_info(codes)["list_date"]
    for c, d in _CODE_CHANGE_LIST_DATE.items():
        if c in li.index:
            li[c] = d
    first = valid.idxmax().where(valid.any())
    first = first.where(first > valid.index[0], pd.Timestamp("1990-01-01"))
    return li.fillna(first)


# ======================================================================
# 定期调整日历
# ======================================================================
def review_schedule(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """2020-06 起每半年一次的审核:截止日、公告日代理、权重基准日、生效日(都在交易日历上)。

    生效日 = 6 月 / 12 月第二个星期五的**下一交易日**;截止日 = 4/30、10/31 及之前最后一个交易日。
    2020-06 期的窗口只有 2020-01-02 ~ 2020-04-30(面板起点),只作初始状态,不进入校验。
    """
    rows = []
    di = dates
    for y in range(dates[0].year, dates[-1].year + 1):
        for m, cm, cd in ((6, 4, 30), (12, 10, 31)):
            first = pd.Timestamp(y, m, 1)
            fri1 = first + pd.Timedelta(days=(4 - first.weekday()) % 7)
            fri2 = fri1 + pd.Timedelta(days=7)
            e = int(np.searchsorted(di.values, np.datetime64(fri2), side="right"))
            c = int(np.searchsorted(di.values, np.datetime64(pd.Timestamp(y, cm, cd)), side="right")) - 1
            if e >= len(di) or c < 0 or di[c] < di[0] + pd.Timedelta(days=60):
                continue
            ws = int(np.searchsorted(di.values, np.datetime64(di[c] - pd.DateOffset(years=1)), side="right"))
            rows.append(dict(review=f"{y}{'H1' if m == 6 else 'H2'}", second_friday=fri2,
                             effective=di[e], eff_idx=e, cutoff=di[c], cut_idx=c,
                             win_start=di[ws], ws_idx=ws, n_win_days=c - ws + 1,
                             announce=di[e - ANNOUNCE_LAG], ann_idx=e - ANNOUNCE_LAG,
                             base=di[e - BASE_LAG], base_idx=e - BASE_LAG))
    return pd.DataFrame(rows)


# ======================================================================
# 选样
# ======================================================================
def buffer_select(ranked: list, old: set | None, n: int, new_in: int, old_keep: int,
                  max_change: int | None) -> tuple[list, dict]:
    """缓冲区 + 单次调整上限的选样。ranked 为按日均总市值降序的候选(已过流动性筛)。

    1. 名次 <= new_in 的新样本、名次 <= old_keep 的老样本优先入选;
    2. 优先集合超过 n 时按名次截断,不足 n 时按名次从剩余候选补足;
    3. 新进数量超过 max_change 时,只保留名次最好的 max_change 个新样本,
       空出的名额按名次回填仍在候选中的老样本(老样本不够再放行新样本)。
    old 为 None(首期)时直接取前 n。返回 (成分列表, 诊断计数)。
    """
    diag = dict(n_cand=len(ranked), old_kept_beyond_n=0, new_blocked_by_buffer=0,
                new_blocked_by_cap=0)
    if old is None:
        return list(ranked[:n]), diag
    rank = {c: i + 1 for i, c in enumerate(ranked)}
    prio = [c for c in ranked if (c in old and rank[c] <= old_keep) or
            (c not in old and rank[c] <= new_in)]
    sel = prio[:n]
    if len(sel) < n:
        s = set(sel)
        sel += [c for c in ranked if c not in s][:n - len(sel)]
    s = set(sel)
    diag["new_blocked_by_buffer"] = sum(1 for c in ranked[:n] if c not in old and c not in s)
    if max_change is not None:
        new = [c for c in sel if c not in old]          # sel 按名次有序 -> new 也有序
        if len(new) > max_change:
            kept = [c for c in sel if c in old] + new[:max_change]
            s = set(kept)
            need = n - len(kept)
            add = [c for c in ranked if c in old and c not in s][:need]
            if len(add) < need:                         # 老样本不够,继续按名次放行新样本
                add += new[max_change:max_change + need - len(add)]
            sel = sorted(kept + add, key=lambda c: rank[c])
            diag["new_blocked_by_cap"] = len(set(new) - set(sel))
    diag["old_kept_beyond_n"] = sum(1 for c in sel if c in old and rank[c] > n)
    return sel, diag


@dataclass
class ProxySpec:
    """代理变体(敏感性对照用)。主口径 = 默认值。"""
    buffer: bool = True          # 缓冲区规则
    liquidity: bool = True       # 成交额剔除
    cap: bool = True             # 单次调整 10% 上限
    weight: str = "circ"         # circ | total | banded | equal
    first_review: str | None = None   # 从哪一期开始(更晚 = 换一个初始状态,测初始化敏感性)

    @property
    def tag(self) -> str:
        t = [self.weight]
        if not self.buffer:
            t.append("nobuf")
        if not self.liquidity:
            t.append("noliq")
        if not self.cap:
            t.append("nocap")
        if self.first_review:
            t.append("from" + self.first_review)
        return "_".join(t)


def _window_mean(x: np.ndarray, ws: int, ce: int) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(x[ws:ce + 1], axis=0)


def select_members(inp: dict, valid: pd.DataFrame, amount: pd.DataFrame, sched: pd.DataFrame,
                   spec: ProxySpec = ProxySpec()) -> tuple[dict, pd.DataFrame]:
    """逐期联合选样(300 -> 500 -> 1000,后者依赖前者的结果)。

    返回 members[review][code] = 按名次排序的成分列表,以及逐期诊断表。
    """
    import warnings
    codes = valid.columns
    dates = valid.index
    V = valid.to_numpy(bool)
    MV = inp["mv"].to_numpy(np.float64)
    AMT = amount.to_numpy(np.float64)
    ST_ = inp["st"].to_numpy(bool)
    mvsrc = inp["mvsrc"].to_numpy(np.int8)
    board = np.array([_board(c) for c in codes])
    lst = listing_dates(codes, valid).reindex(codes).to_numpy("datetime64[ns]")
    # 已退市(退市日 <= 公告日代理)的不得入选。面板内股票都是存活股,这条只对幸存者偏差
    # 实验里加回的退市股起作用:否则 2021-09 已被吸收合并的 600068 仍会凭窗口内的历史市值入选。
    from afactor import newdata as ND
    dl = ND.listing_info(codes)["delist_date"].reindex(codes).to_numpy("datetime64[ns]")

    members, prev, diags = {}, {k: None for k in CSI_RULES}, []
    for _, r in sched.iterrows():
        ws, ce = int(r.ws_idx), int(r.cut_idx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mv_avg = _window_mean(np.where(V, MV, np.nan), ws, ce)
            amt_avg = _window_mean(np.where(V, AMT, np.nan), ws, ce)
        cut = np.datetime64(r.cutoff)
        age_ok = np.where(np.isin(board, ["star", "gem"]),
                          lst <= np.datetime64(r.cutoff - pd.DateOffset(years=1)),
                          lst <= np.datetime64(r.cutoff - pd.DateOffset(months=3)))
        alive = ~(dl <= np.datetime64(r.announce))          # NaT 比较为 False -> 视为存续
        base_ok = ((board != "bse") & ~ST_[int(r.ann_idx)] & np.isfinite(mv_avg)
                   & np.isfinite(amt_avg) & (amt_avg > 0) & (lst <= cut) & alive)
        # 主板上市不满一季度:上市以来日均总市值排前 30 可豁免
        young_main = base_ok & ~age_ok & (board == "main")
        space = base_ok & age_ok
        if young_main.any():
            pool = np.flatnonzero(space | young_main)
            order = pool[np.argsort(-mv_avg[pool], kind="stable")]
            top = set(order[:NEW_LISTING_TOPK])
            space = space | np.array([young_main[i] and i in top for i in range(len(codes))])
        sp = np.flatnonzero(space)
        sp_by_mv = sp[np.argsort(-mv_avg[sp], kind="stable")]
        top300 = set(codes[sp_by_mv[:TOP_EXCLUDE]])

        rev_members, taken = {}, set()
        for code, rule in CSI_RULES.items():
            cand = sp if code == "000300" else np.array(
                [i for i in sp if codes[i] not in taken and codes[i] not in top300], dtype=int)
            n_before_liq = len(cand)
            if spec.liquidity and len(cand):
                k_keep = int(np.floor(len(cand) * (1.0 - rule["liq_cut"])))
                by_amt = cand[np.argsort(-amt_avg[cand], kind="stable")]
                cand = by_amt[:k_keep]
            ranked = list(codes[cand[np.argsort(-mv_avg[cand], kind="stable")]])
            old = prev[code] if spec.buffer else None
            mc = int(round(rule["n"] * MAX_CHANGE_FRAC)) if (spec.cap and spec.buffer) else None
            sel, dg = buffer_select(ranked, old, rule["n"], rule["new_in"], rule["old_keep"], mc)
            rev_members[code] = sel
            taken |= set(sel)
            o = set(prev[code]) if prev[code] is not None else set()
            idx = [codes.get_loc(c) for c in sel]
            diags.append(dict(
                review=r.review, effective=r.effective.date(), cutoff=r.cutoff.date(),
                n_win_days=int(r.n_win_days), index=code, name=INDEX_NAMES[code],
                variant=spec.tag, n_space=int(space.sum()), n_st_excluded=int(
                    ((board != "bse") & ST_[int(r.ann_idx)] & np.isfinite(mv_avg)).sum()),
                n_before_liq=n_before_liq, n_after_liq=len(ranked), n_members=len(sel),
                n_new=len(set(sel) - o) if o else np.nan, n_out=len(o - set(sel)) if o else np.nan,
                overlap_prev=len(set(sel) & o) / len(o) if o else np.nan,
                mv_avg_min_yi=float(np.nanmin(mv_avg[idx]) / 1e8),
                mv_avg_med_yi=float(np.nanmedian(mv_avg[idx]) / 1e8),
                mv_avg_max_yi=float(np.nanmax(mv_avg[idx]) / 1e8),
                n_gem=int(sum(board[i] == "gem" for i in idx)),
                n_star=int(sum(board[i] == "star" for i in idx)),
                n_mv_fallback=int(sum(mvsrc[int(r.base_idx), i] != 1 for i in idx)),
                **dg))
        members[r.review] = rev_members
        prev = {k: v for k, v in rev_members.items()}
    return members, pd.DataFrame(diags)


# ======================================================================
# 权重与漂移
# ======================================================================
def _band(f: np.ndarray) -> np.ndarray:
    out = np.full_like(f, np.nan)
    lo = -np.inf
    for hi, b in _BANDS:
        m = (f > lo) & (f <= hi)
        out[m] = f[m] if b is None else b
        lo = hi
    return out


def base_weights(inp: dict, codes: pd.Index, members: list, b: int, scheme: str) -> np.ndarray:
    """基准日 b 收盘的目标权重(N 维,成分外为 0)。市值取 b 日及之前最后一个有效值。"""
    idx = np.array([codes.get_loc(c) for c in members], dtype=int)
    w = np.zeros(len(codes))
    if scheme == "equal":
        w[idx] = 1.0
    else:
        circ = inp["circ"].iloc[:b + 1, idx].ffill().iloc[-1].to_numpy(np.float64)
        mv = inp["mv"].iloc[:b + 1, idx].ffill().iloc[-1].to_numpy(np.float64)
        if scheme == "circ":
            x = circ
        elif scheme == "total":
            x = mv
        elif scheme == "banded":
            f = np.clip(circ / mv, 0, 1)
            x = mv * _band(f)
        else:
            raise ValueError(scheme)
        x = np.where(np.isfinite(x) & (x > 0), x, 0.0)
        w[idx] = x
    s = w.sum()
    return w / s if s > 0 else w


def daily_returns(p, inp: dict, kind: str) -> np.ndarray:
    """收盘到收盘的逐日收益 R[t] = P[t]/P[t-1]-1(停牌按前值,复牌跳空计入)。

    kind="tr" 全收益:前复权价(含现金分红再投资)
    kind="pr" 价格指数口径:现金分红除息日改用未复权收益(不补分红),送转日仍用前复权收益。
              判别:(1+r_tr)/(1+r_raw)-1 ∈ (1e-4, 0.10) 视为纯现金分红;>=0.10 视为送转
              (送转同日的分红部分会漏剥,量级可忽略);缺未复权价的 21 只直接用全收益。
    """
    p = _base_panel(p)
    v = p["valid"]
    cp = p.raw("close_px").where(v).ffill()
    r_tr = (cp / cp.shift(1) - 1.0).to_numpy(np.float64)
    r_tr = np.where(np.isfinite(r_tr), r_tr, 0.0)
    if kind == "tr":
        return r_tr
    raw = inp["raw"].where(v).ffill()
    r_raw = (raw / raw.shift(1) - 1.0).to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (1.0 + r_tr) / (1.0 + r_raw) - 1.0
    div_day = np.isfinite(ratio) & (ratio > 1e-4) & (ratio < 0.10)
    return np.where(div_day, r_raw, r_tr)


def drift_weights(w0_list: list, b_list: list, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """在各期基准日 b_k 放入 w0_k,之后按 R 漂移,直到下一期基准日。

    返回 D(T x N,D[t] = t 日收盘、按该期成分漂移后的权重;b_0 之前为 0)
    与 D_prev(T x N,D_prev[t] = 用于 t 日 14:50 决策的权重 = 本期 D[t-1],
    期首 t=b_k+1 时即 w0_k —— 不会跨期取到旧成分)。
    """
    T, N = R.shape
    D = np.zeros((T, N))
    Dp = np.zeros((T, N))
    bs = list(b_list) + [T - 1]
    for k, w0 in enumerate(w0_list):
        b, b_next = bs[k], bs[k + 1]
        g = np.cumprod(1.0 + R[b + 1:b_next + 1], axis=0)   # b+1..b_next 的累计增长
        seg = np.vstack([w0[None, :], w0[None, :] * g])      # b..b_next
        seg = seg / seg.sum(axis=1, keepdims=True)
        # 本期覆盖 t ∈ [b+1, b_next](期末 b_next 的收盘权重仍属本期,下一期在 b_next+1 接管)
        D[b + 1:b_next + 1] = seg[1:]
        Dp[b + 1:b_next + 1] = seg[:-1]
        if k == 0:
            D[b] = 0.0
    return D, Dp


def index_from_weights(D: np.ndarray, R: np.ndarray) -> np.ndarray:
    """收盘到收盘指数收益:I[t] = Σ D[t-1] · R[t](t 日收益用 t-1 收盘权重)。"""
    out = np.full(R.shape[0], np.nan)
    has = D.sum(axis=1) > 0
    out[1:] = np.where(has[:-1], (D[:-1] * R[1:]).sum(axis=1), np.nan)
    return out


# ======================================================================
# 主构建
# ======================================================================
def build_proxy(p, spec: ProxySpec = ProxySpec(), kinds=("pr", "tr"), inp: dict | None = None,
                valid: pd.DataFrame | None = None, amount: pd.DataFrame | None = None,
                R: dict | None = None, keep_weights: bool = True) -> dict:
    """构建三只代理指数。返回 dict:
        members, diag, sched, base_w[code](各期基准权重列表)
        W[code]        T x N 收盘口径权重(按全收益漂移;成分外 0)   —— keep_weights=True 时
        W_tail[code]   T x N 尾盘口径权重(t 日 14:50 已知)          —— keep_weights=True 时
        ret[(code, kind)]  收盘到收盘代理日收益(Series,打在实现日 t;首期生效前 NaN)
    inp/valid/amount/R 可外部传入(幸存者偏差实验用扩展股票池)。
    """
    p = _base_panel(p)
    inp = inp if inp is not None else load_inputs(p)
    valid = valid if valid is not None else p["valid"]
    amount = amount if amount is not None else p.raw("amount")
    dates, codes = valid.index, valid.columns
    sched = review_schedule(dates)
    if spec.first_review:
        sched = sched[sched["review"] >= spec.first_review].reset_index(drop=True)
    members, diag = select_members(inp, valid, amount, sched, spec)
    need = set(kinds) | ({"tr"} if keep_weights else set())
    if R is None:
        R = {k: daily_returns(p, inp, k) for k in need}
    out = dict(members=members, diag=diag, sched=sched, W={}, W_tail={}, ret={}, base_w={})
    b_list = sched["base_idx"].tolist()
    for code in CSI_RULES:
        w0s = [base_weights(inp, codes, members[rv][code], b, spec.weight)
               for rv, b in zip(sched["review"], b_list)]
        out["base_w"][code] = w0s
        for kind in sorted(need):
            D, Dp = drift_weights(w0s, b_list, R[kind])
            if kind in kinds:
                out["ret"][(code, kind)] = pd.Series(index_from_weights(D, R[kind]), index=dates)
            if kind == "tr" and keep_weights:
                out["W"][code], out["W_tail"][code] = D, Dp
            del D, Dp
    return out


# ======================================================================
# 对下游的接口
# ======================================================================
def proxy_weights(p, code: str = "000852", timing: str | None = None,
                  refresh: bool = False) -> pd.DataFrame:
    """T x N 的代理基准权重(float32,成分外为 0,每行和为 1;首期生效前全 0)。

    timing="close"(默认,Panel):W[t] = 信号日 t **收盘**时点的权重,供 open10 等
                    "收盘出信号、次日成交"口径使用;
    timing="tail"(TailPanel 默认):W[t] = 信号日 t 14:50 已知的权重(= 上一收盘权重,
                    换期日直接切到新成分),供 tail1450 等尾盘口径使用。
    缓存:cache/_tmp/proxy_csi{1000,500,300}_weights[_tail].parquet。
    第一个生效日为 2020-06-15;**2020 年只作预热**(2020-06 期仅有 4 个月排名窗口)。
    """
    if timing is None:
        timing = "tail" if getattr(p, "_is_tail", False) else "close"
    base = WEIGHT_CACHE[code]
    f = base if timing == "close" else base.with_name(base.stem + "_tail.parquet")
    dates, codes = p.dates, p.codes
    if not refresh and f.exists() and _meta_ok(f"weights_{code}"):
        W = pd.read_parquet(f)
        if W.index.equals(dates) and W.columns.equals(codes):
            return W
    res = build_proxy(p, ProxySpec(), kinds=("tr",))
    for c in CSI_RULES:
        for tm, arr in (("close", res["W"][c]), ("tail", res["W_tail"][c])):
            fb = WEIGHT_CACHE[c]
            ff = fb if tm == "close" else fb.with_name(fb.stem + "_tail.parquet")
            pd.DataFrame(arr.astype(np.float32), index=dates, columns=codes).to_parquet(ff)
        _meta_write(f"weights_{c}")
    save_membership(res, p)
    return pd.read_parquet(f)


def save_membership(res: dict, p) -> pd.DataFrame:
    """成分名单长表(review, effective, index, code, rank_in_index, base_weight)落盘。"""
    rows = []
    codes = _base_panel(p).codes
    for k, r in enumerate(res["sched"].itertuples()):
        for code in CSI_RULES:
            w0 = res["base_w"][code][k]
            for j, c in enumerate(res["members"][r.review][code]):
                rows.append((r.review, r.effective, code, c, j + 1, float(w0[codes.get_loc(c)])))
    df = pd.DataFrame(rows, columns=["review", "effective", "index", "code", "rank", "base_weight"])
    df.to_parquet(MEMBER_CACHE)
    return df


def index_close(code: str) -> pd.Series:
    """真实宽基指数收盘(价格指数),**先截到 2025-12-31**。"""
    from afactor import newdata as ND
    s = ND.index_daily()[code].dropna()
    return s[s.index <= SEAL_END]


def index_returns_on_signal_days(p, mode: str, code: str = "000852") -> pd.Series:
    """把真实指数收益对齐到引擎的信号日索引(与 strategy.benchmark 的 index 完全相同)。

    复用 scripts/run_ir_analysis.py::index_rets_on 的推导:open10 与 tail1450 的持有区间
    都覆盖交易日 t+1,故 r[t] = close(t+1)/close(t)-1 打在信号日 t,**不按 lag 再 shift**。
    索引取 p.dates[:T-1-lag](与 benchmark() 一致)。指数只有收盘价,主口径(开-开)与之
    有执行时点差,属量级参照;严格同口径请用 exec_time_bench_returns(2026-10-10 起的推荐基准)。
    """
    from afactor import strategy as ST
    lag = ST.EXEC_MODES[mode]["lag"]
    px = index_close(code).reindex(p.dates).ffill()
    r = px.shift(-1) / px - 1.0
    idx = p.dates[:len(p.dates) - 1 - lag]
    return r.reindex(idx).fillna(0.0).rename(code)


def weighted_exec_returns(p, mode: str, W: pd.DataFrame, nan_policy: str = "zero") -> pd.Series:
    """给定信号日权重 W[t],按引擎同一成交价口径算组合收益,打在信号日 t。

    与 strategy.benchmark 同一套索引:g[te] = px[te+1]/px[te]-1,te = t+lag,
    r[t] = Σ_i W[t]_i · g_i[t+lag],索引 p.dates[:T-1-lag]。
    nan_policy="zero"   成交价缺失(停牌/未上市)的成分收益记 0、权重保留 —— 与
                        run_strategy 对持仓的处理一致(复牌跳空同样丢失,是引擎口径);
    nan_policy="renorm" 在有收益的成分上重新归一 —— 与 benchmark() 的 mean 一致;
                        W=池等权时应与 ST.benchmark 逐位一致(金标准见 run_index_proxy.py)。
    W 全 0 的日子返回 NaN。
    """
    from afactor import strategy as ST
    lag = ST.EXEC_MODES[mode]["lag"]
    px, _ = ST.exec_prices(p, mode)
    g = (px.shift(-1) / px - 1.0).shift(-lag).reindex(index=p.dates, columns=p.codes)
    Wv = W.reindex(index=p.dates, columns=p.codes).fillna(0.0).to_numpy(np.float64)
    G = g.to_numpy(np.float64)
    fin = np.isfinite(G)
    num = np.where(fin, Wv * np.where(fin, G, 0.0), 0.0).sum(axis=1)
    tot = Wv.sum(axis=1)
    if nan_policy == "zero":
        r = np.where(tot > 0, num, np.nan)
    elif nan_policy == "renorm":
        den = np.where(fin, Wv, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(den > 0, num / den, np.where(tot > 0, 0.0, np.nan))
    else:
        raise ValueError(nan_policy)
    s = pd.Series(r, index=p.dates)
    return s.iloc[:len(p.dates) - 1 - lag]


def engine_consistent_bench_returns(p, mode: str, code: str = "000852",
                                    nan_policy: str = "zero") -> pd.Series:
    """⚠ 已弃用(2026-10-10,前缀 fixa):请改用 exec_time_bench_returns(p, mode)。保留只为复现旧账本。

    旧口径:代理**收盘**权重 W[t] × 引擎成交价收益 px[t+1]->px[t+2](open10 用 twap_open10,
    tail1450 用 twap_tail8),全收益,不含成本。权重时点:尾盘口径自动取 timing="tail"(14:50 已知)。
    返回 Series 打在信号日 t,索引与 strategy.benchmark(p, ..., mode)["rets"] 相同;首期生效前为 NaN。

    为什么弃用(指增审查 P0,research/INDEX_ENHANCEMENT_CSI1000.md 审查记录 #1):W[t] 是信号日 t 收盘的权重,
    引擎却从 px[t+1](次日开盘 TWAP / 全日 VWAP)才开始计收益;每天把权重重置回过时一个隔夜的收盘权重,
    收盘 -> 成交价之间涨了的股票被系统性低配,而这段涨跌与成交价 -> 下一成交价正相关,于是**系统性偏低**:
    2021-2025 open10 约 -0.8pp/年、vwap1 约 -2.5pp/年(research/ledger/fixa_bench_compare.csv)。
    exec_time_bench_returns 先把权重漂移到成交时点,= "在引擎成交价上计价的同一只指数"。
    """
    import warnings
    warnings.warn("engine_consistent_bench_returns 已弃用(收盘权重 × 成交价收益,系统性偏低 open10 约 0.8pp/年、"
                  "vwap1 约 2.5pp/年);请用 index_proxy.exec_time_bench_returns", DeprecationWarning, stacklevel=2)
    from afactor import strategy as ST
    timing = "tail" if ST.EXEC_MODES[mode]["tail"] else "close"
    W = proxy_weights(p, code, timing=timing)
    return weighted_exec_returns(p, mode, W, nan_policy).rename(f"proxy_{code}_{mode}")


# ======================================================================
# 推荐基准:成交时点权重的代理(2026-10-10 由 optimizer 搬入,前缀 fixa;optimizer 里的旧名仍可用)
# ======================================================================
def exec_time_weights(W: np.ndarray, px: np.ndarray, anchor: np.ndarray, lag: int = 1) -> np.ndarray:
    """把"W[t] 所在时点"的市值权重漂移到成交时点 te = t+lag:Wx[t] ∝ W[t]·px[te]/anchor[t]。

    为什么(指增审查 P0/P2):W[t] = 信号日 t **收盘**的代理权重,而引擎从 px[t+1](open10 = 次日开盘 TWAP,
    vwap1 = 次日全日 VWAP)才开始计收益。旧基准 engine_consistent_bench_returns 直接用 W[t] 乘
    px[t+1]→px[t+2] 的收益,等于**每天把权重重置回过时一个隔夜的收盘权重**:收盘到成交价之间涨了的股票被系统性低配。
    收盘→成交价与成交价→下一成交价的涨跌正相关(vwap1 的全日平均价口径尤甚,Working 1960),
    于是旧基准系统性偏低:open10 约 0.75pp/年、vwap1 约 2.4pp/年(research/ledger/ie_bench_compare.csv)。
    按股本不变的市值加权,成交时点的真实权重正是 W[t]·px[te]/anchor[t] 归一;用它乘 px[te]→px[te+1] 的收益,
    两次成分调整之间连乘 = Σ 股本·px[结束] / Σ 股本·px[开始],即"在引擎成交价上计价的同一只指数"(单测验证)。

    W       T×N  W[t] 的时点由 anchor 给出(收盘口径 W = proxy_weights(timing="close"),anchor = 收盘价)
    px      T×N  引擎成交价(与 anchor 同一复权口径:本项目都是前复权)
    anchor  T×N  W[t] 对应时点的价格,停牌按前值
    比值缺失 / 非正(te 停牌、无成交)记 1 = 保持原权重。W 全 0 的行保持全 0。只在评价里用(px[te] 决策时点不可知)。
    """
    T, N = W.shape
    ratio = np.full((T, N), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio[:T - lag] = px[lag:] / anchor[:T - lag]
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0)
    Wx = W * ratio
    s = Wx.sum(axis=1, keepdims=True)
    return np.where(s > 0, Wx / np.where(s > 0, s, 1.0), 0.0)


def exec_time_bench_arrays(W: np.ndarray, px: np.ndarray, anchor: np.ndarray, lag: int = 1) -> np.ndarray:
    """r[t] = Σ_i Wx[t]_i·g_i[te],te = t+lag,g[te] = px[te+1]/px[te] − 1(缺失记 0,与引擎对持仓的处理、
    与旧基准的 nan_policy="zero" 一致:停牌复牌跳空同样丢失)。W 全 0 的行 NaN;长度 T,最后 lag+1 行 NaN。"""
    T, N = W.shape
    g = np.full((T, N), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        g[:T - 1] = px[1:] / px[:-1] - 1.0
    G = np.full((T, N), np.nan)
    G[:T - lag] = g[lag:]                                   # G[t] = g[t+lag]
    Gz = np.where(np.isfinite(G), G, 0.0)
    Wx = exec_time_weights(W, px, anchor, lag)
    r = np.where(W.sum(axis=1) > 0, (Wx * Gz).sum(axis=1), np.nan)
    r[T - 1 - lag:] = np.nan
    return r


def exec_time_bench_returns(p, mode: str, W: pd.DataFrame | None = None, return_weights: bool = False,
                            code: str = "000852"):
    """**推荐的中证指数代理基准**:与策略严格同口径(成交时点权重 × 引擎成交价收益,全收益,不含成本)。

    W=None 时取 proxy_weights(p, code, timing="close")(PIT 代理的信号日收盘权重);也可传入任意收盘时点权重
    (指增优化器传它自己读的代理权重,结果逐位相同)。返回 Series 打在信号日 t,索引 p.dates[:T−1−lag]
    (与 strategy.benchmark / 旧基准相同);首期生效前为 NaN。return_weights=True 时另返回 Wx(T×N ndarray,
    事后归因的主动权重 A = 持仓 − Wx 用它,与基准收益逐日对账)。
    只支持收盘出信号的口径(open*/vwap1);尾盘口径的 W 是 14:50 权重,锚价应为 t−1 收盘,本研究未用到。
    与旧基准的差(2021-2025,research/ledger/fixa_bench_compare.csv):open10 高约 0.8pp/年、vwap1 高约 2.5pp/年。"""
    from afactor import strategy as ST
    em = ST.EXEC_MODES[mode]
    if em["tail"]:
        raise NotImplementedError("尾盘口径的锚价应为 t−1 收盘,本研究未用到")
    lag = int(em["lag"])
    if W is None:
        W = proxy_weights(p, code, timing="close")
    px = ST.exec_prices(p, mode)[0].reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
    anchor = (p.raw("close_px").where(p["valid"]).ffill()        # 与 daily_returns 同一收盘价(停牌按前值)
              .reindex(index=p.dates, columns=p.codes).to_numpy(np.float64))
    Wv = W.reindex(index=p.dates, columns=p.codes).fillna(0.0).to_numpy(np.float64)
    r = exec_time_bench_arrays(Wv, px, anchor, lag)
    s = pd.Series(r, index=p.dates).iloc[:len(p.dates) - 1 - lag].rename(f"proxy_exec_{mode}")
    if return_weights:
        return s, exec_time_weights(Wv, px, anchor, lag)
    return s
