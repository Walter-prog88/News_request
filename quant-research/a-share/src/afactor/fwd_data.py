"""前向检验原始数据:路径、代码格式转换、分片读取、1 分钟 -> baostock 口径 5 分钟聚合(前缀 fwd_data)。

**只有标注为"前向检验"(前缀 fwd)的代码可以 import 本模块**。研究流程(<= 2025-12-31)不得读取
cache/forward/ 下的任何文件;本模块不做封存守卫,因为前向检验本来就要读 2026 数据。
2026-01-01 ~ 2026-08-30 的数据只能作为特征回看/预热输入,不得据此计算或报告任何策略收益、IC、超额。

本模块**只做数据层**:读分片、改代码格式、把 1 分钟宽表聚合成 baostock 口径的 5 分钟宽表。
不计算任何收益率、因子、IC。

------------------------------------------------------------------------------
目录(cache/forward/raw/,每类一个子目录,按股票分片,断点续跑)
------------------------------------------------------------------------------
  bs_m5/parts/<bs代码>.csv.gz   baostock 5 分钟线(不复权 adjustflag=3),2026-08-03 ~ 2026-10-09
                                列 date,time,code,open,high,low,close,volume,amount(原样字符串)
  bs_daily/parts/<bs代码>.csv   baostock 日线(不复权),2026-07-01 ~ 2026-10-09
  bs_adjf/parts/<bs代码>.csv    baostock 复权因子事件,2026-01-01 ~ 2026-10-09(只有除权日才有行)
  bs_profit/parts/<bs代码>.csv  baostock 季频盈利(含 totalShare / liqaShare / pubDate / statDate)
  bs_meta/                      股票清单、交易日历、query_stock_basic、下载清单
  sina_m5_bj/parts/<代码>.csv   新浪 5 分钟线(北交所补充,baostock 不支持 BJ;另含 30 只沪深抽样做跨源质检)
  sina_daily_bj/ sina_qfq_bj/   北交所新浪日线(2026-07-01~10-09)与前复权因子事件(全历史)
  ak_*/                         akshare(东方财富 / 新浪 / 中证指数)整表快照,文件名带取数日期
每个 bs_* 目录下 _status.csv 逐只记录(代码, 行数, 状态, 尝试次数, 时间),_failures.json 为最终失败清单。
空结果(停牌整段 / 无除权 / 未披露)写只有表头的分片,状态 "empty",与失败区分。

------------------------------------------------------------------------------
baostock 5 分钟线口径(2026-10-10 验证,见 research/FORWARD_DATA_QA.md §3)
------------------------------------------------------------------------------
  每天 48 根,time 为 bar 结束时刻:09:35, 09:40, ..., 11:30(24 根), 13:05, ..., 15:00(24 根)。
  与项目 1 分钟数据(241 根,idx0=09:30 开盘集合竞价)的对应:
    09:35 = idx0..5(含开盘集合竞价,6 根)
    09:40 = idx6..10, ..., 11:30 = idx116..120
    13:05 = idx121..125, ..., 15:00 = idx236..240(含 15:00 收盘集合竞价)
  bar 的 close = 组内最后一分钟 close;amount = 组内求和;open = 组内第一分钟 open。
  1 分钟数据没有 high/low/volume,不能对照这三项。

  重叠段 08-03~08-17 实测(数字见 research/ledger/fwd_data_qa.csv 第 4 节):
    - **沪市**逐根几乎完全一致(收盘价 99.9%、成交额 99.5%,不一致的 97% 是相邻两根之间的划分差);
    - **深市**逐根只是近似(收盘价 72.5% 一致、91.3% 在 1 个 tick 内;成交额逐根中位相对误差 0.35%、p90 约 4%),
      但每日 48 根合计与 1 分钟日合计逐位一致 —— 差异来自两家对深市成交在 bar 边界附近的归属不同;
    - baostock 的 open 不是组内第一笔成交价(常等于上一根 close),high/low 也不是真实极值,下游不要用;
    - 无成交时 baostock 给 0 价(停牌日 48 根全 0、开盘后尚无成交的 bar 为 0),须当 NaN。
"""
from __future__ import annotations
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from afactor.config import CACHE, N_BARS

FWD = CACHE / "forward"
RAW = FWD / "raw"
DIRS = {
    "m5": RAW / "bs_m5",
    "daily": RAW / "bs_daily",
    "adjf": RAW / "bs_adjf",
    "profit": RAW / "bs_profit",
    "meta": RAW / "bs_meta",
    "sina_bj": RAW / "sina_m5_bj",
    "sina_daily_bj": RAW / "sina_daily_bj",
    "sina_qfq_bj": RAW / "sina_qfq_bj",
}
FWD_START = "2026-09-01"        # 预注册前向评估窗口起点(由下游前向检验脚本使用)
M5_START, M5_END = "2026-08-03", "2026-10-09"
OVERLAP_START, OVERLAP_END = "2026-08-03", "2026-08-17"   # 与 1 分钟数据重叠段(拼接校验用)
DAILY_START, DAILY_END = "2026-07-01", "2026-10-09"
ADJF_START, ADJF_END = "2026-01-01", "2026-10-09"

# baostock 5 分钟 bar 结束时刻(48 根)
BS5_TIMES = ([f"{h:02d}:{m:02d}" for h, m in
              [(9 + (35 + 5 * k) // 60, (35 + 5 * k) % 60) for k in range(24)]]
             + [f"{h:02d}:{m:02d}" for h, m in
                [(13 + (5 + 5 * k) // 60, (5 + 5 * k) % 60) for k in range(24)]])
assert BS5_TIMES[0] == "09:35" and BS5_TIMES[23] == "11:30"
assert BS5_TIMES[24] == "13:05" and BS5_TIMES[47] == "15:00" and len(BS5_TIMES) == 48


def bs5_groups() -> np.ndarray:
    """241 根 1 分钟 bar -> 48 根 baostock 5 分钟 bar 的组号(长度 241,取值 0..47)。

    第一组 6 根(idx0..5,含开盘集合竞价),其余每组 5 根;11:30 = idx116..120,
    13:05 = idx121..125,15:00 = idx236..240(含收盘集合竞价)。
    """
    g = np.empty(N_BARS, dtype=np.int64)
    g[0:6] = 0
    for k in range(1, 24):
        g[5 * k + 1: 5 * k + 6] = k
    for j in range(24):
        g[121 + 5 * j: 126 + 5 * j] = 24 + j
    return g


def to_bs(code: str) -> str:
    """000001.SZ -> sz.000001;北交所返回 None(baostock 不支持 bj.)。"""
    num, ex = code.split(".")
    ex = ex.upper()
    if ex == "BJ":
        return None
    return f"{ex.lower()}.{num}"


def from_bs(code: str) -> str:
    """sz.000001 -> 000001.SZ"""
    ex, num = code.split(".")
    return f"{num}.{ex.upper()}"


def is_a_share_bs(code: str) -> bool:
    """baostock 代码是否沪深 A 股(排除指数 sh.000/sz.399、基金、债券、B 股)。"""
    ex, num = code.split(".")
    if ex == "sh":
        return num[:3] in ("600", "601", "603", "605", "688", "689")
    if ex == "sz":
        return num[:3] in ("000", "001", "002", "003", "300", "301", "302")
    return False


def part_path(kind: str, bs_code: str) -> Path:
    ext = ".csv.gz" if kind == "m5" else ".csv"
    return DIRS[kind] / "parts" / f"{bs_code}{ext}"


def read_status(kind: str) -> pd.DataFrame:
    """逐只下载状态(同一代码多次运行取最后一条)。"""
    f = DIRS[kind] / "_status.csv"
    if not f.exists():
        return pd.DataFrame(columns=["code", "n", "status", "attempts", "ts"])
    s = pd.read_csv(f, dtype={"code": str, "status": str})
    return s.drop_duplicates("code", keep="last").reset_index(drop=True)


def _read_parts(kind: str, dtype=None) -> pd.DataFrame:
    fs = sorted(f for f in (DIRS[kind] / "parts").glob("*.csv*")
                if not f.name.endswith(".tmp") and ".redo" not in f.name)
    out = []
    for f in fs:
        if f.stat().st_size == 0:
            continue
        d = pd.read_csv(f, dtype=dtype or str)
        if len(d):
            out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def load_m5(codes=None, cache: bool = True) -> pd.DataFrame:
    """baostock 5 分钟线长表(不复权)。列:code(项目格式) dt date time open high low close volume amount。

    dt = bar 结束时刻(datetime64)。cache=True 时合并结果落盘 cache/forward/m5_long.parquet,
    分片有更新(数量或最新修改时间变化)时自动重建。
    """
    f = FWD / "m5_long.parquet"
    meta_f = FWD / "m5_long.meta.json"
    parts = sorted((DIRS["m5"] / "parts").glob("*.csv.gz"))
    sig = {"n": len(parts), "mtime": max((p.stat().st_mtime for p in parts), default=0)}
    if cache and f.exists() and meta_f.exists() and json.loads(meta_f.read_text()) == sig:
        d = pd.read_parquet(f)
    else:
        rows = []
        for p in parts:
            x = pd.read_csv(p, dtype={"date": str, "time": str, "code": str})
            if not len(x):
                continue
            # 逐分片转换类型再合并,避免千万行字符串列占内存
            y = pd.DataFrame({
                "code": from_bs(x["code"].iloc[0]),
                "dt": pd.to_datetime(x["time"].str[:12], format="%Y%m%d%H%M"),
                "time": x["time"].str[8:10] + ":" + x["time"].str[10:12],
            })
            for c in ("open", "high", "low", "close", "amount"):
                y[c] = pd.to_numeric(x[c], errors="coerce").astype(np.float64)
            y["volume"] = pd.to_numeric(x["volume"], errors="coerce").astype(np.float64)
            if (x["code"] != x["code"].iloc[0]).any():
                raise ValueError(f"分片 {p.name} 含多个代码")
            rows.append(y)
        d = pd.concat(rows, ignore_index=True)
        d["code"] = d["code"].astype("category")
        d["date"] = d["dt"].dt.normalize()
        d["time"] = d["time"].astype("category")
        d = d[["code", "dt", "date", "time", "open", "high", "low", "close", "volume", "amount"]]
        d = d.sort_values(["code", "dt"]).reset_index(drop=True)
        if cache:
            d.to_parquet(f, index=False)
            meta_f.write_text(json.dumps(sig))
    if codes is not None:
        d = d[d["code"].isin(set(codes))].reset_index(drop=True)
    return d


def load_sina_m5(role: str | None = "bj") -> pd.DataFrame:
    """新浪 5 分钟线(北交所补充 + 沪深 30 只抽样),列同 load_m5。role='bj' / 'shsz_sample' / None(全部)。

    新浪接口只返回最近 1970 根 bar(2026-10-10 取数:08-05 14:05 ~ 10-09),最早一天不完整;
    口径与 baostock / 1 分钟数据的差异见 research/FORWARD_DATA_QA.md §3.7。
    """
    d = _read_parts("sina_bj")
    if d.empty:
        return d
    roles = pd.read_csv(DIRS["sina_bj"] / "codes.csv", dtype=str)
    if role is not None:
        d = d[d["code"].isin(set(roles.loc[roles.role == role, "code"]))]
    out = pd.DataFrame({"code": d["code"].values, "dt": pd.to_datetime(d["day"]).values})
    for c in ("open", "high", "low", "close", "amount", "volume"):
        out[c] = pd.to_numeric(d[c], errors="coerce").values
    out["date"] = out["dt"].dt.normalize()
    out["time"] = out["dt"].dt.strftime("%H:%M")
    out = out[["code", "dt", "date", "time", "open", "high", "low", "close", "volume", "amount"]]
    return out.sort_values(["code", "dt"]).reset_index(drop=True)


def load_sina_daily_bj() -> pd.DataFrame:
    """北交所新浪日线(不复权),2026-07-01~10-09:date open high low close volume amount
    outstanding_share(流通股本,股)turnover(换手率,小数)。新浪不给停牌行与 preclose。"""
    d = _read_parts("sina_daily_bj")
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d["date"])
    for c in ("open", "high", "low", "close", "volume", "amount", "outstanding_share", "turnover"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.drop(columns=["fetched_at"], errors="ignore").sort_values(["code", "date"]).reset_index(drop=True)


def load_sina_qfq_bj() -> pd.DataFrame:
    """北交所新浪前复权因子事件(全历史):code date qfq_factor(以最新为 1)。"""
    d = _read_parts("sina_qfq_bj")
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d["date"])
    d["qfq_factor"] = pd.to_numeric(d["qfq_factor"], errors="coerce")
    return d.drop(columns=["fetched_at"], errors="ignore").sort_values(["code", "date"]).reset_index(drop=True)


def load_daily() -> pd.DataFrame:
    """baostock 日线长表(不复权),code 为项目格式;数值列转 float。"""
    d = _read_parts("daily")
    d["code"] = d["code"].map(from_bs)
    d["date"] = pd.to_datetime(d["date"])
    for c in ("open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("tradestatus", "isST"):
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("Int64")
    return d.sort_values(["code", "date"]).reset_index(drop=True)


def load_adjf() -> pd.DataFrame:
    """baostock 复权因子事件表:code dividOperateDate foreAdjustFactor backAdjustFactor adjustFactor。"""
    d = _read_parts("adjf")
    if d.empty:
        return d
    d["code"] = d["code"].map(from_bs)
    d["dividOperateDate"] = pd.to_datetime(d["dividOperateDate"])
    for c in ("foreAdjustFactor", "backAdjustFactor", "adjustFactor"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.sort_values(["code", "dividOperateDate"]).reset_index(drop=True)


def load_profit() -> pd.DataFrame:
    """baostock 季频盈利表(含股本)。PIT 使用 pubDate(披露日),statDate 为报告期。"""
    d = _read_parts("profit")
    if d.empty:
        return d
    d["code"] = d["code"].map(from_bs)
    for c in ("pubDate", "statDate"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    for c in ("roeAvg", "npMargin", "gpMargin", "netProfit", "epsTTM", "MBRevenue",
              "totalShare", "liqaShare"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.sort_values(["code", "statDate"]).reset_index(drop=True)


def minute_to_bs5(wide: pd.DataFrame, how: str) -> pd.DataFrame:
    """1 分钟宽表 [分钟 x 股票](每天 241 行)-> baostock 口径 5 分钟宽表 [bar 结束时刻 x 股票]。

    how='last'(收盘价)/ 'first'(开盘价)/ 'sum'(成交额)。
    要求每天恰好 241 行、时间为 09:30..15:00 的上海时间;不满足的日子直接报错,
    由调用方先做清洗(见 clean_minute_month)。整日 NaN(停牌/缺失)保持 NaN。
    """
    days = wide.index.normalize()
    cnt = pd.Series(1, index=wide.index).groupby(days).size()
    if (cnt != N_BARS).any():
        raise ValueError(f"每日行数不是 {N_BARS}: {cnt[cnt != N_BARS].to_dict()}")
    g = bs5_groups()
    ud = cnt.index
    arr = wide.to_numpy(dtype=np.float64).reshape(len(ud), N_BARS, wide.shape[1])
    out = np.full((len(ud), 48, wide.shape[1]), np.nan)
    for k in range(48):
        sl = arr[:, g == k, :]
        if how == "last":
            out[:, k, :] = sl[:, -1, :]
        elif how == "first":
            out[:, k, :] = sl[:, 0, :]
        elif how == "sum":
            # 整组 NaN -> NaN;部分 NaN 不应出现(NaN 为整日级)
            allnan = np.isnan(sl).all(axis=1)
            s = np.nansum(sl, axis=1)
            s[allnan] = np.nan
            out[:, k, :] = s
        else:
            raise ValueError(how)
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=int(t[:2]), minutes=int(t[3:]))
                            for d in ud for t in BS5_TIMES], name="dt")
    return pd.DataFrame(out.reshape(len(ud) * 48, wide.shape[1]), index=idx, columns=wide.columns)


def clean_minute_month(wide: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """修复 2026 年 1 分钟原始文件里发现的时间戳问题(只做结构修复,不改数值)。

    2026-10-10 发现(research/FORWARD_DATA_QA.md §2):202606 / 202608 文件里个别交易日的一部分股票
    用 UTC 时间记录(01:30..07:00 = 上海 09:30..15:00),同一天另一部分股票用上海时间,
    导致当天出现 482 行。修复:UTC 行 +8 小时,再与上海时间行按股票合并(同一格两边都有值时
    检查是否一致,不一致计数返回,取上海时间行的值——两版孰对见 QA 报告 §2.2)。
    202604 的 04-08 / 04-09 缺 09:30(开盘集合竞价)整行:补成 NaN 行,diag["missing_slots"] 列出;
    这两天聚合出的 09:35 成交额不含开盘集合竞价,下游需自行决定是否剔除。
    返回 (修复后宽表(每天恰好 241 行), 诊断字典)。
    """
    t = wide.index
    hhmm = t.hour * 100 + t.minute
    utc_mask = (hhmm >= 130) & (hhmm <= 700)
    diag = {"utc_rows": int(utc_mask.sum()), "conflict_cells": 0, "utc_days": [],
            "missing_slots": {}}
    if utc_mask.any():
        a = wide[~utc_mask]
        b = wide[utc_mask].copy()
        b.index = b.index + pd.Timedelta(hours=8)
        diag["utc_days"] = sorted({str(x.date()) for x in b.index})
        b = b.reindex(a.index.union(b.index))
        a = a.reindex(b.index)
        both = a.notna().to_numpy() & b.notna().to_numpy()
        if both.any():
            av, bv = a.to_numpy()[both], b.to_numpy()[both]
            diag["conflict_cells"] = int((np.abs(av - bv) > 1e-6 * np.maximum(np.abs(av), 1)).sum())
        wide = a.where(a.notna(), b).sort_index()
    # 补齐到每天 241 根(2026-04-08/09 缺 09:30 开盘集合竞价那一行):缺的格子保持 NaN
    days = wide.index.normalize().unique()
    grid = pd.DatetimeIndex([d + o for d in days for o in minute_offsets()], name=wide.index.name)
    missing = grid.difference(wide.index)
    if len(missing):
        diag["missing_slots"] = {str(k.date()): [x.strftime("%H:%M") for x in v]
                                 for k, v in pd.Series(missing, index=missing).groupby(missing.normalize())}
    extra = wide.index.difference(grid)
    if len(extra):
        raise ValueError(f"非标准时间戳 {len(extra)} 个,例如 {list(extra[:3])}")
    return wide.reindex(grid), diag


def minute_offsets() -> list:
    """241 根 1 分钟 bar 的日内时刻(上海时间):09:30, 09:31..11:30, 13:01..15:00。"""
    am = [pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=i) for i in range(121)]
    pm = [pd.Timedelta(hours=13, minutes=1) + pd.Timedelta(minutes=i) for i in range(120)]
    return am + pm
