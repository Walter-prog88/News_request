"""lh_data 第 3 步:把逐股日线拼成 [日期 x 股票] 宽表 parquet。不计算任何策略收益。

两个数据源(提速方案,见 scripts/lh_fetch_sina.py 文件头与 research/LH_DATA_QA.md §0):
  价格 / 量额:全部股票都用新浪 cache/long_daily/parts_sina/(不复权 OHLC + 新浪后复权因子)——baostock 的后复权
    漏掉了股权分置改革的送股对价(QA 两源对比发现),所以退市股的价格也改用新浪;新浪无数据 / 无因子时退回 baostock。
  现存股(universe.csv status=1):换手率 = 新浪 成交量 / 流通股本;上市区间延伸到 2025-12-31。
  已退市股(status=0):换手率、isST、上市区间(首行..末行)取 baostock cache/long_daily/parts/(新浪退市股无股本表)。
  新浪行先经 sina_to_bs() 转成 baostock 同一套字段再进 process():
    后复权价 = 不复权价 x 新浪 hfq 因子;后复权前收 = 上一行后复权收盘(新浪无停牌行,上一行 = 上一交易日;
      后复权口径下"除权参考价 x 当日因子 = 上一交易日后复权收盘",与 baostock preclose 同义);上市首行用新浪发行价;
      2005-01-04 之前上市的股票窗口首行前收未知 = NaN(只影响该股窗口首日)。
    停牌:新浪停牌日**没有行**。上市区间(2005-01-04 或上市日 .. 2025-12-31,现存股必然在市)内的无行交易日
      补成 baostock 式停牌行:tradestatus=0、量额 0、价格 = 最近收盘(之前无成交则 NaN)。
    turn = 新浪 turnover x 100(新浪 = 成交量 / 流通股本,小数);pctChg 新浪没有 = NaN;
    isST:新浪没有,取 StLookup(baostock isST > 项目 st_info_latest.csv(2016-08-09 起)> 视为非 ST),
      逐股来源写进 _codes.parquet 的 st_src。

输出 cache/long_daily/panel/<字段>.parquet(行 = 2005-01-04..2025-12-31 交易日历,列 = 000001.SZ 格式代码),
外加 _codes.parquet(逐股元数据)与 _meta.json(口径、开盘 10 分钟成交额占比)。

口径(LongPanel 与 research/LH_DATA_QA.md 引用这里;下文"行"= baostock 行或新浪经 sina_to_bs 转换后的行):
  交易日 trade   = 有行 & tradestatus==1 & volume>0 & amount>0
  listed         = 有行(上市首行..最后一行;停牌日 baostock 给行、新浪由构建补行,价格=前收,量额为 0/空)
  isST 用法见 StLookup(st_known 标记当日 ST 身份是否有来源)
  suspended      = listed & ~trade
  open_px/high_px/low_px/close_px/preclose_px  后复权价,**非交易日为 NaN**(与项目 Panel 的 close_px 语义一致)
  close_raw      不复权收盘价(非交易日 NaN)
  volume(股)/amount(元)/turn(%)/pct_chg(%)  非交易日 NaN
  is_st          baostock 当日 isST(上市期间每行都有,含停牌日)
  ret            持有收益 = 后复权 close / 上一行 close - 1,逐 baostock 行计算:停牌日 = 0,
                 复牌日 = 跨停牌的累计跳空(项目 Panel 的 ret_c2c 在停牌/复牌日为 NaN,复牌跳空丢失)
  twap_open10    **不是 TWAP**:交易日 = 后复权日开盘价;上市期间的停牌日 = 最近一个交易日的后复权收盘价(估值价,
                 使引擎对停牌持仓按"停牌前收盘 -> 复牌开盘"记跳空);首个交易日前、最后一行之后 = NaN
  days_listed    自上市日起的交易日序号(上市首日 = 1;2005 年前上市的按工作日数近似补偿)
  涨跌停         用不复权价精确判定:f = close_raw / close(当日复权比例),各价 = round(后复权价 * f, 2),
                 前收(交易所除权参考价)= round(preclose * f, 2),涨停价 = 四舍五入(前收 * (1+L), 2)。
                 L 按当时规则(见 band_rules()),limit_up = 收盘价 == 涨停价,one_word_up = limit_up & (最高-最低) < 0.4% x 收盘(同 Panel 阈值)。
                 收盘越出规则带(> 半个价位)时先试 10% 带(ST 5% 股的退市整理期 / 2025-07 起规则变更),
                 仍越出 -> 记为"无涨跌幅限制日" no_limit_day(股改复牌首日、恢复上市首日等),当日不标涨跌停;
                 out_band 单独标出这些"越带兜底"日(验收 Q5 用)。
  一字板        one_word_up/down = 不复权 最高 == 最低 且 日收益(后复权收盘/后复权前收 - 1)> +4.5% / < -4.5%
                 (任务书口径:**不依赖 ST 身份**,锁板即不可成交;两个数据源同一规则)。
                 旧的精确口径(收盘 = 当日涨跌停价 且 振幅 < 0.4%)保留为 ow_lim_up / ow_lim_down,验收里对比两者。
用法: python3 scripts/lh_build_panel.py [--allow-partial]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "cache" / "long_daily"
PARTS = OUT / "parts"
PARTS_SINA = OUT / "parts_sina"
PARTS_ST = OUT / "parts_st"
PANEL = OUT / "panel"
ST_INFO_START = pd.Timestamp("2016-08-09")       # 项目 st_info_latest.csv 的起始日
OW_RET = 0.045                                   # 一字板:|日收益| > 4.5%
LOG = ROOT / "logs" / "lh_build_panel.log"
START, END = "2005-01-01", "2025-12-31"

GEM_20PCT_START = pd.Timestamp("2020-08-24")     # 创业板注册制:存量股 10% -> 20%,新股前 5 日无涨跌幅
MAIN_REG_START = pd.Timestamp("2023-04-10")      # 主板注册制首批上市:新股前 5 日无涨跌幅
IPO_44_START = pd.Timestamp("2014-01-01")        # 新股首日 +44% / -36% 规则(2013-12 新股改革后首批)
ST_10PCT_START = pd.Timestamp("2025-07-07")      # 沪深主板风险警示股 5% -> 10%(由 QA 实测确认,见 LH_DATA_QA.md)

F_FLOAT = ["open_px", "high_px", "low_px", "close_px", "preclose_px", "close_raw", "volume", "amount", "turn",
           "pct_chg", "ret", "twap_open10", "days_listed", "lim_up_rate", "lim_dn_rate"]
F_BOOL = ["listed", "trade", "suspended", "is_st", "limit_up", "limit_down", "one_word_up", "one_word_down",
          "no_limit_day", "open_at_up", "open_at_dn", "ow_lim_up", "ow_lim_down", "out_band"]
BS_COLS = ["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "turn", "tradestatus",
           "pctChg", "isST", "close_raw"]


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def to_ts_code(bs_code: str) -> str:
    ex, num = bs_code.split(".")
    return f"{num}.{ex.upper()}"


def board(num: str) -> str:
    if num.startswith(("688", "689")):
        return "star"
    if num.startswith(("300", "301", "302")):
        return "gem"
    return "main"


def round_half_up(x: np.ndarray, nd: int = 2) -> np.ndarray:
    s = 10.0 ** nd
    return np.floor(x * s + 0.5 + 1e-7) / s


def band_rules(num: str, dates: pd.DatetimeIndex, is_st: np.ndarray, k_from_ipo: np.ndarray,
               ipo: pd.Timestamp | None) -> tuple[np.ndarray, np.ndarray]:
    """逐行涨跌幅上限(上行 bu、下行 bd;NaN = 无涨跌幅限制)。

    常规:主板 10%;创业板 2020-08-24 前 10%、之后 20%;科创板 20%。
    风险警示(ST/*ST,当日 isST):只把 10% 降到 5%(主板、改革前创业板);2025-07-07 起主板 ST 回到 10%;
      创业板改革后、科创板的 ST 仍为 20%。
    新股:注册制(科创板全部;创业板 2020-08-24 起上市;主板 2023-04-10 起上市)上市前 5 个交易日无涨跌幅;
      其余新股首日:2014-01-01 前无涨跌幅限制,之后 +44% / -36%(相对发行价 = baostock 首日 preclose)。
    其余无涨跌幅的日子(股改复牌首日、恢复上市首日、退市整理首日等)没有事件表,由 build 时"收盘越出规则带"兜底识别。
    """
    n = len(dates)
    b = board(num)
    if b == "star":
        base = np.full(n, 0.20)
    elif b == "gem":
        base = np.where(dates >= GEM_20PCT_START, 0.20, 0.10)
    else:
        base = np.full(n, 0.10)
    st = is_st.astype(bool)
    st5 = st & (base == 0.10)
    if b == "main":
        st5 &= dates < ST_10PCT_START
    bu = np.where(st5, 0.05, base).astype(np.float64)
    bd = bu.copy()
    if ipo is not None and ipo >= pd.Timestamp("2005-01-04"):
        reg = (b == "star") or (b == "gem" and ipo >= GEM_20PCT_START) or (b == "main" and ipo >= MAIN_REG_START)
        if reg:
            m = k_from_ipo < 5
            bu[m] = np.nan; bd[m] = np.nan
        else:
            m = k_from_ipo == 0
            if ipo < IPO_44_START:
                bu[m] = np.nan; bd[m] = np.nan
            else:
                bu[m] = 0.44; bd[m] = 0.36
    return bu, bd


def process(bs_code: str, d: pd.DataFrame, cal: pd.DatetimeIndex, ipo: pd.Timestamp | None) -> dict:
    """单只股票 -> 各字段 (行位置, 值)。"""
    num = bs_code.split(".")[1]
    dates = pd.to_datetime(d["date"])
    pos = cal.get_indexer(dates)
    off_cal = int((pos < 0).sum())
    keep = pos >= 0
    d = d.loc[keep.tolist()].reset_index(drop=True)
    dates = dates[keep].reset_index(drop=True)
    pos = pos[keep]
    num_cols = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg", "close_raw"]
    x = {c: pd.to_numeric(d[c], errors="coerce").to_numpy(np.float64) for c in num_cols}
    ts = pd.to_numeric(d["tradestatus"], errors="coerce").to_numpy()
    st = (pd.to_numeric(d["isST"], errors="coerce").fillna(0).to_numpy() == 1)
    trade = (ts == 1) & (np.nan_to_num(x["volume"]) > 0) & (np.nan_to_num(x["amount"]) > 0) \
        & np.isfinite(x["close"]) & (x["close"] > 0)
    n = len(d)
    # 持有收益:逐 baostock 行(停牌行 close = 前收 -> 0;复牌行带跳空)
    c = x["close"]
    ret = np.full(n, np.nan)
    if n > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            ret[1:] = c[1:] / c[:-1] - 1.0
    # 估值价 / 执行价:交易日开盘,停牌日 = 最近交易日收盘(ffill),首个交易日之前 NaN
    last_close = pd.Series(np.where(trade, c, np.nan)).ffill().to_numpy()
    twap = np.where(trade, x["open"], last_close)
    # 不复权还原与精确涨跌停
    with np.errstate(divide="ignore", invalid="ignore"):
        f = x["close_raw"] / c
    o_r, h_r, l_r, c_r = (round_half_up(x[k] * f) for k in ("open", "high", "low", "close"))
    pc_r = round_half_up(x["preclose"] * f)
    first_trade = int(np.flatnonzero(trade)[0]) if trade.any() else 0
    k_from_ipo = np.arange(n) - first_trade
    bu, bd = band_rules(num, pd.DatetimeIndex(dates), st, k_from_ipo, ipo)
    eps = 0.0051

    def px_band(bu_, bd_):
        return round_half_up(pc_r * (1 + bu_)), round_half_up(pc_r * (1 - bd_))

    up, dn = px_band(bu, bd)
    out_band = trade & np.isfinite(bu) & ((c_r > up + eps) | (c_r < dn - eps))
    # 越出规则带:5% 带先试 10% 带
    esc = out_band & (bu == 0.05)
    n_esc10 = 0
    if esc.any():
        up10, dn10 = px_band(np.full(n, 0.10), np.full(n, 0.10))
        ok10 = esc & (c_r <= up10 + eps) & (c_r >= dn10 - eps)
        bu = np.where(ok10, 0.10, bu); bd = np.where(ok10, 0.10, bd)
        up = np.where(ok10, up10, up); dn = np.where(ok10, dn10, dn)
        out_band = out_band & ~ok10
        n_esc10 = int(ok10.sum())
    no_limit = (trade & ~np.isfinite(bu)) | out_band
    out_band_flag = out_band.copy()
    bu = np.where(out_band, np.nan, bu); bd = np.where(out_band, np.nan, bd)
    has = trade & np.isfinite(bu)
    lim_up = has & (np.abs(c_r - up) < 0.005)
    lim_dn = has & (np.abs(c_r - dn) < 0.005)
    # 一字板:收盘 = 涨(跌)停价 且 日内振幅 < 0.4%(与项目 Panel 的 rng < 0.004 同阈值;低价股即最高 == 最低)
    with np.errstate(divide="ignore", invalid="ignore"):
        flat = (h_r - l_r) < np.maximum(0.004 * c_r, 0.005)
    ow_lim_up = lim_up & flat
    ow_lim_dn = lim_dn & flat
    # 任务书口径的一字板:不复权 最高 == 最低 且 |日收益| > 4.5%(不依赖 ST 身份与涨跌停价)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_day = c / x["preclose"] - 1.0
    hl_eq = np.abs(h_r - l_r) < 0.005
    ow_up = trade & hl_eq & (r_day > OW_RET)
    ow_dn = trade & hl_eq & (r_day < -OW_RET)
    # 开盘即在涨(跌)停价:日开盘价执行时买(卖)不进的直接判据(供敏感性分析,引擎默认仍用一字板)
    open_up = has & (np.abs(o_r - up) < 0.005)
    open_dn = has & (np.abs(o_r - dn) < 0.005)
    # 上市天数
    if ipo is not None and ipo >= pd.Timestamp("2005-01-04"):
        dl = (pos - pos[first_trade] + 1).astype(np.float64)
    else:
        base_off = np.busday_count((ipo or pd.Timestamp("2005-01-04")).date(), cal[0].date()) if ipo is not None else 0
        dl = (pos + 1 + max(int(base_off), 0)).astype(np.float64)

    def tr(v):
        return np.where(trade, v, np.nan)

    res = dict(
        open_px=tr(x["open"]), high_px=tr(x["high"]), low_px=tr(x["low"]), close_px=tr(c),
        preclose_px=tr(x["preclose"]), close_raw=tr(x["close_raw"]), volume=tr(x["volume"]),
        amount=tr(x["amount"]), turn=tr(x["turn"]), pct_chg=tr(x["pctChg"]), ret=ret, twap_open10=twap,
        days_listed=dl, lim_up_rate=np.where(trade, bu, np.nan), lim_dn_rate=np.where(trade, bd, np.nan),
        listed=np.ones(n, bool), trade=trade, suspended=~trade, is_st=st, limit_up=lim_up, limit_down=lim_dn,
        one_word_up=ow_up, one_word_down=ow_dn, no_limit_day=no_limit, open_at_up=open_up, open_at_dn=open_dn,
        ow_lim_up=ow_lim_up, ow_lim_down=ow_lim_dn, out_band=out_band_flag,
    )
    # 诊断:停牌行收盘是否等于上一交易日收盘(twap 估值价口径的前提)
    sus_rows = np.flatnonzero(~trade & (np.arange(n) > first_trade))
    sus_mismatch = int(np.sum(np.abs(c[sus_rows] / last_close[sus_rows] - 1) > 1e-6)) if len(sus_rows) else 0
    meta = dict(n_rows=n, n_trade=int(trade.sum()), off_cal=off_cal,
                n_ts0=int((ts != 1).sum()), n_ts0_volpos=int(((ts != 1) & (np.nan_to_num(x["volume"]) > 0)).sum()),
                n_ts1_vol0=int(((ts == 1) & ~(np.nan_to_num(x["volume"]) > 0)).sum()),
                first_row=dates.iloc[0], last_row=dates.iloc[-1],
                first_trade=dates.iloc[first_trade] if trade.any() else pd.NaT,
                last_trade=dates.iloc[int(np.flatnonzero(trade)[-1])] if trade.any() else pd.NaT,
                n_no_limit=int(no_limit.sum()), n_esc10=n_esc10,
                sus_close_mismatch=sus_mismatch, ipo_pre_close=float(x["preclose"][first_trade] * f[first_trade])
                if trade.any() else np.nan, ipo_open_raw=float(o_r[first_trade]) if trade.any() else np.nan)
    return dict(pos=pos, fields=res, meta=meta)


def open10_ratio() -> dict:
    """项目 Panel(2020-2025)开盘 10 分钟成交额 / 全日成交额 的中位比例(股票-日,amount>0)。

    open10_amt_ratio      全部股票-日中位
    open10_amt_ratio_pool 只看 Panel 决策池(universe:上市满 6 日、adv20 >= 2000 万)的股票-日中位
                          —— 预注册 §6.3 的 0.114 就是这个口径;LongPanel 默认用它(见 LongPanel 文档)。"""
    from afactor.config import DAILY
    from afactor.panel import Panel
    a = pd.read_parquet(DAILY / "amt_open10.parquet")
    b = pd.read_parquet(DAILY / "amount.parquet")
    a = a.loc[: END]; b = b.loc[: END]
    rr = a / b.where(b > 0)
    r = rr.stack()
    r = r[np.isfinite(r)]
    by_year = r.groupby(r.index.get_level_values(0).year).median()
    u = Panel()["universe"].reindex(index=rr.index, columns=rr.columns).fillna(False).astype(bool)
    rp = rr.where(u).stack()
    rp = rp[np.isfinite(rp)]
    by_year_p = rp.groupby(rp.index.get_level_values(0).year).median()
    return dict(open10_amt_ratio=float(r.median()), open10_amt_ratio_mean=float(r.mean()),
                open10_amt_ratio_p10=float(r.quantile(0.10)), open10_amt_ratio_p90=float(r.quantile(0.90)),
                open10_amt_ratio_n=int(len(r)), open10_amt_ratio_by_year={int(k): float(v) for k, v in by_year.items()},
                open10_amt_ratio_pool=float(rp.median()), open10_amt_ratio_pool_p10=float(rp.quantile(0.10)),
                open10_amt_ratio_pool_n=int(len(rp)),
                open10_amt_ratio_pool_by_year={int(k): float(v) for k, v in by_year_p.items()})


def _read_gz(f: Path) -> pd.DataFrame:
    with gzip.open(f, "rt") as fh:
        return pd.read_csv(fh, dtype=str, keep_default_na=False)


class StLookup:
    """逐股逐日 ST 身份(新浪源股票用;baostock 源股票直接用自己行里的 isST)。

    优先级(逐日):baostock isST(全字段 parts/ 或 ST 专抓 parts_st/,在其行覆盖的日期上)
                  > 项目 st_info_latest.csv(2016-08-09 起的每日风险警示名单;名单缺的交易日沿用前一名单日)
                  > 视为非 ST(并在 st_src 里写明 "none")。
    """

    def __init__(self, cal: pd.DatetimeIndex, parts_dir: Path | None = PARTS, st_dir: Path | None = PARTS_ST,
                 st_info: pd.DataFrame | None = None):
        self.cal = cal
        self.parts_dir, self.st_dir = parts_dir, st_dir
        if st_info is None:
            st_info = self._read_st_info()
        self.info_start = st_info.index.min() if len(st_info) else pd.Timestamp("2100-01-01")
        m = st_info.reindex(cal[cal >= self.info_start], method="ffill") if len(st_info) else st_info
        self.info = m.fillna(False).astype(bool)
        self.info_pos0 = int(cal.searchsorted(self.info_start)) if len(st_info) else len(cal)

    @staticmethod
    def _read_st_info() -> pd.DataFrame:
        f = ROOT / "data_8:21" / "st_info_latest.csv"
        if not f.exists():
            return pd.DataFrame(dtype=bool)
        st = pd.read_csv(f, usecols=["stock_code", "trade_date"], encoding="utf-8-sig")
        st["d"] = pd.to_datetime(st["trade_date"]).dt.normalize()
        st = st[st["d"] <= pd.Timestamp(END)]                         # 先截断(2026 封存)
        st["bs"] = st["stock_code"].map(lambda c: f"{c.split('.')[1].lower()}.{c.split('.')[0]}")
        st["v"] = True
        return st.pivot_table(index="d", columns="bs", values="v", aggfunc="any").fillna(False).astype(bool)

    def _bs_rows(self, bs_code: str) -> pd.DataFrame | None:
        for dd in (self.parts_dir, self.st_dir):
            if dd is not None and (dd / f"{bs_code}.csv.gz").exists():
                d = _read_gz(dd / f"{bs_code}.csv.gz")
                return d[["date", "isST"]]
        return None

    def get(self, bs_code: str) -> tuple[np.ndarray, np.ndarray, str]:
        """返回 (is_st[bool, len(cal)], known[bool, len(cal)], 来源说明)。"""
        T = len(self.cal)
        st = np.zeros(T, bool)
        known = np.zeros(T, bool)
        src = []
        if bs_code in self.info.columns:
            st[self.info_pos0:] = self.info[bs_code].to_numpy()
        if len(self.info):
            known[self.info_pos0:] = True
            src.append("st_info")
        b = self._bs_rows(bs_code)
        if b is not None and len(b):
            pos = self.cal.get_indexer(pd.to_datetime(b["date"]))
            ok = pos >= 0
            v = pd.to_numeric(b["isST"], errors="coerce").fillna(0).to_numpy() == 1
            st[pos[ok]] = v[ok]
            known[pos[ok]] = True
            src.insert(0, "baostock")
        return st, known, "+".join(src) if src else "none"


def sina_to_bs(bs_code: str, s: pd.DataFrame, cal: pd.DatetimeIndex, ipo: pd.Timestamp | None,
               st: np.ndarray | None, bs: pd.DataFrame | None = None) -> tuple[pd.DataFrame | None, dict]:
    """新浪逐股行 -> baostock 同字段的行(含补出的停牌行)。

    bs = None(现存股):上市区间 = 2005-01-04(或上市首行)..cal 末日,换手率 = 新浪 turnover x 100。
    bs = 该股 baostock 行(退市股):上市区间 = baostock 首行..末行(与新浪行取并),换手率按日期取 baostock turn
         (新浪退市股没有股本表),停牌以"新浪无行"为准。
    新浪偶有与上一行 OHLC+量额完全相同的重复行(如退市后挂在周末的残行),去掉(akshare 用 drop_duplicates 做同样的事)。"""
    s = s.copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s[(s["date"] >= cal[0]) & (s["date"] <= cal[-1])]
    n0 = len(s)
    k6 = ["open", "high", "low", "close", "volume", "amount"]
    s = s.sort_values("date")
    s = s[~(s[k6].to_numpy() == s[k6].shift(1).to_numpy()).all(axis=1)]      # 与上一行完全相同 = 残行
    pos = cal.get_indexer(s["date"])
    info = dict(sina_rows=n0, sina_dup_rows=n0 - len(s), sina_off_cal=int((pos < 0).sum()))
    s = s[pos >= 0]
    pos = pos[pos >= 0]
    if len(s) == 0:
        return None, info
    num = ["open", "high", "low", "close", "volume", "amount", "turnover", "hfq_factor", "prevclose_ipo"]
    x = {k: pd.to_numeric(s[k], errors="coerce").to_numpy(np.float64) for k in num}
    end = len(cal) - 1
    if bs is None:
        start = 0 if (ipo is None or ipo < cal[0]) else int(pos[0])
    else:
        bpos = cal.get_indexer(pd.to_datetime(bs["date"]))
        bpos = bpos[bpos >= 0]
        start = int(bpos.min()) if len(bpos) else int(pos[0])
        end = max(int(bpos.max()) if len(bpos) else int(pos[-1]), int(pos[-1]))
    start = min(start, int(pos[0]))
    idx = np.arange(start, end + 1)
    n = len(idx)
    loc = pos - start
    has = np.zeros(n, bool); has[loc] = True

    def put(v):
        a = np.full(n, np.nan); a[loc] = v; return a
    f = x["hfq_factor"]
    c_raw = put(x["close"])
    c_adj = put(x["close"] * f)
    c_adj_ff = pd.Series(c_adj).ffill().to_numpy()
    c_raw_ff = pd.Series(c_raw).ffill().to_numpy()
    o = np.where(has, put(x["open"] * f), c_adj_ff)
    h = np.where(has, put(x["high"] * f), c_adj_ff)
    lo = np.where(has, put(x["low"] * f), c_adj_ff)
    c = np.where(has, c_adj, c_adj_ff)
    pre = np.full(n, np.nan)
    pre[1:] = c[:-1]                                     # 上一行后复权收盘(停牌行 = 最近收盘)
    # 上市首行:新浪发行价 x 因子(只在窗口内上市的股票)
    pi = put(x["prevclose_ipo"] * f)
    first = int(loc[0])
    if ipo is not None and ipo >= cal[0] and np.isfinite(pi[first]):
        pre[first] = pi[first]
    pre = np.where(has, pre, c_adj_ff)                   # 停牌行 preclose = 收盘(baostock 同)
    vol = np.where(has, put(x["volume"]), 0.0)
    amt = np.where(has, put(x["amount"]), 0.0)
    if bs is None:
        turn = put(x["turnover"] * 100.0)
    else:
        tb = pd.Series(pd.to_numeric(bs["turn"], errors="coerce").to_numpy(),
                       index=pd.to_datetime(bs["date"])).groupby(level=0).last()
        turn = np.where(has, tb.reindex(cal[idx]).to_numpy(np.float64), np.nan)
    d = pd.DataFrame(dict(
        date=cal[idx], code=bs_code, open=o, high=h, low=lo, close=c, preclose=pre, volume=vol, amount=amt,
        turn=turn, tradestatus=np.where(has, 1, 0), pctChg=np.nan,
        isST=(st[idx].astype(int) if st is not None else 0), close_raw=np.where(has, c_raw, c_raw_ff)))
    info.update(sina_fill_rows=int((~has).sum()), sina_lead_fill=int(loc[0]))
    return d[BS_COLS], info


def build_panel(parts_dir: Path, panel_dir: Path, uni: pd.DataFrame, cal: pd.DatetimeIndex,
                ratio_meta: dict, allow_partial: bool = False, sina_dir: Path | None = None,
                st: StLookup | None = None) -> dict:
    """逐股日线 -> panel_dir 下的宽表 parquet。uni 需含 code / ipoDate / outDate(baostock 代码),可选 status。

    sina_dir 给出时(主源选择,写进 _codes.parquet 的 src):
      sina             现存股(status == "1")且新浪有数据(含后复权因子);
      sina+bs          退市股,新浪有行情与因子:价格 / 量额取新浪,换手率 / isST / 上市区间取 baostock;
      baostock         退市股,新浪无行或无因子;
      baostock_fallback 现存股,新浪无数据(如 sh.689009)。"""
    t0 = time.time()
    have = {p.name[:-7] for p in parts_dir.glob("*.csv.gz")}
    empty = {p.name[:-6] for p in parts_dir.glob("*.empty")}
    have_s, empty_s, nos = set(), set(), set()
    if sina_dir is not None and sina_dir.exists():
        have_s = {p.name[:-7] for p in sina_dir.glob("*.csv.gz")}
        empty_s = {p.name[:-6] for p in sina_dir.glob("*.empty")}
        nos = {p.name[:-7] for p in sina_dir.glob("*.nosina")}
    status = dict(zip(uni["code"], uni["status"].astype(str))) if "status" in uni.columns else {}
    src_of = {}
    n_nohfq = 0
    for c in uni["code"]:
        if c in have_s:
            sd = _read_gz(sina_dir / f"{c}.csv.gz")
            if pd.to_numeric(sd["hfq_factor"], errors="coerce").notna().all():
                if status.get(c) == "1":
                    src_of[c] = "sina"
                    continue
                if c in have:                       # 退市股:新浪价格 + baostock 换手率 / ST / 上市区间
                    src_of[c] = "sina+bs"
                    continue
            else:
                n_nohfq += 1
        if c in have:
            src_of[c] = "baostock_fallback" if (status.get(c) == "1" and sina_dir is not None) else "baostock"
    missing = [c for c in uni["code"] if c not in src_of and c not in empty and c not in empty_s]
    if missing and not allow_partial:
        raise SystemExit(f"还有 {len(missing)} 只未下载完(--allow-partial 可先用已下载部分)")
    codes_bs = [c for c in uni["code"] if c in src_of]
    codes = pd.Index([to_ts_code(c) for c in codes_bs], name=None)
    T, N = len(cal), len(codes_bs)
    n_src = pd.Series(src_of).value_counts().to_dict()
    log(f"构建面板:交易日 {T}({cal[0].date()}..{cal[-1].date()}),股票 {N} {n_src}"
        f"(缺 {len(missing)},空 {len(empty) + len(empty_s)},新浪无 {len(nos)})")
    arr = {k: np.full((T, N), np.nan, np.float32) for k in F_FLOAT}
    arr.update({k: np.zeros((T, N), bool) for k in F_BOOL})
    st_known = np.zeros((T, N), bool)
    ipo_map = dict(zip(uni["code"], uni["ipoDate"].fillna("")))
    out_map = dict(zip(uni["code"], uni["outDate"].fillna("")))
    metas = []
    for j, bc in enumerate(codes_bs):
        ipo_s = ipo_map.get(bc) or ""
        ipo = pd.Timestamp(ipo_s) if ipo_s else None
        extra = dict(src=src_of[bc])
        if src_of[bc] in ("sina", "sina+bs"):
            stv, known, st_src = st.get(bc) if st is not None else (None, np.zeros(T, bool), "none")
            bsr = _read_gz(parts_dir / f"{bc}.csv.gz") if src_of[bc] == "sina+bs" else None
            d, inf = sina_to_bs(bc, _read_gz(sina_dir / f"{bc}.csv.gz"), cal, ipo, stv, bs=bsr)
            extra.update(inf, st_src=st_src)
            st_known[:, j] = known
        else:
            d = _read_gz(parts_dir / f"{bc}.csv.gz")
            extra.update(st_src="baostock_self")
        r = process(bc, d, cal, ipo)
        p = r["pos"]
        for k, v in r["fields"].items():
            arr[k][p, j] = v
        if src_of[bc] not in ("sina", "sina+bs"):
            st_known[p, j] = True
        m = r["meta"]; m.update(code=codes[j], bs_code=bc, ipoDate=ipo_s, outDate=out_map.get(bc) or "", **extra)
        metas.append(m)
        if (j + 1) % 500 == 0:
            log(f"  {j+1}/{N} {time.time()-t0:.0f}s")
    panel_dir.mkdir(parents=True, exist_ok=True)
    arr["st_known"] = st_known
    for k in F_FLOAT + F_BOOL + ["st_known"]:
        df = pd.DataFrame(arr.pop(k), index=cal, columns=codes)
        df.to_parquet(panel_dir / f"{k}.parquet")
        del df
    pd.DataFrame(metas).to_parquet(panel_dir / "_codes.parquet")
    meta = dict(built=time.strftime("%Y-%m-%d %H:%M:%S"), start=str(cal[0].date()), end=str(cal[-1].date()),
                n_dates=T, n_codes=N, n_missing=len(missing), missing=missing[:50], n_empty=len(empty) + len(empty_s),
                n_nosina=len(nos), n_sina_nohfq=n_nohfq, sources={k: int(v) for k, v in n_src.items()},
                rules=dict(GEM_20PCT_START=str(GEM_20PCT_START.date()), MAIN_REG_START=str(MAIN_REG_START.date()),
                           IPO_44_START=str(IPO_44_START.date()), ST_10PCT_START=str(ST_10PCT_START.date()),
                           one_word=f"不复权 最高 == 最低 且 |后复权日收益| > {OW_RET:.1%}(不依赖 ST)",
                           ow_lim="收盘=涨跌停价 且 (最高-最低) < max(0.4%*收盘, 0.005 元)",
                           st="新浪源:baostock isST > st_info_latest(2016-08-09 起) > 非 ST;baostock 源:自身 isST"),
                partial=bool(missing))
    meta.update(ratio_meta)
    (panel_dir / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    log(f"完成 {time.time()-t0:.0f}s;开盘 10 分钟成交额占比中位 {meta['open10_amt_ratio']:.4f}"
        f"(决策池 {meta.get('open10_amt_ratio_pool', float('nan')):.4f})")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--no-sina", action="store_true", help="只用 baostock(旧口径,对照用)")
    ap.add_argument("--out", default=str(PANEL))
    args = ap.parse_args()
    uni = pd.read_csv(OUT / "universe.csv", dtype=str, keep_default_na=False)
    cal = pd.to_datetime(pd.read_csv(OUT / "trade_dates.csv")["date"])
    cal = pd.DatetimeIndex(cal[(cal >= START) & (cal <= END)])
    st = StLookup(cal)
    build_panel(PARTS, Path(args.out), uni, cal, open10_ratio(), allow_partial=args.allow_partial,
                sina_dir=None if args.no_sina else PARTS_SINA, st=st)


if __name__ == "__main__":
    main()
