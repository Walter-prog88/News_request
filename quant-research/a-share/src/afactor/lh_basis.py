"""lh_basis:中金所股指期货基差 / 年化贴水 / 空头展期(对冲)成本。只做数据与统计,不做策略回测。

数据: scripts/lh_basis_fetch.py 抓的中金所月度历史包(cache/_tmp/lh_basis_raw/YYYYMM.zip),
      现货收盘 = newdata.index_daily()(000300/000905/000852/000016),
      全收益指数 = 中证 H00300/H00905/H00852/H00016(股息部分)。全部截断到 2025-12-31。

口径(写在任何统计之前,scripts/lh_basis_build.py 严格按此执行):
  基差           basis      = F_close - S_close
  基差率         basis_pct  = basis / S_close
  年化基差率     basis_ann  = basis_pct * 365 / 剩余自然日(到最后交易日);到期日当天(0 天)记 NaN。
                 负值 = 贴水。对空头对冲者,持有到期的事前年化成本 = -basis_ann。
  最后交易日     已到期合约 = 该合约在交易所日行情中出现的最后一天(并用"合约月第三个周五、遇假顺延"规则核验);
                 2025-12-31 仍在交易的合约 = 中金所 2025-12-31 交易参数中的"最后交易日"(合约规格,上市即公布)。
  剩余交易日     td_left(t) = (t, 最后交易日] 内的交易日数(2026 年部分用工作日近似,只影响 2026 年到期合约,
                 它们在 2025-12-31 时 td_left >= 11,不触发任何换月阈值)。

  持仓规则(T 日收盘决定 T->T+1 持有哪张合约,换月在 T 日收盘完成):
    m1   当月:  td_left > R 的合约中到期最近的一张(R 天前换到下月)
    m2   次月:  已挂牌合约中到期第二近的一张;当月合约到期后第一个交易日收盘换月(R 不适用;到期日当天新合约尚未挂牌)
    q1   当季:  td_left > R 的季月(3/6/9/12)合约中最近的一张(季月合约到期前 R 天换到下一季)
    q2   下季:  td_left > R 的季月合约中第二近的一张
    main 主力:  td_left > R 的合约中前一交易日持仓量最大者(只用 T-1 已知持仓量)
  默认 R = 2(在最后交易日前第 2 个交易日收盘换月,避开最后两天);敏感性 R = 0 / 5。

  对冲日序列(每单位期货名义本金,收益按前一日收盘计):
    fut_ret      = F_held(t) / F_held(t-1) - 1          (held = t-1 收盘决定的合约,同一张合约的收盘价)
    spot_ret     = S(t)/S(t-1) - 1                       (价格指数)
    tr_ret       = TR(t)/TR(t-1) - 1                     (全收益指数;缺失时 NaN)
    hedge_cost   = fut_ret - spot_ret                    (空头对冲相对"零成本做空价格指数"的拖累,正 = 成本)
    hedge_cost_tr= fut_ret - tr_ret = hedge_cost - div_ret(相对"零成本做空全收益指数",正 = 成本)
    rf_ret       = SHIBOR3M(t-1) × 自然日间隔 / 365
    hedge_cost_vs_cash = hedge_cost_tr + rf_ret          ("多全收益指数 + 空期货"这一合成现金相对真实现金的拖累;
                                                          无套利定价下为 0。正值 = 用期货对冲比"卖股票拿现金"贵多少,
                                                          不含被对冲那部分股票相对指数的超额)
    roll_tc      = 换月当日 2 × tc_side(平旧 + 开新;默认 tc_side = 1bp/边,含手续费与半个价差,假设值)
    short_ret    = -fut_ret - roll_tc                    (空头腿每单位名义本金的日收益,下游直接用)
  年化 = 日均 × 242(项目统一口径 strategy.TRADING_DAYS)。

  保证金/资金占用不在本模块:空头腿只需保证金(约 10%~40%,随时期变化,见 research/LH_FUTURES_BASIS.md),
  名义本金不占资金;下游做"部分对冲"时须自行扣除保证金占用的现金。

下游接口:
    from afactor import lh_basis as LB
    h = LB.load_hedge("IC", rule="q1", roll_days=2)          # DataFrame,index=date
    c = LB.hedge_cost_daily("IM", rule="m2")                 # Series: 日对冲成本(相对价格指数)
    LB.short_leg_ret("IC", rule="q1")                        # Series: 空头腿日收益(含换月成本)
    或直接读 parquet: cache/_tmp/lh_basis_hedge.parquet(长表: variety, rule, roll_days 三个键)
"""
from __future__ import annotations

import re
import zipfile
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TMP

CUTOFF = pd.Timestamp("2025-12-31")
RAW_DIR = TMP / "lh_basis_raw"
P_CONTRACTS = TMP / "lh_basis_contracts.parquet"
P_HEDGE = TMP / "lh_basis_hedge.parquet"
P_SPOT = TMP / "lh_basis_spot.parquet"
P_EXPIRY = TMP / "lh_basis_expiry.parquet"

VARIETIES = {"IF": "000300", "IC": "000905", "IM": "000852", "IH": "000016"}
TR_CODE = {"000300": "H00300", "000905": "H00905", "000852": "H00852", "000016": "H00016"}
MULT = {"IF": 300, "IH": 300, "IC": 200, "IM": 200}
RULES = ("m1", "m2", "q1", "q2", "main")
ROLL_DAYS = (0, 2, 5)
DEFAULT_ROLL = 2
TC_SIDE = 1e-4
ANN = 242

# 极端时期(标注用;日期为成交日,含首尾)
REGIMES = [
    ("2015股灾", "2015-06-15", "2015-09-06"),
    ("2015限仓期(严)", "2015-09-07", "2017-02-16"),
    ("限仓松绑1-2", "2017-02-17", "2018-12-02"),
    ("限仓松绑3", "2018-12-03", "2019-04-21"),
    ("2018熊市", "2018-01-29", "2019-01-03"),
    ("2020疫情冲击", "2020-01-23", "2020-03-31"),
    ("2024初小微盘危机", "2024-01-02", "2024-02-07"),
    ("2024-09政策急涨", "2024-09-24", "2024-10-08"),
]

_SYM = re.compile(r"^(IF|IC|IM|IH)(\d{2})(\d{2})$")
_COLS = {"合约代码": "symbol", "今开盘": "open", "最高价": "high", "最低价": "low", "成交量": "volume",
         "成交金额": "turnover", "持仓量": "oi", "今收盘": "close", "今结算": "settle", "前结算": "pre_settle"}


# --------------------------------------------------------------------------- 原始数据
def parse_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """读全部月度包 -> 合约日面板(只保留 IF/IC/IM/IH 期货,去掉期权与国债期货)。"""
    rows = []
    for zp in sorted(raw_dir.glob("[0-9]" * 6 + ".zip")):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                m = re.match(r"(\d{8})_1\.csv$", name)
                if not m:
                    continue
                txt = z.read(name).decode("gbk", errors="replace")
                d = pd.read_csv(StringIO(txt), dtype=str)
                d.columns = [c.strip() for c in d.columns]
                d = d[[c for c in _COLS if c in d.columns]].rename(columns=_COLS)
                d["symbol"] = d["symbol"].astype(str).str.strip()
                d = d[d["symbol"].str.match(_SYM)]
                d["date"] = pd.Timestamp(m.group(1))
                rows.append(d)
    p = pd.concat(rows, ignore_index=True)
    for c in ["open", "high", "low", "close", "settle", "pre_settle", "volume", "turnover", "oi"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p[p["date"] <= CUTOFF]
    p["variety"] = p["symbol"].str[:2]
    yy = p["symbol"].str[2:4].astype(int)
    mm = p["symbol"].str[4:6].astype(int)
    p["cmonth"] = (2000 + yy) * 100 + mm
    p = p.drop_duplicates(["date", "symbol"]).sort_values(["variety", "date", "symbol"])
    return p.reset_index(drop=True)


def third_friday_rule(cmonth: int, cal: pd.DatetimeIndex) -> pd.Timestamp:
    """合约月第三个周五;若非交易日,顺延到下一个交易日(只在 cal 覆盖范围内有效)。"""
    y, m = divmod(int(cmonth), 100)
    d = pd.Timestamp(y, m, 1)
    fridays = pd.date_range(d, d + pd.offsets.MonthEnd(0), freq="W-FRI")
    tf = fridays[2]
    if tf > cal.max():
        return tf
    return cal[cal.searchsorted(tf)]


def load_live_specs(raw_dir: Path = RAW_DIR) -> pd.Series:
    """2025-12-31 交易参数里的 合约代码 -> 最后交易日。"""
    f = raw_dir / "contract_info_20251231.parquet"
    if not f.exists():
        return pd.Series(dtype="datetime64[ns]")
    d = pd.read_parquet(f)
    d = d[d["合约代码"].str.match(_SYM)]
    return pd.to_datetime(d.set_index("合约代码")["最后交易日"])


def build_expiry(panel: pd.DataFrame, live: pd.Series) -> pd.DataFrame:
    cal = pd.DatetimeIndex(sorted(panel["date"].unique()))
    g = panel.groupby("symbol").agg(variety=("variety", "first"), cmonth=("cmonth", "first"),
                                    first=("date", "min"), last_obs=("date", "max"))
    g["rule"] = [third_friday_rule(cm, cal) for cm in g["cmonth"]]
    live_mask = g["last_obs"] >= cal.max()
    g["expiry"] = g["last_obs"]
    g["source"] = "observed_last_day"
    for s in g.index[live_mask]:
        if s in live.index:
            g.loc[s, "expiry"] = live[s]
            g.loc[s, "source"] = "cffex_spec_20251231"
        else:
            g.loc[s, "expiry"] = g.loc[s, "rule"]
            g.loc[s, "source"] = "third_friday_rule"
    g["rule_match"] = g["expiry"] == g["rule"]
    return g


# --------------------------------------------------------------------------- 现货
def load_spot() -> pd.DataFrame:
    """价格指数收盘 + 全收益指数收盘(宽表,列 = 指数代码 / H 代码)。"""
    if P_SPOT.exists():
        return pd.read_parquet(P_SPOT)
    from . import newdata
    px = newdata.index_daily()[list(VARIETIES.values())]
    px = px[px.index <= CUTOFF]
    out = px.copy()
    for code, h in TR_CODE.items():
        f = RAW_DIR / f"tr_{h}.parquet"
        if f.exists():
            s = pd.read_parquet(f)[h]
            s.index = pd.to_datetime(s.index)
            out[h] = s[s.index <= CUTOFF]
    f = RAW_DIR / "shibor_3m.parquet"
    if f.exists():
        r = pd.read_parquet(f)["shibor_3m"]
        r.index = pd.to_datetime(r.index)
        r = r[r.index <= CUTOFF]
        out["shibor_3m"] = r.reindex(out.index.union(r.index)).ffill().reindex(out.index)
    out.index.name = "date"
    out.to_parquet(P_SPOT)
    return out


# --------------------------------------------------------------------------- 合约日面板
def build_contracts(panel: pd.DataFrame, expiry: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    cal = pd.DatetimeIndex(sorted(panel["date"].unique()))
    p = panel.merge(expiry[["expiry"]], left_on="symbol", right_index=True, how="left")
    p["spot_code"] = p["variety"].map(VARIETIES)
    sp = spot[[x for x in VARIETIES.values() if x in spot.columns]].stack().rename("spot").reset_index()
    sp.columns = ["date", "spot_code", "spot"]
    p = p.merge(sp, on=["date", "spot_code"], how="left")
    p["dte_cal"] = (p["expiry"] - p["date"]).dt.days
    pos_t = cal.searchsorted(p["date"])
    in_cal = p["expiry"] <= cal.max()
    pos_e = np.where(in_cal, cal.searchsorted(p["expiry"].where(in_cal, cal.max())), len(cal) - 1)
    extra = np.where(in_cal, 0,
                     np.busday_count((cal.max() + pd.Timedelta(days=1)).date(),
                                     (p["expiry"] + pd.Timedelta(days=1)).values.astype("datetime64[D]")))
    p["td_left"] = (pos_e - pos_t) + extra
    p["basis"] = p["close"] - p["spot"]
    p["basis_pct"] = p["basis"] / p["spot"]
    p["basis_ann"] = np.where(p["dte_cal"] > 0, p["basis_pct"] * 365 / p["dte_cal"].clip(lower=1), np.nan)
    p["basis_settle_ann"] = np.where(p["dte_cal"] > 0,
                                     (p["settle"] / p["spot"] - 1) * 365 / p["dte_cal"].clip(lower=1), np.nan)
    p["is_quarterly"] = (p["cmonth"] % 100) % 3 == 0
    return p.sort_values(["variety", "date", "expiry"]).reset_index(drop=True)


# --------------------------------------------------------------------------- 持仓规则
def select_held(c: pd.DataFrame, variety: str, rule: str, roll_days: int = DEFAULT_ROLL) -> pd.Series:
    """T 日收盘决定的持有合约(index=date)。只用 T 日已知信息(主力用 T-1 持仓量)。"""
    v = c[(c["variety"] == variety) & c["close"].notna()].copy()
    if rule == "main":
        v = v.sort_values(["symbol", "date"])
        v["oi_prev"] = v.groupby("symbol")["oi"].shift(1)
        v["oi_prev"] = v["oi_prev"].fillna(-1)               # 新上市合约首日没有前一日持仓
    out = {}
    for dt, g in v.groupby("date", sort=True):
        g = g.sort_values("expiry")
        if rule == "m2":
            # 不按 td_left 过滤:到期日当天下下月合约尚未挂牌,过滤会让 m2 临时跳到远季合约再跳回。
            # 到期日当天 2nd = 下月;次日新合约挂牌后 2nd = 新的下月 -> 换月发生在当月合约到期后第一个交易日收盘。
            out[dt] = g["symbol"].iloc[1] if len(g) > 1 else None
            continue
        e = g[g["td_left"] > roll_days]
        if rule == "m1":
            pick = e
        elif rule in ("q1", "q2"):
            pick = e[e["is_quarterly"]]
        elif rule == "main":
            pick = e.sort_values(["oi_prev", "expiry"], ascending=[False, True])
        else:
            raise ValueError(rule)
        k = 1 if rule == "q2" else 0
        out[dt] = pick["symbol"].iloc[k] if len(pick) > k else None
    return pd.Series(out, name="held").sort_index()


def build_hedge(c: pd.DataFrame, spot: pd.DataFrame, variety: str, rule: str,
                roll_days: int = DEFAULT_ROLL, tc_side: float = TC_SIDE) -> pd.DataFrame:
    held = select_held(c, variety, rule, roll_days)
    v = c[c["variety"] == variety]
    key = list(zip(v["date"], v["symbol"]))
    cd = dict(zip(key, v["close"]))
    bd = dict(zip(key, v["basis_ann"]))
    dd = dict(zip(key, v["dte_cal"]))
    dates = held.index
    code = VARIETIES[variety]
    s = spot[code].reindex(dates)
    df = pd.DataFrame(index=dates)
    df["held"] = held
    df["held_prev"] = held.shift(1)
    # fut_ret(t) 用 t-1 收盘决定的合约:同一张合约 t 与 t-1 的收盘
    prev_dates = [None] + list(dates[:-1])
    f_now = np.array([cd.get((d, h), np.nan) for d, h in zip(dates, df["held_prev"])], dtype=float)
    f_prev = np.array([cd.get((d0, h), np.nan) for d0, h in zip(prev_dates, df["held_prev"])], dtype=float)
    df["fut_ret"] = f_now / f_prev - 1
    df["spot_ret"] = s / s.shift(1) - 1
    h = TR_CODE[code]
    if h in spot.columns:
        tr = spot[h].reindex(dates)
        df["tr_ret"] = tr / tr.shift(1) - 1
    else:
        df["tr_ret"] = np.nan
    df["div_ret"] = df["tr_ret"] - df["spot_ret"]
    if "shibor_3m" in spot.columns:                    # 前一交易日的 SHIBOR 3M,按自然日 ACT/365 计息
        r = spot["shibor_3m"].reindex(dates)
        gap = dates.to_series().diff().dt.days
        df["rf_ret"] = (r.shift(1) / 100 * gap / 365).values
    else:
        df["rf_ret"] = np.nan
    df["hedge_cost"] = df["fut_ret"] - df["spot_ret"]
    df["hedge_cost_tr"] = df["fut_ret"] - df["tr_ret"]
    # "合成现金"(多全收益指数 + 空期货)相对真实现金(SHIBOR 3M)的拖累;按无套利定价应为 0
    df["hedge_cost_vs_cash"] = df["hedge_cost_tr"] + df["rf_ret"]
    df["roll"] = (df["held"] != df["held_prev"]) & df["held_prev"].notna() & df["held"].notna()
    df["roll_tc"] = df["roll"].astype(float) * 2 * tc_side
    df["short_ret"] = -df["fut_ret"] - df["roll_tc"]
    df["basis_ann_held"] = [bd.get((d, x), np.nan) for d, x in zip(dates, df["held"])]
    df["dte_held"] = [dd.get((d, x), np.nan) for d, x in zip(dates, df["held"])]
    df["hedge_cost_ann_exante"] = -df["basis_ann_held"]
    df["regime"] = regime_label(dates)
    df["variety"], df["rule"], df["roll_days"] = variety, rule, roll_days
    df.index.name = "date"
    return df


def regime_label(dates: pd.DatetimeIndex) -> pd.Series:
    lab = pd.Series("", index=dates)
    for name, a, b in REGIMES:
        m = (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
        lab[m] = np.where(lab[m] == "", name, lab[m] + "|" + name)
    return lab.values


# --------------------------------------------------------------------------- 下游接口
def load_contracts() -> pd.DataFrame:
    return pd.read_parquet(P_CONTRACTS)


def load_hedge(variety: str = "IC", rule: str = "q1", roll_days: int = DEFAULT_ROLL) -> pd.DataFrame:
    """单品种单规则的对冲日序列(index=date)。品种上市前没有行(下游需自行决定上市前怎么处理)。"""
    h = pd.read_parquet(P_HEDGE, filters=[("variety", "==", variety), ("rule", "==", rule),
                                          ("roll_days", "==", roll_days)])
    return h.set_index("date").sort_index()


def hedge_cost_daily(variety: str = "IC", rule: str = "q1", roll_days: int = DEFAULT_ROLL,
                     vs: str = "price", include_roll_tc: bool = True) -> pd.Series:
    """日对冲成本(每单位期货名义本金,正 = 成本)。vs='price' 相对价格指数,'tr' 相对全收益指数,
    'cash' 相对"卖出指数持有现金"(= tr 口径 + SHIBOR 3M)。"""
    h = load_hedge(variety, rule, roll_days)
    s = h[{"price": "hedge_cost", "tr": "hedge_cost_tr", "cash": "hedge_cost_vs_cash"}[vs]]
    if include_roll_tc:
        s = s + h["roll_tc"]
    return s.rename(f"hedge_cost_{variety}_{rule}_R{roll_days}_{vs}")


def hedge_cost_ann(variety: str = "IC", rule: str = "q1", roll_days: int = DEFAULT_ROLL,
                   window: int = 242, vs: str = "price") -> pd.Series:
    """滚动 window 日已实现年化对冲成本(日均 × 242)。前 window 日不足时 NaN(min_periods=window//2)。"""
    s = hedge_cost_daily(variety, rule, roll_days, vs)
    return (s.rolling(window, min_periods=window // 2).mean() * ANN).rename(s.name + "_ann")


def short_leg_ret(variety: str = "IC", rule: str = "q1", roll_days: int = DEFAULT_ROLL) -> pd.Series:
    """空头腿日收益(每单位期货名义本金,含换月成本)。不含保证金现金利息。"""
    return load_hedge(variety, rule, roll_days)["short_ret"].rename(f"short_{variety}_{rule}")
