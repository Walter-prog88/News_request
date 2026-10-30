"""神龙摆尾 · 信号与逐日触价模拟(近端全 A,2020-2026)。

规则完全沿用 befox/shenlong_review 已登记的口径,不因近端收益重新调参:
  背景 base : 近 22 日收盘振幅 ≤25%,且 22 日低点较其前 104 日高点跌 ≥20%
  首板      : 涨停 + base + 前 22 日无涨停 + 大实体阳线(实体 ≥ 前 20 日实体中位数 ×1.5)
  箱体      : [首板开盘, 首板收盘]
  整理      : 收盘跌破首板开盘则作废;至少 5 个交易日
  确认      : 收阳 且 收盘 > 首板收盘 且(主版)前一日收盘在箱内
  执行      : 次一交易日开盘买入;止损 = 首板开盘(盘中触价),止盈 = 实际买价 ×1.5,不设时间强平

价格一律在**后复权空间**(raw × 累计复权因子)里比较,除权除息不制造假跳空;
涨停识别与涨跌停可交易性判断另用不复权收盘 + 官方前收盘。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "research" / "shenlong" / "panel"


class Data:
    def __init__(self, drop_bj: bool = True):
        g = lambda n: pd.read_parquet(P / f"{n}.parquet")
        self.o, self.h, self.l, self.c = g("open"), g("high"), g("low"), g("close")
        self.amt = g("amount")
        self.pre, self.cumf = g("preclose"), g("cumf")
        self.st, self.lim = g("is_st"), g("limit_rate")
        self.turn, self.circ = g("turnover"), g("circ_mv")
        cols = list(self.c.columns)
        if drop_bj:
            import csv
            db = set(pd.read_csv(ROOT / "data_8:21" / "daily_basic.csv",
                                 usecols=["ts_code"], dtype=str).ts_code.unique())
            cols = [x for x in cols if not x.endswith(".BJ") and x in db]
        for a in ("o", "h", "l", "c", "amt", "pre", "cumf", "st", "lim", "turn", "circ"):
            setattr(self, a, getattr(self, a)[cols])
        self.dates = self.c.index
        self.cols = cols
        # 后复权空间(锚点无关,只用于比较与收益)
        f = self.cumf.ffill().bfill()
        f = f / f.iloc[0]
        self.O, self.H, self.L, self.C = (x.astype("float64") * f for x in (self.o, self.h, self.l, self.c))

    # ---------- 条件层 ----------
    def limit_up(self) -> pd.DataFrame:
        expect = np.floor(self.pre.astype("float64") * (1 + self.lim.astype("float64")) * 100 + 0.5) / 100
        tol = np.maximum(0.011, self.c.astype("float64") * 0.0005)
        ok = (self.c.astype("float64") - expect).abs() <= tol
        return ok & self.c.notna() & self.pre.notna() & ~self.st & ~self.st.shift(1, fill_value=False)

    def background(self) -> pd.DataFrame:
        c = self.C
        lo = c.shift(1).rolling(22, min_periods=22).min()
        hi = c.shift(1).rolling(22, min_periods=22).max()
        old = c.shift(23).rolling(104, min_periods=104).max()
        return (((hi / lo - 1) <= 0.25) & ((1 - lo / old) >= 0.20)).fillna(False)

    def big_body(self) -> pd.DataFrame:
        body = (self.C - self.O).abs()
        med = body.shift(1).rolling(20, min_periods=20).median()
        return ((self.C > self.O) & (body >= 1.5 * med)).fillna(False)

    def eligible(self, min_hist: int = 250) -> pd.DataFrame:
        have = self.C.notna() & self.O.notna() & self.H.notna() & self.L.notna()
        hist = have.cumsum() >= min_hist
        return have & hist & ~self.st

    def tradable_buy(self) -> pd.DataFrame:
        """开盘价已在涨停价上(±0.1%)则当日买不到。"""
        up = np.floor(self.pre.astype("float64") * (1 + self.lim.astype("float64")) * 100 + 0.5) / 100
        return ~(self.o.astype("float64") >= up * 0.999) & self.o.notna()

    def tradable_sell(self) -> pd.DataFrame:
        dn = np.ceil(self.pre.astype("float64") * (1 - self.lim.astype("float64")) * 100 - 0.5) / 100
        return ~(self.o.astype("float64") <= dn * 1.001) & self.o.notna()


# ---------- 信号:逐股状态机 ----------
def shenlong_signals(d: Data, *, return_box="close", require_volume=False,
                     min_consolidation=5, need_background=True) -> pd.DataFrame:
    limit = d.limit_up().to_numpy()
    base = d.background().to_numpy() if need_background else np.ones(d.C.shape, bool)
    big = d.big_body().to_numpy()
    elig = d.eligible().to_numpy()
    prior_limit = (pd.DataFrame(limit, index=d.dates, columns=d.cols)
                   .shift(1, fill_value=False).rolling(22, min_periods=22).sum().to_numpy() == 0)
    O, C, V = d.O.to_numpy(), d.C.to_numpy(), d.amt.to_numpy()
    valid = (d.C.notna() & d.O.notna() & d.H.notna() & d.L.notna()).to_numpy()
    n, m = C.shape
    rows = []
    for j in range(m):
        board, count = -1, 0
        Cj, Oj, Vj = C[:, j], O[:, j], V[:, j]
        for i in range(n):
            if not valid[i, j]:
                board = -1
                continue
            if board >= 0:
                floor, ceiling = Oj[board], Cj[board]
                if Cj[i] < floor:
                    board = -1
                else:
                    prior_close = floor <= Cj[i - 1] <= ceiling
                    prior_body = prior_close and floor <= Oj[i - 1] <= ceiling
                    conf = (Cj[i] > Oj[i]) and (Cj[i] > ceiling)
                    if require_volume:
                        conf = conf and (Vj[i] > Vj[i - 1])
                    if return_box == "close":
                        conf = conf and prior_close
                    elif return_box == "body":
                        conf = conf and prior_body
                    if count >= min_consolidation and conf:
                        rows.append((d.cols[j], d.dates[i], d.dates[board], floor, ceiling,
                                     Cj[i], count, bool(prior_close), bool(prior_body),
                                     bool(Vj[i] > Vj[i - 1])))
                        board = -1
                        continue
                    count += 1
            if board < 0 and limit[i, j] and big[i, j] and elig[i, j] and prior_limit[i, j] and (
                    base[i, j] or not need_background):
                board, count = i, 0
    return pd.DataFrame(rows, columns=["code", "signal_date", "board_date", "board_open",
                                       "board_close", "signal_close", "consolidation",
                                       "prior_close_in_box", "prior_body_in_box", "vol_up"])


# ---------- 逐日触价模拟 ----------
def simulate(d: Data, sig: pd.DataFrame, *, target_mult=1.5, cost=0.0015,
             limit_guard=True, stop_override=None) -> pd.DataFrame:
    """次日开盘进,盘中触价出;T+1(买入当日不卖),同日双触先止损。"""
    di = {t: k for k, t in enumerate(d.dates)}
    ci = {c: k for k, c in enumerate(d.cols)}
    O, H, L, C = (x.to_numpy() for x in (d.O, d.H, d.L, d.C))
    can_b, can_s = d.tradable_buy().to_numpy(), d.tradable_sell().to_numpy()
    valid = (d.C.notna() & d.O.notna()).to_numpy()
    n = len(d.dates)
    has_frac = "stop_frac" in sig.columns
    out = []
    for k, r in enumerate(sig.itertuples(index=False)):
        j = ci[r.code]
        s = di[r.signal_date]
        e = s + 1
        while e < n and not valid[e, j]:
            e += 1
        if e >= n:
            out.append((r.code, r.signal_date, None, np.nan, np.nan, np.nan, "no_session", np.nan, np.nan))
            continue
        if e != s + 1:                       # 停牌:原口径只试紧邻下一交易日
            out.append((r.code, r.signal_date, None, np.nan, np.nan, np.nan, "halted", np.nan, np.nan))
            continue
        if limit_guard and not can_b[e, j]:
            out.append((r.code, r.signal_date, None, np.nan, np.nan, np.nan, "gap_limit", np.nan, np.nan))
            continue
        entry = O[e, j]
        if has_frac:
            stop = entry * (1 - r.stop_frac)
        elif stop_override is None:
            stop = r.board_open
        else:
            stop = entry * (1 - stop_override)
        target = entry * target_mult
        if not np.isfinite(entry) or entry <= 0 or stop >= entry:
            out.append((r.code, r.signal_date, None, np.nan, np.nan, np.nan, "bad_entry", np.nan, np.nan))
            continue
        kind, px, xd = "open", np.nan, None
        pend = None
        for i in range(e, n):
            if not valid[i, j]:
                continue
            if pend is not None:             # 上一日触价但不可卖 → 次日开盘出
                if can_s[i, j] or not limit_guard:
                    kind, px, xd = pend, O[i, j], d.dates[i]
                    break
                continue
            if i == e:                        # T+1:买入当日不卖,记录意图
                if L[i, j] <= stop:
                    pend = "stop"
                elif H[i, j] >= target:
                    pend = "target"
                continue
            if L[i, j] <= stop:
                p = min(O[i, j], stop)
                if limit_guard and not can_s[i, j]:
                    pend = "stop"; continue
                kind, px, xd = "stop", p, d.dates[i]
                break
            if H[i, j] >= target:
                p = max(O[i, j], target)
                if limit_guard and not can_s[i, j]:
                    pend = "target"; continue
                kind, px, xd = "target", p, d.dates[i]
                break
        if xd is None:
            last = np.where(valid[e:, j])[0]
            xd, px, kind = d.dates[e + last[-1]], C[e + last[-1], j], "open_end"
        ret = (px * (1 - cost)) / (entry * (1 + cost)) - 1
        hold = di[xd] - e
        out.append((r.code, r.signal_date, xd, entry, stop, px, kind, ret, hold))
    t = pd.DataFrame(out, columns=["code", "signal_date", "exit_date", "entry", "stop",
                                   "exit_px", "exit_kind", "ret", "hold"])
    return pd.concat([sig.reset_index(drop=True), t.drop(columns=["code", "signal_date"])], axis=1)
