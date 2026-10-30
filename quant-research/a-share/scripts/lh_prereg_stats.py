"""长历史预注册(LONG_HISTORY_PREREG.md)用到的全部"非策略收益"统计,前缀 lh_prereg。

纪律:本脚本**不读取任何 2005-2019 的个股价格**,不计算任何策略收益。它只做四件事:
  1. 取宽基指数日线价格(baostock,带超时与最大行数保护),核实熊/牛市窗口的高低点;
  2. 用 2020-2025 研究期 cache/daily 的 amount / close_px 是否存在,计算 V1000 池(ADV20 >= 2000 万)
     的日均覆盖率 —— 长历史股票池的 ADV 分位门槛由它换算(不涉及收益);
  3. 同一面板上计算开盘 10 分钟成交额占全日成交额的比例(长历史参与率近似用,不涉及收益);
  4. 用 2021-2025(已揭开的研究期)COMP_v5 排雷日收益对各宽基做 beta / R²,预选对冲工具。

产出(research/ledger/):
  lh_prereg_windows.csv        各指数在各熊/牛市搜索区间内的收盘高低点与涨跌
  lh_prereg_given_windows.csv  任务书原窗口在各指数上的涨跌(核对用)
  lh_prereg_pool_coverage.csv  覆盖率与开盘 10 分钟成交额占比(全期 / 分年)
  lh_prereg_hedge_r2.csv       对冲工具预选的 beta / R²(开盘对开盘,按引擎实现日对齐)
  原始指数价格落盘 cache/_tmp/lh_prereg_index_prices.csv

用法:python3 scripts/lh_prereg_stats.py [--skip-fetch]
"""
from __future__ import annotations
import argparse
import signal
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
LEDGER = ROOT / "research" / "ledger"
TMP = ROOT / "cache" / "_tmp"
IDX_FILE = TMP / "lh_prereg_index_prices.csv"
INDEX_CODES = ["sh.000001", "sz.399317", "sh.000300", "sh.000905", "sh.000852"]
MAX_ROWS = 10000           # 2005-2025 约 5100 个交易日,超过即视为 rs.next() 空转
END = "2025-12-31"         # 纪律 1:只用 <= 2025-12-31


class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


def fetch_index_prices() -> pd.DataFrame:
    import baostock as bs
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(30)
    lg = bs.login()
    signal.alarm(0)
    if lg.error_code != "0":
        raise RuntimeError(lg.error_msg)
    out, fails = [], []
    try:
        for code in INDEX_CODES:
            try:
                signal.alarm(90)
                r = bs.query_history_k_data_plus(code, "date,open,close", start_date="2005-01-01",
                                                 end_date=END, frequency="d")
                rows = []
                while r.error_code == "0" and r.next():
                    rows.append(r.get_row_data())
                    if len(rows) > MAX_ROWS:
                        raise RuntimeError(f"{code} 行数超过 {MAX_ROWS},疑似空转")
                signal.alarm(0)
                if not rows:
                    fails.append((code, r.error_msg or "empty"))
                    continue
                d = pd.DataFrame(rows, columns=["date", "open", "close"])
                d["code"] = code
                out.append(d)
                print(f"  {code}: {len(d)} 行 {d['date'].iloc[0]} ~ {d['date'].iloc[-1]}", flush=True)
            except _Timeout:
                fails.append((code, "timeout"))
            finally:
                signal.alarm(0)
    finally:
        bs.logout()
    if fails:
        print("  失败:", fails, flush=True)
    df = pd.concat(out, ignore_index=True)
    df = df[df["date"] <= END]
    TMP.mkdir(parents=True, exist_ok=True)
    df.to_csv(IDX_FILE, index=False)
    return df


def load_index_panel(skip_fetch: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(IDX_FILE, dtype={"code": str}) if (skip_fetch and IDX_FILE.exists()) else fetch_index_prices()
    raw["date"] = pd.to_datetime(raw["date"])
    close = raw.pivot(index="date", columns="code", values="close").astype(float)
    opn = raw.pivot(index="date", columns="code", values="open").astype(float)
    # 中证1000:baostock 缺 2011 全年,收盘用项目冻结快照(newdata.index_daily,ETF 项目来源)补齐
    from afactor import newdata as ND
    snap = ND.index_daily()
    close["sh.000852"] = snap["000852"].reindex(close.index).combine_first(close.get("sh.000852"))
    names = {"sh.000001": "上证综指", "sz.399317": "国证A股", "sh.000300": "沪深300",
             "sh.000905": "中证500", "sh.000852": "中证1000"}
    return close.loc[:END].rename(columns=names), opn.loc[:END].rename(columns=names)


def windows(close: pd.DataFrame) -> None:
    bears = {"BEAR08": ("2007-06-01", "2008-01-31", "2008-12-31"),
             "BEAR15": ("2015-04-01", "2015-07-31", "2016-03-31"),
             "BEAR18": ("2017-10-01", "2018-03-31", "2019-02-28"),
             "BEAR22": ("2021-07-01", "2022-01-31", "2022-12-31"),
             "BEAR24": ("2023-05-01", "2023-09-30", "2024-03-31")}
    bulls = {"BULL06": ("2005-01-01", "2006-01-31", "2007-12-31"),
             "BULL14": ("2013-06-01", "2014-07-31", "2015-07-31"),
             "BULL19": ("2018-12-01", "2019-01-31", "2019-06-30")}
    rows = []
    for ep, (a, b, c) in bears.items():
        for col in close.columns:
            s = close[col].dropna()
            pk = s.loc[a:b].idxmax()
            tr = s.loc[pk:c].idxmin()
            rows.append(dict(episode=ep, kind="bear", index=col, search=f"峰值∈[{a},{b}],谷底∈[峰值,{c}]",
                             start=pk.date(), end=tr.date(), chg=s[tr] / s[pk] - 1,
                             n_days=int(len(s.loc[pk:tr]))))
    for ep, (a, b, c) in bulls.items():
        for col in close.columns:
            s = close[col].dropna()
            tr = s.loc[a:b].idxmin()
            pk = s.loc[tr:c].idxmax()
            rows.append(dict(episode=ep, kind="bull", index=col, search=f"谷底∈[{a},{b}],峰值∈[谷底,{c}]",
                             start=tr.date(), end=pk.date(), chg=s[pk] / s[tr] - 1,
                             n_days=int(len(s.loc[tr:pk]))))
    w = pd.DataFrame(rows)
    w.to_csv(LEDGER / "lh_prereg_windows.csv", index=False)
    print(w.to_string())

    fixed = {  # 预注册固定窗口(主口径 = 上证综指收盘极值;BULL06 起点 = 评价起点;BULL14 起点 = 任务书突破日)
        "A:BEAR08": ("2007-10-16", "2008-11-04"), "A:BEAR15": ("2015-06-12", "2016-01-28"),
        "A:BEAR18": ("2018-01-24", "2019-01-03"), "A:BEAR22": ("2021-09-13", "2022-04-26"),
        "A:BEAR24": ("2023-05-08", "2024-02-05"),
        "A:BULL06": ("2006-01-04", "2007-10-16"), "A:BULL14": ("2014-07-01", "2015-06-12"),
        "A:BULL19": ("2019-01-03", "2019-04-19"),
        "B:BEAR08": ("2008-01-15", "2008-11-04"), "B:BEAR15": ("2015-06-12", "2015-09-15"),
        "B:BEAR18": ("2017-10-10", "2018-10-18"), "B:BEAR22": ("2021-12-01", "2022-04-26"),
        "B:BEAR24": ("2023-05-04", "2024-02-05"),
        "task:BEAR22": ("2021-12-13", "2022-10-31"), "task:BEAR24": ("2023-07-31", "2024-02-05"),
    }
    rows = []
    for k, (a, b) in fixed.items():
        r = dict(window=k, start=a, end=b)
        for col in close.columns:
            s = close[col].dropna()
            r[col] = s.loc[:b].iloc[-1] / s.loc[:a].iloc[-1] - 1
        rows.append(r)
    g = pd.DataFrame(rows)
    g.to_csv(LEDGER / "lh_prereg_given_windows.csv", index=False)
    print(g.round(4).to_string())

    # 正文引用的上证关键点位(核对任务书窗口用)
    s = close["上证综指"].dropna()
    kp = [("2014 年收盘最低", s.loc["2014"].idxmin()), ("任务书 BULL14 起点", pd.Timestamp("2014-07-01")),
          ("2022 年 4 月谷底", pd.Timestamp("2022-04-26")), ("2022 年 10 月谷底", pd.Timestamp("2022-10-31")),
          ("2023 年收盘最高", s.loc["2023"].idxmax()), ("任务书 BEAR24 起点", pd.Timestamp("2023-07-31")),
          ("2021 年收盘最高", s.loc["2021"].idxmax()), ("国证A股峰值日", pd.Timestamp("2021-12-13"))]
    k = pd.DataFrame([dict(item=a, date=d.date(), sse_close=float(s.loc[d])) for a, d in kp])
    k.to_csv(LEDGER / "lh_prereg_index_keypoints.csv", index=False)
    print(k.to_string())


def coverage() -> None:
    D = ROOT / "cache" / "daily"
    amt = pd.read_parquet(D / "amount.parquet").astype(np.float64).loc[:END]
    amt = amt[[c for c in amt.columns if not c.startswith(("92", "8", "4"))]]   # 长历史只有沪深
    cls = pd.read_parquet(D / "close_px.parquet").reindex_like(amt)            # 只用"是否有收盘价"
    a10 = pd.read_parquet(D / "amt_open10.parquet").astype(np.float64).reindex_like(amt)
    valid = (amt > 0) & cls.notna()
    dl = valid.cumsum().where(valid)
    adv20 = amt.rolling(20, min_periods=10).mean()
    elig = valid & (dl >= 6) & adv20.notna()
    pool = elig & (adv20 >= 2e7)
    n_el, n_pool = elig.sum(1), pool.sum(1)
    cov = (n_pool / n_el).where(n_el >= 100).loc["2020-02-03":]
    share = (a10 / amt).where(pool & (amt > 0) & (a10 >= 0))
    rows = [dict(scope="2020-02-03~2025-12-31", n_days=int(cov.notna().sum()), cov_mean=cov.mean(),
                 cov_median=cov.median(), cov_p10=cov.quantile(.1), cov_p90=cov.quantile(.9),
                 n_elig_mean=n_el.loc[cov.index].mean(), n_pool_mean=n_pool.loc[cov.index].mean(),
                 open10_share_median=float(np.nanmedian(share.loc[cov.index].to_numpy())),
                 open10_share_p10=float(np.nanquantile(share.loc[cov.index].to_numpy(), .1)))]
    for y, cy in cov.groupby(cov.index.year):
        sy = share.loc[cy.index].to_numpy()
        rows.append(dict(scope=str(y), n_days=int(cy.notna().sum()), cov_mean=cy.mean(), cov_median=cy.median(),
                         cov_p10=cy.quantile(.1), cov_p90=cy.quantile(.9),
                         n_elig_mean=n_el.loc[cy.index].mean(), n_pool_mean=n_pool.loc[cy.index].mean(),
                         open10_share_median=float(np.nanmedian(sy)), open10_share_p10=float(np.nanquantile(sy, .1))))
    c = pd.DataFrame(rows)
    c.to_csv(LEDGER / "lh_prereg_pool_coverage.csv", index=False)
    print(c.round(4).to_string())


def hedge_r2(opn: pd.DataFrame) -> None:
    f = pd.read_parquet(LEDGER / "freeze_ref_daily.parquet")       # 2021-2025 研究期,已揭开
    ser = {"T1_excl|Y1|strategy": "COMP_v5排雷(Y=1)", "C|bench_pool5": "同频池等权"}
    rows = []
    for sc, sname in ser.items():
        y = f[sc].dropna()
        for col in opn.columns:
            o = opn[col]
            g = o.shift(-2) / o.shift(-1) - 1.0                     # 行 t = open[t+1] -> open[t+2]
            cors = {}
            for k in range(-2, 3):
                x = g.shift(-k).reindex(y.index)
                m = x.notna() & y.notna()
                cors[k] = float(np.corrcoef(y[m], x[m])[0, 1])
            x = g.reindex(y.index)
            m = x.notna() & y.notna()
            cc = float(np.corrcoef(y[m], x[m])[0, 1])
            b = float(np.cov(y[m], x[m])[0, 1] / np.var(x[m], ddof=1))
            rows.append(dict(series=sname, index=col, window="2021-2025", corr=cc, r2=cc ** 2, beta=b,
                             corr_peak_shift=max(cors, key=cors.get), n=int(m.sum())))
    h = pd.DataFrame(rows)
    h.to_csv(LEDGER / "lh_prereg_hedge_r2.csv", index=False)
    print(h.round(4).to_string())


def data_status() -> None:
    """长历史数据的现状(只看代码清单、文件名与上市/退市日期,不读任何价格)。"""
    H = ROOT / "cache" / "history_daily"
    rows = []
    uni = pd.read_csv(H / "universe.csv", dtype=str)
    pre = uni["code"].str[:5]
    is_idx = pre.isin(["sh.00", "sz.39"])
    rows.append(dict(item="universe.csv 代码总数", value=len(uni), note="2005-2019 每年末 query_all_stock 并集"))
    rows.append(dict(item="universe.csv 其中指数代码(sh.000*/sz.399*)", value=int(is_idx.sum()),
                     note="fetch_daily_history.py 的过滤 code[3:4] in (0,3,6) 放进了指数"))
    rows.append(dict(item="universe.csv 其中股票代码", value=int((~is_idx).sum()), note=""))
    parts = sorted(p.name for p in (H / "parts").glob("*.csv"))
    n_idx_parts = sum(1 for p in parts if p.startswith(("sh.000", "sz.399")))
    rows.append(dict(item="parts/ 已抓文件数", value=len(parts), note="写本文时的快照"))
    rows.append(dict(item="parts/ 其中指数文件", value=n_idx_parts, note=""))
    sb = pd.read_csv(ROOT / "data_8:21" / "stock_basic.csv", dtype=str)
    dd = pd.to_datetime(sb["delist_date"], format="%Y%m%d", errors="coerce")
    for y in range(2005, 2026):
        rows.append(dict(item=f"stock_basic 退市数 {y}", value=int((dd.dt.year == y).sum()), note="delist_date 年份"))
    rows.append(dict(item="stock_basic 退市数 2005-2019 合计", value=int(dd.dt.year.between(2005, 2019).sum()), note=""))
    rows.append(dict(item="stock_basic 退市数 2020-2025 合计", value=int(dd.dt.year.between(2020, 2025).sum()), note=""))
    s = pd.DataFrame(rows)
    s.to_csv(LEDGER / "lh_prereg_data_status.csv", index=False)
    print(s.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true", help="复用 cache/_tmp 里已取的指数价格")
    a = ap.parse_args()
    close, opn = load_index_panel(a.skip_fetch)
    windows(close)
    coverage()
    hedge_r2(opn)
    data_status()


if __name__ == "__main__":
    main()
