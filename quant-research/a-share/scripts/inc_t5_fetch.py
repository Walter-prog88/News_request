"""T5 数据抓取(前缀 inc_t5;预注册 research/INC_T5_PREREG.md §12 步骤 1)。可断点续抓,全部落盘到 cache/_tmp/inc_t5_fetch/。

子命令:
  em_div     akshare stock_fhps_em,报告期 20050331..20250930(每年 4 个),每个报告期一个 parquet
  bs_div     baostock query_dividend_data:293 只退市股在市年份(yearType=report,按预案公告年份),
             另抽 40 只幸存股 x 2016-2024 作 D3(b) 两源核对
  em_fin     C 轨:akshare stock_lrb_em / stock_zcfz_em / stock_yjbb_em,报告期 20060331..20161231
所有请求带超时(signal.alarm)+ 3 次重试;只读 <= 2025-12-31 的信息(公告日晚于截止日的行在读取层丢弃)。
日志 logs/inc_t5_fetch.log。
用法: python3 scripts/inc_t5_fetch.py em_div bs_div em_fin
"""
from __future__ import annotations

import signal
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

OUT = ROOT / "cache" / "_tmp" / "inc_t5_fetch"
LOG = ROOT / "logs" / "inc_t5_fetch.log"
OUT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(exist_ok=True)


def log(*a):
    s = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")


class _TO(Exception):
    pass


def _alarm(sig, frm):
    raise _TO()


def with_timeout(fn, sec: int, *a, **kw):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(sec)
    try:
        return fn(*a, **kw)
    finally:
        signal.alarm(0)


def retry(fn, sec=90, tries=3, sleep=2.0, *a, **kw):
    last = None
    for k in range(tries):
        try:
            return with_timeout(fn, sec, *a, **kw)
        except Exception as e:      # noqa: BLE001
            last = e
            time.sleep(sleep * (k + 1))
    raise RuntimeError(f"failed after {tries}: {type(last).__name__} {last}")


def periods(y0: int, y1: int, last: str | None = None) -> list[str]:
    out = []
    for y in range(y0, y1 + 1):
        for md in ("0331", "0630", "0930", "1231"):
            p = f"{y}{md}"
            if last is not None and p > last:
                continue
            out.append(p)
    return out


# ---------------------------------------------------------------- EM 分红
def cmd_em_div():
    import akshare as ak
    d = OUT / "em_div"
    d.mkdir(exist_ok=True)
    ps = periods(2005, 2025, "20250930")
    ok = fail = 0
    for p in ps:
        f = d / f"{p}.parquet"
        if f.exists():
            ok += 1
            continue
        t0 = time.time()
        try:
            df = retry(ak.stock_fhps_em, 120, 3, 3.0, date=p)
            df.to_parquet(f)
            ok += 1
            log(f"[em_div] {p} rows={len(df)} {time.time() - t0:.0f}s")
        except Exception as e:      # noqa: BLE001
            if "NoneType" in str(e):
                # akshare 对"该报告期没有任何分红预案"的空响应会抛 TypeError('NoneType' ...):记为空表(已逐期核对)
                pd.DataFrame().to_parquet(f)
                ok += 1
                log(f"[em_div] {p} EMPTY (TypeError NoneType -> 该报告期无预案)")
            else:
                fail += 1
                log(f"[em_div] {p} FAILED {e}")
        time.sleep(1.0)
    log(f"[em_div] done ok={ok} fail={fail} of {len(ps)}")


# ---------------------------------------------------------------- baostock 分红
BS_FIELDS = ["code", "dividPreNoticeDate", "dividAgmPumDate", "dividPlanAnnounceDate", "dividPlanDate",
             "dividRegistDate", "dividOperateDate", "dividPayDate", "dividStockMarketDate",
             "dividCashPsBeforeTax", "dividCashPsAfterTax", "dividStocksPs", "dividCashStock",
             "dividReserveToStockPs"]


def _bs_query(bs, code, year):
    rs = bs.query_dividend_data(code=code, year=str(year), yearType="report")
    if rs.error_code != "0":
        raise RuntimeError(f"bs error {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def cmd_bs_div():
    import baostock as bs
    from afactor.lh_panel import LongPanel
    p = LongPanel()
    info = p.code_info
    out = info["outDate"].fillna("").astype(str)
    is_del = ((out != "") & (out <= "2025-12-31")).to_numpy()
    rng = np.random.default_rng(20261011)
    surv = np.flatnonzero(~is_del & (pd.to_datetime(info["ipoDate"]) <= pd.Timestamp("2015-01-01")).to_numpy())
    samp = rng.choice(surv, 40, replace=False)
    jobs = []
    for j in np.flatnonzero(is_del):
        y0 = max(2005, int(str(info["ipoDate"].iloc[j])[:4]) if str(info["ipoDate"].iloc[j]) not in ("", "nan") else 2005)
        y1 = int(out.iloc[j][:4])
        jobs.append(("del", info["bs_code"].iloc[j], info.index[j], y0, y1))
    for j in samp:
        jobs.append(("surv", info["bs_code"].iloc[j], info.index[j], 2016, 2024))
    f_done = OUT / "bs_div_done.txt"
    done = set(f_done.read_text().split("\n")) if f_done.exists() else set()
    fres = OUT / "bs_div_rows.csv"
    lg = bs.login()
    log("[bs_div] login", lg.error_code, lg.error_msg, f"jobs={len(jobs)} (del={int(is_del.sum())}, surv={len(samp)})")
    n_req = n_fail = 0
    t0 = time.time()
    first = not fres.exists()
    for kind, bsc, code, y0, y1 in jobs:
        for y in range(y0, y1 + 1):
            key = f"{kind}|{code}|{y}"
            if key in done:
                continue
            try:
                df = retry(lambda: _bs_query(bs, bsc, y), 40, 3, 1.0)
                n_req += 1
                if len(df):
                    df.insert(0, "kind", kind)
                    df.insert(1, "lp_code", code)
                    df.insert(2, "qyear", y)
                    df.to_csv(fres, mode="a", header=first, index=False)
                    first = False
                with open(f_done, "a") as fh:
                    fh.write(key + "\n")
            except Exception as e:      # noqa: BLE001
                n_fail += 1
                log(f"[bs_div] FAIL {key} {e}")
                try:
                    bs.logout()
                except Exception:       # noqa: BLE001
                    pass
                time.sleep(2)
                bs.login()
            if n_req % 200 == 0 and n_req:
                log(f"[bs_div] req={n_req} fail={n_fail} {time.time() - t0:.0f}s")
    bs.logout()
    log(f"[bs_div] done req={n_req} fail={n_fail} {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- EM 财报(C 轨)
def cmd_em_fin(y0: int = 2006, y1: int = 2016):
    import akshare as ak
    d = OUT / "em_fin"
    d.mkdir(exist_ok=True)
    fns = {"lrb": ak.stock_lrb_em, "zcfz": ak.stock_zcfz_em, "yjbb": ak.stock_yjbb_em}
    ps = periods(y0, y1)
    ok = fail = 0
    for p in ps:
        for nm, fn in fns.items():
            f = d / f"{nm}_{p}.parquet"
            if f.exists():
                ok += 1
                continue
            t0 = time.time()
            try:
                df = retry(fn, 150, 3, 3.0, date=p)
                df.to_parquet(f)
                ok += 1
                log(f"[em_fin] {nm} {p} rows={len(df)} {time.time() - t0:.0f}s")
            except Exception as e:      # noqa: BLE001
                fail += 1
                log(f"[em_fin] {nm} {p} FAILED {e}")
            time.sleep(1.0)
    log(f"[em_fin] done ok={ok} fail={fail} of {len(ps) * 3}")


def cmd_em_fin05():
    """TTM / 同比需要上一年同期:补抓 2005 年四个报告期(C 轨 2006 年报告用,可见日同样取法定截止日)。"""
    cmd_em_fin(2005, 2005)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        {"em_div": cmd_em_div, "bs_div": cmd_bs_div, "em_fin": cmd_em_fin, "em_fin05": cmd_em_fin05}[c]()
