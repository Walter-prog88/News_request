"""补 2005-01 ~ 2019-12 全 A 日线(行业层状态判断用;README §10.7 / DATA_WISHLIST #1)。

来源 baostock(免费,2005 年起)。输出独立目录 cache/history_daily/,**不碰 data_8:21/daily_basic.csv**。
股票清单 = 每年最后一个交易日 query_all_stock 的并集 -> 含已退市股,部分消除幸存者偏差。
字段:后复权 OHLC(adjustflag=1,收益跨分红正确)+ 不复权 close + volume/amount/turn/tradestatus/isST。
断点续跑:每只股票一个 csv 分片,已存在即跳过。
"""
from __future__ import annotations
import os, sys, time, json, argparse
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
import baostock as bs

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cache" / "history_daily"
PARTS = OUT / "parts"
START, END = "2005-01-01", "2019-12-31"
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST"


def bs_login():
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(lg.error_msg)


def rows(res):
    out = []
    while (res.error_code == "0") and res.next():
        out.append(res.get_row_data())
    return out


def universe() -> pd.DataFrame:
    """每年末的全部 A 股并集(含后来退市的)。"""
    f = OUT / "universe.csv"
    if f.exists():
        return pd.read_csv(f, dtype=str)
    bs_login()
    recs = {}
    for y in range(2005, 2020):
        for d in (f"{y}-12-31", f"{y}-12-30", f"{y}-12-29", f"{y}-12-28"):
            r = bs.query_all_stock(day=d)
            rr = rows(r)
            if rr:
                for code, status, name in rr:
                    if code[:2] in ("sh", "sz") and code[3:4] in ("0", "3", "6"):
                        recs.setdefault(code, {"code": code, "name": name, "first_seen": d})
                print(f"  {d}: {len(rr)} 只,累计 {len(recs)}", flush=True)
                break
    bs.logout()
    df = pd.DataFrame(list(recs.values()))
    OUT.mkdir(parents=True, exist_ok=True); df.to_csv(f, index=False)
    return df


def _init():
    bs_login()


def fetch_one(code: str) -> tuple[str, int, str]:
    part = PARTS / f"{code}.csv"
    if part.exists():
        return code, -1, "skip"
    try:
        r = bs.query_history_k_data_plus(code, FIELDS, start_date=START, end_date=END,
                                         frequency="d", adjustflag="1")       # 后复权
        if r.error_code != "0":
            return code, 0, r.error_msg
        a = rows(r)
        r2 = bs.query_history_k_data_plus(code, "date,code,close", start_date=START, end_date=END,
                                          frequency="d", adjustflag="3")      # 不复权收盘(算市值用)
        b = rows(r2)
        if not a:
            part.write_text(""); return code, 0, "empty"
        da = pd.DataFrame(a, columns=r.fields)
        db = pd.DataFrame(b, columns=r2.fields).rename(columns={"close": "close_raw"})
        d = da.merge(db[["date", "close_raw"]], on="date", how="left")
        d.to_csv(part, index=False)
        return code, len(d), "ok"
    except Exception as e:
        return code, 0, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    uni = universe()
    codes = uni["code"].tolist()[: args.limit] if args.limit else uni["code"].tolist()
    todo = [c for c in codes if not (PARTS / f"{c}.csv").exists()]
    print(f"股票 {len(codes)} 只,待抓 {len(todo)} 只,{args.workers} 线程", flush=True)
    # baostock 连接**不是线程安全的**(多线程共用一个连接会"接收数据异常"),改多进程、每进程独立登录
    t0 = time.time(); ok = fail = 0; errs = []
    with Pool(args.workers, initializer=_init) as pool:
        for i, (code, n, msg) in enumerate(pool.imap_unordered(fetch_one, todo, chunksize=4), 1):
            if msg == "ok" or msg == "empty": ok += 1
            elif msg != "skip": fail += 1; errs.append((code, msg))
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}  ok {ok} fail {fail}  {time.time()-t0:.0f}s", flush=True)
    (OUT / "errors.json").write_text(json.dumps(errs, ensure_ascii=False, indent=1))
    print(f"完成:ok {ok} fail {fail},耗时 {time.time()-t0:.0f}s;失败清单 errors.json", flush=True)


if __name__ == "__main__":
    main()
