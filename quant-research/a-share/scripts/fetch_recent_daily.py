"""补 2026-08-18 起的全 A 日线(我们的面板止于 2026-08-17,而交易者数据到 08-27)。
只抓少量天数,append-only 写到 cache/recent_daily/,不碰 cache/sealed。"""
from __future__ import annotations
import os, sys, time, json, argparse
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
import baostock as bs

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cache" / "recent_daily"; PARTS = OUT / "parts"
START, END = "2026-08-18", "2026-09-05"
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST"

def _init():
    lg = bs.login()
    if lg.error_code != "0": raise RuntimeError(lg.error_msg)

def rows(res):
    out = []
    while (res.error_code == "0") and res.next(): out.append(res.get_row_data())
    return out

def fetch_one(code):
    part = PARTS / f"{code}.csv"
    if part.exists(): return code, -1, "skip"
    try:
        r = bs.query_history_k_data_plus(code, FIELDS, start_date=START, end_date=END,
                                          frequency="d", adjustflag="3")   # 不复权(近期无除权影响,且与持仓成本可比)
        if r.error_code != "0": return code, 0, r.error_msg
        a = rows(r)
        if not a: part.write_text(""); return code, 0, "empty"
        pd.DataFrame(a, columns=r.fields).to_csv(part, index=False)
        return code, len(a), "ok"
    except Exception as e:
        return code, 0, f"{type(e).__name__}: {e}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    # 股票清单:用我们面板的股票代码(含所有在市股票)
    codes = [l.strip() for l in open(ROOT / "cache" / "recent_daily" / "codes.txt")]
    todo = [c for c in codes if not (PARTS / f"{c}.csv").exists()]
    print(f"股票 {len(codes)} 只,待抓 {len(todo)},{a.workers} 进程,窗口 {START}~{END}", flush=True)
    t0 = time.time(); ok = fail = 0; errs = []
    with Pool(a.workers, initializer=_init) as pool:
        for i, (c, n, msg) in enumerate(pool.imap_unordered(fetch_one, todo, chunksize=8), 1):
            if msg in ("ok", "empty"): ok += 1
            elif msg != "skip": fail += 1; errs.append((c, msg))
            if i % 500 == 0: print(f"  {i}/{len(todo)} ok {ok} fail {fail} {time.time()-t0:.0f}s", flush=True)
    (OUT / "errors.json").write_text(json.dumps(errs, ensure_ascii=False, indent=1))
    print(f"完成 ok {ok} fail {fail} 耗时 {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
