"""lh_data 第 1-2 步:2005-01-01 ~ 2025-12-31 沪深 A 股(含已退市)日线下载。不计算任何策略收益。

来源 baostock(免费,2005 年起,含已退市股)。输出 cache/long_daily/,**不碰** cache/history_daily/(旧抓取只到
2019、清单还混进了指数,统一重抓,不混用)。

股票清单(universe.csv)
  1. 2005-2025 每年最后一个交易日 query_all_stock 的并集(任务口径);
  2. 额外加 2004 年最后一个交易日的清单 —— 2005 年内退市的股票不在 2005 年末清单里(覆盖扩展,报告里单列);
  3. 再与 baostock query_stock_basic(type=1 股票,含 outDate)交叉:上市与退市落在两个年末之间的股票
     (年末快照抓不到)也补进来,source 列注明来源。
  只保留沪深 A 股普通股:sh.60* / sh.68* / sz.00* / sz.30*(剔除指数 sh.000/sz.399、B 股 sh.900/sz.200、北交所)。

下载(每只股票两次请求)
  后复权 adjustflag=1: date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST
  不复权 adjustflag=3: date,close  -> 合并为 close_raw
  存 cache/long_daily/parts/<code>.csv.gz(原样字符串,先写临时文件再改名,存在即跳过 = 可续跑);
  无数据的股票写 <code>.empty 标记。

健壮性(baostock 客户端的两个已知坑)
  - socketutil.send_msg 的 `while True: recv` 在服务端断连时 recv 返回 b'' 会 100% CPU 空转 -> 每个请求
    signal.alarm(REQ_TIMEOUT) 超时;超时/异常后丢弃旧连接、重新 login(残留字节会让后续请求错位)。
  - 读取循环必须调用 get_row_data(),否则 next() 永远为 True;另加最大行数保护 MAX_ROWS。
  - 多进程(baostock 连接不是线程安全的),每进程单独 login;失败写 errors.json 与日志,多轮重试,可续跑。

用法: python3 scripts/lh_fetch_daily.py [--workers 6] [--passes 4] [--limit N] [--universe-only]
            [--only-delisted] [--codes-file F]
  提速方案(2026-10-10 主控实测):baostock 按 IP 限速(单股全历史约 27 秒,加进程总吞吐不变),
  所以 baostock 只抓**已退市**股票(--only-delisted:universe.csv 的 status=0;新浪退市股无股本表 / isST,
  退市股的换手率、isST、上市区间取这里,价格见 lh_build_panel 文件头);
  现存股票走新浪(scripts/lh_fetch_sina.py)。已抓的 parts 一律复用(存在即跳过)。
  --st-only:新浪没有 isST / tradestatus。对现存股只抓 date,tradestatus,isST(不复权,窗口 --st-start..--st-end)
  写 cache/long_daily/parts_st/<code>.csv.gz,供面板的 ST 身份与停牌核对(见 lh_build_panel.st_lookup)。
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import signal
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import baostock as bs

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cache" / "long_daily"
PARTS = OUT / "parts"
LOG = ROOT / "logs" / "lh_fetch_daily.log"
START, END = "2005-01-01", "2025-12-31"
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST"
REQ_TIMEOUT = 150          # 秒;全字段 20 年约 30 秒(分 3 页,每页 2000 行)
MAX_ROWS = 8000            # 2005-2025 约 5100 个交易日
A_PREFIX = ("sh.60", "sh.68", "sz.00", "sz.30")


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{os.getpid()}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


class _Alarm(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Alarm("baostock 请求超时")


def _with_timeout(fn, seconds: int):
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(int(seconds))
    try:
        return fn()
    finally:
        signal.alarm(0)


def rows(res, cap: int = MAX_ROWS) -> list:
    out = []
    while (res.error_code == "0") and res.next():
        out.append(res.get_row_data())
        if len(out) > cap:
            raise RuntimeError(f"超过最大行数 {cap}")
    return out


def _drop_socket() -> None:
    import baostock.common.context as ctx
    s = getattr(ctx, "default_socket", None)
    try:
        if s is not None:
            s.close()
    except Exception:
        pass


def bs_login(retries: int = 5) -> None:
    last = None
    for i in range(retries):
        try:
            lg = _with_timeout(bs.login, 40)
            if lg.error_code == "0":
                return
            last = lg.error_msg
        except Exception as e:      # noqa: BLE001
            last = repr(e)
        _drop_socket()
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"login 失败: {last}")


# ------------------------------------------------------------------ 股票清单
def trade_dates() -> pd.DataFrame:
    f = OUT / "trade_dates.csv"
    if f.exists():
        return pd.read_csv(f, dtype=str)
    r = _with_timeout(lambda: bs.query_trade_dates(start_date="2004-12-01", end_date=END), 120)
    d = pd.DataFrame(_with_timeout(lambda: rows(r, 20000), 120), columns=r.fields)
    d = d[d["is_trading_day"] == "1"][["calendar_date"]].rename(columns={"calendar_date": "date"})
    d.to_csv(f, index=False)
    return d


def universe() -> pd.DataFrame:
    f = OUT / "universe.csv"
    if f.exists():
        return pd.read_csv(f, dtype=str)
    OUT.mkdir(parents=True, exist_ok=True)
    bs_login()
    td = pd.to_datetime(trade_dates()["date"])
    ye = td.groupby(td.dt.year).max()
    ye = ye[(ye.index >= 2004) & (ye.index <= 2025)]
    recs: dict[str, dict] = {}
    snap_rows = []
    for y, d in ye.items():
        day = d.strftime("%Y-%m-%d")
        for attempt in range(4):
            try:
                r = _with_timeout(lambda: bs.query_all_stock(day=day), 120)
                rr = _with_timeout(lambda: rows(r, 20000), 120)
                break
            except Exception as e:      # noqa: BLE001
                log(f"query_all_stock {day} 失败 {e!r},重连重试")
                _drop_socket(); bs_login()
        else:
            raise RuntimeError(f"query_all_stock {day} 连续失败")
        n_a = 0
        for code, status, name in rr:
            if not code.startswith(A_PREFIX):
                continue
            n_a += 1
            snap_rows.append((y, day, code, status, name))
            rec = recs.setdefault(code, dict(code=code, name=name, first_seen=day, last_seen=day,
                                             n_snap=0, in_snap_2004_only=True))
            rec["last_seen"] = day
            rec["name_last"] = name
            rec["n_snap"] += 1
            if y >= 2005:
                rec["in_snap_2004_only"] = False
        log(f"  年末 {day}: 返回 {len(rr)} 行,沪深 A 股 {n_a} 只,累计 {len(recs)}")
    pd.DataFrame(snap_rows, columns=["year", "day", "code", "tradeStatus", "name"]).to_csv(
        OUT / "universe_snapshots.csv", index=False)
    # baostock 证券基本资料:补"两个年末之间上市又退市"的股票,并拿 ipoDate / outDate
    r = _with_timeout(bs.query_stock_basic, 120)
    sb = pd.DataFrame(_with_timeout(lambda: rows(r, 100000), 120), columns=r.fields)
    sb.to_csv(OUT / "bs_stock_basic.csv", index=False)
    bs.logout()
    sb = sb[(sb["type"] == "1") & sb["code"].str.startswith(A_PREFIX)].copy()
    alive = (sb["ipoDate"] <= END) & ((sb["outDate"] == "") | (sb["outDate"] >= START))
    df = pd.DataFrame(list(recs.values()))
    df["source"] = "snapshot"
    df.loc[df["in_snap_2004_only"].astype(bool), "source"] = "snapshot2004"
    extra = sb[alive & ~sb["code"].isin(df["code"])]
    if len(extra):
        df = pd.concat([df, pd.DataFrame(dict(code=extra["code"], name=extra["code_name"],
                                              name_last=extra["code_name"], first_seen="", last_seen="",
                                              n_snap=0, in_snap_2004_only=False,
                                              source="stock_basic_bs"))], ignore_index=True)
    df = df.merge(sb[["code", "code_name", "ipoDate", "outDate", "status"]], on="code", how="left")
    df = df.sort_values("code").reset_index(drop=True)
    df.to_csv(f, index=False)
    log(f"股票清单 {len(df)} 只:{df['source'].value_counts().to_dict()}")
    return df


# ------------------------------------------------------------------ 逐股下载
def _init(st_win=None):
    signal.signal(signal.SIGALRM, _on_alarm)
    if st_win:                     # macOS 默认 spawn,子进程不继承父进程改过的全局变量
        ST_WIN[:] = list(st_win)
    bs_login()


def _query(code: str, fields: str, adj: str) -> pd.DataFrame:
    def go():
        r = bs.query_history_k_data_plus(code, fields, start_date=START, end_date=END,
                                         frequency="d", adjustflag=adj)
        if r.error_code != "0":
            raise RuntimeError(f"error_code={r.error_code} {r.error_msg}")
        return pd.DataFrame(rows(r), columns=r.fields)
    return _with_timeout(go, REQ_TIMEOUT)


ST_FIELDS = "date,tradestatus,isST"
ST_WIN = [START, END]
PARTS_ST = OUT / "parts_st"


def fetch_st(code: str) -> tuple[str, int, str, float]:
    """只抓 date,tradestatus,isST(不复权),窗口 ST_WIN。"""
    part = PARTS_ST / f"{code}.csv.gz"
    if part.exists() or (PARTS_ST / f"{code}.empty").exists():
        return code, -1, "skip", 0.0
    t0 = time.time()
    try:
        def go():
            r = bs.query_history_k_data_plus(code, ST_FIELDS, start_date=ST_WIN[0], end_date=ST_WIN[1],
                                             frequency="d", adjustflag="3")
            if r.error_code != "0":
                raise RuntimeError(f"error_code={r.error_code} {r.error_msg}")
            return pd.DataFrame(rows(r), columns=r.fields)
        d = _with_timeout(go, REQ_TIMEOUT)
        if d.empty:
            (PARTS_ST / f"{code}.empty").write_text("")
            return code, 0, "empty", time.time() - t0
        tmp = PARTS_ST / f".{code}.tmp"
        with gzip.open(tmp, "wt") as fh:
            d.to_csv(fh, index=False)
        os.replace(tmp, part)
        return code, len(d), "ok", time.time() - t0
    except Exception as e:              # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        _drop_socket()
        try:
            bs_login(retries=3)
        except Exception as e2:         # noqa: BLE001
            msg += f" | relogin 失败 {e2}"
        return code, 0, msg, time.time() - t0


def fetch_one(code: str) -> tuple[str, int, str, float]:
    part = PARTS / f"{code}.csv.gz"
    if part.exists() or (PARTS / f"{code}.empty").exists():
        return code, -1, "skip", 0.0
    t0 = time.time()
    try:
        a = _query(code, FIELDS, "1")
        if a.empty:
            (PARTS / f"{code}.empty").write_text("")
            return code, 0, "empty", time.time() - t0
        b = _query(code, "date,close", "3").rename(columns={"close": "close_raw"})
        if len(b) != len(a) or not a["date"].equals(b["date"]):
            # 两次请求的日期集应完全一致;不一致说明中途数据异常,记失败下一轮重抓
            raise RuntimeError(f"复权/不复权行数或日期不一致 {len(a)} vs {len(b)}")
        d = a.copy()
        d["close_raw"] = b["close_raw"].to_numpy()
        tmp = PARTS / f".{code}.tmp"
        with gzip.open(tmp, "wt") as fh:
            d.to_csv(fh, index=False)
        os.replace(tmp, part)
        return code, len(d), "ok", time.time() - t0
    except Exception as e:              # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        # 超时/异常后连接里可能残留上一请求的字节 -> 丢弃旧连接重新登录
        _drop_socket()
        try:
            bs_login(retries=3)
        except Exception as e2:         # noqa: BLE001
            msg += f" | relogin 失败 {e2}"
        return code, 0, msg, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--universe-only", action="store_true")
    ap.add_argument("--only-delisted", action="store_true", help="只抓 universe.csv 中 status=0(已退市)的股票")
    ap.add_argument("--codes-file", help="只抓该文件(每行一个 baostock 代码)里的股票")
    ap.add_argument("--st-only", action="store_true", help="只抓 date,tradestatus,isST 到 parts_st/")
    ap.add_argument("--st-start", default=START)
    ap.add_argument("--st-end", default=END)
    args = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    uni = universe()
    if args.universe_only:
        return
    codes = uni["code"].tolist()
    if args.only_delisted:
        codes = uni.loc[uni["status"].astype(str) == "0", "code"].tolist()
    if args.codes_file:
        keep = {ln.strip() for ln in Path(args.codes_file).read_text().splitlines() if ln.strip()}
        codes = [c for c in codes if c in keep]
    if args.limit:
        codes = codes[: args.limit]
    work, pdir = fetch_one, PARTS
    if args.st_only:
        ST_WIN[:] = [args.st_start, args.st_end]
        PARTS_ST.mkdir(parents=True, exist_ok=True)
        work, pdir = fetch_st, PARTS_ST
        # 已有全字段 parts 的股票已经有 isST,不重抓
        codes = [c for c in codes if not (PARTS / f"{c}.csv.gz").exists()]
    for ps in range(1, args.passes + 1):
        todo = [c for c in codes if not (pdir / f"{c}.csv.gz").exists()
                and not (pdir / f"{c}.empty").exists()]
        if not todo:
            break
        log(f"第 {ps} 轮:清单 {len(codes)} 只,待抓 {len(todo)} 只,{args.workers} 进程")
        t0 = time.time(); ok = fail = 0; errs = []; secs = []
        with Pool(args.workers, initializer=_init, initargs=(list(ST_WIN),)) as pool:
            for i, (code, n, msg, sec) in enumerate(pool.imap_unordered(work, todo, chunksize=1), 1):
                if msg in ("ok", "empty"):
                    ok += 1; secs.append(sec)
                elif msg != "skip":
                    fail += 1; errs.append((code, msg)); log(f"  失败 {code}: {msg}")
                if i % 50 == 0 or i == len(todo):
                    el = time.time() - t0
                    eta = el / i * (len(todo) - i)
                    log(f"  {i}/{len(todo)} ok {ok} fail {fail} 已用 {el/60:.1f} 分 "
                        f"单股中位 {pd.Series(secs).median() if secs else float('nan'):.1f}s 预计剩余 {eta/60:.0f} 分")
        (pdir / "_errors.json" if args.st_only else OUT / "errors.json").write_text(
            json.dumps(errs, ensure_ascii=False, indent=1))
        log(f"第 {ps} 轮结束:ok {ok} fail {fail},耗时 {(time.time()-t0)/60:.1f} 分")
        if fail and ps < args.passes:
            time.sleep(30)
    left = [c for c in codes if not (pdir / f"{c}.csv.gz").exists() and not (pdir / f"{c}.empty").exists()]
    log(f"全部结束:未完成 {len(left)} 只 {left[:20]}")


if __name__ == "__main__":
    main()
