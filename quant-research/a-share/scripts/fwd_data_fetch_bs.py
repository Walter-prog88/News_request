"""前向检验数据下载(baostock):5 分钟线 / 日线 / 复权因子 / 季频股本(前缀 fwd_data)。

用户明确要求下载(2026-10-10)。**只下载原始数据,不计算任何收益、因子、IC。**
输出 cache/forward/raw/bs_<kind>/parts/<bs代码>.csv[.gz],原样保存 baostock 返回的字符串;
断点续跑:分片已存在即跳过(先写临时文件再原子改名,半截文件不会被当作完成)。

股票清单 = Panel().codes(cache/daily 的 5565 只)∪ query_all_stock(day=2026-10-09) 的沪深 A 股。
北交所(.BJ)baostock 不支持(返回 10004011"股票代码未标识sh或sz"),不请求,记入清单的 uncovered。

稳健性(baostock 的 rs.next() 在连接异常时可能空转 / 翻页静默失败):
  - 每个请求(含全部翻页)signal.alarm 60 秒超时;超时异常继承 BaseException,
    以免被 baostock 内部的 `except Exception` 吞掉;
  - socket 级别再设 60 秒超时;
  - 翻页结束后必须 error_code == "0",否则视为失败(翻页失败时 baostock 只返回第一页);
  - 每种数据设最大行数保护;
  - 单只股票失败重试 3 次,每次失败后重新 login(新 socket);最终失败写 _failures.json。

用法:
  python3 scripts/fwd_data_fetch_bs.py --kind meta
  python3 scripts/fwd_data_fetch_bs.py --kind m5 --workers 4
  python3 scripts/fwd_data_fetch_bs.py --kind daily,adjf,profit --workers 2
  python3 scripts/fwd_data_fetch_bs.py --kind m5 --redo sz.000001,sh.600000   # 强制重抓指定代码
  python3 scripts/fwd_data_fetch_bs.py --kind profit --workers 2 --reverse     # 倒序,与正序进程两头并行
"""
from __future__ import annotations
import os
import sys
import json
import time
import signal
import argparse
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
import baostock as bs
import baostock.common.context as bs_ctx

from afactor import fwd_data as FD

TIMEOUT = 60
RETRIES = 3

KINDS = {
    "m5": dict(fields="date,time,code,open,high,low,close,volume,amount",
               start=FD.M5_START, end=FD.M5_END, freq="5", max_rows=48 * 80),
    "daily": dict(fields="date,code,open,high,low,close,preclose,volume,amount,turn,"
                         "tradestatus,pctChg,isST",
                  start=FD.DAILY_START, end=FD.DAILY_END, freq="d", max_rows=200),
    "adjf": dict(start=FD.ADJF_START, end=FD.ADJF_END, max_rows=200),
    "profit": dict(quarters=[(2025, 4), (2026, 1), (2026, 2)], max_rows=50),
}


class FwdTimeout(BaseException):
    """继承 BaseException:baostock.util.socketutil.send_msg 里有 `except Exception`,会吞掉普通异常。"""


def _alarm(signum, frame):
    raise FwdTimeout(f"request > {TIMEOUT}s")


def _login():
    old = getattr(bs_ctx, "default_socket", None)
    if old is not None:
        try:
            old.close()
        except Exception:
            pass
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT)
    try:
        lg = bs.login()
    finally:
        signal.alarm(0)
    if lg.error_code != "0":
        raise RuntimeError(f"login: {lg.error_msg}")
    sock = getattr(bs_ctx, "default_socket", None)
    if sock is not None:
        sock.settimeout(TIMEOUT)


def _collect(res, max_rows: int) -> tuple[list, list]:
    """读完全部翻页。翻页后 error_code != 0 -> 失败;超过 max_rows -> 失败(防空转)。"""
    if res is None:
        raise RuntimeError("result is None")
    if res.error_code != "0":
        raise RuntimeError(f"{res.error_code} {res.error_msg}")
    out = []
    while res.next():
        out.append(res.get_row_data())
        if len(out) > max_rows:
            raise RuntimeError(f"rows > {max_rows}")
    if res.error_code != "0":
        raise RuntimeError(f"paging {res.error_code} {res.error_msg}")
    return out, list(res.fields)


def _query(kind: str, code: str) -> pd.DataFrame:
    k = KINDS[kind]
    if kind in ("m5", "daily"):
        r = bs.query_history_k_data_plus(code, k["fields"], start_date=k["start"], end_date=k["end"],
                                         frequency=k["freq"], adjustflag="3")
        rows, fields = _collect(r, k["max_rows"])
        return pd.DataFrame(rows, columns=fields)
    if kind == "adjf":
        r = bs.query_adjust_factor(code=code, start_date=k["start"], end_date=k["end"])
        rows, fields = _collect(r, k["max_rows"])
        return pd.DataFrame(rows, columns=fields)
    if kind == "profit":
        frames = []
        for y, q in k["quarters"]:
            r = bs.query_profit_data(code=code, year=y, quarter=q)
            rows, fields = _collect(r, k["max_rows"])
            d = pd.DataFrame(rows, columns=fields)
            d.insert(0, "q_req", f"{y}Q{q}")
            frames.append(d)
        return pd.concat(frames, ignore_index=True)
    raise ValueError(kind)


_KIND = None


def _init(kind: str):
    global _KIND
    _KIND = kind
    _login()


def fetch_one(code: str):
    kind = _KIND
    part = FD.part_path(kind, code)
    if part.exists():
        return code, -1, "skip", 0
    last = ""
    for att in range(1, RETRIES + 1):
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(TIMEOUT)
        try:
            d = _query(kind, code)
            signal.alarm(0)
            tmp = part.with_name(part.name + ".tmp")
            if kind == "m5":
                d.to_csv(tmp, index=False, compression="gzip")
            else:
                d.to_csv(tmp, index=False)
            os.replace(tmp, part)
            return code, len(d), ("ok" if len(d) else "empty"), att
        except FwdTimeout as e:
            signal.alarm(0)
            last = f"timeout: {e}"
        except Exception as e:                       # noqa: BLE001
            signal.alarm(0)
            last = f"{type(e).__name__}: {e}"
        time.sleep(2 * att)
        try:
            _login()
        except BaseException as e:                   # noqa: BLE001
            signal.alarm(0)
            last += f" | relogin {type(e).__name__}: {e}"
    return code, 0, f"fail: {last}", RETRIES


# ---------------- 股票清单与元数据 ----------------
def _all_rows(res, max_rows=20000):
    rows, fields = _collect(res, max_rows)
    return pd.DataFrame(rows, columns=fields)


def build_meta():
    """股票清单(两路并集)、交易日历、query_stock_basic。只在主进程跑一次。"""
    from afactor.panel import Panel
    out = FD.DIRS["meta"]
    out.mkdir(parents=True, exist_ok=True)
    _login()
    signal.alarm(TIMEOUT * 3)
    try:
        allst = _all_rows(bs.query_all_stock(day="2026-10-09"))
        cal = _all_rows(bs.query_trade_dates(start_date="2025-12-01", end_date="2026-10-31"))
        basic = _all_rows(bs.query_stock_basic())
    finally:
        signal.alarm(0)
    bs.logout()
    allst.to_csv(out / "query_all_stock_20261009.csv", index=False)
    cal.to_csv(out / "trade_dates_20251201_20261031.csv", index=False)
    basic.to_csv(out / "query_stock_basic_20261010.csv", index=False)

    panel_codes = list(Panel().codes)
    bs_a = sorted(c for c in allst["code"] if FD.is_a_share_bs(c))
    rows = []
    for c in panel_codes:
        b = FD.to_bs(c)
        rows.append(dict(code=c, bs_code=b, in_panel=1, in_bs_list=int(b in set(bs_a)) if b else 0,
                         covered=int(b is not None)))
    pset = {FD.to_bs(c) for c in panel_codes}
    for b in bs_a:
        if b not in pset:
            rows.append(dict(code=FD.from_bs(b), bs_code=b, in_panel=0, in_bs_list=1, covered=1))
    u = pd.DataFrame(rows).sort_values("code").reset_index(drop=True)
    u.to_csv(out / "universe.csv", index=False)
    print(f"Panel {len(panel_codes)} 只(其中 BJ {sum(c.endswith('.BJ') for c in panel_codes)} 只不覆盖);"
          f"baostock 沪深 A 股 {len(bs_a)} 只;并集 {len(u)},可请求 {int(u.covered.sum())}", flush=True)
    return u


def universe() -> list[str]:
    f = FD.DIRS["meta"] / "universe.csv"
    if not f.exists():
        build_meta()
    u = pd.read_csv(f, dtype=str)
    return sorted(u.loc[u["covered"] == "1", "bs_code"])


def run_kind(kind: str, workers: int, limit: int | None, redo: list[str], reverse: bool = False):
    d = FD.DIRS[kind]
    (d / "parts").mkdir(parents=True, exist_ok=True)
    for c in redo:
        p = FD.part_path(kind, c)
        if p.exists():
            p.rename(p.with_name(p.name + f".redo{int(time.time())}"))   # 不删除,改名留档
    codes = universe()
    if limit:
        codes = codes[:limit]
    if reverse:                     # 与正序进程从两头往中间抓,已存在的分片会被跳过
        codes = codes[::-1]
    todo = [c for c in codes if not FD.part_path(kind, c).exists()]
    print(f"[{kind}] 股票 {len(codes)},待抓 {len(todo)},{workers} 进程", flush=True)
    t0 = time.time()
    n_ok = n_empty = n_fail = 0
    stat_f = d / "_status.csv"
    new = not stat_f.exists()
    with open(stat_f, "a") as sf, Pool(workers, initializer=_init, initargs=(kind,)) as pool:
        if new:
            sf.write("code,n,status,attempts,ts\n")
        for i, (c, n, msg, att) in enumerate(pool.imap_unordered(fetch_one, todo, chunksize=4), 1):
            if msg == "skip":
                continue
            if msg == "ok":
                n_ok += 1
            elif msg == "empty":
                n_empty += 1
            else:
                n_fail += 1
            sf.write(f"{c},{n},\"{msg.replace(chr(34), chr(39))}\",{att},{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\n")
            sf.flush()
            if i % 200 == 0:
                print(f"  [{kind}] {i}/{len(todo)} ok {n_ok} empty {n_empty} fail {n_fail} "
                      f"{time.time() - t0:.0f}s", flush=True)
    st = FD.read_status(kind)
    fails = st[st["status"].str.startswith("fail")]
    fails = fails[~fails["code"].map(lambda c: FD.part_path(kind, c).exists())]
    (d / "_failures.json").write_text(json.dumps(fails.to_dict("records"), ensure_ascii=False, indent=1))
    print(f"[{kind}] 完成 ok {n_ok} empty {n_empty} fail {n_fail};最终未完成 {len(fails)};"
          f"耗时 {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, help="meta / m5 / daily / adjf / profit,逗号分隔")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="只抓前 N 只(探针用)")
    ap.add_argument("--redo", default="", help="强制重抓的 bs 代码,逗号分隔")
    ap.add_argument("--reverse", action="store_true", help="倒序抓(与正序进程并行,两头往中间)")
    a = ap.parse_args()
    redo = [c for c in a.redo.split(",") if c]
    for kind in a.kind.split(","):
        if kind == "meta":
            build_meta()
        else:
            run_kind(kind, a.workers, a.limit, redo, a.reverse)


if __name__ == "__main__":
    main()
