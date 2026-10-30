"""前向检验数据下载(akshare):业绩预告/快报、三张财报、宽基指数日线、中证1000 成分与权重(前缀 fwd_data)。

用户明确要求下载(2026-10-10)。**只下载原始数据,不计算任何收益。**
输出 cache/forward/raw/ak_*/,文件名带取数日期(fetched=YYYYMMDD),同一天重跑覆盖、跨天重跑并存(快照语义)。

为什么东方财富数据中心要同时存 akshare 结果和原始字段:
  akshare 的 stock_lrb_em / stock_yjyg_em 等把接口返回的 "columns=ALL" 原始表**按位置**改成中文列名,
  东方财富一旦调整字段顺序就会静默错位。这里用与 akshare 完全相同的 URL / reportName / filter
  再抓一份原始英文字段(em_raw_*),质检脚本按数值逐列核对 akshare 中文列 = 哪个原始字段。

超时:所有 requests 调用默认 timeout=60 秒(猴子补丁 Session.request),每个函数整体再加 signal.alarm;
超时异常继承 BaseException,避免被 akshare 内部的 except Exception 吞掉。失败写 _failures.json,可重跑。

用法:python3 scripts/fwd_data_fetch_ak.py [--only em,index,cons,mapcheck]
"""
from __future__ import annotations
import os
import sys
import json
import time
import signal
import argparse
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
import requests

_orig_request = requests.Session.request


def _request_with_timeout(self, method, url, **kw):
    kw.setdefault("timeout", 60)
    return _orig_request(self, method, url, **kw)


requests.Session.request = _request_with_timeout

import pandas as pd
import akshare as ak

from afactor import fwd_data as FD

FETCH_DAY = pd.Timestamp.now().strftime("%Y%m%d")
OUT_EM = FD.RAW / "ak_em"
OUT_IDX = FD.RAW / "ak_index"
OUT_CONS = FD.RAW / "ak_csindex"
REPORT_DATES = ["20260630", "20260930"]
INDEXES = ["sh000852", "sh000300", "sh000905", "sh000001"]
CONS_INDEXES = ["000852", "000905", "000300"]     # 任务要求 000852;另两个顺带(各 2 次请求)


class FwdTimeout(BaseException):
    pass


def _alarm(signum, frame):
    raise FwdTimeout("call timeout")


def call(fn, seconds: int = 600):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)


# ---------------- 东方财富数据中心原始字段 ----------------
_EM = {
    # name: (url, reportName, sortColumns, filter 模板(含 BJ 排除与否与 akshare 保持一致))
    "yjyg": ("https://datacenter.eastmoney.com/securities/api/data/v1/get", "RPT_PUBLIC_OP_NEWPREDICT",
             "NOTICE_DATE,SECURITY_CODE", " (REPORT_DATE='{d}')"),
    "yjkb": ("https://datacenter.eastmoney.com/securities/api/data/v1/get", "RPT_FCI_PERFORMANCEE",
             "UPDATE_DATE,SECURITY_CODE",
             '(SECURITY_TYPE_CODE in ("058001001","058001008"))(TRADE_MARKET_CODE!="069001017")'
             "(REPORT_DATE='{d}')"),
    "lrb": ("https://datacenter-web.eastmoney.com/api/data/v1/get", "RPT_DMSK_FN_INCOME",
            "NOTICE_DATE,SECURITY_CODE",
            '(SECURITY_TYPE_CODE in ("058001001","058001008"))(TRADE_MARKET_CODE!="069001017")'
            "(REPORT_DATE='{d}')"),
    "zcfz": ("https://datacenter-web.eastmoney.com/api/data/v1/get", "RPT_DMSK_FN_BALANCE",
             "NOTICE_DATE,SECURITY_CODE",
             '(SECURITY_TYPE_CODE in ("058001001","058001008"))(TRADE_MARKET_CODE!="069001017")'
             "(REPORT_DATE='{d}')"),
    "xjll": ("https://datacenter-web.eastmoney.com/api/data/v1/get", "RPT_DMSK_FN_CASHFLOW",
             "NOTICE_DATE,SECURITY_CODE",
             '(SECURITY_TYPE_CODE in ("058001001","058001008"))(TRADE_MARKET_CODE!="069001017")'
             "(REPORT_DATE='{d}')"),
}
_AK = {"yjyg": ak.stock_yjyg_em, "yjkb": ak.stock_yjkb_em, "lrb": ak.stock_lrb_em,
       "zcfz": ak.stock_zcfz_em, "xjll": ak.stock_xjll_em}


def em_raw(name: str, date: str) -> pd.DataFrame:
    url, rep, sort, flt = _EM[name]
    d = "-".join([date[:4], date[4:6], date[6:]])
    params = {"sortColumns": sort, "sortTypes": "-1,-1", "pageSize": "500", "pageNumber": "1",
              "reportName": rep, "columns": "ALL", "filter": flt.format(d=d)}
    s = requests.Session()
    j = s.get(url, params=params).json()
    if not j.get("result"):
        return pd.DataFrame()                     # 该报告期尚无数据(接口返回 result=None)
    pages = int(j["result"]["pages"])
    if pages > 100:
        raise RuntimeError(f"pages {pages} > 100")
    frames = [pd.DataFrame(j["result"]["data"])]
    for p in range(2, pages + 1):
        params["pageNumber"] = str(p)
        jj = s.get(url, params=params).json()
        frames.append(pd.DataFrame(jj["result"]["data"]))
        time.sleep(0.3)
    out = pd.concat(frames, ignore_index=True)
    n_total = int(j["result"].get("count", len(out)))
    if len(out) != n_total:
        print(f"  [warn] {name} {date}: 行数 {len(out)} != count {n_total}", flush=True)
    return out


def fetch_em(fails: list, manifest: list):
    OUT_EM.mkdir(parents=True, exist_ok=True)
    for name in ["yjyg", "yjkb", "lrb", "zcfz", "xjll"]:
        for date in REPORT_DATES:
            for src in ("ak", "raw"):
                f = OUT_EM / f"{src}_{name}_{date}_fetched{FETCH_DAY}.csv"
                t0 = time.time()
                try:
                    if src == "ak":
                        d = call(lambda: _AK[name](date=date))
                        d.insert(0, "报告日期", date)       # 与 data_8:21/source_raw/eastmoney_*_raw.csv 同口径
                    else:
                        d = call(lambda: em_raw(name, date))
                    d.to_csv(f, index=False)
                    manifest.append(dict(kind=f"em_{src}_{name}", report_date=date, file=str(f.relative_to(FD.RAW)),
                                         rows=len(d), cols=d.shape[1], secs=round(time.time() - t0, 1)))
                    print(f"  {src}_{name}_{date}: {d.shape}", flush=True)
                except BaseException as e:                     # noqa: BLE001
                    if isinstance(e, KeyboardInterrupt):
                        raise
                    msg = f"{type(e).__name__}: {e}"
                    # akshare 在该报告期没有数据时会因 result=None 抛 TypeError,记为空而非失败
                    if src == "ak" and ("NoneType" in msg or "not subscriptable" in msg):
                        manifest.append(dict(kind=f"em_{src}_{name}", report_date=date, file=None, rows=0,
                                             cols=0, secs=round(time.time() - t0, 1), note="no data: " + msg))
                        print(f"  {src}_{name}_{date}: 无数据 ({msg})", flush=True)
                    else:
                        fails.append(dict(kind=f"em_{src}_{name}", report_date=date, error=msg))
                        print(f"  {src}_{name}_{date}: FAIL {msg}", flush=True)
                time.sleep(1)


def fetch_mapcheck(fails: list, manifest: list):
    """只为"东财字段 ↔ 项目 pkl 字段"的数值核对:项目三张表 pkl(到 2026-08-21)的 2026-06-30 行很少
    (资产负债表为 0),再抓一份 2026-03-31 的东财原始字段,在一季报上逐字段比对。不用于任何信号。"""
    OUT_EM.mkdir(parents=True, exist_ok=True)
    for name in ["lrb", "zcfz", "xjll"]:
        date = "20260331"
        f = OUT_EM / f"raw_{name}_{date}_fetched{FETCH_DAY}.csv"
        t0 = time.time()
        try:
            d = call(lambda: em_raw(name, date))
            d.to_csv(f, index=False)
            manifest.append(dict(kind=f"em_raw_{name}", report_date=date, file=str(f.relative_to(FD.RAW)),
                                 rows=len(d), cols=d.shape[1], secs=round(time.time() - t0, 1),
                                 note="字段映射核对用"))
            print(f"  raw_{name}_{date}: {d.shape}", flush=True)
        except BaseException as e:                             # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            fails.append(dict(kind=f"em_raw_{name}", report_date=date, error=f"{type(e).__name__}: {e}"))
        time.sleep(1)


def fetch_index(fails: list, manifest: list):
    OUT_IDX.mkdir(parents=True, exist_ok=True)
    for sym in INDEXES:
        f = OUT_IDX / f"{sym}_fetched{FETCH_DAY}.csv"
        try:
            d = call(lambda: ak.stock_zh_index_daily(symbol=sym), 300)
            d.to_csv(f, index=False)
            manifest.append(dict(kind="index_daily_sina", code=sym, file=str(f.relative_to(FD.RAW)), rows=len(d),
                                 first=str(d["date"].iloc[0]), last=str(d["date"].iloc[-1])))
            print(f"  {sym}: {d.shape} {d['date'].iloc[0]} ~ {d['date'].iloc[-1]}", flush=True)
        except BaseException as e:                             # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            fails.append(dict(kind="index_daily_sina", code=sym, error=f"{type(e).__name__}: {e}"))
        time.sleep(1)


def fetch_cons(fails: list, manifest: list):
    OUT_CONS.mkdir(parents=True, exist_ok=True)
    for code in CONS_INDEXES:
        for what, fn in (("cons_weight", ak.index_stock_cons_weight_csindex), ("cons", ak.index_stock_cons_csindex)):
            f = OUT_CONS / f"{code}_{what}_fetched{FETCH_DAY}.csv"
            try:
                d = call(lambda: fn(symbol=code), 300)
                d.to_csv(f, index=False)
                dates = sorted(pd.Series(d["日期"]).astype(str).unique().tolist())
                manifest.append(dict(kind=f"csindex_{what}", code=code, file=str(f.relative_to(FD.RAW)),
                                     rows=len(d), data_dates=dates, fetched=FETCH_DAY))
                print(f"  {code} {what}: {d.shape} 日期 {dates}", flush=True)
            except BaseException as e:                         # noqa: BLE001
                if isinstance(e, KeyboardInterrupt):
                    raise
                fails.append(dict(kind=f"csindex_{what}", code=code, error=f"{type(e).__name__}: {e}"))
            time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="em,index,cons")
    a = ap.parse_args()
    fails, manifest = [], []
    t0 = time.time()
    if "index" in a.only:
        fetch_index(fails, manifest)
    if "cons" in a.only:
        fetch_cons(fails, manifest)
    if "em" in a.only.split(","):
        fetch_em(fails, manifest)
    if "mapcheck" in a.only:
        fetch_mapcheck(fails, manifest)
    FD.RAW.mkdir(parents=True, exist_ok=True)
    tag = a.only.replace(",", "_")
    (FD.RAW / f"_ak_manifest_{tag}_{FETCH_DAY}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1, default=str))
    (FD.RAW / f"_ak_failures_{tag}_{FETCH_DAY}.json").write_text(
        json.dumps(fails, ensure_ascii=False, indent=1, default=str))
    print(f"完成:{len(manifest)} 个文件,失败 {len(fails)},耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
