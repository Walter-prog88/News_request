"""北交所 5 分钟线补充下载(新浪,akshare.stock_zh_a_minute)(前缀 fwd_data)。

为什么要有它:baostock 不支持北交所(bj. 代码返回 10004011),而面板 5565 只里有 336 只 .BJ;
东方财富历史分钟接口(push2his)在本机网络被代理拦截。新浪分钟接口只返回**最近 1970 根** 5 分钟 bar
(约 41 个交易日),2026-10-10 取数时覆盖 2026-08-05 下午 ~ 10-09 —— **窗口每天往后滚,必须当天抓**。
它只是 baostock 的补充,单独存放在 cache/forward/raw/sina_m5_bj/,下游可自行决定是否使用。

同时按固定随机种子抽 30 只沪深股票(--sample)一并抓新浪 5 分钟,只用于质检"新浪 vs baostock"的口径差异。
北交所另抓新浪日线(--kind daily,2026-07-01~10-09,含 outstanding_share / turnover)与前复权因子事件
(--kind qfq,stock_zh_a_daily adjust="qfq-factor",全历史),对应 baostock 的 bs_daily / bs_adjf。

**只下载原始数据,不计算任何收益。** 每个请求 60 秒超时,单只失败重试 3 次,断点续跑,失败清单落盘。
用法:python3 scripts/fwd_data_fetch_sina_bj.py [--kind m5|daily|qfq] [--sample 30]
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

import numpy as np
import pandas as pd
import akshare as ak

from afactor import fwd_data as FD
from afactor.config import load_raw_minute

OUT = FD.RAW / "sina_m5_bj"
OUT_K = {"m5": OUT, "daily": FD.RAW / "sina_daily_bj", "qfq": FD.RAW / "sina_qfq_bj"}
FETCH_TS = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


class FwdTimeout(BaseException):
    pass


def _alarm(signum, frame):
    raise FwdTimeout("timeout 60s")


def sina_symbol(code: str) -> str:
    num, ex = code.split(".")
    return ex.lower() + num


def _call(kind: str, code: str) -> pd.DataFrame:
    if kind == "m5":
        return ak.stock_zh_a_minute(symbol=sina_symbol(code), period="5", adjust="")
    if kind == "daily":
        return ak.stock_zh_a_daily(symbol=sina_symbol(code), start_date="20260701", end_date="20261009", adjust="")
    if kind == "qfq":
        return ak.stock_zh_a_daily(symbol=sina_symbol(code), adjust="qfq-factor")
    raise ValueError(kind)


def fetch(code: str, kind: str = "m5") -> tuple[int, str, int]:
    out = OUT_K[kind]
    part = out / "parts" / f"{code}.csv"
    if part.exists():
        return -1, "skip", 0
    last = ""
    for att in range(1, 4):
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(60)
        try:
            d = _call(kind, code)
            signal.alarm(0)
            if d is None:
                d = pd.DataFrame()
            if len(d) > 5000:
                raise RuntimeError(f"rows {len(d)} > 5000")
            d.insert(0, "code", code)
            d["fetched_at"] = FETCH_TS
            tmp = part.with_name(part.name + ".tmp")
            d.to_csv(tmp, index=False)
            os.replace(tmp, part)
            return len(d), ("ok" if len(d) else "empty"), att
        except FwdTimeout as e:
            signal.alarm(0)
            last = f"timeout: {e}"
        except Exception as e:                           # noqa: BLE001
            signal.alarm(0)
            last = f"{type(e).__name__}: {str(e)[:200]}"
            if "NoneType" in last or "KeyError" in last:   # 新浪对无数据代码返回空 -> akshare 内部报错
                part.write_text("code,fetched_at\n")
                return 0, "empty", att
        time.sleep(3 * att)
    return 0, f"fail: {last}", 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=30, help="另抽 N 只沪深股票做跨源质检(仅 m5)")
    ap.add_argument("--kind", default="m5", choices=["m5", "daily", "qfq"])
    a = ap.parse_args()
    out = OUT_K[a.kind]
    (out / "parts").mkdir(parents=True, exist_ok=True)
    u = pd.read_csv(FD.DIRS["meta"] / "universe.csv", dtype=str)
    bj = set(u.loc[u.code.str.endswith(".BJ"), "code"])
    bj |= {c for c in load_raw_minute("close_adj", "202608").columns if c.endswith(".BJ")}
    rng = np.random.default_rng(20261010)
    shsz = sorted(u.loc[u.covered == "1", "code"])
    sample = sorted(rng.choice(shsz, size=a.sample, replace=False).tolist()) if (a.sample and a.kind == "m5") else []
    todo = sorted(bj) + sample
    pd.DataFrame({"code": todo, "role": ["bj"] * len(bj) + ["shsz_sample"] * len(sample)}).to_csv(
        out / "codes.csv", index=False)
    print(f"北交所 {len(bj)} 只 + 沪深抽样 {len(sample)} 只", flush=True)
    stat_f = out / "_status.csv"
    new = not stat_f.exists()
    t0 = time.time()
    with open(stat_f, "a") as sf:
        if new:
            sf.write("code,n,status,attempts,ts\n")
        for i, c in enumerate(todo, 1):
            n, msg, att = fetch(c, a.kind)
            if msg != "skip":
                sf.write(f"{c},{n},\"{msg.replace(chr(34), chr(39))}\",{att},{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\n")
                sf.flush()
                time.sleep(0.5)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    st = pd.read_csv(stat_f, dtype=str).drop_duplicates("code", keep="last")
    fails = st[st.status.str.startswith("fail") & ~st.code.map(lambda c: (out / "parts" / f"{c}.csv").exists())]
    (out / "_failures.json").write_text(json.dumps(fails.to_dict("records"), ensure_ascii=False, indent=1))
    print(f"完成:失败 {len(fails)},耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
