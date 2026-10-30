"""lh_data 第 2(b) 步:**现存**沪深 A 股 2005-01-01 ~ 2025-12-31 日线,数据源新浪(akshare 的 stock_zh_a_daily 同源)。
不计算任何策略收益。

为什么换源(提速方案,主控 2026-10-10 实测):baostock 按 IP 限速,单股全历史约 27 秒,6 进程 5475 只要 7 小时,
加进程总吞吐不变;东方财富 push2his 在本机被代理拒绝。新浪不限进程级登录,8 线程约 0.85 秒/只(akshare 两次调用)。
退市股:新浪仍有行情与复权因子但没有股本表,换手率 / isST / 上市区间仍取 baostock(scripts/lh_fetch_daily.py --only-delisted)。

每只股票 3 个 HTTP 请求(akshare 的 adjust="" 与 adjust="hfq" 两次调用共 5 个请求,其中 2 个重复;这里直接请求
akshare 用的同三个新浪地址,结果与 akshare 逐项相同,见 --verify):
  1. hisdata_klc2/klc_kl.js  不复权 OHLC、volume(股)、amount(元)、postVol/postAmt(科创/创业板盘后固定价格交易)、
                             prevclose(只有上市首行有 = 发行价)。**停牌日没有行**。
  2. StockService.getAmountBySymbol  流通股本变动表(万股)-> outstanding_share(股),按日期 as-of 向后取最近一次变动
  3. hfq.js                  后复权因子表(除权日, 因子)-> hfq_factor,按日期 as-of 向后取
akshare 用 outer merge + ffill 再 drop_duplicates 去掉股本/因子变动日产生的假行;这里只在真实交易日上 as-of 取值,
没有假行,也不对后复权价四舍五入到 2 位(akshare 会 round(2),低价股后复权收益因此有取整误差)。

输出 cache/long_daily/parts_sina/<baostock 代码>.csv.gz(先写临时文件再改名,存在即跳过 = 可续跑),字段:
  date, open, high, low, close(不复权), volume, amount, post_vol, post_amt, prevclose_ipo,
  outstanding_share(股), turnover(= volume / outstanding_share,**小数**,不是 %), hfq_factor
  后复权价 = 不复权价 x hfq_factor(由 lh_build_panel 计算)。
  窗口内无行的写 <code>.empty;新浪查不到(多为退市股)的写 <code>.nosina。

--delisted:对**已退市**股票(status=0)也抓新浪不复权行情与后复权因子(新浪对退市股仍保留 hist 与 hfq.js,
只是股本表为空)。用途:baostock 的后复权**漏掉了股权分置改革的送股对价**(2005-2007 复牌日收益系统性偏低,
QA 两源对比发现),退市股的价格因此也改用新浪,baostock 只提供换手率 / isST / 上市区间(见 lh_build_panel)。
  退市股:只发 2 个请求(hist + hfq),outstanding_share / turnover 为 NaN;hfq 表为空的股票 hfq_factor = NaN,构建时退回 baostock。

健壮性:每请求 (连接 5 秒, 读取 30 秒) 超时;单股失败重试 3 次(间隔 2/4/8 秒);全局连续失败 >= 8 次视为被限流,
所有线程暂停 60 x k 秒(k 递增,上限 10 分钟)并把并发降一档;失败写 parts_sina/_errors.json 与日志,多轮重试。

用法: python3 scripts/lh_fetch_sina.py [--threads 8] [--passes 3] [--limit N] [--codes-file F] [--verify N]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cache" / "long_daily"
PARTS = OUT / "parts_sina"
LOG = ROOT / "logs" / "lh_fetch_sina.log"
START, END = "2005-01-01", "2025-12-31"
TIMEOUT = (5, 30)
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Referer": "https://finance.sina.com.cn/"}

from akshare.stock.cons import (hk_js_decode, zh_sina_a_stock_amount_url,           # noqa: E402
                                zh_sina_a_stock_hfq_url, zh_sina_a_stock_hist_url)
from akshare.utils import demjson                                                    # noqa: E402


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def sina_symbol(bs_code: str) -> str:
    return bs_code.replace(".", "")


_tl = threading.local()


def _ctx():
    if not hasattr(_tl, "js"):
        import py_mini_racer
        js = py_mini_racer.MiniRacer()
        js.eval(hk_js_decode)
        _tl.js = js
        s = requests.Session()
        s.headers.update(HEADERS)
        _tl.sess = s
    return _tl.js, _tl.sess


class NoSina(Exception):
    """新浪没有这只股票(返回空/无法解析),不重试。"""


class Throttle:
    """全局限流状态:连续失败计数、暂停到某时刻、当前允许并发。"""

    def __init__(self, threads: int):
        self.lock = threading.Lock()
        self.consec_fail = 0
        self.pause_until = 0.0
        self.n_pause = 0
        self.allowed = threads
        self.sem = threading.BoundedSemaphore(threads)
        self.reduced = 0

    def wait(self):
        while True:
            with self.lock:
                dt = self.pause_until - time.time()
            if dt <= 0:
                return
            time.sleep(min(dt, 5))

    def ok(self):
        with self.lock:
            self.consec_fail = 0

    def fail(self):
        with self.lock:
            self.consec_fail += 1
            if self.consec_fail >= 8 and time.time() >= self.pause_until:
                self.n_pause += 1
                sec = min(60 * self.n_pause, 600)
                self.pause_until = time.time() + sec
                self.consec_fail = 0
                # 并发降一档:占住一个信号量名额不还(最少保留 2 线程)
                if self.allowed > 2:
                    self.allowed -= 1
                    self.reduced += 1
                    threading.Thread(target=self.sem.acquire, daemon=True).start()
                log(f"  !! 连续失败,疑似限流:全部暂停 {sec}s,并发降到 {self.allowed}")


def _get(url: str) -> str:
    _, s = _ctx()
    r = s.get(url, timeout=TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.text


def fetch_raw(bs_code: str, delisted: bool = False) -> pd.DataFrame:
    """3 个请求 -> 逐交易日 DataFrame(全历史,未截窗口)。delisted=True:不请求股本表,hfq 为空时因子记 NaN。"""
    sym = sina_symbol(bs_code)
    js, _ = _ctx()
    txt = _get(zh_sina_a_stock_hist_url.format(sym))
    if "=" not in txt or len(txt) < 50:
        raise NoSina(f"hist 空响应 len={len(txt)}")
    lst = js.call("d", txt.split("=")[1].split(";")[0].replace('"', ""))
    d = pd.DataFrame(lst)
    if d.empty:
        raise NoSina("hist 解码为空")
    d["date"] = pd.to_datetime(d["date"].astype(str).str[:10])
    for c in ("prevclose", "postVol", "postAmt"):
        if c not in d.columns:
            d[c] = np.nan
    d = d.rename(columns={"prevclose": "prevclose_ipo", "postVol": "post_vol", "postAmt": "post_amt"})
    num = ["open", "high", "low", "close", "volume", "amount", "post_vol", "post_amt", "prevclose_ipo"]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce")
    d = d.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    # 流通股本(万股);退市股新浪返回 null,不请求
    if delisted:
        sh = pd.DataFrame(dict(date=pd.to_datetime([]), outstanding_share=np.array([], float)))
    else:
        t2 = _get(zh_sina_a_stock_amount_url.format(sym, sym))
        seg = t2[t2.find("["): t2.rfind("]") + 1]
        if not seg:
            raise NoSina("股本表空")
        sh = pd.DataFrame(demjson.decode(seg))
        sh.columns = ["date", "outstanding_share"]
        sh["date"] = pd.to_datetime(sh["date"])
        sh["outstanding_share"] = pd.to_numeric(sh["outstanding_share"], errors="coerce") * 10000
        sh = sh.dropna().sort_values("date").drop_duplicates("date", keep="last")
    # 后复权因子
    t3 = _get(zh_sina_a_stock_hfq_url.format(sym))
    hf = pd.DataFrame(eval(t3.split("=")[1].split("\n")[0])["data"])   # 与 akshare 相同的解析方式(新浪 JS 对象)
    if hf.empty and delisted:
        hf = pd.DataFrame(dict(d=pd.to_datetime(["1900-01-01"]), f=[np.nan]))
    elif hf.empty:
        raise NoSina("hfq 因子表空")
    hf.columns = ["date", "hfq_factor"]
    hf["date"] = pd.to_datetime(hf["date"])
    hf["hfq_factor"] = pd.to_numeric(hf["hfq_factor"], errors="coerce")
    hf = hf.dropna().sort_values("date").drop_duplicates("date", keep="last")
    d = pd.merge_asof(d, sh, on="date", direction="backward")
    d = pd.merge_asof(d, hf, on="date", direction="backward")
    d["turnover"] = d["volume"] / d["outstanding_share"]
    return d


COLS = ["date", "open", "high", "low", "close", "volume", "amount", "post_vol", "post_amt", "prevclose_ipo",
        "outstanding_share", "turnover", "hfq_factor"]


def fetch_one(bs_code: str, th: Throttle, retries: int = 3, delisted: bool = False) -> tuple[str, int, str, float]:
    part = PARTS / f"{bs_code}.csv.gz"
    if part.exists() or (PARTS / f"{bs_code}.empty").exists() or (PARTS / f"{bs_code}.nosina").exists():
        return bs_code, -1, "skip", 0.0
    t0 = time.time()
    last = ""
    with th.sem:
        for k in range(retries + 1):
            th.wait()
            try:
                d = fetch_raw(bs_code, delisted=delisted)
                th.ok()
                w = d[(d["date"] >= START) & (d["date"] <= END)].copy()
                if w.empty:
                    (PARTS / f"{bs_code}.empty").write_text("")
                    return bs_code, 0, "empty", time.time() - t0
                w["date"] = w["date"].dt.strftime("%Y-%m-%d")
                tmp = PARTS / f".{bs_code}.tmp"
                with gzip.open(tmp, "wt") as fh:
                    w[COLS].to_csv(fh, index=False)
                os.replace(tmp, part)
                return bs_code, len(w), "ok", time.time() - t0
            except NoSina as e:
                th.ok()                      # 正常响应,只是没有这只股票
                (PARTS / f"{bs_code}.nosina").write_text(str(e))
                return bs_code, 0, "nosina", time.time() - t0
            except Exception as e:           # noqa: BLE001
                last = f"{type(e).__name__}: {str(e)[:160]}"
                th.fail()
                if k < retries:
                    time.sleep(2 ** (k + 1) + random.random())
    return bs_code, 0, last, time.time() - t0


def verify(codes: list[str]) -> None:
    """与 akshare.stock_zh_a_daily(adjust="" / "hfq")逐项对比(不复权价、量额、换手率、后复权价到 0.01)。"""
    import akshare as ak
    for c in codes:
        mine = fetch_raw(c)
        mine = mine[(mine["date"] >= START) & (mine["date"] <= END)].set_index("date")
        sym = sina_symbol(c)
        a0 = ak.stock_zh_a_daily(symbol=sym, start_date="20050101", end_date="20251231", adjust="")
        a1 = ak.stock_zh_a_daily(symbol=sym, start_date="20050101", end_date="20251231", adjust="hfq")
        a0.index = pd.to_datetime(a0["date"]); a1.index = pd.to_datetime(a1["date"])
        j = mine.index.intersection(a0.index)
        dc = (mine.loc[j, "close"] - a0.loc[j, "close"]).abs().max()
        dv = (mine.loc[j, "volume"] - a0.loc[j, "volume"]).abs().max()
        dt = (mine.loc[j, "turnover"] / a0.loc[j, "turnover"] - 1).abs().max()
        j1 = mine.index.intersection(a1.index)
        dh = ((mine.loc[j1, "close"] * mine.loc[j1, "hfq_factor"]).round(2) - a1.loc[j1, "close"]).abs().max()
        log(f"verify {c}: 行 mine {len(mine)} ak {len(a0)}/{len(a1)} 共同 {len(j)}/{len(j1)} "
            f"|close差|max {dc:.4f} |volume差|max {dv:.0f} |turnover相对差|max {dt:.2e} |hfq close差|max {dh:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--codes-file")
    ap.add_argument("--verify", type=int, default=0, help="只做与 akshare 的对比校验,抽 N 只")
    ap.add_argument("--delisted", action="store_true", help="抓已退市股(status=0):只要 hist + hfq")
    args = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    uni = pd.read_csv(OUT / "universe.csv", dtype=str, keep_default_na=False)
    codes = uni.loc[uni["status"] == ("0" if args.delisted else "1"), "code"].tolist()   # 现存股(status=1)/退市股
    if args.codes_file:
        keep = {ln.strip() for ln in Path(args.codes_file).read_text().splitlines() if ln.strip()}
        codes = [c for c in uni["code"] if c in keep]
    if args.verify:
        rng = random.Random(0)
        verify(rng.sample(codes, args.verify))
        return
    if args.limit:
        codes = codes[: args.limit]
    th = Throttle(args.threads)
    for ps in range(1, args.passes + 1):
        todo = [c for c in codes if not any((PARTS / f"{c}{s}").exists() for s in (".csv.gz", ".empty", ".nosina"))]
        if not todo:
            break
        log(f"第 {ps} 轮:清单 {len(codes)} 只,待抓 {len(todo)} 只,{th.allowed} 线程")
        t0 = time.time(); ok = fail = 0; errs = []; secs = []; tags = {}
        with ThreadPoolExecutor(args.threads) as ex:
            futs = [ex.submit(fetch_one, c, th, 3, args.delisted) for c in todo]
            for i, fu in enumerate(as_completed(futs), 1):
                code, n, msg, sec = fu.result()
                if msg in ("ok", "empty", "nosina"):
                    ok += 1; secs.append(sec); tags[msg] = tags.get(msg, 0) + 1
                elif msg != "skip":
                    fail += 1; errs.append((code, msg)); log(f"  失败 {code}: {msg}")
                if i % 200 == 0 or i == len(todo):
                    el = time.time() - t0
                    log(f"  {i}/{len(todo)} ok {ok} {tags} fail {fail} 已用 {el/60:.1f} 分 "
                        f"吞吐 {el/i:.2f}s/只 单股中位 {np.median(secs) if secs else float('nan'):.2f}s "
                        f"预计剩余 {el/i*(len(todo)-i)/60:.0f} 分")
        (PARTS / "_errors.json").write_text(json.dumps(errs, ensure_ascii=False, indent=1))
        log(f"第 {ps} 轮结束:ok {ok} fail {fail},耗时 {(time.time()-t0)/60:.1f} 分")
        if fail and ps < args.passes:
            time.sleep(60)
    left = [c for c in codes if not any((PARTS / f"{c}{s}").exists() for s in (".csv.gz", ".empty", ".nosina"))]
    nos = [c for c in codes if (PARTS / f"{c}.nosina").exists()]
    log(f"全部结束:未完成 {len(left)} 只 {left[:20]};新浪无数据 {len(nos)} 只 {nos[:20]}")


if __name__ == "__main__":
    main()
