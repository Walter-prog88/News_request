"""lh_basis 第 1 步:抓取中金所股指期货历史日行情 + 全收益指数 + 未到期合约规格。

数据源(都只取 <= 2025-12-31):
  1. 中金所月度历史包  http://www.cffex.com.cn/sj/historysj/YYYYMM/zip/YYYYMM.zip
     (akshare.get_cffex_daily 的底层来源;这里按月整包下载,一个月一个请求,比逐日快 20 倍)
     2010-04 .. 2025-12 共 189 个月,原样存 cache/_tmp/lh_basis_raw/YYYYMM.zip
  2. 中证全收益指数 H00300 / H00905 / H00852 / H00016(akshare.stock_zh_index_hist_csindex)
     用来把"对价格指数的对冲成本"换算成"对全收益指数的对冲成本"(股息部分)
  4. SHIBOR 3M(akshare.rate_interbank),"期货对冲 vs 卖股持现金"的比较口径
  3. 2025-12-31 当日中金所交易参数(akshare.futures_contract_info_cffex),
     只为拿到 2025-12-31 仍在交易、到期日在 2026 年的合约的"最后交易日"(合约规格,上市时即公布)

纪律: 并行 4 线程;每个请求有超时 + 重试;失败写日志,可续跑(已存在且校验通过的文件跳过)。
用法: python3 scripts/lh_basis_fetch.py  [--workers 4]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "cache" / "_tmp" / "lh_basis_raw"
RAW.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "logs" / "lh_basis_fetch.log"
CUTOFF = "20251231"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def months(start="2010-04", end="2025-12"):
    return [p.strftime("%Y%m") for p in pd.period_range(start, end, freq="M")]


def zip_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 200:
        return False
    try:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith("_1.csv")]
            return len(names) >= 5          # 一个月至少 5 个交易日(2010-04 从 16 日开始也有 11 天)
    except zipfile.BadZipFile:
        return False


def fetch_month(ym: str, retries: int = 4) -> tuple[str, str]:
    out = RAW / f"{ym}.zip"
    if zip_ok(out):
        return ym, "cached"
    url = f"http://www.cffex.com.cn/sj/historysj/{ym}/zip/{ym}.zip"
    err = ""
    for k in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=(10, 60))
            if r.status_code != 200:
                err = f"http {r.status_code}"
            else:
                tmp = out.with_suffix(".part")
                tmp.write_bytes(r.content)
                if zip_ok(tmp):
                    tmp.rename(out)
                    return ym, f"ok {len(r.content)}B"
                err = f"bad zip ({len(r.content)}B, head={r.content[:40]!r})"
                tmp.unlink(missing_ok=True)
        except Exception as e:                         # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        time.sleep(1.5 * (k + 1))
    return ym, f"FAIL {err}"


def fetch_tr_indices() -> None:
    """中证全收益指数(收盘),直接请求 akshare.stock_zh_index_hist_csindex 用的同一个 csindex 接口。
    本机 python3.9 的 LibreSSL 经代理连 csindex 会 SSLEOFError,所以用 curl(-m 150 超时)取 JSON。按年分段(长区间响应常被截断)。"""
    import subprocess

    def one(code):
        out = RAW / f"tr_{code}.parquet"
        if out.exists():
            log(f"tr {code} cached")
            return
        parts = []
        for y0 in range(2005, 2026):
            s, e = f"{y0}0101", f"{y0}1231"
            url = (f"https://www.csindex.com.cn/csindex-home/perf/index-perf?indexCode={code}"
                   f"&startDate={s}&endDate={e}")
            part = RAW / "tr_parts" / f"{code}_{y0}.parquet"      # 按年缓存,失败可续跑
            part.parent.mkdir(exist_ok=True)
            if part.exists():
                parts.append(pd.read_parquet(part))
                continue
            for k in range(6):
                try:
                    r = subprocess.run(["curl", "-s", "-m", "150", url], capture_output=True, timeout=180)
                    js = json.loads(r.stdout.decode("utf-8"))
                    d = pd.DataFrame(js["data"])[["tradeDate", "close"]]
                    if len(d) > 400 or len(d) < 200:
                        raise ValueError(f"rows {len(d)} outside [200, 400] guard")
                    d.to_parquet(part)
                    parts.append(d)
                    break
                except Exception as ex:                # noqa: BLE001
                    log(f"tr {code} {s}-{e} try{k} {type(ex).__name__}: {ex}")
                    time.sleep(2 * (k + 1))
        if len(parts) < 21:
            log(f"tr {code} FAIL: only {len(parts)}/21 years")
            return
        # 按年请求时,部分年份的 12-31 不在返回里(2008/2009/2010 实测缺);对末日 < 12-31 的年份补一个跨年小窗口
        for y0 in range(2005, 2025):
            py = parts[y0 - 2005]
            if str(py["tradeDate"].max())[4:] == "1231":
                continue
            part = RAW / "tr_parts" / f"{code}_{y0}_yearend.parquet"
            if part.exists():
                parts.append(pd.read_parquet(part))
                continue
            url = (f"https://www.csindex.com.cn/csindex-home/perf/index-perf?indexCode={code}"
                   f"&startDate={y0}1224&endDate={y0 + 1}0106")
            for k in range(6):
                try:
                    r = subprocess.run(["curl", "-s", "-m", "60", url], capture_output=True, timeout=90)
                    d = pd.DataFrame(json.loads(r.stdout.decode("utf-8"))["data"])[["tradeDate", "close"]]
                    if not 1 <= len(d) <= 20:
                        raise ValueError(f"rows {len(d)}")
                    d.to_parquet(part)
                    parts.append(d)
                    break
                except Exception as ex:                # noqa: BLE001
                    log(f"tr {code} yearend {y0} try{k} {type(ex).__name__}: {ex}")
                    time.sleep(2 * (k + 1))
        d = pd.concat(parts).drop_duplicates("tradeDate")
        d["date"] = pd.to_datetime(d["tradeDate"], format="%Y%m%d")
        d = d[d["date"] <= CUTOFF].set_index("date")["close"].astype(float).sort_index().rename(code).to_frame()
        d.to_parquet(out)
        log(f"tr {code} rows={len(d)} {d.index.min().date()}..{d.index.max().date()}")

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, ["H00300", "H00905", "H00852", "H00016"]))


def fetch_shibor() -> None:
    """SHIBOR 3 个月(年化 %,akshare.rate_interbank,2006-10 起)。用于"期货对冲 vs 卖股持现金"的比较口径。"""
    import akshare as ak
    import signal

    out = RAW / "shibor_3m.parquet"
    if out.exists():
        log("shibor cached")
        return

    def handler(signum, frame):                        # noqa: ARG001
        raise TimeoutError("shibor timeout")

    for k in range(3):
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(180)
            d = ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="3月")
            signal.alarm(0)
            d["date"] = pd.to_datetime(d["报告日"])
            d = d[d["date"] <= CUTOFF].set_index("date")["利率"].astype(float).rename("shibor_3m").to_frame()
            d.to_parquet(out)
            log(f"shibor rows={len(d)} {d.index.min().date()}..{d.index.max().date()}")
            return
        except Exception as ex:                        # noqa: BLE001
            signal.alarm(0)
            log(f"shibor try{k} {type(ex).__name__}: {ex}")
            time.sleep(3 * (k + 1))
    log("shibor FAIL")


def fetch_live_specs() -> None:
    """2025-12-31 交易参数(含最后交易日)。若当日无文件,向前找最近一个交易日。"""
    import akshare as ak
    out = RAW / "contract_info_20251231.parquet"
    if out.exists():
        log("contract info cached")
        return
    for d in ["20251231", "20251230", "20251229"]:
        try:
            df = ak.futures_contract_info_cffex(date=d)
            if len(df):
                df = df.astype(str)
                df.to_parquet(out)
                log(f"contract info {d} rows={len(df)}")
                return
        except Exception as ex:                        # noqa: BLE001
            log(f"contract info {d} {type(ex).__name__}: {ex}")
    log("contract info FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", choices=["zips", "tr", "shibor", "specs"], default=None)
    a = ap.parse_args()
    if a.only == "shibor":
        return fetch_shibor()
    if a.only == "specs":
        return fetch_live_specs()
    if a.only == "tr":
        return fetch_tr_indices()
    ms = months()
    log(f"start: {len(ms)} months, workers={a.workers}")
    fails = []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch_month, m): m for m in ms}
        for f in as_completed(futs):
            ym, st = f.result()
            if st.startswith("FAIL"):
                fails.append(ym)
            if not st.startswith("cached"):
                log(f"{ym} {st}")
    log(f"zips done: {len(ms) - len(fails)}/{len(ms)} ok, fails={fails}")
    (RAW / "fetch_status.json").write_text(json.dumps({"fails": fails, "n": len(ms)}))
    fetch_shibor()
    fetch_tr_indices()
    fetch_live_specs()
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
