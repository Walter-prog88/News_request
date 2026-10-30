"""从新浪行情接口抓取全市场快照:行业分类 + 总市值/流通市值/换手率/PE/PB。

注意:这是**当前快照**,不是历史序列。用途与其局限见 README §9。
"""
from __future__ import annotations
import json, os, re, sys, time
import urllib.request
import pandas as pd

BOARDS = "https://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php"
NODE = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        "Market_Center.getHQNodeData?page={p}&num=100&sort=symbol&asc=1&node={n}&symbol=")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
OUT = os.path.join(os.path.dirname(__file__), "..", "data_8:21", "sina_snapshot.csv")


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("gbk", errors="replace")


def main():
    txt = get(BOARDS)
    boards = {}
    for m in re.finditer(r'"(new_[a-z0-9]+)":"new_[a-z0-9]+,([^,]+),', txt):
        boards[m.group(1)] = m.group(2)
    print(f"行业板块 {len(boards)} 个", flush=True)
    rows = []
    for i, (node, name) in enumerate(sorted(boards.items()), 1):
        for page in range(1, 12):
            try:
                s = get(NODE.format(p=page, n=node))
            except Exception as e:
                print(f"  {name} p{page} 失败 {e}"); break
            s = s.strip()
            if not s or s in ("null", "[]"):
                break
            try:
                data = json.loads(s)
            except json.JSONDecodeError:
                break
            if not data:
                break
            for d in data:
                rows.append({"symbol": d.get("symbol"), "code": d.get("code"),
                             "name": d.get("name"), "industry": name, "node": node,
                             "mktcap_wan": d.get("mktcap"), "nmc_wan": d.get("nmc"),
                             "turnoverratio": d.get("turnoverratio"),
                             "pe": d.get("per"), "pb": d.get("pb"),
                             "close": d.get("trade")})
            if len(data) < 100:
                break
            time.sleep(0.15)
        if i % 10 == 0:
            print(f"  [{i}/{len(boards)}] 累计 {len(rows)} 行", flush=True)
        time.sleep(0.2)
    df = pd.DataFrame(rows).drop_duplicates("symbol")
    # 转成本项目的 ts_code 格式
    df["ts_code"] = df["symbol"].str.upper().str.replace(
        r"^SH(\d+)$", r"\1.SH", regex=True).str.replace(r"^SZ(\d+)$", r"\1.SZ", regex=True)
    df["total_mv"] = pd.to_numeric(df["mktcap_wan"], errors="coerce") * 1e4
    df["circ_mv"] = pd.to_numeric(df["nmc_wan"], errors="coerce") * 1e4
    df.to_csv(OUT, index=False)
    print(f"\n共 {len(df)} 只股票,{df['industry'].nunique()} 个行业 -> {OUT}")
    print(df.groupby("industry").size().sort_values(ascending=False).head(8).to_string())


if __name__ == "__main__":
    main()
