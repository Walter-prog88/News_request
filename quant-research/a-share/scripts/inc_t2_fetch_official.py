"""INC_T2 步骤 0a:官方名单原料抓取(前缀 inc_t2)。只做网络读取与落盘,不解析、不算任何收益。

预注册 research/INC_T2_PREREG.md §3.2 / §12.5。子命令:

  ann      读中证指数官网 12 份半年度定期调整公告的详情接口(queryAnnouncementById,JSON,公开网页数据),
           原样落盘 cache/_tmp/inc_t2_official_raw/ann_<id>.json,sha256 记入 research/ledger/inc_t2_official_files.csv。
           2020H1 ~ 2022H2 的公告正文(HTML)里直接含沪深300 / 中证500 调入调出表;2023H1 起名单只在 pdf 附件。
  bs       baostock 补取三期"更新延迟"后的首个快照(2021H1 -> 2021-08-16、2021H2 -> 2022-01-17、
           2022H2 -> 2023-04-03),与设计者已落盘的生效前后快照(inc_t2_bs_members_300_500.parquet)配合做差集。
  attach   下载公告附件(xlsx / pdf,含中证1000 名单)。**预注册 §12.5 规定须先征得用户同意**;
           本脚本只有在命令行显式给出 --user-consent 时才会下载(执行代理在 2026-10-11 本轮没有得到用户同意,未运行)。

所有输出不含 2026 年信息:公告 publishDate <= 2025-11-28;baostock 快照日期 <= 2023-04-03。
"""
from __future__ import annotations
import os, sys, json, time, hashlib, urllib.request, urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "cache" / "_tmp"
RAW = TMP / "inc_t2_official_raw"
LEDGER = ROOT / "research" / "ledger"
CAL = LEDGER / "inc_t2_review_calendar.csv"
ANN_JSON = TMP / "inc_t2_csindex_announcements_2020_2025.json"
FILES_LEDGER = LEDGER / "inc_t2_official_files.csv"
API = "https://www.csindex.com.cn/csindex-home/announcement/queryAnnouncementById?id={}"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# baostock 更新延迟三期:用"延迟后首个体现更新的快照"(见 inc_t2_bs_gap_probe.csv)
BS_DELAYED = {"2021H1": "2021-08-16", "2021H2": "2022-01-17", "2022H2": "2023-04-03"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def calendar_rows() -> list[dict]:
    import csv
    with open(CAL, newline="") as f:
        return list(csv.DictReader(f))


def ann_ids() -> dict:
    """review -> csindex 公告 id(从调样日历的 note 字段读,与设计者核对过的 12 期一致)。"""
    out = {}
    for r in calendar_rows():
        note = r["note"]
        i = note.index("csindex id ") + len("csindex id ")
        out[r["review"]] = int(note[i:].split(";")[0].strip())
    return out


def write_files_ledger(rows: list[dict]) -> None:
    import csv
    old = []
    if FILES_LEDGER.exists():
        with open(FILES_LEDGER, newline="") as f:
            old = [r for r in csv.DictReader(f)]
    keep = {r["file"]: r for r in old}
    for r in rows:
        keep[r["file"]] = r
    cols = ["review", "kind", "source_url", "file", "bytes", "sha256", "fetched_at", "note"]
    with open(FILES_LEDGER, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(keep.values(), key=lambda x: (x["review"], x["kind"], x["file"])):
            w.writerow({c: r.get(c, "") for c in cols})


def cmd_ann() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for rv, i in ann_ids().items():
        f = RAW / f"ann_{i}.json"
        if not f.exists():
            req = urllib.request.Request(API.format(i), headers=UA)
            b = urllib.request.urlopen(req, timeout=60).read()
            f.write_bytes(b)
            time.sleep(1.0)
        d = json.loads(f.read_bytes())["data"]
        assert d["publishDate"] <= "2025-11-28", "封存:公告日期越界"
        rows.append(dict(review=rv, kind="announcement_json", source_url=API.format(i),
                         file=str(f.relative_to(ROOT)), bytes=f.stat().st_size, sha256=sha256(f),
                         fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                         note=f"publishDate {d['publishDate']}; {d['title']}"))
        print(rv, i, d["publishDate"], f.stat().st_size, flush=True)
    write_files_ledger(rows)


def cmd_bs() -> None:
    import baostock as bs
    import pandas as pd
    out = TMP / "inc_t2_bs_members_delayed.parquet"
    lg = bs.login()
    assert lg.error_code == "0", lg.error_msg
    frames = []
    try:
        for rv, q in BS_DELAYED.items():
            for idx, fn in (("000300", bs.query_hs300_stocks), ("000905", bs.query_zz500_stocks)):
                rs = fn(date=q)
                assert rs.error_code == "0", rs.error_msg
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                df = pd.DataFrame(rows, columns=rs.fields)
                df["query_date"] = q
                df["index"] = idx
                df["review"] = rv
                frames.append(df)
                print(rv, idx, q, len(df), df["updateDate"].unique()[:3], flush=True)
                time.sleep(0.5)
    finally:
        bs.logout()
    D = pd.concat(frames, ignore_index=True)
    D.to_parquet(out, index=False)
    write_files_ledger([dict(review=rv, kind="baostock_delayed_snapshot", source_url="baostock query_hs300/zz500_stocks",
                             file=str(out.relative_to(ROOT)), bytes=out.stat().st_size, sha256=sha256(out),
                             fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                             note="dates " + ",".join(f"{k}:{v}" for k, v in BS_DELAYED.items()))
                        for rv in ["ALL"]])


def attachment_urls() -> list[tuple[str, str, str]]:
    """(review, fileName, url):接口 enclosureList 优先;2020H1~2021H1 附件在正文 href 里(/indices/info/files/*.xlsx)。"""
    import re
    ids = ann_ids()
    meta = {a["id"]: a for a in json.load(open(ANN_JSON))}
    out = []
    for rv, i in ids.items():
        encl = meta.get(i, {}).get("enclosures", [])
        if encl:
            for e in encl:
                out.append((rv, e["fileName"], e["fileUrl"]))
            continue
        f = RAW / f"ann_{i}.json"
        if f.exists():
            c = json.loads(f.read_bytes())["data"].get("content", "")
            for h in re.findall(r'href="([^"]+\.(?:xlsx|xls|pdf))"', c):
                url = h if h.startswith("http") else "https://www.csindex.com.cn" + h
                out.append((rv, h.rsplit("/", 1)[-1], url))
    return out


def cmd_attach(consent: bool) -> None:
    lst = attachment_urls()
    for rv, name, url in lst:
        print(rv, name, url)
    if not consent:
        print("\n未下载:预注册 §12.5 规定附件下载须先征得用户同意。用户同意后运行\n"
              "  python3 scripts/inc_t2_fetch_official.py attach --user-consent\n"
              "下载到 cache/_tmp/inc_t2_official_raw/attach/(独立目录),再用 python3 -I 解析。")
        return
    dst = RAW / "attach"
    dst.mkdir(parents=True, exist_ok=True)
    rows = []
    seen: dict = {}
    for rv, name, url in lst:
        parts = urllib.parse.urlsplit(url)
        safe = urllib.parse.urlunsplit(parts._replace(path=urllib.parse.quote(urllib.parse.unquote(parts.path))))
        ext = name.rsplit(".", 1)[-1].lower()
        seen[rv] = seen.get(rv, 0) + 1                   # 同一期可能有多个附件(如 2020H1、2021H1 各两份)
        f = dst / f"{rv}_{seen[rv]}.{ext}"
        if not f.exists():
            req = urllib.request.Request(safe, headers={"User-Agent": "Mozilla/5.0"})
            f.write_bytes(urllib.request.urlopen(req, timeout=120).read())
            time.sleep(1.0)
        rows.append(dict(review=rv, kind="attachment", source_url=url, file=str(f.relative_to(ROOT)),
                         bytes=f.stat().st_size, sha256=sha256(f), fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                         note=name))
    write_files_ledger(rows)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "ann":
        cmd_ann()
    elif cmd == "bs":
        cmd_bs()
    elif cmd == "attach":
        cmd_attach("--user-consent" in sys.argv[2:])
    else:
        print(__doc__)
