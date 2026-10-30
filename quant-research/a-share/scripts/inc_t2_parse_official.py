"""INC_T2 步骤 0b:解析中证指数官网公告正文(只用标准库;按项目纪律用 `python3 -I` 运行,读取下载来的原料)。

输入  cache/_tmp/inc_t2_official_raw/ann_<id>.json(inc_t2_fetch_official.py ann 抓取)
输出  cache/_tmp/inc_t2_official_html_changes.csv   review, ann_id, index, direction(in/out), code6, name, row
      cache/_tmp/inc_t2_official_html_reserve.csv   备选名单(只记录,不用)
      cache/_tmp/inc_t2_official_text_counts.csv    review, index, n_text(正文"更换 N 只"), effective_text

正文表格的版式:每行 [调出代码, 调出名称, 调入代码, 调入名称];标题行含"调出名单 / 调入名单 / 证券代码"。
本脚本不读任何价格或收益。用法:python3 -I scripts/inc_t2_parse_official.py <项目根目录>
"""
import csv
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

INDEX_PAT = [("000300", r"沪深\s*300"), ("000905", r"中证\s*500"), ("000852", r"中证\s*1\s*000"),
             ("000903", r"中证\s*100(?!0)"), ("HK100", r"中证香港\s*100"), ("RED", r"中证红利"),
             ("A500", r"中证\s*A\s*500"), ("A50", r"中证\s*A\s*50(?!0)"), ("A100", r"中证\s*A\s*100")]


class TableText(HTMLParser):
    """把 HTML 拆成顺序事件流:('text', s) 与 ('row', [cells]);表格内的文字只进 row。"""

    def __init__(self):
        super().__init__()
        self.events = []
        self.in_table = 0
        self.row = None
        self.cell = None
        self.buf = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._flush()
            self.in_table += 1
        elif tag == "tr" and self.in_table:
            self.row = []
        elif tag in ("td", "th") and self.in_table:
            self.cell = []
        elif tag in ("p", "br", "div") and not self.in_table:
            self._flush()

    def handle_endtag(self, tag):
        if tag == "table":
            self.in_table = max(0, self.in_table - 1)
        elif tag == "tr" and self.in_table and self.row is not None:
            self.events.append(("row", self.row))
            self.row = None
        elif tag in ("td", "th") and self.in_table and self.cell is not None:
            if self.row is not None:
                self.row.append(norm("".join(self.cell)))
            self.cell = None
        elif tag in ("p", "div") and not self.in_table:
            self._flush()

    def handle_data(self, data):
        if self.in_table and self.cell is not None:
            self.cell.append(data)
        elif not self.in_table:
            self.buf.append(data)

    def _flush(self):
        s = norm("".join(self.buf))
        if s:
            self.events.append(("text", s))
        self.buf = []

    def close(self):
        super().close()
        self._flush()


def norm(s: str) -> str:
    s = s.replace("\xa0", " ").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", s).strip()


def section_of(text: str):
    """标题文字 -> (index_key, kind);kind ∈ {'change', 'reserve'}。"""
    kind = "reserve" if "备选名单" in text else ("change" if "调整名单" in text else None)
    if kind is None:
        return None
    for key, pat in INDEX_PAT:
        if re.search(pat + r"\s*指数", text):
            return key, kind
    return None


def parse_counts(text: str) -> dict:
    """正文'XX指数更换 N 只'。"""
    t = text.replace(" ", "")
    out = {}
    for key, pat in INDEX_PAT:
        m = re.search(pat.replace(r"\s*", "") + r"指数更换(\d+)只", t)
        if m:
            out[key] = int(m.group(1))
    return out


def main(root: Path) -> None:
    raw = root / "cache" / "_tmp" / "inc_t2_official_raw"
    cal = list(csv.DictReader(open(root / "research" / "ledger" / "inc_t2_review_calendar.csv", newline="")))
    changes, reserve, counts = [], [], []
    for r in cal:
        rv = r["review"]
        aid = int(r["note"].split("csindex id ")[1].split(";")[0])
        d = json.loads((raw / f"ann_{aid}.json").read_bytes())["data"]
        html = d.get("content") or ""
        p = TableText()
        p.feed(html)
        p.close()
        full = norm(re.sub(r"<[^>]+>", " ", html))
        cnt = parse_counts(full)
        m = re.search(r"于\s*([0-9 ]+年\s*[0-9 ]+月\s*[0-9 ]+日)\s*收[市盘]后生效", full)
        eff = m.group(1).replace(" ", "") if m else ""
        for key in ("000300", "000905", "000852"):
            counts.append(dict(review=rv, ann_id=aid, index=key, n_text=cnt.get(key, ""), effective_text=eff))
        sec = None
        nrow = 0
        for kind, val in p.events:
            if kind == "text":
                s = section_of(val)
                if s:
                    sec = s
                continue
            cells = [c for c in val]
            joined = " ".join(cells)
            s = section_of(joined) if len(cells) <= 2 else None
            if s:                         # 标题写在表格第一行的版式
                sec = s
                continue
            if sec is None:
                continue
            codes = [c for c in cells if re.fullmatch(r"\d{6}", c)]
            if not codes:
                continue                  # 表头行
            nrow += 1
            key, kd = sec
            if kd == "reserve":
                # [排序, 代码, 名称, 排序, 代码, 名称]
                for i, c in enumerate(cells):
                    if re.fullmatch(r"\d{6}", c):
                        rank = cells[i - 1] if i >= 1 else ""
                        name = cells[i + 1] if i + 1 < len(cells) else ""
                        reserve.append(dict(review=rv, ann_id=aid, index=key, rank=rank, code6=c, name=name))
                continue
            if len(cells) == 4:
                pairs = [("out", cells[0], cells[1]), ("in", cells[2], cells[3])]
            elif len(cells) == 2:
                pairs = [("unknown", cells[0], cells[1])]
            else:
                pairs = [("unknown", c, "") for c in codes]
            for direction, code, name in pairs:
                if re.fullmatch(r"\d{6}", code or ""):
                    changes.append(dict(review=rv, ann_id=aid, index=key, direction=direction, code6=code,
                                        name=name, row=nrow))
    tmp = root / "cache" / "_tmp"
    for name, rows, cols in (("inc_t2_official_html_changes.csv", changes,
                              ["review", "ann_id", "index", "direction", "code6", "name", "row"]),
                             ("inc_t2_official_html_reserve.csv", reserve,
                              ["review", "ann_id", "index", "rank", "code6", "name"]),
                             ("inc_t2_official_text_counts.csv", counts,
                              ["review", "ann_id", "index", "n_text", "effective_text"])):
        with open(tmp / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for x in rows:
                w.writerow(x)
        print(name, len(rows))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
