"""INC_T2 步骤 0c:官方名单定稿 + Gate 0(不读任何价格 / 收益)。

来源(预注册 §3.2):
  一级  中证指数官网公告。2020H1 ~ 2022H2 公告正文含沪深300 表(2021H1 起另含中证500 表),由
        inc_t2_parse_official.py 解析;2023H1 起名单只在 pdf 附件 —— **附件下载需用户同意,本轮未下载**,
        中证1000 全部 12 期、以及 300/500 在 2020 两期(500)与 2023H1 ~ 2025H2 的一级名单因此缺失。
  二级  baostock 历史成分快照差集(生效前快照 vs 生效后首个体现更新的快照;三期延迟见 inc_t2_bs_gap_probe.csv)。
定稿规则(偏离预注册,见报告"偏离"节):一级有则用一级,并与二级逐只核对;一级缺失时用二级,
        以公告正文"更换 N 只"计数作 Gate 0 核对。中证1000 没有任何官方来源 -> Gate 0 不通过。

输出:
  research/ledger/inc_t2_official_changes.csv     review, index, code, direction(in/out), source, name
  research/ledger/inc_t2_official_vs_baostock.csv 每期 × 指数 × 方向:一级数、二级数、交集、差异代码
  research/ledger/inc_t2_gate0.csv                每期 × 指数:正文计数、定稿计数、来源、是否通过
  cache/_tmp/inc_t2_official_members.parquet      每期生效前 / 后的 300、500 官方成分(长表)
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from afactor.config import TMP, LEDGER

CAL = LEDGER / "inc_t2_review_calendar.csv"
BS_DELAYED = {"2021H1": "2021-08-16", "2021H2": "2022-01-17", "2022H2": "2023-04-03"}


def code6_to_ts(c: str) -> str:
    c = str(c).zfill(6)
    if c[0] == "6":
        return c + ".SH"
    if c[0] in "03":
        return c + ".SZ"
    if c[0] in "48":
        return c + ".BJ"
    return c + ".??"


def bs_to_ts(c: str) -> str:
    ex, num = c.split(".")
    return f"{num}.{ex.upper()}"


def main():
    t0 = time.time()
    cal = pd.read_csv(CAL, parse_dates=["C_cutoff", "A_official_publish", "A_plus1_first_tradable",
                                        "E_minus1_passive_close", "E_effective"])
    html = pd.read_csv(TMP / "inc_t2_official_html_changes.csv", dtype={"code6": str})
    html = html[html["index"].isin(["000300", "000905"])].copy()
    assert (html["direction"] != "unknown").all(), "正文表格存在方向不明的行"
    html["code"] = html["code6"].map(code6_to_ts)
    cnt = pd.read_csv(TMP / "inc_t2_official_text_counts.csv", dtype={"index": str})
    cnt["index"] = cnt["index"].str.zfill(6)
    bsm = pd.read_parquet(TMP / "inc_t2_bs_members_300_500.parquet")
    bsd = pd.read_parquet(TMP / "inc_t2_bs_members_delayed.parquet")
    for d in (bsm, bsd):
        d["ts"] = d["code"].map(bs_to_ts)

    rows, cmp_rows, g0, mem = [], [], [], []
    for r in cal.itertuples():
        rv = r.review
        em1 = r.E_minus1_passive_close.strftime("%Y-%m-%d")
        e = r.E_effective.strftime("%Y-%m-%d")
        for idx in ("000300", "000905"):
            pre = set(bsm.query("index == @idx and query_date == @em1")["ts"])
            post_e = bsm.query("index == @idx and query_date == @e")
            if rv in BS_DELAYED:
                q = BS_DELAYED[rv]
                post = set(bsd.query("index == @idx and query_date == @q")["ts"])
                post_src = f"baostock 延迟快照 {q}"
            else:
                post = set(post_e["ts"])
                post_src = f"baostock 快照 {e}"
            bs_in, bs_out = post - pre, pre - post
            h = html[(html["review"] == rv) & (html["index"] == idx)]
            h_in, h_out = set(h.loc[h["direction"] == "in", "code"]), set(h.loc[h["direction"] == "out", "code"])
            has_html = len(h) > 0
            n_text = cnt.query("review == @rv and index == @idx")["n_text"]
            n_text = int(n_text.iloc[0]) if len(n_text) and pd.notna(n_text.iloc[0]) else np.nan
            if has_html:
                fin_in, fin_out, src = h_in, h_out, "csindex_html"
            else:
                fin_in, fin_out, src = bs_in, bs_out, "baostock"
            names = dict(zip(h["code"], h["name"]))
            for d_, s_ in (("in", fin_in), ("out", fin_out)):
                for c in sorted(s_):
                    rows.append(dict(review=rv, index=idx, code=c, direction=d_, source=src, name=names.get(c, "")))
                cmp_rows.append(dict(review=rv, index=idx, direction=d_, n_html=(len(h_in) if d_ == "in" else len(h_out)) if has_html else np.nan,
                                     n_baostock=len(bs_in) if d_ == "in" else len(bs_out), baostock_post=post_src,
                                     n_common=len((h_in if d_ == "in" else h_out) & (bs_in if d_ == "in" else bs_out)) if has_html else np.nan,
                                     only_html=";".join(sorted((h_in if d_ == "in" else h_out) - (bs_in if d_ == "in" else bs_out))) if has_html else "",
                                     only_baostock=";".join(sorted((bs_in if d_ == "in" else bs_out) - (h_in if d_ == "in" else h_out))) if has_html else ""))
            ok_cnt = (len(fin_in) == n_text and len(fin_out) == n_text) if np.isfinite(n_text) else None
            agree = (fin_in == bs_in and fin_out == bs_out) if has_html else None
            passed = bool((ok_cnt is None or ok_cnt) and (agree is None or agree)
                          and len(fin_in) == len(fin_out) and len(fin_in) > 0)
            g0.append(dict(review=rv, window=r.window, index=idx, n_text=n_text, n_in=len(fin_in), n_out=len(fin_out),
                           source=src, html_eq_baostock=agree, count_eq_text=ok_cnt,
                           gate0_pass=passed,
                           note=("正文无该指数计数(在附件),以 baostock 前后快照差集定稿" if not np.isfinite(n_text) else "")))
            after = (pre - fin_out) | fin_in
            for c in pre:
                mem.append((rv, idx, "before", c))
            for c in after:
                mem.append((rv, idx, "after", c))
        g0.append(dict(review=rv, window=r.window, index="000852",
                       n_text=int(cnt.query("review == @rv and index == '000852'")["n_text"].fillna(-1).iloc[0]) if rv >= "2023H1" else np.nan,
                       n_in=np.nan, n_out=np.nan, source="none", html_eq_baostock=None, count_eq_text=None,
                       gate0_pass=False,
                       note="中证1000 官方名单只在公告附件(xlsx/pdf);附件下载须用户同意(预注册 §12.5),本轮未下载;无第二来源"))
    R = pd.DataFrame(rows)
    R.to_csv(LEDGER / "inc_t2_official_changes.csv", index=False)
    C = pd.DataFrame(cmp_rows)
    C.to_csv(LEDGER / "inc_t2_official_vs_baostock.csv", index=False)
    G = pd.DataFrame(g0)
    G.to_csv(LEDGER / "inc_t2_gate0.csv", index=False)
    M = pd.DataFrame(mem, columns=["review", "index", "when", "code"])
    M.to_parquet(TMP / "inc_t2_official_members.parquet", index=False)
    print(G.to_string())
    print(C[["review", "index", "direction", "n_html", "n_baostock", "n_common", "only_html", "only_baostock"]].to_string())
    print(f"done {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
