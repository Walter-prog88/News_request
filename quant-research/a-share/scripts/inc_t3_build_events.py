"""INC-T3 S0:数据验收 G1-G4 + 事件表(读任何收益之前)。

预注册 research/INC_T3_PREREG.md §3、§10、§12 第 1 步。本脚本**不读取任何价格收益**:
只用东财时间表(丢掉其事后收益列)、新浪解禁页公告日期、流通股本 / 上市日 / ST / 决策池布尔面板、
以及 mvfix 跳升事件表的(代码、日期、比例、送转标签)四列。

用法
  python3 scripts/inc_t3_build_events.py events      # G1 / G2 / G3 + 事件表落盘
  python3 scripts/inc_t3_build_events.py sina        # G4 新浪抽样(网络,约 5 分钟;已抓的页面缓存复用)
  python3 scripts/inc_t3_build_events.py manifest    # 写 inc_t3_prereg_manifest.json(S0 结束)
产出
  research/ledger/inc_t3_unlock_events.csv、inc_t3_sina_announce_sample.csv、inc_t3_acceptance.csv、
  inc_t3_prereg_manifest.json;cache/_tmp/inc_t3_sina_xsjj/*.html
"""
from __future__ import annotations
import os, sys, time, re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t3_common as C                                   # noqa: E402
from afactor.panel import Panel                             # noqa: E402
from afactor import strategy as ST, newdata as ND           # noqa: E402

ACC = C.LEDGER / "inc_t3_acceptance.csv"
SINA_CSV = C.LEDGER / "inc_t3_sina_announce_sample.csv"


def upsert_acceptance(rows: list[dict]):
    new = pd.DataFrame(rows)
    if ACC.exists():
        old = pd.read_csv(ACC)
        old = old[~old["item"].isin(new["item"])]
        new = pd.concat([old, new], ignore_index=True)
    new["written_at"] = new.get("written_at", pd.Series(dtype=object)).fillna(
        pd.Timestamp.now().isoformat(timespec="seconds"))
    new.to_csv(ACC, index=False)


# ====================================================================== G1-G3 + 事件表
def cmd_events():
    t0 = time.time()
    rows = []
    # ---- G1 快照 ----
    raw_md5 = C.raw_concat_md5()
    pq_md5 = C.md5(C.EM_PARQUET)
    em = C.load_em()
    full = pd.read_parquet(C.EM_PARQUET, columns=["FREE_DATE"])
    g1 = (raw_md5 == C.EM_RAW_MD5) and (pq_md5 == C.EM_PARQUET_MD5) and full["FREE_DATE"].max() <= C.CUT
    rows.append(dict(gate="G1", item="snapshot_md5", value=f"raw {raw_md5} / parquet {pq_md5} / max FREE_DATE "
                     f"{full['FREE_DATE'].max().date()}", threshold="= 预注册 md5 且 <= 2025-12-31", passed=g1))
    print(rows[-1], flush=True)
    if not g1:
        upsert_acceptance(rows)
        raise SystemExit("G1 不通过:停")

    # ---- 映射 ----
    p = Panel()
    dates, codes = p.dates, p.codes
    T, N = len(dates), len(codes)
    em["code"] = [C.ts_code(s, m) for s, m in zip(em["SECURITY_CODE"], em["MARKET_TYPE"])]
    em["j"] = codes.get_indexer(em["code"])
    n_all = len(em)
    in_panel = em["j"] >= 0
    rows.append(dict(gate="G2", item="map_share", value=f"{in_panel.mean():.4f} ({int(in_panel.sum())}/{n_all})",
                     threshold=">= 0.90", passed=bool(in_panel.mean() >= 0.90)))
    print(rows[-1], flush=True)
    ev = em[in_panel].copy()
    # 事件日 D:FREE_DATE 非交易日取其后第一个交易日
    dpos = np.searchsorted(dates.to_numpy(), ev["FREE_DATE"].to_numpy(), side="left")
    ok = dpos < T
    ev = ev[ok].copy(); dpos = dpos[ok]
    ev["d_pos"] = dpos
    ev["D"] = dates[dpos]
    ev["D_shifted"] = ev["D"] != ev["FREE_DATE"]

    # ---- 规模变量 ----
    fs = ND.float_share(dates, codes).to_numpy(np.float64)          # 股
    tref = ev["d_pos"].to_numpy() - 21
    ev["t_ref_pos"] = tref
    jj = ev["j"].to_numpy()
    fs_ref = np.where(tref >= 0, fs[np.clip(tref, 0, T - 1), jj], np.nan)
    ev["float_share_ref"] = fs_ref
    able = ev["ABLE_FREE_SHARES"].to_numpy(np.float64) * 1e4         # 万股 -> 股
    cur = ev["CURRENT_FREE_SHARES"].to_numpy(np.float64) * 1e4
    rp = able / fs_ref
    rp[~(fs_ref > 0)] = np.nan
    fb = ev["FREE_RATIO"].to_numpy(np.float64) * np.where(cur > 0, able / np.where(cur > 0, cur, 1.0), np.nan)
    ev["ratio_plan_src"] = np.where(np.isfinite(rp), "float_share_ref", np.where(np.isfinite(fb), "fallback", "nan"))
    ev["ratio_plan"] = np.where(np.isfinite(rp), rp, fb)
    # 东财 FREE_RATIO 的分母实测是**解禁后**流通股本(= 实际 /(解禁前流通 + 实际),上界 1;
    # FREE_RATIO/(1-FREE_RATIO)×计划/实际 与 ratio_plan 的比值中位 1.000)。预注册 §3.3 把 ratio_actual 定义为
    # "实际解禁 / 解禁前流通",按这一语义换算;东财原值另存 ratio_actual_em(偏离清单 #1,读收益之前)。
    fr = ev["FREE_RATIO"].astype(np.float64)
    ev["ratio_actual_em"] = fr
    ev["ratio_actual"] = np.where(fr < 1.0, fr / (1.0 - fr), np.nan)
    ev["ratio_current_ref"] = cur / np.where(fs_ref > 0, fs_ref, np.nan)
    ge5 = ev["FREE_RATIO"] >= 0.05
    miss_fs = float((ev.loc[ge5, "ratio_plan_src"] != "float_share_ref").mean())
    nan_rp = float(ev.loc[ge5, "ratio_plan"].isna().mean())
    rows.append(dict(gate="G2", item="ratio_plan_nan_share_ge5", value=f"{nan_rp:.4f}(float_share 缺失需退回 {miss_fs:.4f})",
                     threshold="<= 0.03", passed=bool(nan_rp <= 0.03)))
    print(rows[-1], flush=True)
    # 计划 vs 东财口径的一致性(描述)
    both = ev[np.isfinite(rp) & (ev["FREE_RATIO"] > 0) & (cur > 0)]
    rel = (both["ratio_plan"] / (both["FREE_RATIO"] * both["ABLE_FREE_SHARES"] / both["CURRENT_FREE_SHARES"]))
    rel2 = (both["ratio_plan"] / (both["ratio_actual"] * both["ABLE_FREE_SHARES"] / both["CURRENT_FREE_SHARES"]))
    rows.append(dict(gate="info", item="ratio_plan_vs_em", value=f"ratio_plan / (FREE_RATIO×计划/实际) 中位 {rel.median():.3f},"
                     f"p90 {rel.quantile(.9):.3f}(东财分母为解禁后流通);换算 FREE_RATIO/(1-FREE_RATIO) 后中位 "
                     f"{rel2.median():.3f}, p10 {rel2.quantile(.1):.3f}, p90 {rel2.quantile(.9):.3f}",
                     threshold="描述", passed=True))

    # ---- 类型 / 层 / 分组 ----
    ev["first_type"] = ev["FREE_SHARES_TYPE"].map(C.first_type)
    ev["tg"] = ev["FREE_SHARES_TYPE"].map(C.type_group)
    ev["tier"] = ev["FREE_SHARES_TYPE"].map(C.tier_of)
    ev["size_bucket"] = C.size_bucket(ev["ratio_plan"])
    ev["size_bucket_actual"] = C.size_bucket(ev["ratio_actual"])
    ev["K_pos"] = np.where(ev["tier"] == "A", ev["d_pos"] - C.K_A, ev["d_pos"] - C.K_B)
    ev["K_date"] = [dates[k] if k >= 0 else pd.NaT for k in ev["K_pos"]]
    ev["t_ref_date"] = [dates[k] if k >= 0 else pd.NaT for k in tref]

    # ---- 在池(D-21 的 V1000 池且非 ST)----
    U = ST.universe(p, C.VERSION).to_numpy(bool)
    st = p["is_st"].reindex(index=dates, columns=codes).fillna(False).to_numpy(bool)
    inp = np.zeros(len(ev), bool)
    okr = tref >= 0
    inp[okr] = U[tref[okr], jj[okr]] & ~st[tref[okr], jj[okr]]
    ev["in_pool"] = inp
    # ---- 上市年龄(D-21)----
    age = C.listing_age_panel(dates, codes)
    ev["age_ref"] = np.where(okr, age[np.clip(tref, 0, T - 1), jj], np.nan)
    ev["age_bucket"] = C.age_bucket(ev["age_ref"].to_numpy())
    li = ND.listing_info(codes)
    ev["list_date"] = li["list_date"].reindex(ev["code"]).to_numpy()
    del age, fs

    # ---- 重叠:同一股票在 [D-20, D+60] 内另有 ratio_plan >= 5% 的事件 ----
    ev = ev.sort_values(["code", "d_pos"]).reset_index(drop=True)
    ev["event_idx"] = np.arange(len(ev))
    big = ev[ev["ratio_plan"] >= 0.05][["code", "d_pos", "event_idx"]]
    bmap = {c: g[["d_pos", "event_idx"]].to_numpy() for c, g in big.groupby("code")}
    ov = np.zeros(len(ev), bool)
    for i, (c, d, ei) in enumerate(zip(ev["code"], ev["d_pos"], ev["event_idx"])):
        a = bmap.get(c)
        if a is None:
            continue
        m = (a[:, 0] >= d - 20) & (a[:, 0] <= d + 60) & (a[:, 1] != ei)
        ov[i] = bool(m.any())
    ev["overlap"] = ov

    # ---- G3 交叉验证 ----
    jp = pd.read_csv(C.JUMPS, usecols=["code", "date", "r", "adj_label"])
    jp["date"] = pd.to_datetime(jp["date"])
    nz = jp[(jp["r"] >= 1.05) & (~jp["adj_label"].astype(bool))].copy()
    em_all = em.copy()
    key_free = set(zip(em_all["code"], em_all["FREE_DATE"]))
    hit = np.array([(c, d) in key_free for c, d in zip(nz["code"], nz["date"])])
    rows.append(dict(gate="G3", item="jump_same_day_hit", value=f"{hit.mean():.4f} ({int(hit.sum())}/{len(nz)})",
                     threshold=">= 0.75", passed=bool(hit.mean() >= 0.75)))
    print(rows[-1], flush=True)
    m_jump = nz[hit].merge(em_all[["code", "FREE_DATE", "FREE_RATIO"]], left_on=["code", "date"],
                           right_on=["code", "FREE_DATE"], how="left")
    cc = np.corrcoef(np.log(m_jump["r"]), np.log1p(m_jump["FREE_RATIO"]))[0, 1]
    rows.append(dict(gate="info", item="jump_size_corr", value=f"{cc:.3f}", threshold="描述(设计者 0.946)", passed=True))
    # 反向:东财 >= 5% 批次被跳升(|lag|<=3 交易日)命中
    jset = {}
    for c, g in nz.groupby("code"):
        jset[c] = dates.get_indexer(g["date"])
    rev = []
    for c, d in zip(ev.loc[ev["FREE_RATIO"] >= 0.05, "code"], ev.loc[ev["FREE_RATIO"] >= 0.05, "d_pos"]):
        a = jset.get(c)
        rev.append(bool(a is not None and (np.abs(a - d) <= 3).any()))
    rows.append(dict(gate="info", item="em_ge5_reverse_hit", value=f"{np.mean(rev):.4f}(面板内 n={len(rev)})",
                     threshold="描述(设计者全表 0.913)", passed=True))
    # C 层事件(事后上界,§7.5):非送转 r >= 1.05 的全部跳升,单独存
    nz["j"] = codes.get_indexer(nz["code"])
    nz["d_pos"] = dates.get_indexer(nz["date"])
    nz = nz[(nz["j"] >= 0) & (nz["d_pos"] >= 0)].copy()
    nz["ratio_jump"] = nz["r"] - 1.0
    nz[["code", "date", "d_pos", "j", "r", "ratio_jump"]].to_csv(C.LEDGER / "inc_t3_c_tier_events.csv", index=False)

    # ---- 落盘 ----
    keep = ["event_idx", "SECURITY_CODE", "code", "j", "MARKET_TYPE", "FREE_DATE", "D", "d_pos", "D_shifted",
            "t_ref_pos", "t_ref_date", "K_pos", "K_date", "FREE_SHARES_TYPE", "first_type", "tg", "tier",
            "ABLE_FREE_SHARES", "CURRENT_FREE_SHARES", "FREE_RATIO", "TOTAL_RATIO", "BATCH_HOLDER_NUM",
            "float_share_ref", "ratio_plan", "ratio_plan_src", "ratio_actual", "ratio_actual_em", "ratio_current_ref",
            "size_bucket", "size_bucket_actual", "in_pool", "age_ref", "age_bucket", "list_date", "overlap"]
    out = ev[keep]
    out.to_csv(C.EVENTS_CSV, index=False)
    ev_md5 = C.md5(C.EVENTS_CSV)
    rows.append(dict(gate="info", item="events_csv", value=f"{len(out)} 行,md5 {ev_md5}", threshold="描述", passed=True))
    # 描述性计数
    yr = out["D"].dt.year
    for tag, m in (("all", np.ones(len(out), bool)), ("in_pool", out["in_pool"].to_numpy()),
                   ("in_pool_ge10", (out["in_pool"] & (out["ratio_plan"] >= 0.10)).to_numpy()),
                   ("in_pool_ge5", (out["in_pool"] & (out["ratio_plan"] >= 0.05)).to_numpy())):
        vc = yr[m].value_counts().sort_index()
        rows.append(dict(gate="info", item=f"n_events_{tag}", value=f"{int(m.sum())}(" + "/".join(
            f"{y}:{n}" for y, n in vc.items()) + ")", threshold="描述", passed=True))
    vc = out[out["in_pool"] & (out["ratio_plan"] >= 0.05)].groupby(["tier", "tg"]).size()
    rows.append(dict(gate="info", item="in_pool_ge5_by_tier_tg", value="; ".join(f"{a}-{b}:{n}" for (a, b), n in vc.items()),
                     threshold="描述", passed=True))
    upsert_acceptance(rows)
    print(pd.DataFrame(rows).to_string())
    print(f"done events {time.time()-t0:.0f}s")


# ====================================================================== G4 新浪抽样
def _parse_sina(html: str, code6: str) -> pd.DataFrame:
    tb = html[html.find('id="dataTable"'):]
    recs = []
    for tr in re.findall(r"<tr>(.*?)</tr>", tb, flags=re.S):
        tds = re.findall(r"<td>(.*?)</td>", tr, flags=re.S)
        tds = [re.sub(r"<[^>]+>", "", x).strip() for x in tds]
        if len(tds) < 7:
            continue
        recs.append(dict(s_code=tds[0].zfill(6), s_name=tds[1], s_free_date=tds[2], s_qty_wan=tds[3],
                         s_mv_yi=tds[4], s_batch=tds[5], s_announce=tds[6]))
    d = pd.DataFrame(recs)
    if len(d) == 0:
        return d
    d["s_free_date"] = pd.to_datetime(d["s_free_date"], errors="coerce")
    d["s_announce"] = pd.to_datetime(d["s_announce"], errors="coerce")
    d["s_qty_wan"] = pd.to_numeric(d["s_qty_wan"].str.replace(",", ""), errors="coerce")
    d = d[(d["s_code"] == code6) & (d["s_free_date"] <= C.CUT)]       # 只用 <= 2025-12-31 的解禁行
    return d


def _fetch(sym: str, sess) -> str | None:
    f = C.SINA_DIR / f"{sym}.html"
    if f.exists() and f.stat().st_size > 5000:
        return f.read_bytes().decode("gbk", errors="replace")
    for url in (f"https://vip.stock.finance.sina.com.cn/q/go.php/vInvestConsult/kind/xsjj/index.phtml?symbol={sym}",
                f"http://vip.stock.finance.sina.com.cn/q/go.php/vInvestConsult/kind/xsjj/index.phtml?symbol={sym}"):
        for _ in range(3):                                   # 首次 + 重试 2 次
            try:
                r = sess.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200 and len(r.content) > 5000:
                    f.write_bytes(r.content)
                    time.sleep(0.25)
                    return r.content.decode("gbk", errors="replace")
            except Exception:
                time.sleep(1.0)
    return None


def cmd_sina(n_target: int = 420):
    import requests
    t0 = time.time()
    C.SINA_DIR.mkdir(parents=True, exist_ok=True)
    ev = C.load_events()
    A = ev[(ev["tier"] == "A") & (ev["D"] >= pd.Timestamp("2020-01-01"))].copy()
    # 分层:首发 / 定增 / 追加承诺 / 其他(股权分置等);小层至少 40 批,其余按批次数比例
    strata = A["tg"].where(A["tg"].isin(["首发", "定增", "追加承诺"]), "其他A")
    A["stratum"] = strata
    cnt = A["stratum"].value_counts()
    alloc = {}
    small = [s for s in cnt.index if cnt[s] * n_target / len(A) < 40]
    for s in small:
        alloc[s] = int(min(40, cnt[s]))
    rest = n_target - sum(alloc.values())
    big = [s for s in cnt.index if s not in small]
    tot_big = cnt[big].sum()
    for s in big:
        alloc[s] = int(round(rest * cnt[s] / tot_big))
    rng = np.random.default_rng([C.SEED, 30])
    chosen, used_codes = [], set()
    for s in sorted(alloc):                                  # 层顺序固定
        g = A[A["stratum"] == s]
        order = rng.permutation(len(g))
        k = 0
        for i in order:
            r = g.iloc[i]
            if r["code"] in used_codes:
                continue
            chosen.append(r); used_codes.add(r["code"]); k += 1
            if k >= alloc[s]:
                break
    S = pd.DataFrame(chosen).reset_index(drop=True)
    print(f"抽样 {len(S)} 批 / {S['code'].nunique()} 只;分层 {alloc}", flush=True)
    sess = requests.Session()
    out = []
    for i, r in S.iterrows():
        code6 = r["code"][:6]
        sym = ("sh" if r["code"].endswith(".SH") else "sz") + code6
        html = _fetch(sym, sess)
        rec = dict(event_idx=int(r["event_idx"]), code=r["code"], stratum=r["stratum"], tg=r["tg"],
                   first_type=r["first_type"], FREE_DATE=r["FREE_DATE"], D=r["D"],
                   CURRENT_FREE_SHARES=r["CURRENT_FREE_SHARES"], ABLE_FREE_SHARES=r["ABLE_FREE_SHARES"],
                   fetched=html is not None, n_rows=np.nan, date_match=False, qty_match=False,
                   s_qty_wan=np.nan, s_announce=pd.NaT, lead_days=np.nan)
        if html is not None:
            d = _parse_sina(html, code6)
            rec["n_rows"] = len(d)
            if len(d):
                hit = d[d["s_free_date"] == r["FREE_DATE"]]
                if len(hit):
                    rec["date_match"] = True
                    q = hit["s_qty_wan"].to_numpy(np.float64)
                    rel = np.abs(q - r["CURRENT_FREE_SHARES"]) / max(abs(r["CURRENT_FREE_SHARES"]), 1e-9)
                    b = int(np.nanargmin(rel)) if np.isfinite(rel).any() else 0
                    h = hit.iloc[b]
                    rec.update(qty_match=bool(rel[b] <= 0.01), s_qty_wan=h["s_qty_wan"], s_announce=h["s_announce"],
                               lead_days=float((r["FREE_DATE"] - h["s_announce"]).days)
                               if pd.notna(h["s_announce"]) else np.nan)
        out.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(S)} [{time.time()-t0:.0f}s]", flush=True)
    R = pd.DataFrame(out)
    R["ok_30d"] = R["lead_days"] >= 30
    R.to_csv(SINA_CSV, index=False)
    m = R["date_match"]
    share_matched = float(R.loc[m, "ok_30d"].mean()) if m.any() else np.nan
    share_cons = float(R["ok_30d"].fillna(False).mean())
    mq = m & R["qty_match"]
    share_qty = float(R.loc[mq, "ok_30d"].mean()) if mq.any() else np.nan
    by = R[m].groupby("stratum")["ok_30d"].agg(["mean", "size"])
    rows = [dict(gate="G4", item="sina_announce_le_D-30_matched",
                 value=f"{share_matched:.4f}(日期匹配 {int(m.sum())}/{len(R)} 批,{R.loc[m,'code'].nunique()} 只;"
                       f"日期+数量匹配子集 {share_qty:.4f}, n={int(mq.sum())};未匹配按失败计 {share_cons:.4f})",
                 threshold=">= 0.95(日期匹配样本)", passed=bool(share_matched >= 0.95)),
            dict(gate="info", item="sina_by_stratum",
                 value="; ".join(f"{s}:{v['mean']:.3f}(n={int(v['size'])})" for s, v in by.iterrows()),
                 threshold="描述", passed=True),
            dict(gate="info", item="sina_lead_days_quantiles",
                 value=f"p5 {R.loc[m,'lead_days'].quantile(.05):.0f} / p50 {R.loc[m,'lead_days'].median():.0f} / "
                       f"min {R.loc[m,'lead_days'].min():.0f}(自然日)", threshold="描述", passed=True),
            dict(gate="info", item="sina_fetch", value=f"抓取成功 {int(R['fetched'].sum())}/{len(R)};"
                 f"{time.time()-t0:.0f}s", threshold="描述", passed=True)]
    upsert_acceptance(rows)
    print(pd.DataFrame(rows).to_string())


def cmd_sina_retry():
    """G4 补抓(执行者第二段,读任何收益之前):首轮抓取里有一批页面 http 200 但 dataTable 为空表
    (连续出现,疑似限流返回空表;旧 _fetch 只看字节数 > 5000,把空表当成功并缓存)。
    这里对 n_rows == 0 的样本重新抓取,存到 inc_t3_sina_xsjj_retry/(不覆盖、不删除首轮缓存),
    间隔 2 秒、空表最多重试 3 次;重算 G4 并把两种分母(日期匹配 / 全部 420 批)都落盘。"""
    import requests
    t0 = time.time()
    rdir = C.TMP / "inc_t3_sina_xsjj_retry"
    rdir.mkdir(parents=True, exist_ok=True)
    R = pd.read_csv(SINA_CSV, parse_dates=["FREE_DATE", "D", "s_announce"])
    if "retry" not in R.columns:
        R["retry"] = False
    sess = requests.Session()
    todo = R.index[(R["n_rows"].fillna(0) == 0)]
    print(f"空表样本 {len(todo)} 批,补抓", flush=True)
    for i in todo:
        r = R.loc[i]
        code6 = r["code"][:6]
        sym = ("sh" if r["code"].endswith(".SH") else "sz") + code6
        f = rdir / f"{sym}.html"
        html, d = None, pd.DataFrame()
        for k in range(4):
            if f.exists() and k == 0:
                html = f.read_bytes().decode("gbk", errors="replace")
            else:
                try:
                    resp = sess.get(f"https://vip.stock.finance.sina.com.cn/q/go.php/vInvestConsult/kind/xsjj/"
                                    f"index.phtml?symbol={sym}", timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                    if resp.status_code == 200:
                        f.write_bytes(resp.content)
                        html = resp.content.decode("gbk", errors="replace")
                except Exception:
                    html = None
                time.sleep(2.0)
            if html is not None:
                d = _parse_sina(html, code6)
                if len(d):
                    break
        R.loc[i, "retry"] = True
        R.loc[i, "n_rows"] = len(d)
        if len(d):
            hit = d[d["s_free_date"] == r["FREE_DATE"]]
            if len(hit):
                q = hit["s_qty_wan"].to_numpy(np.float64)
                rel = np.abs(q - r["CURRENT_FREE_SHARES"]) / max(abs(r["CURRENT_FREE_SHARES"]), 1e-9)
                b = int(np.nanargmin(rel)) if np.isfinite(rel).any() else 0
                h = hit.iloc[b]
                R.loc[i, "date_match"] = True
                R.loc[i, "qty_match"] = bool(rel[b] <= 0.01)
                R.loc[i, "s_qty_wan"] = h["s_qty_wan"]
                R.loc[i, "s_announce"] = h["s_announce"]
                R.loc[i, "lead_days"] = float((r["FREE_DATE"] - h["s_announce"]).days) if pd.notna(h["s_announce"]) else np.nan
        print(f"  {r['code']} rows {len(d)} match {bool(R.loc[i, 'date_match'])} [{time.time()-t0:.0f}s]", flush=True)
    R["date_match"] = R["date_match"].astype(bool)
    R["ok_30d"] = R["lead_days"] >= 30
    R.to_csv(SINA_CSV, index=False)
    m = R["date_match"]
    share_matched = float(R.loc[m, "ok_30d"].mean())
    share_cons = float(R["ok_30d"].fillna(False).mean())
    mq = m & R["qty_match"].astype(bool)
    by = R[m].groupby("stratum")["ok_30d"].agg(["mean", "size"])
    nonbt = m & (R["first_type"] != "股权分置限售股份")
    rows = [dict(gate="G4", item="sina_announce_le_D-30_matched",
                 value=f"{share_matched:.4f}(补抓后日期匹配 {int(m.sum())}/{len(R)} 批;日期+数量匹配子集 "
                       f"{float(R.loc[mq, 'ok_30d'].mean()):.4f}, n={int(mq.sum())};未匹配按失败计 {share_cons:.4f};"
                       f"仍未匹配 {int((~m).sum())} 批为新浪表内无该批次(覆盖缺口),非公告晚)",
                 threshold=">= 0.95(日期匹配样本;分母口径见报告偏离 #2)", passed=bool(share_matched >= 0.95)),
            dict(gate="info", item="sina_by_stratum",
                 value="; ".join(f"{s}:{v['mean']:.3f}(n={int(v['size'])})" for s, v in by.iterrows())
                       + f";剔除股权分置 {float(R.loc[nonbt, 'ok_30d'].mean()):.4f}(n={int(nonbt.sum())})",
                 threshold="描述", passed=True),
            dict(gate="info", item="sina_lead_days_quantiles",
                 value=f"p5 {R.loc[m,'lead_days'].quantile(.05):.0f} / p50 {R.loc[m,'lead_days'].median():.0f} / "
                       f"min {R.loc[m,'lead_days'].min():.0f}(自然日);< 30 日的只有股权分置限售股(按持有人申请解禁,非规则日期)",
                 threshold="描述", passed=True),
            dict(gate="info", item="sina_retry", value=f"首轮空表 {len(todo)} 批补抓;{time.time()-t0:.0f}s",
                 threshold="描述", passed=True)]
    upsert_acceptance(rows)
    print(pd.DataFrame(rows).to_string())


def cmd_manifest():
    pay = dict(written_at=pd.Timestamp.now().isoformat(timespec="seconds"), stage="executor S0 end (before any return)",
               prereg_md5=C.md5(C.PREREG), em_raw_concat_md5=C.raw_concat_md5(), em_parquet_md5=C.md5(C.EM_PARQUET),
               unlock_events_md5=C.md5(C.EVENTS_CSV), comp_v5_score_md5=C.md5(C.COMP_PATH),
               capacity_grid_md5=C.md5(C.CAPGRID), songzhuan_events_md5=C.md5(C.JUMPS),
               sina_sample_md5=C.md5(SINA_CSV) if SINA_CSV.exists() else None)
    assert pay["prereg_md5"] == C.PREREG_MD5, "预注册文件被改动"
    assert pay["comp_v5_score_md5"] == C.COMP_MD5 and pay["capacity_grid_md5"] == C.CAPGRID_MD5
    assert pay["songzhuan_events_md5"] == C.JUMPS_MD5
    C.write_json(C.LEDGER / "inc_t3_prereg_manifest.json", pay)
    print(pay)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "events"
    {"events": cmd_events, "sina": cmd_sina, "sina_retry": cmd_sina_retry, "manifest": cmd_manifest}[cmd]()
