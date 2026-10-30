"""5 分钟聚合规则 vs baostock 逐 bar 核对(前缀 m5):2025-12,随机 30 只沪深股票。

聚合规则见 afactor.minute5(bar0 = idx0..5 含开盘集合竞价,之后每 5 根一组,bar47 含收盘集合竞价)。
baostock 用不复权价(adjustflag=3);1 分钟数据是前复权(基准 2026-08),两者只差逐日常数因子,
所以价格按"当日 15:00 收盘之比"归一后逐 bar 比较;成交额与复权无关,直接比。
网络:每只股票一次请求,超时 60s + 最大行数保护,失败重试 2 次并记录;原始返回缓存在 cache/_tmp/m5_bs_raw/,可续跑。
审查 P1(2026-10-10 修复):超时异常改为 BaseException(minute5.BsTimeout),任何失败后 minute5.bs_relogin() 换新 socket,
并校验响应代码 / 日期属于本次请求(minute5.bs_query_rows)。旧版超时会被 baostock 吞掉、共享 socket 错位,
之后每只股票静默拿到上一只的数据;本次核对当时 0 次失败,缓存数据不受影响(逐股收盘吻合率 m5_baostock_by_stock.csv)。
`python scripts/m5_baostock_check.py --pairs` 只用缓存重算逐股票日的原料配对(第二层误差蒙特卡罗的输入),不联网。
另抓同期 baostock 日线(不复权)核对日成交额,并把 5 分钟原料公式分别作用在"baostock 原始 bar"与"1 分钟精确聚合 bar"上,
量化第二层误差(baostock 数据本身与聚合口径的差异),与第一层误差(5 分钟 vs 1 分钟真值)并列。
产出:research/ledger/m5_baostock_by_stock.csv、m5_baostock_by_bar.csv、m5_baostock_summary.csv、
     m5_baostock_daily.csv(日线核对)、m5_baostock_feature_err.csv(原料两层误差)
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, TMP, load_raw_minute
from afactor import minute5 as M5

MONTH, START, END = "202512", "2025-12-01", "2025-12-31"
N_STOCKS, SEED = 30, 20251201
RAW_DIR = TMP / "m5_bs_raw"
TOL_PX = 1e-4          # 归一后价格相对误差容差(float32 前复权价 ~1e-7,不复权价两位小数)


def pick_codes(C: pd.DataFrame) -> list[str]:
    ok = C.notna().all(axis=0)
    cand = [c for c in C.columns[ok] if c.endswith((".SH", ".SZ"))]
    rng = np.random.default_rng(SEED)
    return sorted(rng.choice(cand, N_STOCKS, replace=False).tolist())


def fetch_all(codes: list[str]) -> tuple[dict, list]:
    import baostock as bs
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out, fails = {}, []
    need = [c for c in codes if not (RAW_DIR / f"{c}.parquet").exists()]
    if need:
        lg = M5.bs_login(30)
        print(f"login {lg.error_code} {lg.error_msg}", flush=True)
        for c in need:
            for attempt in range(3):
                try:
                    # 失败时 fetch_bs_5m 已重登(新 socket)并校验了响应代码 / 日期
                    df = M5.fetch_bs_5m(c, START, END, adjustflag="3", timeout=60, max_rows=5000)
                    df.to_parquet(RAW_DIR / f"{c}.parquet")
                    print(f"  {c}: {len(df)} 行", flush=True)
                    break
                except (M5.BsTimeout, Exception) as e:
                    print(f"  {c}: 第 {attempt + 1} 次失败 {type(e).__name__}: {e}", flush=True)
                    time.sleep(2)
            else:
                fails.append(c)
        try:
            bs.logout()
        except Exception:
            pass
    for c in codes:
        f = RAW_DIR / f"{c}.parquet"
        if f.exists():
            out[c] = pd.read_parquet(f)
    return out, fails


def fetch_daily(codes: list[str]) -> pd.DataFrame:
    """baostock 日线(不复权)成交额与收盘,缓存 cache/_tmp/m5_bs_raw/daily_<code>.parquet。"""
    import baostock as bs
    need = [c for c in codes if not (RAW_DIR / f"daily_{c}.parquet").exists()]
    if need:
        M5.bs_login(30)
        for c in need:
            for attempt in range(3):
                try:
                    d = M5.bs_query_rows(M5.bs_code(c), "date,code,open,close,amount", START, END, frequency="d",
                                         adjustflag="3", timeout=60, max_rows=100)
                    for k in ("open", "close", "amount"):
                        d[k] = pd.to_numeric(d[k], errors="coerce")
                    d["date"] = pd.to_datetime(d["date"])
                    d.drop(columns=["code"]).set_index("date").to_parquet(RAW_DIR / f"daily_{c}.parquet")
                    break
                except (M5.BsTimeout, Exception) as e:
                    print(f"  日线 {c}: 第 {attempt + 1} 次失败 {type(e).__name__}: {e}", flush=True)
                    time.sleep(2)
        try:
            bs.logout()
        except Exception:
            pass
    out = []
    for c in codes:
        f = RAW_DIR / f"daily_{c}.parquet"
        if f.exists():
            out.append(pd.read_parquet(f).assign(code=c))
    return pd.concat(out) if out else pd.DataFrame()


FEAT_CMP = ["close_px", "open_auction_px", "amount", "ret_intraday", "rvol", "rng", "smart_money", "amt_gini",
            "amt_time_wgt", "amt_first30_share", "gap_share", "twap_open10__mid4", "amt_open10__sum2",
            "vwap__mid_h", "amt_openauc_share__bar0"]
LOG_FIELDS = {"close_px", "open_auction_px", "amount", "twap_open10__mid4", "amt_open10__sum2", "vwap__mid_h"}


PAIRS_OUT = TMP / "m5_bs_feature_pairs.parquet"   # 逐股票日:ours5 / bs5 / bs5r(第二层误差蒙特卡罗的输入)


def feature_pairs(codes, P5, O5, V5, bsd, FAC: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
    """逐股票日的原料三件套:ours5 = 1 分钟精确聚合 bar、bs5 = baostock 原始 bar、bs5r = baostock bar 经日线修正
    (minute5.reconcile_with_daily,前向检验推荐的做法)。同一套 5 分钟公式 build_daily_5m._day_features_5m。"""
    from afactor.build_daily_5m import _day_features_5m
    days = sorted(set(P5.index.normalize()))
    ok_codes = [c for c in codes if c in bsd and not bsd[c].empty]
    recs = []
    for d in days:
        sl = P5.index.normalize() == d
        p_o, o_o, v_o = (X.loc[sl, ok_codes].to_numpy(np.float64) for X in (P5, O5, V5))
        bp = np.column_stack([bsd[c]["close"].reindex(P5.index[sl]).to_numpy(np.float64) for c in ok_codes])
        bo = np.column_stack([bsd[c]["open"].reindex(P5.index[sl]).to_numpy(np.float64) for c in ok_codes])
        bv = np.column_stack([bsd[c]["amount"].reindex(P5.index[sl]).to_numpy(np.float64) for c in ok_codes])
        fac = FAC.loc[d, ok_codes].to_numpy(np.float64)       # 不复权 -> 前复权(日线收盘 / 1 分钟收盘)
        f_o = _day_features_5m(p_o, o_o, v_o)
        f_b = _day_features_5m(bp / fac, bo / fac, bv)
        dd = D[D.index == d].set_index("code").reindex(ok_codes)
        f_r = _day_features_5m(*M5.reconcile_with_daily(bp / fac, bo / fac, bv, dd["close"].to_numpy() / fac,
                                                        dd["open"].to_numpy() / fac, dd["amount"].to_numpy()))
        for i, c in enumerate(ok_codes):
            recs.append(dict(date=d, code=c, **{f"{k}|ours5": f_o[k][i] for k in FEAT_CMP},
                             **{f"{k}|bs5": f_b[k][i] for k in FEAT_CMP},
                             **{f"{k}|bs5r": f_r[k][i] for k in FEAT_CMP}))
    return pd.DataFrame(recs)


def feature_errors(codes, P5, O5, V5, bsd, FAC: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
    """同一套 5 分钟公式分别作用于 baostock bar 与 1 分钟精确聚合 bar;再与 1 分钟真值(cache/daily)比。
    价格/成交额类报 |log 差|(bp),其余报 |差| 与相对真值截面离散度的比例。semibeta_mp 需全市场,不在此列。
    第三种:baostock bar 先用日线修正(minute5.reconcile_with_daily)再算 —— 前向检验推荐的做法。"""
    from afactor.config import DAILY
    R = feature_pairs(codes, P5, O5, V5, bsd, FAC, D)
    R.to_parquet(PAIRS_OUT)
    days = sorted(R["date"].unique())
    ok_codes = sorted(R["code"].unique())
    rows = []
    for k in FEAT_CMP:
        base = k.split("__")[0]
        tru = pd.read_parquet(DAILY / f"{base}.parquet", columns=ok_codes).reindex(days).stack()
        tru.index.names = ["date", "code"]
        t = R.set_index(["date", "code"])[[f"{k}|ours5", f"{k}|bs5", f"{k}|bs5r"]].join(tru.rename("truth1"))
        for ex in ("SZ", "SH", "全部"):
            tt = t if ex == "全部" else t[t.index.get_level_values("code").str.endswith(ex)]
            a, b, br, tr = tt[f"{k}|ours5"], tt[f"{k}|bs5"], tt[f"{k}|bs5r"], tt["truth1"]
            if k in LOG_FIELDS:
                l1 = (np.log(a / tr) * 1e4).abs(); l2 = (np.log(b / a) * 1e4).abs()
                l2r = (np.log(br / a) * 1e4).abs()
                unit = "bp(|log 差|)"
            else:
                sd = tr.std()
                l1 = (a - tr).abs() / sd; l2 = (b - a).abs() / sd; l2r = (br - a).abs() / sd
                unit = "|差|/真值离散度"
            rows.append(dict(field=k, exchange=ex, n=int(l2.notna().sum()), unit=unit,
                             layer1_5m_vs_1m_median=float(l1.median()), layer1_p90=float(l1.quantile(0.9)),
                             layer2_bs_vs_agg_median=float(l2.median()), layer2_p90=float(l2.quantile(0.9)),
                             layer2_exact_frac=float((l2 < 1e-6).mean()),
                             layer2r_reconciled_median=float(l2r.median()), layer2r_p90=float(l2r.quantile(0.9))))
    return pd.DataFrame(rows)


def main_pairs():
    """只用缓存(cache/_tmp/m5_bs_raw/)重算逐股票日配对并落盘,不联网、不改任何 ledger。"""
    C = load_raw_minute("close_adj", MONTH); O = load_raw_minute("open_adj", MONTH)
    V = load_raw_minute("amt", MONTH)
    codes = pick_codes(C)
    miss = [c for c in codes if not (RAW_DIR / f"{c}.parquet").exists() or not (RAW_DIR / f"daily_{c}.parquet").exists()]
    if miss:
        raise SystemExit(f"缓存缺失,需先联网跑一次 main(): {miss}")
    P5, O5, V5 = M5.aggregate_wide(C[codes], O[codes], V[codes])
    bsd = {c: pd.read_parquet(RAW_DIR / f"{c}.parquet") for c in codes}
    D = pd.concat([pd.read_parquet(RAW_DIR / f"daily_{c}.parquet").assign(code=c) for c in codes])
    our_close = C[codes].groupby(C.index.normalize()).last()
    FAC = D.reset_index().pivot(index="date", columns="code", values="close").reindex(
        index=our_close.index, columns=codes) / our_close
    R = feature_pairs(codes, P5, O5, V5, bsd, FAC, D)
    R.to_parquet(PAIRS_OUT)
    print(f"配对 {len(R)} 个股票日 -> {PAIRS_OUT}")


def main():
    C = load_raw_minute("close_adj", MONTH); O = load_raw_minute("open_adj", MONTH)
    V = load_raw_minute("amt", MONTH)
    codes = pick_codes(C)
    print("样本:", codes, flush=True)
    P5, O5, V5 = M5.aggregate_wide(C[codes], O[codes], V[codes])
    bsd, fails = fetch_all(codes)
    D = fetch_daily(codes)
    # 逐日复权因子 = baostock 日线收盘(不复权)/ 1 分钟 15:00 收盘(前复权)。
    # 不用 baostock 5 分钟最后一根收盘:上交所股票约 1/4 的交易日它不等于日线收盘(见 m5_baostock_daily.csv)。
    our_close = C[codes].groupby(C.index.normalize()).last()
    FAC = D.reset_index().pivot(index="date", columns="code", values="close").reindex(
        index=our_close.index, columns=codes) / our_close

    bar_rows, stock_rows = [], []
    for c in codes:
        if c not in bsd or bsd[c].empty:
            stock_rows.append(dict(code=c, status="fetch_failed")); continue
        b = bsd[c]
        ours = pd.DataFrame({"close": P5[c], "open": O5[c], "amount": V5[c]})
        j = ours.join(b[["open", "close", "amount", "volume"]], rsuffix="_bs", how="outer")
        day = j.index.normalize()
        n_bs_per_day = b.groupby(b.index.normalize()).size()
        fac = pd.Series(FAC[c].reindex(day).to_numpy(), index=j.index)
        j["close_rel_err"] = j["close_bs"] / (j["close"] * fac) - 1.0
        j["open_rel_err"] = j["open_bs"] / (j["open"] * fac) - 1.0
        j["amt_ratio"] = j["amount_bs"] / j["amount"]
        j["bar"] = j.index.strftime("%H:%M")
        j["code"] = c
        j["factor"] = fac
        bar_rows.append(j)
        both = j.dropna(subset=["close", "close_bs"])
        amt_ok = both[(both["amount"] > 0) & (both["amount_bs"] > 0)]
        stock_rows.append(dict(
            code=c, status="ok", n_days_bs=int(n_bs_per_day.size), bars_per_day_bs=int(n_bs_per_day.median()),
            bars_per_day_min=int(n_bs_per_day.min()), n_bars_ours=int(j["close"].notna().sum()),
            n_bars_bs=int(j["close_bs"].notna().sum()), n_bars_matched=len(both),
            factor_min=float(fac.min()), factor_max=float(fac.max()),
            close_match_frac=float((both["close_rel_err"].abs() < TOL_PX).mean()),
            close_max_abs_err=float(both["close_rel_err"].abs().max()),
            open_match_frac=float((both["open_rel_err"].abs() < TOL_PX).mean()),
            open_max_abs_err=float(both["open_rel_err"].abs().max()),
            amt_ratio_median=float(amt_ok["amt_ratio"].median()),
            amt_ratio_p01=float(amt_ok["amt_ratio"].quantile(0.01)),
            amt_ratio_p99=float(amt_ok["amt_ratio"].quantile(0.99)),
            amt_within_1e3=float(((amt_ok["amt_ratio"] - 1).abs() < 1e-3).mean()),
            amt_zero_mismatch=int(((both["amount"] > 0) != (both["amount_bs"] > 0)).sum()),
            day_amount_ratio=float(both["amount_bs"].sum() / both["amount"].sum())))
    S = pd.DataFrame(stock_rows)
    B = pd.concat(bar_rows) if bar_rows else pd.DataFrame()
    S.to_csv(LEDGER / "m5_baostock_by_stock.csv", index=False)
    g = B.dropna(subset=["close", "close_bs"]).groupby("bar")
    by_bar = pd.DataFrame({
        "n": g.size(),
        "close_match_frac": g["close_rel_err"].apply(lambda x: float((x.abs() < TOL_PX).mean())),
        "open_match_frac": g["open_rel_err"].apply(lambda x: float((x.abs() < TOL_PX).mean())),
        "amt_ratio_median": g["amt_ratio"].median(),
        "amt_ratio_p01": g["amt_ratio"].quantile(0.01),
        "amt_ratio_p99": g["amt_ratio"].quantile(0.99),
    }).reset_index()
    by_bar.to_csv(LEDGER / "m5_baostock_by_bar.csv", index=False)
    both = B.dropna(subset=["close", "close_bs"])
    ak = both[(both["amount"] > 0) & (both["amount_bs"] > 0)]
    summ = pd.DataFrame([dict(
        month=MONTH, n_stocks=len(codes), n_fetch_failed=len(fails), n_bars_matched=len(both),
        n_bars_ours_only=int((B["close"].notna() & B["close_bs"].isna()).sum()),
        n_bars_bs_only=int((B["close"].isna() & B["close_bs"].notna()).sum()),
        close_match_frac=float((both["close_rel_err"].abs() < TOL_PX).mean()),
        open_match_frac=float((both["open_rel_err"].abs() < TOL_PX).mean()),
        amt_ratio_median=float(ak["amt_ratio"].median()),
        amt_ratio_p01=float(ak["amt_ratio"].quantile(0.01)), amt_ratio_p99=float(ak["amt_ratio"].quantile(0.99)),
        amt_within_1e3=float(((ak["amt_ratio"] - 1).abs() < 1e-3).mean()),
        stocks_with_factor_change_in_month=int((S.get("factor_max", pd.Series(dtype=float)) /
                                                S.get("factor_min", pd.Series(dtype=float)) - 1 > 1e-4).sum()),
        tol_px=TOL_PX)])
    summ.to_csv(LEDGER / "m5_baostock_summary.csv", index=False)
    # 日线核对:1 分钟全日成交额/收盘/开盘 vs baostock 日线 vs baostock 5 分钟加总/末根收盘/首根开盘
    if not D.empty:
        ours_day = V[codes].groupby(V.index.normalize()).sum()
        bs5_day = pd.DataFrame({c: bsd[c]["amount"].groupby(bsd[c].index.normalize()).sum() for c in bsd})
        bs5_last = pd.DataFrame({c: bsd[c]["close"].groupby(bsd[c].index.normalize()).last() for c in bsd})
        bs5_first = pd.DataFrame({c: bsd[c]["open"].groupby(bsd[c].index.normalize()).first() for c in bsd})
        our_open = O[codes].groupby(O.index.normalize()).first()
        drows = []
        for c in codes:
            dc = D[D.code == c]
            for dt, r in dc.iterrows():
                drows.append(dict(code=c, exchange=c[-2:], date=dt, amt_daily_bs=r["amount"],
                                  amt_1m_sum=float(ours_day.at[dt, c]) if dt in ours_day.index else np.nan,
                                  amt_5m_bs_sum=float(bs5_day.at[dt, c]) if c in bs5_day and dt in bs5_day.index else np.nan,
                                  close_daily_bs=r["close"], close_5m_bs_last=float(bs5_last.at[dt, c]),
                                  open_daily_bs=r["open"], open_5m_bs_first=float(bs5_first.at[dt, c]),
                                  fac_close=float(r["close"] / our_close.at[dt, c]),
                                  fac_open=float(r["open"] / our_open.at[dt, c])))
        DD = pd.DataFrame(drows)
        DD["daily_over_1m"] = DD.amt_daily_bs / DD.amt_1m_sum
        DD["bs5sum_over_1m"] = DD.amt_5m_bs_sum / DD.amt_1m_sum
        DD["bs5_last_eq_daily_close"] = (DD.close_5m_bs_last - DD.close_daily_bs).abs() < 0.005
        DD["bs5_first_eq_daily_open"] = (DD.open_5m_bs_first - DD.open_daily_bs).abs() < 0.005
        DD["open_close_factor_diff_bp"] = (DD.fac_close / DD.fac_open - 1) * 1e4   # 1 分钟开收盘与日线是否同一复权口径
        DD.to_csv(LEDGER / "m5_baostock_daily.csv", index=False)
        dsum = DD.groupby("exchange").agg(n=("code", "size"),
                                          daily_match_1e4=("daily_over_1m", lambda x: float(((x - 1).abs() < 1e-4).mean())),
                                          bs5sum_match_1e4=("bs5sum_over_1m", lambda x: float(((x - 1).abs() < 1e-4).mean())),
                                          bs5sum_ratio_median=("bs5sum_over_1m", "median"),
                                          bs5sum_ratio_min=("bs5sum_over_1m", "min"),
                                          bs5_last_eq_daily_close=("bs5_last_eq_daily_close", "mean"),
                                          bs5_first_eq_daily_open=("bs5_first_eq_daily_open", "mean"),
                                          oc_factor_diff_abs_max_bp=("open_close_factor_diff_bp", lambda x: float(x.abs().max())))
        print(dsum)
        for ex, r in dsum.iterrows():
            summ[f"daily_bs_eq_1m_{ex}"] = r["daily_match_1e4"]
            summ[f"bs5sum_eq_1m_{ex}"] = r["bs5sum_match_1e4"]
            summ[f"bs5sum_ratio_median_{ex}"] = r["bs5sum_ratio_median"]
            summ[f"bs5sum_ratio_min_{ex}"] = r["bs5sum_ratio_min"]
            summ[f"bs5_last_eq_daily_close_{ex}"] = r["bs5_last_eq_daily_close"]
            summ[f"bs5_first_eq_daily_open_{ex}"] = r["bs5_first_eq_daily_open"]
            summ[f"oc_factor_diff_abs_max_bp_{ex}"] = r["oc_factor_diff_abs_max_bp"]
    # 按交易所拆分逐 bar 吻合率
    for ex in ("SZ", "SH"):
        bb = both[both["code"].str.endswith(ex)]
        ak2 = bb[(bb["amount"] > 0) & (bb["amount_bs"] > 0)]
        summ[f"close_match_frac_{ex}"] = float((bb["close_rel_err"].abs() < TOL_PX).mean())
        summ[f"open_match_frac_{ex}"] = float((bb["open_rel_err"].abs() < TOL_PX).mean())
        summ[f"amt_within_1e3_{ex}"] = float(((ak2["amt_ratio"] - 1).abs() < 1e-3).mean())
        summ[f"close_abs_err_p99_bp_{ex}"] = float(bb["close_rel_err"].abs().quantile(0.99) * 1e4)
    # 缺口位置:15:00 之前的累计成交额 vs 15:00 那根(逐股票日)
    B2 = B.dropna(subset=["amount", "amount_bs"]).copy()
    B2["day"] = B2.index.normalize()
    pre = B2[B2["bar"] < "15:00"].groupby(["code", "day"])[["amount", "amount_bs"]].sum()
    last = B2[B2["bar"] == "15:00"].groupby(["code", "day"])[["amount", "amount_bs"]].sum()
    pr = pre["amount_bs"] / pre["amount"]; lr = last["amount_bs"] / last["amount"]
    for ex in ("SZ", "SH"):
        sel = pr.index.get_level_values("code").str.endswith(ex)
        summ[f"pre1500_cum_within_1e3_{ex}"] = float(((pr[sel] - 1).abs() < 1e-3).mean())
        summ[f"bar1500_amt_ratio_median_{ex}"] = float(lr[sel].median())
        summ[f"bar1500_amt_ratio_p25_{ex}"] = float(lr[sel].quantile(0.25))
        summ[f"bar1500_amt_ratio_min_{ex}"] = float(lr[sel].min())
    summ.to_csv(LEDGER / "m5_baostock_summary.csv", index=False)
    FE = feature_errors(codes, P5, O5, V5, bsd, FAC, D)
    FE.to_csv(LEDGER / "m5_baostock_feature_err.csv", index=False)
    pd.set_option("display.width", 250)
    print(FE.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(S.to_string(index=False))
    print(by_bar.to_string(index=False))
    print(summ.T)
    if fails:
        print("抓取失败:", fails)


if __name__ == "__main__":
    main_pairs() if "--pairs" in sys.argv[1:] else main()
