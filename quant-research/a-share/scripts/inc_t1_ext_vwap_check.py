"""INC-T1 §3.3 第 3 条:窗口 VWAP 近似(amt/mid)的外部核对 —— baostock 5 分钟线(真实成交量)抽样。

抽样(预注册,不看任何收益):V1000 池 2024 年 ADV 十分位各 6 只,十分位内按代码排序取前 6 只沪深股票
(baostock 不含北交所;"池内股票" = 2024 年 ≥ 50% 的交易日在 V1000 决策池内;ADV = 2024 年日均全日成交额)。
期间 2024-01-01 ~ 2024-12-31,baostock 不复权(adjustflag=3)5 分钟线,真实窗口 VWAP = Σamount / Σvolume。
baostock 5 分钟 bar 按结束时刻标注:09:35 bar 含 09:30 开盘竞价(1 分钟 idx0..5),15:00 bar 含收盘竞价(idx236..240)。
因此只比较按 5 分钟边界对齐、且不含竞价的窗口:
  预注册四窗:09:35-10:00(idx6..30)、10:00-10:30(idx31..60 = W3)、13:00-14:00(idx121..180 = W5)、14:30-14:55(idx211..235)
  信息性补充(同样精确对齐):W2 09:41-10:00(idx11..30)、W4 10:31-11:30(idx61..120)、W6 14:01-14:30(idx181..210)
复权约掉:比较同日比值 (窗口 VWAP / 当日 15:00 收盘价),本项目用前复权 1 分钟数据,baostock 用不复权 5 分钟数据。
偏离 dev = ln(本项目比值 / 真实比值),单位 bp。规则:某窗口 |dev| 中位数 > 10bp -> 该窗口主口径改用 twap_W(报告注明)。
窗口映射:09:35-10:00 -> W1、W2;10:00-10:30 -> W3;13:00-14:00 -> W5;14:30-14:55 -> W6、W7(W4、W6 另有精确窗口)。
每只股票 ≤ 120s 超时(minute5.fetch_bs_5m:setitimer + 失败重登),失败记录后跳过;60 只中失败 > 20 只则本条降为信息性。
原始返回缓存 cache/_tmp/inc_t1_bs5m/,可续跑。产出 research/ledger/inc_t1_ext_vwap_check.csv(汇总)
与 inc_t1_ext_vwap_check_detail.csv(逐股票 × 窗口)。
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, TMP, load_raw_minute
from afactor.panel import Panel
from afactor import strategy as ST, minute5 as M5

RAW_DIR = TMP / "inc_t1_bs5m"
START, END = "2024-01-01", "2024-12-31"
N_PER_DECILE = 6
TIMEOUT = 120
RULE_BP = 10.0
# (名称, 1 分钟 idx 区间含两端, 5 分钟 bar 结束时刻区间含两端, 是否预注册窗口, 映射到的本项目窗口)
CMP_WINDOWS = [
    ("09:35-10:00", (6, 30), ("09:40", "10:00"), True, "W1,W2"),
    ("10:00-10:30", (31, 60), ("10:05", "10:30"), True, "W3"),
    ("13:00-14:00", (121, 180), ("13:05", "14:00"), True, "W5"),
    ("14:30-14:55", (211, 235), ("14:35", "14:55"), True, "W6,W7"),
    ("W2 09:41-10:00", (11, 30), ("09:45", "10:00"), False, "W2"),
    ("W4 10:31-11:30", (61, 120), ("10:35", "11:30"), False, "W4"),
    ("W6 14:01-14:30", (181, 210), ("14:05", "14:30"), False, "W6"),
]


def pick_sample(p: Panel) -> pd.DataFrame:
    sel = np.asarray(p.dates.year == 2024)
    U = ST.universe(p, "V1000").to_numpy(bool)[sel]
    amt = p.raw("amount").to_numpy(np.float64)[sel]
    frac = U.mean(axis=0)
    adv = np.nanmean(np.where(amt > 0, amt, np.nan), axis=0)
    codes = np.asarray(p.codes)
    ok = (frac >= 0.5) & np.isfinite(adv) & np.array([c.endswith((".SH", ".SZ")) for c in codes])
    df = pd.DataFrame(dict(code=codes[ok], adv2024=adv[ok], pool_frac=frac[ok]))
    df["decile"] = pd.qcut(df.adv2024.rank(method="first"), 10, labels=False) + 1
    out = (df.sort_values(["decile", "code"]).groupby("decile").head(N_PER_DECILE).reset_index(drop=True))
    return out


def fetch(codes: list[str]) -> tuple[dict, list]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    need = [c for c in codes if not (RAW_DIR / f"{c}.parquet").exists()]
    fails = []
    if need:
        lg = M5.bs_login(30)
        print(f"baostock login {lg.error_code}", flush=True)
        for i, c in enumerate(need, 1):
            t0 = time.time()
            for attempt in range(2):
                try:
                    df = M5.fetch_bs_5m(c, START, END, adjustflag="3", timeout=TIMEOUT, max_rows=20000)
                    df.to_parquet(RAW_DIR / f"{c}.parquet")
                    print(f"  [{i}/{len(need)}] {c}: {len(df)} 行 {time.time() - t0:.1f}s", flush=True)
                    break
                except (M5.BsTimeout, Exception) as e:
                    print(f"  [{i}/{len(need)}] {c}: 第 {attempt + 1} 次失败 {type(e).__name__}: {e}", flush=True)
                    time.sleep(2)
            else:
                fails.append(c)
        try:
            import baostock as bs
            bs.logout()
        except Exception:
            pass
    data = {c: pd.read_parquet(RAW_DIR / f"{c}.parquet") for c in codes if (RAW_DIR / f"{c}.parquet").exists()}
    return data, fails


def ours_ratios(codes: list[str]) -> dict:
    """本项目口径:窗口 Σamt/Σ(amt/mid) 与 idx240 收盘价之比,逐股票日。返回 {窗口名: DataFrame(日 × 股)}。"""
    out = {w[0]: [] for w in CMP_WINDOWS}
    amt_out = {w[0]: [] for w in CMP_WINDOWS}
    for m in [f"2024{k:02d}" for k in range(1, 13)]:
        C = load_raw_minute("close_adj", m); O = load_raw_minute("open_adj", m); V = load_raw_minute("amt", m)
        cc = [c for c in codes if c in C.columns]
        C, O, V = C[cc], O[cc], V[cc]
        days = C.index.normalize()
        for d in days.unique():
            sl = days == d
            P = C[sl].to_numpy(np.float64); Op = O[sl].to_numpy(np.float64); A = V[sl].to_numpy(np.float64)
            if P.shape[0] != 241:
                continue
            mid = (P + Op) / 2.0
            with np.errstate(divide="ignore", invalid="ignore"):
                vol = np.where(mid > 0, A / mid, np.nan)
                for name, (lo, hi), _, _, _ in CMP_WINDOWS:
                    a = np.nansum(A[lo:hi + 1], axis=0); v = np.nansum(vol[lo:hi + 1], axis=0)
                    vw = np.where(v > 0, a / v, np.nan)
                    out[name].append(pd.Series(vw / P[240], index=cc, name=d))
                    amt_out[name].append(pd.Series(a, index=cc, name=d))
        del C, O, V
    return ({k: pd.DataFrame(v) for k, v in out.items()}, {k: pd.DataFrame(v) for k, v in amt_out.items()})


def true_ratios(df: pd.DataFrame) -> dict:
    """baostock 5 分钟:窗口 Σamount/Σvolume 与 15:00 bar 收盘之比,逐日。"""
    df = df.copy()
    df["d"] = df.index.normalize(); df["hm"] = df.index.strftime("%H:%M")
    close = df[df.hm == "15:00"].set_index("d")["close"]
    out = {}
    for name, _, (a, b), _, _ in CMP_WINDOWS:
        w = df[(df.hm >= a) & (df.hm <= b)]
        g = w.groupby("d")[["amount", "volume"]].sum()
        nb = w.groupby("d").size()
        vw = (g.amount / g.volume).where((g.volume > 0) & (nb == nb.max()))
        out[name] = (vw / close.reindex(vw.index)).rename("true")
    return out


def main():
    p = Panel()
    samp = pick_sample(p)
    codes = samp.code.tolist()
    print(f"抽样 {len(codes)} 只:各十分位 {samp.groupby('decile').size().tolist()}", flush=True)
    data, fails = fetch(codes)
    print(f"抓取成功 {len(data)} / 失败 {len(fails)}: {fails}", flush=True)
    ours, ours_amt = ours_ratios(list(data))
    det, allrows = [], []
    for c, df in data.items():
        tr = true_ratios(df)
        for name, _, _, prereg, mapto in CMP_WINDOWS:
            o = ours[name].get(c)
            if o is None:
                continue
            t = tr[name]
            j = pd.concat([o.rename("ours"), t], axis=1).dropna()
            j = j[(j.ours > 0) & (j.true > 0)]
            if j.empty:
                continue
            dev = np.log(j.ours / j.true) * 1e4
            dec = int(samp.set_index("code").loc[c, "decile"])
            det.append(dict(code=c, decile=dec, window=name, prereg=prereg, maps_to=mapto, n_days=len(dev),
                            dev_median_bp=float(dev.median()), absdev_median_bp=float(dev.abs().median()),
                            absdev_p90_bp=float(dev.abs().quantile(0.9))))
            allrows.append(pd.DataFrame(dict(window=name, decile=dec, dev=dev.to_numpy())))
    detail = pd.DataFrame(det)
    alldev = pd.concat(allrows, ignore_index=True)
    rows = []
    n_ok = len(data)
    informative_only = len(fails) > 20
    for name, _, _, prereg, mapto in CMP_WINDOWS:
        x = alldev[alldev.window == name].dev
        if x.empty:
            continue
        med_abs = float(x.abs().median())
        rows.append(dict(window=name, prereg=prereg, maps_to=mapto, n_stock_days=len(x), n_stocks=int(detail[detail.window == name].code.nunique()),
                         dev_median_bp=float(x.median()), absdev_median_bp=med_abs,
                         absdev_p90_bp=float(x.abs().quantile(0.9)), absdev_p99_bp=float(x.abs().quantile(0.99)),
                         share_within_10bp=float((x.abs() <= 10).mean()),
                         small_decile_absdev_median_bp=float(alldev[(alldev.window == name) & (alldev.decile <= 3)].dev.abs().median()),
                         rule_switch_to_twap=bool(prereg and med_abs > RULE_BP and not informative_only),
                         n_fetch_ok=n_ok, n_fetch_fail=len(fails), informative_only=informative_only))
    summ = pd.DataFrame(rows)
    summ.to_csv(LEDGER / "inc_t1_ext_vwap_check.csv", index=False)
    detail.to_csv(LEDGER / "inc_t1_ext_vwap_check_detail.csv", index=False)
    samp.to_csv(LEDGER / "inc_t1_ext_vwap_sample.csv", index=False)
    print(summ.to_string(), flush=True)


if __name__ == "__main__":
    main()
