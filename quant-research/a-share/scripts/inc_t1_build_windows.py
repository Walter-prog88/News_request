"""INC-T1 第 1 步:从原始 1 分钟数据按月流式聚合各成交窗口的价格 / 成交额 / 区间(预注册 §3.2-§3.3)。

窗口(bar 索引,含两端;idx0 = 09:30 开盘集合竞价,idx1..120 = 09:31..11:30,idx121..240 = 13:01..15:00):
  W0 09:30 竞价 [0]        W1 09:31-09:40 [1,10]   W2 09:41-10:00 [11,30]   W3 10:01-10:30 [31,60]
  W4 10:31-11:30 [61,120]  W5 13:01-14:00 [121,180] W6 14:01-14:30 [181,210] W7 14:31-14:57 [211,237]
  W8 15:00 竞价 [240]      W9 全日 [0,240]          W10 上午 [0,120]         W11 下午 [121,240]
  GAP 14:58-14:59 [238,239](只用于成交额加总核对)

每个窗口输出(日期 × 股票,float32):
  twap  窗口内有成交分钟的 mid 等时均值(与 build_daily._twap 逐字同式;全无成交退化为 mid 均值)
  vwap  Σamt / Σ(amt/mid)(与 vwap 字段同一近似;窗口无成交时为 NaN,原样存,使用时再按 §偏离 2 退回 twap)
  amt   Σamt(元)
  hi/lo 窗口内 max(open,close) / min(open,close) 的极值(与 open10_high/open10_low、high_px/low_px 同口径;
        预注册写的是 mid 的极值,执行时改为与 capacity_diagnostics 同法,见报告"偏离预注册"第 1 条)
  单根 bar 窗口(W0、W8):twap = vwap = 该 bar 的 close_adj(竞价成交价),不用 mid。
另存九个时点的 bar 收盘价 pt{idx}(idx = 0,10,30,60,120,180,210,237,240)与 14:57 bar 中价 mid237(W8 的 f_var 用)。

只读 202001~202512 的月份文件(2026 封存不读)。中间结果 cache/_tmp/inc_t1_build/{YYYYMM}.npz,
合并后 cache/_tmp/inc_t1_win_{field}_{W}.parquet(对齐 Panel 的日期 × 股票)。

用法:
  python3 scripts/inc_t1_build_windows.py build     # 逐月聚合(已存在的月份跳过)
  python3 scripts/inc_t1_build_windows.py merge     # 合并成面板
  python3 scripts/inc_t1_build_windows.py checks    # §3.3 第 2 条复现门槛 -> research/ledger/inc_t1_window_build_checks.csv
"""
from __future__ import annotations
import os, sys, time, glob, re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from afactor.config import (TMP, LEDGER, N_BARS, load_raw_minute, raw_months, RESEARCH_END)

EPS = 1e-12
BUILD_DIR = TMP / "inc_t1_build"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

# 窗口 [lo, hi) 半开区间(bar 索引)
WINDOWS = {
    "W0": (0, 1), "W1": (1, 11), "W2": (11, 31), "W3": (31, 61), "W4": (61, 121),
    "W5": (121, 181), "W6": (181, 211), "W7": (211, 238), "W8": (240, 241),
    "W9": (0, 241), "W10": (0, 121), "W11": (121, 241),
}
SINGLE_BAR = {"W0", "W8"}
GAP = (238, 240)
POINTS = [0, 10, 30, 60, 120, 180, 210, 237, 240]
POINT_LABELS = {0: "09:30", 10: "09:40", 30: "10:00", 60: "10:30", 120: "11:30", 180: "14:00",
                210: "14:30", 237: "14:57", 240: "15:00"}
FIELDS = ["twap", "vwap", "amt", "hi", "lo"]


def win_path(field: str, w: str):
    return TMP / f"inc_t1_win_{field}_{w}.parquet"


def extra_names() -> list[str]:
    return ["amt_GAP"] + [f"pt{i}" for i in POINTS] + ["mid237"]


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) < EPS, np.nan, a / np.where(np.abs(b) < EPS, np.nan, b))


def _twap(mid, v, lo, hi):
    """与 build_daily._twap 逐字相同。"""
    m = mid[lo:hi]; vv = v[lo:hi]
    trad = vv > EPS
    num = np.where(trad, m, np.nan)
    cnt = trad.sum(axis=0)
    with np.errstate(invalid="ignore"):
        px = np.nanmean(num, axis=0)
        px = np.where(cnt == 0, np.nanmean(m, axis=0), px)
    return px


def day_windows(P: np.ndarray, O: np.ndarray, V: np.ndarray) -> dict:
    """P/O/V (241, N) float64 -> {name: (N,) float64}。"""
    out = {}
    mid = (O + P) / 2.0
    vol = _safe_div(V, mid)
    hi_m = np.fmax(P, O); lo_m = np.fmin(P, O)
    with np.errstate(all="ignore"):
        for w, (lo, hi) in WINDOWS.items():
            if w in SINGLE_BAR:
                px = P[lo]
                out[f"twap_{w}"] = px
                out[f"vwap_{w}"] = px
                out[f"amt_{w}"] = V[lo]
            else:
                out[f"twap_{w}"] = _twap(mid, V, lo, hi)
                a = np.nansum(V[lo:hi], axis=0)
                out[f"amt_{w}"] = a
                out[f"vwap_{w}"] = _safe_div(a, np.nansum(vol[lo:hi], axis=0))
            out[f"hi_{w}"] = np.nanmax(hi_m[lo:hi], axis=0)
            out[f"lo_{w}"] = np.nanmin(lo_m[lo:hi], axis=0)
        out["amt_GAP"] = np.nansum(V[GAP[0]:GAP[1]], axis=0)
        for i in POINTS:
            out[f"pt{i}"] = P[i]
        out["mid237"] = mid[237]
    # 全日 vwap 与 build_daily 同式:Σ全日 amt / Σ全日 vol(W9 已是这一式)
    return out


def process_month(month: str) -> str:
    f = BUILD_DIR / f"{month}.npz"
    if f.exists():
        return f"{month}: 已存在,跳过"
    t0 = time.time()
    C = load_raw_minute("close_adj", month)
    O = load_raw_minute("open_adj", month)
    V = load_raw_minute("amt", month)
    codes = C.columns
    assert list(O.columns) == list(codes) and list(V.columns) == list(codes), f"{month} 列不一致"
    assert O.index.equals(C.index) and V.index.equals(C.index), f"{month} 索引不一致"
    days = C.index.normalize().values
    uniq, starts, counts = np.unique(days, return_index=True, return_counts=True)
    order = np.argsort(starts)
    uniq, starts, counts = uniq[order], starts[order], counts[order]
    Cv, Ov, Vv = C.to_numpy(), O.to_numpy(), V.to_numpy()          # float32,不整月转 float64(省内存)
    feats, good, bad = {}, [], []
    for d, s, n in zip(uniq, starts, counts):
        if n != N_BARS:
            bad.append((str(pd.Timestamp(d).date()), int(n)))
            continue
        P_ = Cv[s:s + n].astype(np.float64); O_ = Ov[s:s + n].astype(np.float64); V_ = Vv[s:s + n].astype(np.float64)
        valid = ~np.isnan(P_).all(axis=0)
        fd = day_windows(P_, O_, V_)
        for k, val in fd.items():
            feats.setdefault(k, []).append(np.where(valid, val, np.nan).astype(np.float32))
        good.append(pd.Timestamp(d))
    del C, O, V, Cv, Ov, Vv
    arrs = {k: np.asarray(v, dtype=np.float32) for k, v in feats.items()}
    tmp = BUILD_DIR / f"{month}.tmp.npz"
    np.savez(tmp, dates=np.array([str(d.date()) for d in good]), codes=np.array(list(codes), dtype=object), **arrs)
    os.replace(tmp, f)
    return f"{month}: {len(good)} 日 x {len(codes)} 股 x {len(arrs)} 面板, {time.time() - t0:.1f}s" + \
        (f" 异常日={bad}" if bad else "")


def research_months() -> list[str]:
    end = pd.Timestamp(RESEARCH_END).strftime("%Y%m")
    ms = [m for m in raw_months("close_adj") if m <= end]
    assert all(m < "202601" for m in ms)
    return ms


def cmd_build():
    ms = research_months()
    print(f"研究期月份 {len(ms)}:{ms[0]} ~ {ms[-1]}", flush=True)
    t0 = time.time()
    for i, m in enumerate(ms, 1):
        print(f"[{i}/{len(ms)}] {process_month(m)}  累计 {time.time() - t0:.0f}s", flush=True)
    print("build done", flush=True)


def cmd_merge():
    from afactor.panel import Panel
    p = Panel()
    dates, codes = p.dates, p.codes
    di = {d: i for i, d in enumerate(dates.strftime("%Y-%m-%d"))}
    ci = {c: i for i, c in enumerate(codes)}
    files = [f for f in glob.glob(str(BUILD_DIR / "*.npz")) if re.fullmatch(r"\d{6}\.npz", os.path.basename(f))]
    files = sorted(files)
    assert len(files) == 72, f"月份文件 {len(files)} != 72"
    meta = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        d = [str(x) for x in z["dates"]]; c = [str(x) for x in z["codes"]]
        missing_c = [x for x in c if x not in ci]
        meta[f] = (np.array([di[x] for x in d]),
                   np.array([ci.get(x, -1) for x in c]), len(missing_c))
        names = [k for k in z.files if k not in ("dates", "codes")]
        z.close()
    print(f"{len(files)} 个月,{len(names)} 个面板;面板外股票数(逐月最大) {max(v[2] for v in meta.values())}", flush=True)
    t0 = time.time()
    for k in names:
        buf = np.full((len(dates), len(codes)), np.nan, np.float32)
        for f in files:
            z = np.load(f, allow_pickle=True)
            ri, cj, _ = meta[f]
            keep = cj >= 0
            buf[np.ix_(ri, cj[keep])] = z[k][:, keep]
            z.close()
        df = pd.DataFrame(buf, index=dates, columns=codes)
        df.index.name = "date"
        if k in ("amt_GAP", "mid237") or k.startswith("pt"):
            out = TMP / f"inc_t1_win_{k}.parquet"
        else:
            fld, w = k.split("_", 1)
            out = win_path(fld, w)
        df.to_parquet(out)
        print(f"  {k} -> {out.name}  {time.time() - t0:.0f}s", flush=True)
    print("merge done", flush=True)


# ------------------------------------------------------------------ 加载工具(供下游脚本 import)
def load_win(field: str, w: str) -> pd.DataFrame:
    return pd.read_parquet(win_path(field, w)).astype(np.float32)


def load_extra(name: str) -> pd.DataFrame:
    return pd.read_parquet(TMP / f"inc_t1_win_{name}.parquet").astype(np.float32)


def window_price(w: str, kind: str = "vwap") -> pd.DataFrame:
    """成交价面板。kind='vwap'(主)/'twap'。vwap 在窗口无成交(amt_W<=0)时退回 twap
    (该股票日不可成交,引擎 can=False;退回只为保持持有收益路径连续,见报告偏离第 2 条)。"""
    tw = load_win("twap", w)
    if kind == "twap" or w in SINGLE_BAR:
        return tw
    vw = load_win("vwap", w)
    amt = load_win("amt", w)
    return vw.where((amt > 0) & vw.notna(), tw)


# ------------------------------------------------------------------ 门槛
def _rel_err(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    a = a.astype(np.float64); b = b.astype(np.float64)
    both = np.isfinite(a) & np.isfinite(b)
    nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(a[both] - b[both]) / np.maximum(np.abs(b[both]), 1e-12)
    return (float(r.max()) if r.size else np.nan), int(both.sum()), nan_mismatch


ULP_TOL = 2.4e-7     # 2 个 float32 ulp:原始分钟数据在日面板构建之后被压成 float32(scripts/compact_raw.py),
                     # 求和类字段无法逐位复现,只能到 1 ulp(见报告偏离第 3 条)


def _bitwise(a: np.ndarray, b: np.ndarray) -> tuple[bool, int, int, float]:
    a = a.astype(np.float32); b = b.astype(np.float32)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    nanmis = int((np.isnan(a) != np.isnan(b)).sum())
    both = np.isfinite(a) & np.isfinite(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(a[both].astype(np.float64) - b[both]) / np.maximum(np.abs(b[both].astype(np.float64)), 1e-12)
    mr = float(rel.max()) if rel.size else 0.0
    return bool(same.all()) or (nanmis == 0 and mr <= ULP_TOL), int((~same).sum()), int(np.isfinite(b).sum()), mr


def cmd_checks():
    from afactor.panel import Panel
    p = Panel()
    rows = []

    def add(check, item, value, threshold, passed, note=""):
        rows.append(dict(check=check, item=item, value=value, threshold=threshold, passed=bool(passed), note=note))
        print(rows[-1], flush=True)

    # (a) twap_W1 vs twap_open10
    e, n, nm = _rel_err(load_win("twap", "W1").to_numpy(), p.raw("twap_open10").to_numpy())
    add("reproduce", "twap_W1 vs twap_open10 最大相对误差", e, 1e-5, e <= 1e-5 and nm == 0, f"n={n}, NaN 不一致 {nm}")
    # (b) vwap_W9 vs vwap
    e, n, nm = _rel_err(load_win("vwap", "W9").to_numpy(), p.raw("vwap").to_numpy())
    add("reproduce", "vwap_W9 vs vwap 最大相对误差", e, 1e-5, e <= 1e-5 and nm == 0, f"n={n}, NaN 不一致 {nm}")
    # (c) Σ 窗口成交额 vs amount
    tot = None
    n_nan = None
    for w in ["W0", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"]:
        a = load_win("amt", w).to_numpy(np.float64)
        nn = ~np.isfinite(a)
        n_nan = nn if n_nan is None else (n_nan | nn)
        a = np.nan_to_num(a, nan=0.0)                 # 单根 bar 窗口在该 bar 缺失时为 NaN(该 bar 无成交,计 0)
        tot = a if tot is None else tot + a
    tot = tot + np.nan_to_num(load_extra("amt_GAP").to_numpy(np.float64), nan=0.0)
    amount = p.raw("amount").to_numpy(np.float64)
    ok = np.isfinite(amount) & (amount > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(tot[ok] - amount[ok]) / amount[ok]
    add("reproduce", "Σ(W0..W8 + 14:58-59) 成交额 vs amount 最大相对误差", float(rel.max()), 1e-6, rel.max() <= 1e-6,
        f"n={int(ok.sum())}(amount>0 的股票日);某窗口成交额为 NaN(bar 缺失,按 0 计)的股票日 {int((n_nan & ok).sum())};"
        f"float32 存储舍入约 6e-8")
    # (c2) W9/W10+W11 成交额自洽
    a9 = load_win("amt", "W9").to_numpy(np.float64)
    e, n, nm = _rel_err(a9, amount)
    add("reproduce", "amt_W9 vs amount 最大相对误差", e, 1e-6, e <= 1e-6, f"n={n}")
    # (d) 逐位:amt_W1 vs amt_open10、amt_W8 vs amt_close_auction、twap_W8 vs close_px、twap_W0 vs open_auction_px
    for mine, ref in [(("amt", "W1"), "amt_open10"), (("amt", "W8"), "amt_close_auction"),
                      (("twap", "W8"), "close_px"), (("twap", "W0"), "open_auction_px"),
                      (("amt", "W0"), "amt_open_auction")]:
        same, ndiff, n, mr = _bitwise(load_win(*mine).to_numpy(), p.raw(ref).to_numpy())
        add("reproduce_bitwise", f"{mine[0]}_{mine[1]} vs {ref} 逐位(不一致单元数)", ndiff, 0, same,
            f"不一致单元 {ndiff} / 有值 {n};不一致处最大相对差 {mr:.2e}(容 {ULP_TOL:.1e} = 2 float32 ulp)")
    # (e) hi/lo W1 vs open10_high/low(f_var 同法的前提)
    for fld, ref in [("hi", "open10_high"), ("lo", "open10_low")]:
        same, ndiff, n, mr = _bitwise(load_win(fld, "W1").to_numpy(), p.raw(ref).to_numpy())
        add("reproduce_bitwise", f"{fld}_W1 vs {ref} 逐位(不一致单元数)", ndiff, 0, same, f"不一致单元 {ndiff} / 有值 {n};最大相对差 {mr:.2e}")
    for fld, ref in [("hi", "high_px"), ("lo", "low_px")]:
        same, ndiff, n, mr = _bitwise(load_win(fld, "W9").to_numpy(), p.raw(ref).to_numpy())
        add("reproduce_bitwise", f"{fld}_W9 vs {ref} 逐位(不一致单元数)", ndiff, 0, same, f"不一致单元 {ndiff} / 有值 {n};最大相对差 {mr:.2e}")
    # (f) 窗口价格落在全日 [low, high] 内(原料错位防护,按构造恒成立)
    hp = p.raw("high_px").to_numpy(np.float64); lp = p.raw("low_px").to_numpy(np.float64)
    for w in WINDOWS:
        for kind in ("twap", "vwap"):
            x = load_win(kind, w).to_numpy(np.float64)
            m = np.isfinite(x) & np.isfinite(hp)
            out_of = float(((x[m] > hp[m] * (1 + 1e-6)) | (x[m] < lp[m] * (1 - 1e-6))).mean())
            add("sanity", f"{kind}_{w} 落在 [low_px, high_px] 外的比例", out_of, 0.0, out_of == 0.0, f"n={int(m.sum())}")
    # (g) 成交额分窗口占比(信息性,与预注册 §3.4 2024-12 实测对照)
    sel = np.asarray((p.dates >= "2024-12-01") & (p.dates <= "2024-12-31"))
    for w in list(WINDOWS) + ["GAP"]:
        a = (load_extra("amt_GAP") if w == "GAP" else load_win("amt", w)).to_numpy(np.float64)[sel]
        am = amount[sel]
        ok2 = np.isfinite(am) & (am > 0) & np.isfinite(a)
        add("info", f"2024-12 {w} 成交额占全日比 股票日中位", float(np.median(a[ok2] / am[ok2])), np.nan, True,
            "对照预注册 §3.4")
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "inc_t1_window_build_checks.csv", index=False)
    gate = df[df.check.isin(["reproduce", "reproduce_bitwise", "sanity"])]
    print(f"门槛 {int(gate.passed.sum())}/{len(gate)} 通过", flush=True)
    if not gate.passed.all():
        print("!!! 有门槛未通过:", gate[~gate.passed].item.tolist(), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": cmd_build, "merge": cmd_merge, "checks": cmd_checks}[cmd]()
