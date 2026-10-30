"""补最高/最低价字段(2026-08-23,事件驱动引擎"突破前高后回踩"路径需要)。

原始分钟数据只有 close_adj / open_adj / amt,没有分钟 high/low,故:
  high_px  = 当日所有分钟 max(open, close) 的最大值(≈ 最高价,偏低不超过一根分钟 K 的上影)
  low_px   = 当日所有分钟 min(open, close) 的最小值
  pre_high_px / pre_low_px = 09:30-14:49(bar[:230])同口径 —— 14:49 时点可得
  open10_high / open10_low = 09:31-09:40 窗口(与 twap_open10 同窗)—— 早盘退出判定用
写入 cache/daily(2020-2025)与 cache/sealed(2026),与现有面板同格式。**不读取封存期的值做任何评估。**
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from afactor.config import load_raw_minute, raw_months, PRE_CUTOFF, N_BARS, DAILY, SEALED, SEAL_START

OPEN_WIN = (1, 11)


def one_month(month):
    C = load_raw_minute("close_adj", month); O = load_raw_minute("open_adj", month)
    codes = C.columns; days = C.index.normalize().values
    Ca, Oa = C.to_numpy(np.float64), O.to_numpy(np.float64)
    hi_m = np.fmax(Ca, Oa); lo_m = np.fmin(Ca, Oa)
    uniq, starts, counts = np.unique(days, return_index=True, return_counts=True)
    rows = {k: [] for k in ["high_px", "low_px", "pre_high_px", "pre_low_px", "open10_high", "open10_low"]}; good = []
    for d, s, n in zip(uniq, starts, counts):
        if n != N_BARS: continue
        H, L = hi_m[s:s + n], lo_m[s:s + n]
        with np.errstate(all="ignore"):
            rows["high_px"].append(np.nanmax(H, axis=0)); rows["low_px"].append(np.nanmin(L, axis=0))
            rows["pre_high_px"].append(np.nanmax(H[:PRE_CUTOFF], axis=0)); rows["pre_low_px"].append(np.nanmin(L[:PRE_CUTOFF], axis=0))
            rows["open10_high"].append(np.nanmax(H[OPEN_WIN[0]:OPEN_WIN[1]], axis=0)); rows["open10_low"].append(np.nanmin(L[OPEN_WIN[0]:OPEN_WIN[1]], axis=0))
        good.append(pd.Timestamp(d))
    return month, {k: pd.DataFrame(np.asarray(v, np.float32), index=pd.DatetimeIndex(good), columns=codes) for k, v in rows.items()}


def main():
    months = raw_months("close_adj"); t0 = time.time(); parts = {}
    with ProcessPoolExecutor(4) as ex:
        for fu in as_completed([ex.submit(one_month, m) for m in months]):
            m, d = fu.result(); parts[m] = d
            if len(parts) % 10 == 0: print(f"  {len(parts)}/{len(months)}  {time.time()-t0:.0f}s", flush=True)
    ref = pd.read_parquet(DAILY / "close_px.parquet"); ref_s = pd.read_parquet(SEALED / "close_px.parquet")
    for k in ["high_px", "low_px", "pre_high_px", "pre_low_px", "open10_high", "open10_low"]:
        full = pd.concat([parts[m][k] for m in sorted(parts)]).sort_index()
        full = full.reindex(columns=ref.columns)
        full[full.index < pd.Timestamp(SEAL_START)].reindex(ref.index).to_parquet(DAILY / f"{k}.parquet")
        full[full.index >= pd.Timestamp(SEAL_START)].reindex(ref_s.index).to_parquet(SEALED / f"{k}.parquet")
    # 自检:high >= max(open, close) 且 low <= min(open, close),用研究期一天抽样
    h = pd.read_parquet(DAILY / "high_px.parquet"); l = pd.read_parquet(DAILY / "low_px.parquet")
    c = ref; o = pd.read_parquet(DAILY / "open_auction_px.parquet")
    ok = ((h >= np.fmax(c, o) - 1e-6) | h.isna()).mean().mean(); ok2 = ((l <= np.fmin(c, o) + 1e-6) | l.isna()).mean().mean()
    print(f"自检 high>=max(open,close): {ok*100:.3f}%  low<=min: {ok2*100:.3f}%  | 耗时 {time.time()-t0:.0f}s")
    print(f"日振幅 (high/low-1) 中位 {((h/l-1).stack().median())*100:.2f}%")


if __name__ == "__main__":
    main()
