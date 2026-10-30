"""5 分钟口径日频原料(前缀 m5):1 分钟 -> baostock 口径 5 分钟 bar -> 与 cache/daily 同形状的日频原料。

只处理研究期 2020-01 ~ 2025-12(硬性守卫:月份 > 202512 直接报错),产出 cache/m5/<字段>.parquet。
用途:量化"前向检验用 5 分钟线近似 1 分钟线"的误差(scripts/m5_bridge.py,报告 research/MINUTE5_BRIDGE.md)。

用法:
    python -m afactor.build_daily_5m                    # 2020-01 ~ 2025-12 全部月份(已有月度结果跳过)
    python -m afactor.build_daily_5m 202001 202003      # 指定区间
    python -m afactor.build_daily_5m --merge            # 合并月度结果 -> cache/m5/*.parquet
    python -m afactor.build_daily_5m --finalize         # 按 minute5 里选定的候选写出规范字段名

## 哪些原料需要 5 分钟重算(COMP_v5 24 个成员 + 回测引擎)
(a) 只依赖日级量,5 分钟 bar 可以**精确**或几乎精确得到:
    close_px  = bar47 收盘 = 1 分钟 idx240 收盘(逐位相同)
    amount    = Σ 48 根 bar 成交额 = Σ 241 根(只差浮点求和顺序)
    由此派生的 ret_c2c / turnover_proxy / adv20 / 决策池 / 市值 / 统计行业标签不受影响;基本面不受影响。
(a') 依赖开盘价(1 分钟口径 = idx0 收盘,5 分钟口径只能取 bar0 开盘 = 开盘集合竞价价,见 minute5 文档):
    open_auction_px、ret_intraday = 收盘/开盘 - 1、overnight(派生)。
(b) 依赖分钟结构,必须用 5 分钟 bar 重算 —— 本文件的 _day_features_5m,逐个说明映射:
    rvol / smart_money / semibeta_mp   分钟收益 -> 5 分钟收益(路径 = [开盘价, 48 根收盘],48 段)
    gap_share                          分钟间跳空/分钟实体 -> 5 分钟 bar 间跳空 / bar 实体
    amt_gini                           241 根分钟成交额的基尼 -> 48 根 bar 成交额的基尼
    amt_time_wgt                       成交额时间重心:分钟位置 -> bar 内各分钟的中心位置(仍在 0..240 网格上)
    amt_first30_share                  idx1..30(09:31-10:00)-> bar 0..5(09:35-10:00),**含开盘集合竞价**
    amt_openauc_share                  idx0 无法从 5 分钟 bar 拆出,只能用候选近似(minute5.openauc_share_candidates)
    vwap / twap_open10 / amt_open10    候选近似,按 2020 年误差选定(minute5 模块文档)
    rng                                (日内 max-min)/均值:241 个分钟收盘 -> 49 点路径(只用于一字板判定)
"""
from __future__ import annotations
import sys, time, warnings, glob, os, re
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from afactor.config import N_BARS, DAILY, RESEARCH_END, load_raw_minute, raw_months, assert_not_sealed
from afactor.build_daily import _safe_div, _gini, _smart_money
from afactor import minute5 as M5

warnings.filterwarnings("ignore")
EPS = 1e-12
LAST_MONTH = pd.Timestamp(RESEARCH_END).strftime("%Y%m")       # 202512

CANDIDATE_GROUPS = {
    "twap_open10": ("TWAP5_METHOD", M5.twap_open10_candidates),
    "amt_open10": ("AMT_OPEN10_METHOD", M5.amt_open10_candidates),
    "vwap": ("VWAP5_METHOD", M5.vwap_candidates),
    "amt_openauc_share": ("OPENAUC5_METHOD", M5.openauc_share_candidates),
}


def _day_features_5m(P5: np.ndarray, O5: np.ndarray, V5: np.ndarray) -> dict:
    """单日 5 分钟 (48, N) -> {字段: (N,)}。公式逐条对应 build_daily._day_features / build_market._mkt_features,
    只把"分钟"换成"5 分钟 bar",映射见各行注释。"""
    out = {}
    X = M5.price_path(P5, O5)                       # (49, N) 开盘价 + 48 根收盘(对应 1 分钟 P[0..240])
    r = _safe_div(X[1:], X[:-1]) - 1.0              # (48, N),对应 1 分钟 r = P[1:]/P[:-1]-1(240 段)
    r = np.clip(r, -0.5, 0.5)                       # 同 build_daily:防除权残留毛刺
    v1 = V5                                         # 与 r 对齐;1 分钟口径 v1 = V[1:] 不含 idx0,
    #                                                 5 分钟 bar0 的成交额含集合竞价(无法拆分)
    mid = (O5 + P5) / 2.0                           # bar 中价,同 1 分钟 mid = (open+close)/2
    tot_amt = np.nansum(V5, axis=0)

    # ---- (a)/(a') ----
    out["close_px"] = P5[M5.IDX5_CLOSE]
    out["open_auction_px"] = O5[M5.IDX5_0935]       # 1 分钟口径为 P[0](09:30 这一分钟收盘)
    out["amount"] = tot_amt
    out["ret_intraday"] = _safe_div(P5[M5.IDX5_CLOSE], O5[M5.IDX5_0935]) - 1.0

    # ---- (b) 收益结构 ----
    out["rvol"] = np.nanstd(r, axis=0)              # 1 分钟:240 段分钟收益的 std;此处 48 段(尺度约 ×√5,截面排序不受影响)
    out["rng"] = _safe_div(np.nanmax(X, axis=0) - np.nanmin(X, axis=0), np.nanmean(X, axis=0))

    # ---- (b) 量价 ----
    out["smart_money"] = _smart_money(r, v1, _safe_div(v1, mid))   # 同一函数,S=|r|/sqrt(股数近似),取前 20% 成交额
    out["amt_gini"] = _gini(V5)
    out["amt_time_wgt"] = _safe_div(np.nansum(V5 * (M5.BAR5_CENTER / (N_BARS - 1))[:, None], axis=0), tot_amt)
    # 1 分钟:Σ V[1:31](09:31-10:00,不含 idx0);5 分钟 bar0..5(09:35-10:00)只能连集合竞价一起算
    out["amt_first30_share"] = _safe_div(np.nansum(V5[:M5.IDX5_1000 + 1], axis=0), tot_amt)

    # ---- (b) 分钟形态:跳空 vs 实体 ----
    # 1 分钟:gap = O[1:]/P[:-1]-1(240 个分钟边界),body = P/O-1,分母用 body[1:](去掉 idx0)。
    # 5 分钟:gap 只剩 47 个 bar 边界(= 1 分钟口径在这些边界上的 gap,逐位相同),bar 内另 4 个分钟边界的
    #        跳空并入 bar 实体;同样去掉 bar0 的实体。
    gap = _safe_div(O5[1:], P5[:-1]) - 1.0
    body = _safe_div(P5, O5) - 1.0
    ag = np.nansum(np.abs(gap), axis=0)
    out["gap_share"] = _safe_div(ag, ag + np.nansum(np.abs(body[1:]), axis=0))

    # ---- (b) 市场互动:半贝塔"市场涨我跌"(build_market._mkt_features 同式)----
    valid = ~np.isnan(P5).all(axis=0)
    with np.errstate(invalid="ignore"):
        mkt = np.nanmean(np.where(valid[None, :], r, np.nan), axis=1)    # 全市场 5 分钟等权收益
    mkt2 = np.nansum(mkt ** 2)
    n_i = np.clip(r, None, 0)
    Pm = np.clip(mkt, 0, None)
    out["semibeta_mp"] = _safe_div(-np.nansum(n_i * Pm[:, None], axis=0), mkt2)

    # ---- 候选近似(选定前全部落盘,规范字段名由 --finalize 写出)----
    for fld, (_, fn) in CANDIDATE_GROUPS.items():
        cands = fn(V5) if fld in ("amt_open10", "amt_openauc_share") else fn(P5, O5, V5)
        for k, v in cands.items():
            out[f"{fld}__{k}"] = v
    return out


def process_month(month: str) -> str:
    if month > LAST_MONTH:
        raise PermissionError(f"build_daily_5m 只处理研究期(<= {LAST_MONTH}),拒绝 {month}")
    t0 = time.time()
    C = load_raw_minute("close_adj", month)
    O = load_raw_minute("open_adj", month)
    V = load_raw_minute("amt", month)
    codes = C.columns
    assert list(O.columns) == list(codes) and list(V.columns) == list(codes), f"{month} 列不一致"
    assert O.index.equals(C.index) and V.index.equals(C.index), f"{month} 索引不一致"
    days = C.index.normalize().values
    Ca, Oa, Va = C.to_numpy(np.float32), O.to_numpy(np.float32), V.to_numpy(np.float32)
    del C, O, V
    uniq, starts, counts = np.unique(days, return_index=True, return_counts=True)
    order = np.argsort(starts)
    feats, bad, good = {}, [], []
    for d, s, n in zip(uniq[order], starts[order], counts[order]):
        if n != N_BARS:
            bad.append((str(pd.Timestamp(d).date()), int(n)))
            continue
        P_ = Ca[s:s + n].astype(np.float64); O_ = Oa[s:s + n].astype(np.float64)
        V_ = Va[s:s + n].astype(np.float64)
        valid = ~np.isnan(P_).all(axis=0)
        P5, O5, V5 = M5.aggregate_day(P_, O_, V_)
        f = _day_features_5m(P5, O5, V5)
        for k, val in f.items():
            feats.setdefault(k, []).append(np.where(valid, val, np.nan))
        good.append(pd.Timestamp(d))
    assert_not_sealed(good)
    M5.M5_TMP.mkdir(parents=True, exist_ok=True)
    arrs = {k: np.asarray(v, dtype=np.float32) for k, v in feats.items()}
    np.savez_compressed(M5.M5_TMP / f"{month}.npz", dates=np.array([str(d.date()) for d in good]),
                        codes=np.array(list(codes), dtype=object), **arrs)
    return (f"{month}: {len(good)}日 x {len(codes)}股 x {len(arrs)}字段, {time.time() - t0:.0f}s"
            + (f"  异常日={bad}" if bad else ""))


def merge() -> None:
    """月度 npz -> cache/m5/<字段>.parquet,行列对齐 cache/daily/close_px(研究期面板形状)。"""
    ref = pd.read_parquet(DAILY / "close_px.parquet")
    dates, codes = ref.index, ref.columns
    di = {d: i for i, d in enumerate(dates.strftime("%Y-%m-%d"))}
    ci = {c: i for i, c in enumerate(codes)}
    files = sorted(glob.glob(str(M5.M5_TMP / "*.npz")))
    names = None
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            names = sorted(k for k in z.files if k not in ("dates", "codes"))
        break
    M5.M5_DIR.mkdir(parents=True, exist_ok=True)
    BATCH = 8
    for b0 in range(0, len(names), BATCH):
        batch = names[b0:b0 + BATCH]
        buf = {k: np.full((len(dates), len(codes)), np.nan, np.float32) for k in batch}
        for f in files:
            with np.load(f, allow_pickle=True) as z:
                d = [str(x) for x in z["dates"]]; c = [str(x) for x in z["codes"]]
                keep_d = [i for i, x in enumerate(d) if x in di]
                keep_c = [j for j, x in enumerate(c) if x in ci]
                ri = np.array([di[d[i]] for i in keep_d]); cj = np.array([ci[c[j]] for j in keep_c])
                if len(keep_d) != len(d):
                    print(f"  ! {os.path.basename(f)}: {len(d) - len(keep_d)} 个日期不在 cache/daily 索引内,已丢弃")
                for k in batch:
                    a = z[k][np.ix_(keep_d, keep_c)]
                    buf[k][np.ix_(ri, cj)] = a
        for k in batch:
            df = pd.DataFrame(buf[k], index=dates, columns=codes)
            df.index.name = ref.index.name
            df.to_parquet(M5.M5_DIR / f"{k}.parquet")
        print(f"  已写 {b0 + len(batch)}/{len(names)}", flush=True)


def finalize() -> None:
    """把 minute5 里选定的候选复制为规范字段名(twap_open10 / amt_open10 / vwap / amt_openauc_share)。"""
    for fld, (const, _) in CANDIDATE_GROUPS.items():
        meth = getattr(M5, const)
        src = M5.M5_DIR / f"{fld}__{meth}.parquet"
        pd.read_parquet(src).to_parquet(M5.M5_DIR / f"{fld}.parquet")
        print(f"  {fld} <- {src.name}")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--merge":
        merge(); print("done", flush=True); return
    if args and args[0] == "--finalize":
        finalize(); print("done", flush=True); return
    workers = 3
    if "--workers" in args:
        i = args.index("--workers"); workers = int(args[i + 1]); args = args[:i] + args[i + 2:]
    months = [m for m in raw_months("close_adj") if m <= LAST_MONTH]
    if len(args) >= 2:
        months = [m for m in months if args[0] <= m <= args[1]]
    todo = [m for m in months if not (M5.M5_TMP / f"{m}.npz").exists()]
    print(f"待处理 {len(todo)}/{len(months)} 个月,{workers} 进程", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_month, m): m for m in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                print(f"[{i}/{len(todo)}] {fu.result()}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(todo)}] {futs[fu]} FAILED: {type(e).__name__}: {e}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
