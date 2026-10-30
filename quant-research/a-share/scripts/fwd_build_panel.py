"""前向检验面板构建(前缀 fwd):cache/forward/panel/<字段>.parquet,日期 2025-01-02 ~ 2026-10-09。

预注册 research/FORWARD_TEST_PREREG.md §4.1–4.3;**本脚本不计算任何收益、因子、IC**,只把原料拼成与 Panel
同形状的日频面板。分钟类原料一律是 MINUTE5_BRIDGE 定下的 baostock 口径 5 分钟(三段来源,同一套公式):
  ≤ 2025-12-31            cache/m5(1 分钟精确聚合,桥接 C 口径)
  2026-01-01 ~ 08-17      new_data 1 分钟 -> fwd_adapters.minute_month_features(只作回看)
  2026-08-18 ~ 10-09      baostock 5 分钟 -> 日线修正 -> 复权到 08-17 基准 -> fwd_adapters.bs5_day_features
  另:08-03 ~ 08-17 两源重叠,以 1 分钟为准;1 分钟缺失但 baostock 有成交的股票日(QA:08-03~08-11 每天约 80 只)
      用 baostock 补;07-01 ~ 08-02 的 1 分钟缺口没有分钟级替代,只用 baostock 日线补收盘 / 开盘 / 成交额。
日级字段:收盘、开盘(open_auction_px)、成交额在 08-18 起取 baostock 日线(经 reconcile 写进 bar47/bar0);
价格水平:08-17 及以前统一到 08-17 单一前复权基准(见下"复权基准修正");08-18 起 × F_t(以 08-17 为基准的后复权,
F_t 只用 t 及以前的除权事件:ratio_e = 前一交易日收盘 / 除权日 preclose,取自 baostock 日线)。

**复权基准修正(审查 P0,2026-10-10,未预注册的数据修正)**:new_data 2026 年的 1 分钟月文件不是同一个前复权基准
—— 每个月文件各自复权到本月末(06-01、07-01、05-06 等月初出现约 3,300 个虚假跳变),另有月内除权未复权、
单日整体错乘等个别错误;研究期 2025-12 文件也有同类问题(77 只)。修正只动 4 个价格水平字段
(close_px / open_auction_px / twap_open10 / vwap;其余 12 个分钟原料是日内比值,与复权无关),对 <= 08-17 的每个股票日乘
    m(t) = tgt(t) / [close_px(t) / raw_close(t)]
使 close_px(t) ≡ raw_close(t) × tgt(t)(以 08-17 为基准的单一前复权,与 08-18 起的 baostock 段同一基准),同日其余价格字段
乘同一个 m(t)(日内比值不变,float32 舍入级)。2025 年行同样处理(研究期 cache/m5 的基准 -> 08-17 基准,逐股近似常数倍)。tgt 的除权比例:07-02 ~ 08-17 用 baostock 日线的交易所口径(前收 / preclose,与 F_t 同源),
<= 07-01 用 data_8:21 tushare adj_factor(rebase_target)。raw_close 缺失的股票日沿用同月前 / 后一个有效 m;无系数的股票 m = 1。
tgt(t) 只用 08-17 及以前的除权事件,基准日 08-17 早于首个信号日 09-01,不引入前视。m 矩阵存 cache/_tmp/fwd_aux/rebase_mult_0817.parquet。
北交所:baostock 不覆盖,2026-01-01 起北交所列全部 NaN(2025 年保留 cache/m5 原值);预注册 §4.1 市场均值剔除北交所。

阶段(可续跑,中间结果在 cache/_tmp/fwd_*):
    python3 scripts/fwd_build_panel.py m1        # 2026-01 ~ 08 一分钟月度 -> 5 分钟口径(3 进程)
    python3 scripts/fwd_build_panel.py bs5       # baostock 5 分钟 08-03 ~ 10-09 -> 5 分钟口径
    python3 scripts/fwd_build_panel.py daily     # 日线、复权、ST、换手、股本、市值、未复权价
    python3 scripts/fwd_build_panel.py assemble  # 拼成 cache/forward/panel/*.parquet
    python3 scripts/fwd_build_panel.py labels    # 统计行业标签(2026 各季)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from afactor.config import CACHE, TMP, DAILY, LEDGER, ROOT

PANEL_DIR = CACHE / "forward" / "panel"
M1_DIR = TMP / "fwd_m5_months"
BS5_FILE = TMP / "fwd_bs5_feats.npz"
AUX_DIR = TMP / "fwd_aux"
START, END = pd.Timestamp("2025-01-02"), pd.Timestamp("2026-10-09")
M1_END = pd.Timestamp("2026-08-17")           # 1 分钟数据末日
BS_SPLICE = pd.Timestamp("2026-08-18")        # baostock 接管日
OVERLAP0 = pd.Timestamp("2026-08-03")
MONTHS_2026 = [f"2026{m:02d}" for m in range(1, 9)]
for _d in (M1_DIR, AUX_DIR, PANEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =============================================================================================== m1
def _m1_one(month: str) -> str:
    from afactor import fwd_adapters as FA_
    out = M1_DIR / f"{month}.npz"
    if out.exists():
        return f"{month}: 已存在,跳过"
    r = FA_.minute_month_features(month, drop_bj=True, clean=True)
    np.savez_compressed(out, dates=np.array([str(d.date()) for d in r["dates"]]),
                        codes=np.array(list(r["codes"]), dtype=object), **r["fields"])
    json.dump(r["diag"], open(M1_DIR / f"{month}.diag.json", "w"), ensure_ascii=False, default=str)
    return f"{month}: {r['diag']['n_days']} 日 x {r['diag']['n_codes']} 股, {r['diag']['secs']}s, 诊断 {r['diag']}"


def stage_m1(workers: int = 3):
    todo = [m for m in MONTHS_2026 if not (M1_DIR / f"{m}.npz").exists()]
    log(f"[m1] 待处理 {todo}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_m1_one, m): m for m in todo}
        for fu in as_completed(futs):
            try:
                log("[m1]", fu.result())
            except Exception as e:
                log(f"[m1] {futs[fu]} FAILED: {type(e).__name__}: {e}")
                raise


# =============================================================================================== daily
def _cutoff_ctx():
    """前向检验放宽数据截止日(fixb 接口;调用者文件名前缀 fwd,进入 / 退出打印横幅并写审计日志)。"""
    from afactor import config as C
    return C.set_data_cutoff("2026-10-09", reason="前向检验(fwd):构建前向面板;2026-01-01~08-30 只作特征回看/预热")


def _aux_save(name: str, df: pd.DataFrame):
    df.index.name = "date"
    df.columns.name = "ts_code"
    df.to_parquet(AUX_DIR / f"{name}.parquet")


def _aux(name: str) -> pd.DataFrame:
    return pd.read_parquet(AUX_DIR / f"{name}.parquet")


def stage_daily():
    """baostock 日线(07-01 ~ 10-09)-> 复权系数、日级字段;daily_basic(<= 08-17)-> 未复权价 / 换手 / 流通股本;
    ST(研究期名单 + 2026 名单 + baostock isST);PIT 总股本与真实市值(预注册 §4.3)。不计算任何收益。"""
    from afactor import fwd_data as FD, newdata as ND, fundamental as FU, config as C
    t0 = time.time()
    d = FD.load_daily().sort_values(["code", "date"]).reset_index(drop=True)
    for c in ("tradestatus", "isST"):
        d[c] = d[c].astype("float64")
    prevc = d.groupby("code")["close"].shift(1)
    ok_r = (prevc > 0) & (d["preclose"] > 0)
    ratio = (prevc / d["preclose"]).where(ok_r, 1.0)
    d["ratio"] = ratio.where((ratio - 1.0).abs() > 1e-12, 1.0)
    d["B"] = d.groupby("code")["ratio"].cumprod()
    piv = lambda c: d.pivot(index="date", columns="code", values=c).sort_index()
    B = piv("B").ffill()
    B0817 = B.loc[:M1_END].ffill().iloc[-1].fillna(1.0)       # 08-17 收盘时的累计系数(之后上市的 = 1)
    A = B / B0817                                              # 价格 × A = 以 08-17 为基准的复权价(<=08-17 前复权、之后后复权)
    ratio_p = piv("ratio").fillna(1.0)
    raw_close_bs = piv("close")
    ok_bs = (piv("tradestatus") == 1) & (piv("amount") > 0) & (raw_close_bs > 0)
    _aux_save("bs_A", A.astype(np.float64))
    _aux_save("bs_ratio", ratio_p.astype(np.float64))
    _aux_save("bs_close_adj", (raw_close_bs * A).where(ok_bs).astype(np.float64))
    _aux_save("bs_open_adj", (piv("open") * A).where(ok_bs).astype(np.float64))
    _aux_save("bs_amount", piv("amount").where(ok_bs).astype(np.float64))
    _aux_save("bs_raw_close", raw_close_bs.where(raw_close_bs > 0).astype(np.float64))
    _aux_save("bs_ok", ok_bs.astype(np.float32))
    turn = piv("turn")
    _aux_save("bs_turn", turn.where(ok_bs).astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        fl = (piv("volume") / (turn / 100.0)).where((turn > 0) & ok_bs)
    _aux_save("bs_float_implied", fl.astype(np.float64))
    _aux_save("bs_isST", piv("isST").astype(np.float64))
    log(f"[daily] baostock 日线 {d['date'].min().date()}~{d['date'].max().date()},{d['code'].nunique()} 只;"
        f"除权事件(ratio≠1)08-18 起 {int(((ratio_p.loc[BS_SPLICE:] != 1.0)).sum().sum())} 个 [{time.time()-t0:.0f}s]")

    # ---- 面板日期与代码(与 assemble 相同)
    dates, codes = panel_index()
    with _cutoff_ctx():
        # 未复权价 / 换手 / 流通股本:daily_basic(<= 08-17)
        d_db = dates[dates <= M1_END]
        rc = ND.raw_close(d_db, codes)
        tr = ND.turnover_rate(d_db, codes)
        fs = ND.float_share(d_db, codes)
        _aux_save("db_raw_close", rc.astype(np.float64))
        _aux_save("db_turnover", tr.astype(np.float64))
        _aux_save("db_float_share", fs.astype(np.float64))
        log(f"[daily] daily_basic 到 {d_db[-1].date()} [{time.time()-t0:.0f}s]")
        # ST:<= 2026-06-30 用 data_8:21 名单(源到 07-03;与 FU.build_st_panel 同一实现),07-01 起 baostock isST
        d_st = dates[dates <= pd.Timestamp("2026-06-30")]
        st1 = FU.build_st_panel(d_st, codes)
        _aux_save("st_list", st1.astype(np.float32))
        # 总股本(PIT):tushare 资产负债表 total_share(fixb 版本 PIT,源到 06-13)+ 报告期序号
        with C.accept_source_gap(FU.REPORT_TABLES, reason="前向面板:tushare 股本只到 2026-06-13,之后接 baostock 季频 totalShare"):
            rep = FU.build_report_level()
            rep = rep.assign(ref_qi=np.where(rep["total_share"].notna(),
                                             rep["end_date"].dt.year * 4 + (rep["end_date"].dt.quarter - 1), np.nan))
            sh_ts = FU._pit_daily(rep, "total_share", dates, codes)
            qi_ts = FU._pit_daily(rep, "ref_qi", dates, codes)
    _aux_save("sh_ts", sh_ts.astype(np.float64))
    _aux_save("qi_ts", qi_ts.astype(np.float64))
    log(f"[daily] ST 名单与 tushare 股本 [{time.time()-t0:.0f}s]")

    # baostock 季频股本(按 pubDate 生效,frontier 取最新报告期;与 _pit_daily 同一取法)
    pr = FD.load_profit()
    pr = pr[(pr["totalShare"] > 0) & pr["pubDate"].notna() & pr["statDate"].notna()].copy()
    pr["qi"] = pr["statDate"].dt.year * 4 + (pr["statDate"].dt.quarter - 1)
    pr = pr.sort_values(["code", "pubDate", "statDate"])
    pr["front"] = pr.groupby("code")["qi"].cummax()
    pr = pr[pr["qi"] >= pr["front"]].drop_duplicates(["code", "pubDate"], keep="last")
    def _pit(col):
        pv = pr.pivot_table(index="pubDate", columns="code", values=col, aggfunc="last").reindex(columns=codes)
        return pv.reindex(pv.index.union(dates)).ffill().reindex(dates)
    sh_bs, qi_bs = _pit("totalShare"), _pit("qi")
    # 两源合并:取报告期更新的一条;同一报告期以 tushare 为准
    use_bs = qi_bs.notna() & (qi_ts.isna() | (qi_bs > qi_ts))
    S = sh_ts.where(~use_bs, sh_bs)
    QI = qi_ts.where(~use_bs, qi_bs)
    # 除权因子:tushare adj_factor(<= 08-17)接 baostock(08-18 起 × A),股本按"报告期末之后的除权"放大(预注册 §7.4)
    af = pd.read_csv(ROOT / "data_8:21" / "adj_factor.csv", dtype={"ts_code": str})
    af = af[af["trade_date"] >= 20231001]
    af["date"] = pd.to_datetime(af["trade_date"].astype(str))
    AF = af.pivot(index="date", columns="ts_code", values="adj_factor").sort_index().reindex(columns=codes)
    AF = AF.loc[:M1_END].ffill()
    Ab = A.reindex(columns=codes)
    post = Ab.loc[BS_SPLICE:].mul(AF.iloc[-1], axis=1)
    AF = pd.concat([AF, post]).sort_index()
    AF = AF.reindex(AF.index.union(dates)).ffill()
    qend = pd.date_range("2023-09-30", "2026-12-31", freq="Q")
    afq = AF.reindex(AF.index.union(qend)).ffill().reindex(qend)
    afq.index = [int(q.year * 4 + (q.quarter - 1)) for q in qend]
    QIv = QI.to_numpy(np.float64)
    rowpos = {q: i for i, q in enumerate(afq.index)}
    qmap = np.vectorize(lambda q: rowpos.get(int(q), -1) if np.isfinite(q) else -1)(QIv)
    afq_arr = afq.to_numpy(np.float64)
    jj = np.broadcast_to(np.arange(len(codes))[None, :], qmap.shape)
    af_s0 = np.where(qmap >= 0, afq_arr[np.clip(qmap, 0, None), jj], np.nan)
    af_t = AF.reindex(dates).to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mult = af_t / af_s0
    mult = np.where(np.isfinite(mult) & (mult > 0), mult, 1.0)
    S_adj = pd.DataFrame(S.to_numpy(np.float64) * mult, index=dates, columns=codes)
    _aux_save("total_share_pit", S_adj)
    _aux_save("total_share_qi", QI.astype(np.float64))
    log(f"[daily] PIT 总股本:baostock 接管 {int(use_bs.to_numpy().sum())} 格;除权放大 >1% 的格 "
        f"{int((mult > 1.01).sum())} [{time.time()-t0:.0f}s]")


def panel_index() -> tuple[pd.DatetimeIndex, pd.Index]:
    """前向面板的行列:日期 = 研究期 2025 年交易日 + baostock 交易日历 2026-01-01 ~ 10-09;
    代码 = 研究面板全部代码(原顺序)+ 2026-08-18 以后上市、研究面板没有的沪深新股(排序后追加)。"""
    ref = pd.read_parquet(DAILY / "close_px.parquet", columns=["000001.SZ"])
    d25 = ref.index[(ref.index >= START)]
    cal = pd.read_csv(CACHE / "forward" / "raw" / "bs_meta" / "trade_dates_20251201_20261031.csv")
    td = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal["is_trading_day"] == 1, "calendar_date"]))
    d26 = td[(td >= pd.Timestamp("2026-01-01")) & (td <= END)]
    dates = pd.DatetimeIndex(list(d25) + list(d26), name="date")
    rcodes = pd.read_parquet(DAILY / "close_px.parquet").columns
    u = pd.read_csv(CACHE / "forward" / "raw" / "bs_meta" / "universe.csv")
    new = sorted(set(u.loc[(u["in_panel"] == 0) & (u["covered"] == 1), "code"]) - set(rcodes))
    codes = pd.Index(list(rcodes) + new, name="ts_code")
    return dates, codes


# =============================================================================================== bs5
def stage_bs5():
    """baostock 5 分钟(不复权)08-03 ~ 10-09 -> 逐日 16 个 5 分钟口径原料(fwd_adapters.bs5_day_features)。"""
    from afactor import fwd_data as FD, fwd_adapters as FA_
    t0 = time.time()
    m = FD.load_m5()
    tpos = {t: k for k, t in enumerate(FD.BS5_TIMES)}
    m["k"] = m["time"].astype(str).map(tpos)
    assert m["k"].notna().all()
    days = pd.DatetimeIndex(sorted(m["date"].unique()))
    codes = pd.Index(sorted(m["code"].astype(str).unique()))
    ci = codes.get_indexer(m["code"].astype(str))
    di = days.get_indexer(m["date"])
    ki = m["k"].to_numpy(int)
    shp = (len(days), 48, len(codes))
    arr = {}
    for c in ("close", "open", "amount"):
        a = np.full(shp, np.nan)
        a[di, ki, ci] = m[c].to_numpy(np.float64)
        arr[c] = a
    nb = np.zeros((len(days), len(codes)), int)
    np.add.at(nb, (di, ci), 1)
    del m
    log(f"[bs5] 长表 -> 数组 {shp} [{time.time()-t0:.0f}s]")
    A = _aux("bs_A").reindex(index=days, columns=codes)
    cl = _aux("bs_raw_close").reindex(index=days, columns=codes)
    ok = _aux("bs_ok").reindex(index=days, columns=codes).fillna(0).astype(bool)
    dd = FD.load_daily()
    op = dd.pivot(index="date", columns="code", values="open").reindex(index=days, columns=codes)
    am = dd.pivot(index="date", columns="code", values="amount").reindex(index=days, columns=codes)
    out = {}
    for i, day in enumerate(days):
        okd = ok.iloc[i].to_numpy(bool) & (nb[i] == 48)
        f = FA_.bs5_day_features(arr["close"][i], arr["open"][i], arr["amount"][i], cl.iloc[i].to_numpy(np.float64),
                                 op.iloc[i].to_numpy(np.float64), am.iloc[i].to_numpy(np.float64),
                                 A.iloc[i].fillna(1.0).to_numpy(np.float64), okd)
        for k, v in f.items():
            out.setdefault(k, []).append(v)
    np.savez_compressed(BS5_FILE, dates=np.array([str(x.date()) for x in days]),
                        codes=np.array(list(codes), dtype=object),
                        **{k: np.asarray(v, np.float32) for k, v in out.items()})
    log(f"[bs5] 写 {BS5_FILE.name}:{len(days)} 日 x {len(codes)} 股 [{time.time()-t0:.0f}s]")


def _load_npz(f) -> tuple[pd.DatetimeIndex, pd.Index, dict]:
    with np.load(f, allow_pickle=True) as z:
        dts = pd.DatetimeIndex(pd.to_datetime([str(x) for x in z["dates"]]))
        cs = pd.Index([str(x) for x in z["codes"]])
        flds = {k: z[k] for k in z.files if k not in ("dates", "codes")}
    return dts, cs, flds


# =============================================================================================== rebase
PRICE_LEVEL_FIELDS = ("close_px", "open_auction_px", "twap_open10", "vwap")   # 其余分钟原料是日内比值,与复权无关


def load_ts_adj_factor(dates: pd.DatetimeIndex, codes: pd.Index, end=M1_END) -> pd.DataFrame:
    """data_8:21 tushare adj_factor(源到 2026-08-21),只取 <= end 的行,按面板日历前值填充(停牌日沿用)。"""
    af = pd.read_csv(ROOT / "data_8:21" / "adj_factor.csv", dtype={"ts_code": str},
                     usecols=["ts_code", "trade_date", "adj_factor"])
    af = af[(af["trade_date"] >= int(dates[0].strftime("%Y%m%d")) - 10000) & (af["trade_date"] <= int(end.strftime("%Y%m%d")))]
    af["date"] = pd.to_datetime(af["trade_date"].astype(str))
    AF = af.pivot(index="date", columns="ts_code", values="adj_factor").sort_index().reindex(columns=codes)
    return AF.reindex(AF.index.union(dates)).ffill().reindex(dates).loc[:end]


BS_DAILY0 = pd.Timestamp("2026-07-01")         # baostock 日线首日(其后的除权比例 = 前收 / 交易所 preclose)


def rebase_target(dates: pd.DatetimeIndex, codes: pd.Index) -> pd.DataFrame:
    """<= 08-17 各股票日"以 08-17 为基准"的目标复权系数 tgt(t)(复权价 = 未复权收盘 × tgt)。
    除权比例的口径:07-02 ~ 08-17 的事件用 baostock 日线的交易所口径(前收 / 除权日 preclose,与 08-18 起的 F_t、
    1 分钟供应商、研究期面板同一口径);<= 07-01 的事件只有 tushare adj_factor(理论除权价口径,与交易所 preclose
    在低价股上差约 1e-3;另发现 tushare 漏记 689009.SH 2026-08-07 的 CDR 分红,07-02 起用 baostock 即可避开)。
        t >= 07-01:tgt(t) = A(t)                                   (A = fwd_aux/bs_A = 1 / Π_{t<e<=08-17} ratio_e)
        t <  07-01:tgt(t) = AF(t)/AF(07-01) × A(07-01)
    不在 baostock 日线里的股票(07-01 之前退市等)退回 tgt = AF(t)/AF(08-17)。"""
    AF = load_ts_adj_factor(dates, codes)
    d1 = AF.index
    tgt_ts = AF / AF.loc[M1_END]
    A = _aux("bs_A").reindex(columns=codes)
    A = A.reindex(A.index.union(d1)).loc[BS_DAILY0:M1_END].reindex(d1[d1 >= BS_DAILY0])
    tgt = tgt_ts.copy()
    late = d1 >= BS_DAILY0
    tgt.loc[late] = A.to_numpy(np.float64)
    tgt.loc[~late] = (AF.loc[~late] / AF.loc[BS_DAILY0]).mul(A.loc[BS_DAILY0], axis=1).to_numpy(np.float64)
    return tgt.where(tgt.notna(), tgt_ts)


def rebase_mult(close_px: np.ndarray, dates: pd.DatetimeIndex, codes: pd.Index) -> np.ndarray:
    """复权基准修正倍数 m (T, N),float64;> 08-17 的行恒为 1。
    m(t) = tgt(t) / [close_px(t)/raw_close(t)](tgt 见 rebase_target),raw_close 取 daily_basic(<= 08-17,与面板 raw_close 同源)。
    raw_close 或 tgt 缺失而 close_px 有值的格:同一(股票, 月份)内先前值后后值填充(月文件内基准一致),再跨月前值 / 后值,
    仍缺(无 AF 的股票)= 1。"""
    rows = np.asarray(dates <= M1_END)
    d1 = dates[rows]
    c = pd.DataFrame(close_px[rows].astype(np.float64), index=d1, columns=codes)
    rc = _aux("db_raw_close").reindex(index=d1, columns=codes).astype(np.float64)
    tgt = rebase_target(dates, codes)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = tgt * rc / c
    m = m.where(np.isfinite(m) & (m > 0) & c.notna())
    mon = d1.to_period("M")
    m = m.groupby(mon).transform(lambda x: x.ffill().bfill())
    m = m.ffill().bfill().fillna(1.0)
    out = np.ones(close_px.shape, np.float64)
    out[rows] = m.to_numpy(np.float64)
    return out


# =============================================================================================== assemble
def stage_assemble():
    """拼 cache/forward/panel/*.parquet。来源优先级见模块文档;每个格子的来源另存 src_code(诊断)。"""
    from afactor import fwd_adapters as FA_, minute5 as M5
    t0 = time.time()
    dates, codes = panel_index()
    T, N = len(dates), len(codes)
    fields = list(FA_.FIELDS)
    # 来源编码:0 无,1 cache/m5(2025),2 一分钟聚合(2026),3 baostock 5 分钟(1 分钟缺口补齐,08-03~08-17),
    #           4 baostock 5 分钟(08-18 起),5 baostock 日线补日级字段(07-01~08-02 一分钟缺口)
    src = np.zeros((T, N), np.int8)
    buf = {f: np.full((T, N), np.nan, np.float32) for f in fields}
    # 1) 2025:cache/m5
    d25 = dates[dates.year == 2025]
    r25 = dates.get_indexer(d25)
    for f in fields:
        x = pd.read_parquet(M5.M5_DIR / f"{f}.parquet").reindex(index=d25, columns=codes).to_numpy(np.float32)
        buf[f][r25] = x
    src[r25] = np.where(np.isfinite(buf["close_px"][r25]), 1, 0)
    # 2) 2026-01 ~ 08-17:一分钟聚合
    for mth in MONTHS_2026:
        dts, cs, fl = _load_npz(M1_DIR / f"{mth}.npz")
        ri = dates.get_indexer(dts)
        assert (ri >= 0).all(), f"{mth} 有不在交易日历里的日期 {dts[ri < 0]}"
        cj = codes.get_indexer(cs)
        assert (cj >= 0).all()
        for f in fields:
            buf[f][np.ix_(ri, cj)] = fl[f]
        src[np.ix_(ri, cj)] = np.where(np.isfinite(fl["close_px"]), 2, src[np.ix_(ri, cj)])
    m1_dates = dates[(dates >= pd.Timestamp("2026-01-01")) & (dates <= M1_END)]
    log(f"[assemble] 一分钟段 {m1_dates[0].date()}~{m1_dates[-1].date()} [{time.time()-t0:.0f}s]")
    # 3) baostock 5 分钟:08-18 起全部;08-03 ~ 08-17 只补一分钟缺失的股票日
    dts, cs, fl = _load_npz(BS5_FILE)
    ri = dates.get_indexer(dts)
    cj = codes.get_indexer(cs)
    assert (ri >= 0).all() and (cj >= 0).all()
    okb = np.isfinite(fl["close_px"])
    for k, (r, day) in enumerate(zip(ri, dts)):
        if day >= BS_SPLICE:
            take = np.ones(len(cj), bool)
            code_src = 4
        else:
            take = ~np.isfinite(buf["close_px"][r, cj]) & okb[k]
            code_src = 3
        for f in fields:
            row = buf[f][r]
            row[cj[take]] = fl[f][k, take]
        src[r, cj[take & okb[k]]] = code_src
        if day >= BS_SPLICE:
            src[r, cj[take & ~okb[k]]] = 0
    # 北交所:2026-01-01 起无前向数据源(baostock 不覆盖;预注册 §4.1)
    bj = np.array([c.endswith(".BJ") for c in codes])
    r26 = dates >= pd.Timestamp("2026-01-01")
    for f in fields:
        buf[f][np.ix_(r26, bj)] = np.nan
    src[np.ix_(r26, bj)] = 0
    # 4) 07-01 ~ 08-02 一分钟缺口:baostock 日线补收盘 / 开盘 / 成交额 / 日内收益(分钟结构字段留 NaN)
    cl, op, am = _aux("bs_close_adj"), _aux("bs_open_adj"), _aux("bs_amount")
    gap_days = dates[(dates >= pd.Timestamp("2026-07-01")) & (dates < OVERLAP0)]
    n_fill = 0
    for day in gap_days:
        r = dates.get_loc(day)
        c_ = cl.loc[day].reindex(codes).to_numpy(np.float64)
        o_ = op.loc[day].reindex(codes).to_numpy(np.float64)
        a_ = am.loc[day].reindex(codes).to_numpy(np.float64)
        miss = ~np.isfinite(buf["close_px"][r]) & np.isfinite(c_) & ~bj
        buf["close_px"][r, miss] = c_[miss]
        buf["open_auction_px"][r, miss] = o_[miss]
        buf["amount"][r, miss] = a_[miss]
        with np.errstate(invalid="ignore", divide="ignore"):
            buf["ret_intraday"][r, miss] = (c_[miss] / o_[miss] - 1.0)
        src[r, miss] = 5
        n_fill += int(miss.sum())
    log(f"[assemble] 7 月一分钟缺口用日线补 {n_fill} 个股票日 [{time.time()-t0:.0f}s]")
    # 5) 复权基准修正(审查 P0):<= 08-17 的价格水平字段统一到 08-17 单一前复权基准(模块文档)
    mult = rebase_mult(buf["close_px"], dates, codes)
    rows_m = np.asarray(dates <= M1_END)
    for f in PRICE_LEVEL_FIELDS:
        x = buf[f][rows_m].astype(np.float64) * mult[rows_m]
        buf[f][rows_m] = x.astype(np.float32)
    _aux_save("rebase_mult_0817", pd.DataFrame(mult, index=dates, columns=codes))
    chg = np.abs(mult - 1.0) > 1e-4
    log(f"[assemble] 复权基准修正:|m-1|>1e-4 的股票日 {int(chg.sum())}(2026 年 {int(chg[np.asarray(dates.year == 2026)].sum())}),"
        f"涉及 {int(chg.any(axis=0).sum())} 只 [{time.time()-t0:.0f}s]")
    for f in fields:
        df = pd.DataFrame(buf[f], index=dates, columns=codes)
        df.to_parquet(PANEL_DIR / f"{f}.parquet")
    pd.DataFrame(src, index=dates, columns=codes).to_parquet(PANEL_DIR / "_src_code.parquet")

    # ---- 日级辅助字段:未复权价、换手、流通股本、ST、总股本、真实市值、流通市值
    late = np.asarray(dates > M1_END)

    def splice(db_name, bs_name):
        out = _aux(db_name).reindex(index=dates, columns=codes).copy()
        b = _aux(bs_name).reindex(index=dates, columns=codes)
        out.loc[late] = b.loc[late]
        return out
    fin = np.isfinite(buf["close_px"])
    raw_close = splice("db_raw_close", "bs_raw_close").where(fin)    # 与 close_px 同一"有成交"口径
    turnover = splice("db_turnover", "bs_turn").where(fin)
    fs = splice("db_float_share", "bs_float_implied")
    fs.loc[late] = fs.ffill().loc[late]                               # 08-18 起换手为 0(停牌)时沿用前值
    S = _aux("total_share_pit").reindex(index=dates, columns=codes)
    mv = raw_close * S
    circ = raw_close * fs
    st_list = _aux("st_list").reindex(index=dates, columns=codes)
    st_bs = _aux("bs_isST").reindex(index=dates, columns=codes)
    jul = np.asarray(dates >= pd.Timestamp("2026-07-01"))
    is_st = st_list.copy()
    is_st.loc[jul] = st_bs.loc[jul]
    is_st = is_st.ffill().fillna(0.0)
    for nm, df in (("raw_close", raw_close), ("turnover_rate", turnover), ("float_share", fs),
                   ("total_share", S), ("mv", mv), ("circ_mv", circ), ("is_st", is_st)):
        df.astype(np.float32).to_parquet(PANEL_DIR / f"{nm}.parquet")
    meta = dict(built_at=time.strftime("%Y-%m-%d %H:%M:%S %z"), n_dates=T, n_codes=N,
                first=str(dates[0].date()), last=str(dates[-1].date()), fields=fields,
                aux=["raw_close", "turnover_rate", "float_share", "total_share", "mv", "circ_mv", "is_st",
                     "statind_labels", "_src_code"],
                src_codes={0: "无", 1: "cache/m5(2025,1 分钟精确聚合)", 2: "new_data 1 分钟聚合(2026-01~08-17)",
                           3: "baostock 5 分钟补 1 分钟缺口(08-03~08-17)", 4: "baostock 5 分钟(08-18 起)",
                           5: "baostock 日线补日级字段(07-01~08-02 1 分钟缺口)"})
    json.dump(meta, open(PANEL_DIR / "_meta.json", "w"), ensure_ascii=False, indent=1)
    log(f"[assemble] 写 cache/forward/panel:{T} 日 x {N} 股 [{time.time()-t0:.0f}s]")


# =============================================================================================== labels
def stage_labels():
    """统计行业标签:2025 年行用研究期标签文件;2026 各季用 fwd_adapters.statind_labels_for_quarters
    (与 statind.build_labels 同一算法与季度种子)在前向面板收盘价上计算。"""
    from afactor import fwd_adapters as FA_, statind as SI
    t0 = time.time()
    close = pd.read_parquet(PANEL_DIR / "close_px.parquet")
    L = FA_.statind_labels_for_quarters(close, ["2025Q4", "2026Q1", "2026Q2", "2026Q3"])
    ref = SI.load_labels().reindex(index=close.index, columns=close.columns)
    r25 = close.index.year == 2025
    L.loc[r25] = ref.loc[r25].fillna(-1).astype(np.int16).to_numpy()
    L.astype(np.int16).to_parquet(PANEL_DIR / "statind_labels.parquet")
    log(f"[labels] 写 statind_labels:2026 各季有标签股票数 "
        f"{[(str(q), int((L[L.index.to_period('Q') == q] >= 0).any().sum())) for q in pd.period_range('2026Q1', '2026Q4', freq='Q')]}"
        f" [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    st = sys.argv[1:] or ["m1"]
    for s in st:
        globals()[f"stage_{s}"]()
