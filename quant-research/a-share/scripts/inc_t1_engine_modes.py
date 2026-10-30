"""INC-T1 第二层:引擎级单窗口执行模式(预注册 §4.2)与新模式的引擎断言(§3.3 第 4 条)。

不改任何库代码:新模式在进程内注入 ST.EXEC_MODES,并把 ST.exec_prices 替换为
inc_t1_common._patched_exec_prices(mode 以 inc_t1_ 开头时返回 (窗口价格, 窗口成交额),其余转回原函数)。
目标全部是 ndarray(禁止 callable 目标:decision_window_amount 按 mode 前缀映射字段,对新模式会映射错)。

模式(14 个):W1_twap(= 现行 open10 口径,参照)、W1_vwap、W2..W7 / W9..W11 的 vwap、W0 / W8(单价)、W1_twap_lag2。
成本:恒定 AUM + 窗口缩放冲击 s_W(逐年池均值比,§6)+ 一跳半价差,Y=1 主口径;Y 0.5/1.5 只对主候选
(W9、W5、W7 与参照 W1_twap / W1_vwap)。参与率上限 10% × te 当天窗口成交额(POV)。

审查修复(2026-10-11):
  P0  窗口内整窗口封死在涨 / 跌停价的股票日 -> 窗口成交额置 0 -> 引擎 can = False(inc_t1_common.register_mode(seal=True),
      mode 名带 _seal 后缀,与引擎对全日一字板的处理同一约定:方向无关)。修复前的 14 个模式(无后缀)结果保留作前后对照。
  P1  "Δ同窗超额"与"总收益差"改对**同价格口径**的 W1 参照:VWAP 模式对 W1_vwap,TWAP 模式(W1_twap_lag2)对 W1_twap,
      单价窗口 W0 / W8 对 W1_vwap(与第一层基线一致);W1 VWAP−TWAP 的口径差单列(conv_gap),不计入任何方案。
      同时保留对 W1_twap 的旧口径列(*_vs_w1twap)。

用法:
  python3 scripts/inc_t1_engine_modes.py checks [old]      # §3.3 第 4 条 -> research/ledger/inc_t1_engine_checks.csv(默认跑 _seal 模式并追加)
  python3 scripts/inc_t1_engine_modes.py run <strategy> [old]   # 第二层网格(一个策略)-> cache/_tmp/inc_t1_l2_<strategy>.parquet(默认 _seal)
  python3 scripts/inc_t1_engine_modes.py summarize        # -> research/ledger/inc_t1_layer2_grid.csv
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t1_common as C                                   # noqa: E402
from inc_t1_common import ST, IM, Panel, LEDGER, TMP, TRADING_DAYS   # noqa: E402

GOLD = os.path.join(os.path.dirname(__file__), "..", "tests", "golden_engine.json")

MODES = [("W1", "twap", 1), ("W1", "vwap", 1)] + [(w, "vwap", 1) for w in ["W2", "W3", "W4", "W5", "W6", "W7", "W9", "W10", "W11"]] \
    + [("W0", "twap", 1), ("W8", "twap", 1), ("W1", "twap", 2)]
REF_MODE = ("W1", "twap", 1)
Y_SENS_MODES = [("W1", "twap", 1), ("W1", "vwap", 1), ("W9", "vwap", 1), ("W5", "vwap", 1), ("W7", "vwap", 1)]
Y_MAIN = 1.0
Y_SENS = [0.5, 1.5]


def _mode(p, spec, seal: bool = False):
    w, kind, lag = spec
    return C.register_mode(p, w, kind, lag, seal)


# ====================================================================== 断言
def cmd_checks(seal: bool = True):
    p = Panel()
    rows = []

    def add(mode, check, value, lo, hi, passed, note=""):
        rows.append(dict(mode=mode, check=check, value=value, lo=lo, hi=hi, passed=bool(passed), note=note))
        print(rows[-1], flush=True)

    ones = pd.DataFrame(1.0, index=p.dates, columns=p.codes, dtype=np.float32)
    comp = pd.read_parquet(TMP / "comp_v5_score_open10.parquet")
    gold = json.load(open(GOLD))
    windows = {"2021-2025": lambda ix: ix.year >= 2021, "2021-2023": lambda ix: (ix.year >= 2021) & (ix.year <= 2023),
               "2024-2025": lambda ix: ix.year >= 2024}
    for y in range(2021, 2026):
        windows[str(y)] = (lambda yy: (lambda ix: ix.year == yy))(y)
    for spec in MODES:
        t0 = time.time()
        m = _mode(p, spec, seal)
        # (1) 全池等权 r5 vs 同频基准:残差 = mean(s - b + cost) * 242(连续路径,无 mask,旧成本)
        s = ST.run_strategy(ones, p, "V1000", m, scheme="exclude", exclude_frac=0.0, rebalance=5, record=True)
        b = ST.benchmark(p, "V1000", m, rebalance=5)
        assert s["rets"].index.equals(b["rets"].index)
        d = s["rets"] - b["rets"] + s["record"]["daily"]["cost_total"]
        for wn, f in windows.items():
            r = float(d[f(d.index)].mean() * TRADING_DAYS * 100)
            add(m, f"fullpool_r5_vs_samefreq_resid_pp[{wn}]", r, -0.5, 1.5, -0.5 < r < 1.5,
                "常数打分全池等权 5 日调仓 vs benchmark(rebalance=5);1000 万、旧成本、连续路径")
        # (2) exposure <= 1(引擎循环内硬断言 + 相位明细复核)
        ex_max = float(s["record"]["daily_by_phase"]["exposure"].max())
        add(m, "exposure_max_fullpool", ex_max, 0, 1 + 1e-9, ex_max <= 1 + 1e-9)
        # (3) 对齐峰值 k=0:全池常数打分与 COMP_v5 排雷,对同窗同频基准
        cs = {k: s["rets"].corr(b["rets"].shift(k)) for k in (-2, -1, 0, 1, 2)}
        kmax = max(cs, key=cs.get)
        add(m, "align_peak_k_fullpool", kmax, 0, 0, kmax == 0, json.dumps({str(k): round(v, 6) for k, v in cs.items()}))
        e = ST.run_strategy(comp, p, "V1000", m, scheme="exclude", buffer=(0.20, 0.35), rebalance=5, record=True)
        cs = {k: e["rets"].corr(b["rets"].shift(k)) for k in (-2, -1, 0, 1, 2)}
        kmax = max(cs, key=cs.get)
        add(m, "align_peak_k_comp_exclude", kmax, 0, 0, kmax == 0, json.dumps({str(k): round(v, 6) for k, v in cs.items()}))
        ex_max = float(e["record"]["daily_by_phase"]["exposure"].max())
        add(m, "exposure_max_comp_exclude", ex_max, 0, 1 + 1e-9, ex_max <= 1 + 1e-9)
        # (4) 金标准:注入的 W1-TWAP 模式 vs golden a_comp_exclude_std|full(open10)
        if spec == REF_MODE and seal:
            # 封板规则开启后 W1_twap 不再逐位等于金标准(窗口内封死的股票日被判不可成交);只记录差异大小,不作门槛
            g = gold["results"]["a_comp_exclude_std|full"]
            r = e["rets"].to_numpy(np.float64); gr = np.asarray(g["rets"], np.float64)
            assert len(r) == len(gr)
            ndiff = int((np.abs(r - gr) > 1e-9).sum())
            add(m, "info_sealed_vs_golden_n_days_differ", float(ndiff), np.nan, np.nan, True,
                f"信息性:封板规则下 W1_twap 与金标准 a_comp_exclude_std 逐日差 >1e-9 的天数 {ndiff} / {len(r)};"
                f"最大逐日差 {float(np.max(np.abs(r - gr))):.2e};年化 {e['ann_return']:.6f} vs 金标准 {g['ann_return']:.6f}")
        if spec == REF_MODE and not seal:
            g = gold["results"]["a_comp_exclude_std|full"]
            r = e["rets"].to_numpy(np.float64); gr = np.asarray(g["rets"], np.float64)
            assert len(r) == len(gr)
            dmax = float(np.max(np.abs(r - gr)))
            add(m, "golden_a_comp_exclude_std_full_max_abs_diff", dmax, 0, 1e-6, dmax <= 1e-6,
                f"golden ann_return {g['ann_return']:.10f} vs {e['ann_return']:.10f}")
            # 同一配置用原生 open10 模式再跑一遍,确认注入路径与原生路径的差异只来自 1 ulp 的成交额
            o = ST.run_strategy(comp, p, "V1000", "open10", scheme="exclude", buffer=(0.20, 0.35), rebalance=5)
            dmax2 = float(np.max(np.abs(o["rets"].to_numpy() - r)))
            add(m, "native_open10_vs_injected_max_abs_diff", dmax2, 0, 1e-6, dmax2 <= 1e-6)
        C.release_mode(m)
        print(f"  {m} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    f_out = LEDGER / "inc_t1_engine_checks.csv"
    if f_out.exists():                              # 追加:保留另一族(无 / 有 _seal 后缀)的既有行
        old = pd.read_csv(f_out)
        old = old[~old["mode"].isin(df["mode"].unique())]
        df_all = pd.concat([old, df], ignore_index=True)
    else:
        df_all = df
    df_all.to_csv(f_out, index=False)
    print(f"引擎断言(本次 {'_seal' if seal else '无后缀'} 族){int(df.passed.sum())}/{len(df)} 通过;账本共 {len(df_all)} 行", flush=True)
    if not df.passed.all():
        print("!!! 未通过:", df[~df.passed][["mode", "check", "value"]].to_string(), flush=True)


# ====================================================================== 第二层网格
def close_valuation(p: Panel, m: str, rec: dict, rets: pd.Series) -> pd.Series:
    """终点统一计价的修正因子 corr(t):信号日 t 的持有期结束(估值日 d = t+lag+1)时,
    把按窗口价计价的持仓改按 close_px[d] 计价的倍数 = Σ h_end_i·close[d,i]/px[d,i] + 现金。
    h_end = newh·(1+r_i)/(1+port_r)(相位平均持仓;二阶误差见报告)。返回以信号日 t 为索引的 Series。"""
    lag = ST.EXEC_MODES[m]["lag"]
    px = C._PRICES[m][0].to_numpy(np.float64)
    close = p.raw("close_px").to_numpy(np.float64)
    H = rec["holdings"]
    idx = H.index
    pos = p.dates.get_indexer(idx)
    Hm = H.to_numpy(np.float64)
    te = pos + lag
    d = te + 1
    with np.errstate(divide="ignore", invalid="ignore"):
        r = px[d] / px[te] - 1.0
        r = np.where(np.isfinite(r), r, 0.0)
        ratio = close[d] / px[d]
        ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0)
    port = rets.reindex(idx).to_numpy(np.float64)
    h_end = Hm * (1.0 + r) / (1.0 + port)[:, None]
    ex = h_end.sum(axis=1)
    corr = (h_end * ratio).sum(axis=1) + (1.0 - ex)
    return pd.Series(corr, index=idx)


def run_cell(p, W, mask, m, X, Y, s_row, cms):
    if Y not in cms:
        cms[Y] = IM.CostModel(p, impact_y=Y)
    cm = C.WinScaledConstAUMCost(cms[Y], s_row, label="pool_year")
    res = ST.run_weights(W, p, m, X, rebalance=C.REBALANCE, date_mask=mask, cost_model=cm,
                         participation_on_nav=False, record=True)
    daily = res["record"]["daily"]
    corr = close_valuation(p, m, res["record"], res["rets"])
    ex_max = float(res["record"]["daily_by_phase"]["exposure"].max())
    assert ex_max <= 1 + 1e-9
    out = pd.DataFrame(dict(ret=res["rets"], corr=corr.reindex(res["rets"].index)))
    for c in ["cost_total", "cost_commission", "cost_stamp", "cost_spread", "cost_impact", "exposure", "cash",
              "turnover", "unfilled", "desired_turnover", "part_cut", "part_bound_frac"]:
        out[c] = daily[c].reindex(out.index)
    return out, ex_max


def cmd_run(strat: str, seal: bool = True):
    p = Panel()
    scores = C.load_scores(p)
    W = C.strategy_weights(p, scores, strat)
    mask = C.sim_mask(p, strat)
    s_year = C.window_scales(p)
    cms = {}
    out_f = TMP / f"inc_t1_l2_{strat}.parquet"
    done = pd.read_parquet(out_f) if out_f.exists() else None
    have = set() if done is None else set(zip(done["mode"], done["X"], done["Y"]))
    parts = [] if done is None else [done]
    bench_parts = []
    t00 = time.time()
    for spec in MODES:
        m = _mode(p, spec, seal)
        s_row = C.s_row_for(p, s_year, spec[0])
        ys = [Y_MAIN] + (Y_SENS if spec in Y_SENS_MODES else [])
        for X in C.AUMS:
            for Y in ys:
                if (m, X, Y) in have:
                    continue
                t0 = time.time()
                df, ex_max = run_cell(p, W, mask, m, X, Y, s_row, cms)
                df = df.reset_index().rename(columns={"index": "date"})
                df["mode"], df["X"], df["Y"], df["strategy"] = m, X, Y, strat
                parts.append(df)
                pd.concat(parts, ignore_index=True).to_parquet(out_f)
                print(f"[{strat}] {m} X={X:.0e} Y={Y} 年化 {((1 + df.ret).prod() ** (TRADING_DAYS / len(df)) - 1) * 100:.2f}% "
                      f"exp_max {ex_max:.6f}  {time.time() - t0:.0f}s / 累计 {time.time() - t00:.0f}s", flush=True)
        bf = TMP / f"inc_t1_l2_bench_{C.mode_name(*spec)}.parquet"      # 基准只用价格、不加可成交约束:_seal 族共用无后缀的基准
        if not bf.exists():
            b = ST.benchmark(p, "V1000", m, rebalance=C.REBALANCE)["rets"].rename("bench")
            b.to_frame().to_parquet(bf)
        C.release_mode(m)
    print("run done", flush=True)


def _slice_total(df: pd.DataFrame, lag: int, y0: int, y1: int) -> tuple[float, int]:
    """估值日对齐的总收益(收盘统一计价):lag=2 的信号日整体前移一日,使估值日历与 lag=1 相同。"""
    ix = df.index
    shift = lag - 1
    sel_dates = ix[(ix.year >= y0) & (ix.year <= y1)]
    pos = np.flatnonzero(np.isin(ix, sel_dates))
    pos = pos - shift
    pos = pos[pos >= 0]
    a, b = pos[0], pos[-1]
    nav = (1.0 + df["ret"]).cumprod()
    v_end = nav.iloc[b] * df["corr"].iloc[b]
    v_start = (nav.iloc[a - 1] * df["corr"].iloc[a - 1]) if a >= 1 else 1.0
    return float(v_end / v_start), int(b - a + 1)


def ref_modes_for(m: str) -> dict:
    """mode -> 三个参照:ref(同价格口径,判定用)/ w1twap(旧口径)/ w1vwap,均取 m 所在的 seal 族。
    同价格口径:VWAP 模式对 W1_vwap;TWAP 模式(W1_twap、W1_twap_lag2)对 W1_twap;单价窗口 W0 / W8 对 W1_vwap(与第一层基线一致)。"""
    d = C.parse_mode(m)
    sl = bool(d["seal"])
    tw, vw = C.mode_name("W1", "twap", 1, sl), C.mode_name("W1", "vwap", 1, sl)
    if d["lag"] == 2:
        ref = tw
    elif d["window"] in ("W0", "W8") or d["kind"] == "vwap":
        ref = vw
    else:
        ref = tw
    return dict(ref=ref, w1twap=tw, w1vwap=vw)


def cmd_summarize():
    p = Panel()
    rows = []
    for strat in C.STRAT_KEYS:
        f = TMP / f"inc_t1_l2_{strat}.parquet"
        if not f.exists():
            print(f"缺 {f}", flush=True)
            continue
        D = pd.read_parquet(f)
        D["date"] = pd.to_datetime(D["date"])
        ref = {}
        for (m, X, Y), g in D.groupby(["mode", "X", "Y"]):
            g = g.set_index("date").sort_index()
            ref[(m, X, Y)] = g

        def get_ref(rm, X, Y):
            return ref.get((rm, X, Y)) if (rm, X, Y) in ref else ref.get((rm, X, 1.0))

        for (m, X, Y), g in ref.items():
            md = C.parse_mode(m)
            lag = md["lag"]
            rms = ref_modes_for(m)
            if get_ref(rms["ref"], X, Y) is None or get_ref(rms["w1twap"], X, Y) is None or get_ref(rms["w1vwap"], X, Y) is None:
                continue
            unsealed = ref.get((m[:-5], X, Y)) if md["seal"] else None
            b = pd.read_parquet(TMP / f"inc_t1_l2_bench_{C.mode_name(md['window'], md['kind'], lag)}.parquet")["bench"]
            for pname, (y0, y1, role) in C.PERIODS[strat].items():
                sel = (g.index.year >= y0) & (g.index.year <= y1)
                gs = g[sel]
                n = len(gs)
                yrs = n / TRADING_DAYS
                ann = float((1 + gs.ret).prod() ** (1 / yrs) - 1)
                bs = b.reindex(gs.index)
                bann = float((1 + bs).prod() ** (1 / yrs) - 1)
                es = ST.excess_stats(gs.ret, bs)
                R, nn = _slice_total(g, lag, y0, y1)
                tot_ann = R ** (TRADING_DAYS / nn) - 1
                row = dict(strategy=strat, mode=m, window=md["window"], kind=md["kind"], lag=lag, sealed=md["seal"],
                           X=X, Y=Y, period=pname, role=role, n_days=n,
                           ann_return=ann, bench_samewin_ann=bann, excess_geo_pp=(ann - bann) * 100,
                           excess_arith_pp=es.get("excess_arith", np.nan) * 100, excess_t_nw=es.get("t_nw", np.nan),
                           ir_adj=es.get("ir_adj", np.nan), te_adj=es.get("te_adj", np.nan),
                           total_close_ann=tot_ann, ref_mode=rms["ref"])

                # 对三个参照的总收益差(收盘统一计价)与 Δ同窗超额(几何超额差)、日收益差 NW t
                def close_rets(x, lg):
                    nav = (1 + x.ret).cumprod() * x["corr"]
                    v = nav / nav.shift(1) - 1
                    v.index = x.index
                    return v.shift(lg - 1) if lg > 1 else v         # 按估值日对齐:lag2 的 t-1 行与 lag1 的 t 行同一估值日
                for tag, rm in (("ref", rms["ref"]), ("w1twap", rms["w1twap"]), ("w1vwap", rms["w1vwap"])):
                    r0 = get_ref(rm, X, Y)
                    R0, nn0 = _slice_total(r0, 1, y0, y1)
                    tot0_ann = R0 ** (TRADING_DAYS / nn0) - 1
                    s0 = r0[(r0.index.year >= y0) & (r0.index.year <= y1)]
                    b0 = pd.read_parquet(TMP / f"inc_t1_l2_bench_{C.mode_name(*ref_spec(rm)[:3])}.parquet")["bench"].reindex(s0.index)
                    ann0 = float((1 + s0.ret).prod() ** (TRADING_DAYS / len(s0)) - 1)
                    bann0 = float((1 + b0).prod() ** (TRADING_DAYS / len(s0)) - 1)
                    dv = (close_rets(g, lag) - close_rets(r0, 1))
                    dv = dv[(dv.index.year >= y0) & (dv.index.year <= y1)].dropna()
                    sfx = "" if tag == "ref" else f"_vs_{tag}"
                    row[f"total_close_ann_ref{'' if tag == 'ref' else '_' + tag}"] = tot0_ann
                    row[f"total_diff{sfx if tag != 'ref' else '_vs_ref'}_pp"] = (tot_ann - tot0_ann) * 100
                    row[f"d_excess{sfx if tag != 'ref' else '_vs_ref'}_pp"] = ((ann - bann) - (ann0 - bann0)) * 100
                    row[f"daily_close_diff{sfx if tag != 'ref' else '_vs_ref'}_arith_pp"] = float(dv.mean() * TRADING_DAYS * 100) if len(dv) else np.nan
                    row[f"daily_close_diff{sfx if tag != 'ref' else '_vs_ref'}_t_nw"] = C.nw_t(dv.to_numpy()) if len(dv) else np.nan
                if unsealed is not None:
                    Ru, nnu = _slice_total(unsealed, lag, y0, y1)
                    row["total_diff_vs_unsealed_same_mode_pp"] = (tot_ann - (Ru ** (TRADING_DAYS / nnu) - 1)) * 100
                row.update(dict(
                    cost_total_pp=gs.cost_total.mean() * TRADING_DAYS * 100,
                    cost_spread_pp=gs.cost_spread.mean() * TRADING_DAYS * 100,
                    cost_impact_pp=gs.cost_impact.mean() * TRADING_DAYS * 100,
                    turnover=gs.turnover.mean(), unfilled_mean=gs.unfilled.mean(),
                    part_bound_frac_mean=gs.part_bound_frac.mean(), cash_mean=gs.cash.mean(),
                    exposure_mean=gs.exposure.mean()))
                rows.append(row)
    df = pd.DataFrame(rows)
    # 兼容旧列名(判定脚本 / 旧账本使用):total_diff_vs_ref_pp 现在是同价格口径参照
    df.to_csv(LEDGER / "inc_t1_layer2_grid.csv", index=False)
    show = df[(df.Y == 1.0) & df.sealed]
    print(show.pivot_table(index=["strategy", "X", "mode"], columns="period", values="total_diff_vs_ref_pp").round(2).to_string(), flush=True)


def ref_spec(m: str) -> tuple:
    d = C.parse_mode(m)
    return (d["window"], d["kind"], d["lag"], d["seal"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "checks":
        cmd_checks()
    elif cmd == "run":
        cmd_run(sys.argv[2])
    elif cmd == "summarize":
        cmd_summarize()
    else:
        raise SystemExit(cmd)
