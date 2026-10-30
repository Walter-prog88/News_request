"""lh_basis 第 2 步:解析 -> 到期日核验 -> 基差/年化贴水 -> 对冲(展期)日序列 -> 汇总表 + 图。

只用 <= 2025-12-31。不做任何策略回测(只有期货、指数数据)。
输出:
  cache/_tmp/lh_basis_contracts.parquet   合约日面板(含 basis / basis_ann / td_left)
  cache/_tmp/lh_basis_expiry.parquet      合约 -> 最后交易日(及规则核验)
  cache/_tmp/lh_basis_spot.parquet        价格指数 + 全收益指数收盘
  cache/_tmp/lh_basis_hedge.parquet       对冲日序列长表(variety × rule × roll_days)  <- 下游接口
  research/ledger/lh_basis_summary.csv    按年 / 窗口 / 极端时期汇总(主口径 R=2)
  research/ledger/lh_basis_roll_sensitivity.csv  换月阈值 R=0/2/5 全样本对比
  research/ledger/lh_basis_qa.csv         数据质量与口径核验
  research/ledger/lh_basis_drawdown_cond.csv     按现货回撤深度分档的贴水(描述性)
  research/ledger/lh_basis_index_years.csv       指数年收益与股息率(风格错配参考)
  research/ledger/lh_basis_loo.csv               2016 起已实现成本留一年检验
  reports/lh_basis_ann.png / reports/lh_basis_cost_yearly.png
用法: python3 scripts/lh_basis_build.py
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402

from afactor import lh_basis as LB                     # noqa: E402
from afactor.config import LEDGER, REPORTS             # noqa: E402

LOG = ROOT / "logs" / "lh_basis_build.log"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def nw_t(x, q=20):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    s = float((e ** 2).sum() / n)
    for L in range(1, q + 1):
        s += 2.0 * (1.0 - L / (q + 1.0)) * float((e[L:] * e[:-L]).sum() / n)
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def _hedge_job(args):
    v, rule, R = args
    c = pd.read_parquet(LB.P_CONTRACTS)
    spot = pd.read_parquet(LB.P_SPOT)
    return LB.build_hedge(c, spot, v, rule, R).reset_index()


# --------------------------------------------------------------------------- QA
def qa(panel, expiry, c, spot):
    rows = []
    cal_f = pd.DatetimeIndex(sorted(panel["date"].unique()))
    for v, code in LB.VARIETIES.items():
        d = panel[panel["variety"] == v]
        dates = pd.DatetimeIndex(sorted(d["date"].unique()))
        sp = spot[code].dropna()
        sp = sp[(sp.index >= dates.min()) & (sp.index <= dates.max())]
        nper = d.groupby("date")["symbol"].nunique()
        e = expiry[expiry["variety"] == v]
        ex = e[e["source"] == "observed_last_day"]
        cv = c[(c["variety"] == v) & (c["date"] == c["expiry"])]
        rows += [
            (v, "首个交易日", str(dates.min().date())),
            (v, "末个交易日", str(dates.max().date())),
            (v, "期货交易日数", len(dates)),
            (v, "期货有/现货无的日数", int(len(dates.difference(sp.index)))),
            (v, "现货有/期货无的日数", int(len(sp.index.difference(dates)))),
            (v, "每日在市合约数(众数)", int(nper.mode().iloc[0])),
            (v, "每日在市合约数≠4的日数", int((nper != 4).sum())),
            (v, "合约数", len(e)),
            (v, "已到期合约: 观测最后交易日=第三周五顺延规则", f"{int(ex['rule_match'].sum())}/{len(ex)}"),
            (v, "已到期合约: 不符规则的合约", ";".join(f"{s}:{r.expiry.date()}vs{r.rule.date()}"
                                             for s, r in ex[~ex["rule_match"]].iterrows()) or "无"),
            (v, "未到期合约最后交易日来源", ";".join(f"{s}:{r.expiry.date()}({r.source})"
                                          for s, r in e[e["source"] != "observed_last_day"].iterrows())),
            (v, "到期日收盘基差率|F/S-1|中位数(收敛检验)", round(float(cv["basis_pct"].abs().median()), 6)),
            (v, "到期日收盘基差率|F/S-1|最大值", round(float(cv["basis_pct"].abs().max()), 6)),
            (v, "收盘价缺失的合约日", int(d["close"].isna().sum())),
            (v, "收盘价<=0 的合约日", int((d["close"] <= 0).sum())),
        ]
        # 收盘口径 vs 结算价口径的年化基差(2016 前期货 15:15 收盘、现货 15:00 收盘)
        cc = c[(c["variety"] == v) & (c["dte_cal"] >= 7)]
        for lab, m in [("2015及以前", cc["date"] < "2016-01-01"), ("2016及以后", cc["date"] >= "2016-01-01")]:
            if m.any():
                diff = (cc.loc[m, "basis_ann"] - cc.loc[m, "basis_settle_ann"])
                rows.append((v, f"年化基差 收盘口径-结算口径 中位数({lab};剩余>=7天)", round(float(diff.median()), 5)))
                rows.append((v, f"年化基差 收盘口径-结算口径 |差|90分位({lab};剩余>=7天)",
                             round(float(diff.abs().quantile(0.9)), 5)))
    for h in LB.TR_CODE.values():
        s = spot[h].dropna() if h in spot.columns else pd.Series(dtype=float)
        rows.append((h, "全收益指数覆盖", f"{s.index.min().date()}..{s.index.max().date()} n={len(s)}"
                     if len(s) else "缺失"))
    rows.append(("ALL", "期货日历(所有品种并集)", f"{cal_f.min().date()}..{cal_f.max().date()} n={len(cal_f)}"))
    return pd.DataFrame(rows, columns=["variety", "item", "value"])


# --------------------------------------------------------------------------- 汇总
def summarize(h: pd.DataFrame, label_type: str, label: str) -> dict:
    b = h["basis_ann_held"].dropna()
    hc = h["hedge_cost"].dropna()
    n = int(h["fut_ret"].notna().sum())
    out = {
        "period_type": label_type, "period": label, "n_days": n,
        "start": str(h.index.min().date()), "end": str(h.index.max().date()),
        "basis_ann_mean": b.mean(), "basis_ann_median": b.median(),
        "basis_ann_p05": b.quantile(0.05), "basis_ann_p25": b.quantile(0.25),
        "basis_ann_p75": b.quantile(0.75), "basis_ann_p95": b.quantile(0.95),
        "basis_ann_min": b.min(), "basis_ann_max": b.max(),
        # 剩余 < 15 自然日时年化会把小基差放大成噪音,另报一个只用剩余 >= 15 天的均值/中位数
        "basis_ann_mean_dte15": h.loc[h["dte_held"] >= 15, "basis_ann_held"].mean(),
        "basis_ann_median_dte15": h.loc[h["dte_held"] >= 15, "basis_ann_held"].median(),
        "basis_pct_mean": (h["basis_ann_held"] * h["dte_held"] / 365).mean(),
        "dte_held_mean": h["dte_held"].mean(),
        "realized_cost_ann": hc.mean() * LB.ANN,
        "realized_cost_sum": hc.sum(),
        "realized_cost_nw_t": nw_t(hc.values, 20),
        "roll_tc_ann": h["roll_tc"].mean() * LB.ANN,
        "n_rolls": int(h["roll"].sum()),
        "div_ann": h["div_ret"].mean() * LB.ANN if h["div_ret"].notna().any() else np.nan,
        "realized_cost_tr_ann": h["hedge_cost_tr"].mean() * LB.ANN if h["hedge_cost_tr"].notna().any() else np.nan,
        "rf_ann": h["rf_ret"].mean() * LB.ANN if h["rf_ret"].notna().any() else np.nan,
        "cost_vs_cash_ann": h["hedge_cost_vs_cash"].mean() * LB.ANN if h["hedge_cost_vs_cash"].notna().any() else np.nan,
        "short_ret_ann": h["short_ret"].mean() * LB.ANN,
        "spot_ret_sum": h["spot_ret"].sum(),
        "div_sum": h["div_ret"].sum() if h["div_ret"].notna().any() else np.nan,
        "spot_ret_ann": h["spot_ret"].mean() * LB.ANN,
    }
    out["realized_net_cost_ann"] = out["realized_cost_ann"] + out["roll_tc_ann"]
    return out


def build_summary(H: pd.DataFrame, R: int) -> pd.DataFrame:
    rows = []
    sub = H[H["roll_days"] == R]
    for (v, rule), g in sub.groupby(["variety", "rule"]):
        g = g.set_index("date").sort_index()
        g = g.iloc[1:]                                  # 首日无收益
        for y, gy in g.groupby(g.index.year):
            rows.append({"variety": v, "rule": rule, "roll_days": R, **summarize(gy, "year", str(y))})
        windows = [("全样本", None, None), ("开发窗2010-2014", "2010-01-01", "2014-12-31"),
                   ("检验窗2015-2025", "2015-01-01", "2025-12-31"), ("2020-2025(研究期)", "2020-01-01", "2025-12-31"),
                   ("2016起(15:00同步收盘)", "2016-01-01", "2025-12-31")]
        for name, a, b in windows:
            gw = g if a is None else g[(g.index >= a) & (g.index <= b)]
            if len(gw) >= 20:
                rows.append({"variety": v, "rule": rule, "roll_days": R, **summarize(gw, "window", name)})
        for name, a, b in LB.REGIMES:
            gw = g[(g.index >= a) & (g.index <= b)]
            if len(gw) >= 5:
                rows.append({"variety": v, "rule": rule, "roll_days": R, **summarize(gw, "regime", name)})
    return pd.DataFrame(rows)


def drawdown_cond(H: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    """描述性:现货距 242 日高点的回撤(T 日已知)分档下,持有合约的年化贴水与随后 20 日已实现成本。"""
    rows = []
    sub = H[(H["roll_days"] == LB.DEFAULT_ROLL) & H["rule"].isin(["m2", "q1"])]
    for (v, rule), g in sub.groupby(["variety", "rule"]):
        g = g.set_index("date").sort_index()
        s = spot[LB.VARIETIES[v]].reindex(g.index)
        dd = s / s.rolling(242, min_periods=60).max() - 1
        fwd = g["hedge_cost"].rolling(20).sum().shift(-20) * LB.ANN / 20
        bins = pd.cut(dd, [-1, -0.30, -0.20, -0.10, -0.05, 1e-9],
                      labels=["<-30%", "-30~-20%", "-20~-10%", "-10~-5%", "-5~0%"])
        for lab, gg in g.groupby(bins, observed=True):
            rows.append({"variety": v, "rule": rule, "spot_dd_bucket": str(lab), "n_days": len(gg),
                         "basis_ann_mean": gg["basis_ann_held"].mean(),
                         "basis_ann_median": gg["basis_ann_held"].median(),
                         "fwd20_realized_cost_ann_mean": fwd.reindex(gg.index).mean(),
                         "years": ";".join(f"{y}:{n}" for y, n in gg.index.year.value_counts().sort_index().items())})
    return pd.DataFrame(rows)


def index_years(spot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, g in spot[spot.index >= "2005-01-01"].groupby(spot[spot.index >= "2005-01-01"].index.year):
        r = {"year": y}
        for code, h in LB.TR_CODE.items():
            p = spot[code].dropna()
            p = p[p.index.year <= y]
            py = p[p.index.year == y]
            prev = p[p.index.year < y]
            base = prev.iloc[-1] if len(prev) else py.iloc[0]
            r[f"ret_{code}"] = py.iloc[-1] / base - 1 if len(py) else np.nan
            if h in spot.columns and spot[h].notna().any():
                t = spot[h].dropna()
                ty = t[t.index.year == y]
                tprev = t[t.index.year < y]
                if len(ty) and len(tprev):
                    r[f"tr_{h}"] = ty.iloc[-1] / tprev.iloc[-1] - 1
                    r[f"div_{code}"] = r[f"tr_{h}"] - r[f"ret_{code}"]
        r["ret_852_minus_300"] = r.get("ret_000852", np.nan) - r.get("ret_000300", np.nan)
        r["ret_905_minus_300"] = r.get("ret_000905", np.nan) - r.get("ret_000300", np.nan)
        r["ret_852_minus_905"] = r.get("ret_000852", np.nan) - r.get("ret_000905", np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def leave_one_year_out(H: pd.DataFrame) -> pd.DataFrame:
    """主口径(q1/m2, R=2)2016 起已实现年化成本的留一年检验:逐年剔除后重算日均 × 242。"""
    rows = []
    sub = H[(H["roll_days"] == LB.DEFAULT_ROLL) & H["rule"].isin(["m2", "q1"])]
    for (v, rule), g in sub.groupby(["variety", "rule"]):
        g = g.set_index("date").sort_index()
        g = g[g.index >= "2016-01-01"]
        c = (g["hedge_cost"] + g["roll_tc"]).dropna()
        full = c.mean() * LB.ANN
        for y in sorted(set(c.index.year)):
            rest = c[c.index.year != y]
            rows.append({"variety": v, "rule": rule, "dropped_year": y, "full_ann": full,
                         "loo_ann": rest.mean() * LB.ANN, "loo_nw_t": nw_t(rest.values, 20)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 图
def plots(H: pd.DataFrame, summ: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    col = {"IF": "#2a78d6", "IC": "#eb6834", "IM": "#1baf7a", "IH": "#eda100"}   # dataviz 参考调色板槽 1-4
    shade = [("2015股灾", "2015-06-15", "2015-09-06"), ("2015限仓期(严)", "2015-09-07", "2017-02-16"),
             ("2018熊市", "2018-01-29", "2019-01-03"), ("2024初小微盘危机", "2024-01-02", "2024-02-07")]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, rule, ttl in [(axes[0], "m2", "次月合约(m2)年化基差率,20 日均值"),
                          (axes[1], "q1", "当季合约(q1, R=2)年化基差率,只用剩余 >= 15 自然日的日子,20 日均值")]:
        for v in ["IF", "IC", "IM", "IH"]:
            g = H[(H["variety"] == v) & (H["rule"] == rule) & (H["roll_days"] == LB.DEFAULT_ROLL)]
            g = g.set_index("date")
            x = g["basis_ann_held"].where(g["dte_held"] >= 15) if rule == "q1" else g["basis_ann_held"]
            s = x.rolling(20, min_periods=5).mean()
            ax.plot(s.index, s.values * 100, color=col[v], lw=1.1, label=v)
        for _, a, b in shade:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#999", alpha=0.15, lw=0)
        ax.axhline(0, color="#333", lw=0.6)
        ax.set_ylabel("年化 %(负=贴水)")
        ax.set_title(ttl, fontsize=11, loc="left")
        ax.grid(alpha=0.25)
        ax.legend(ncol=4, fontsize=9, frameon=False, loc="lower left")
    axes[1].set_xlabel("灰色阴影:2015 股灾 / 2015-09~2017-02 严格限仓 / 2018 熊市 / 2024 初小微盘危机")
    fig.tight_layout()
    fig.savefig(REPORTS / "lh_basis_ann.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    s = summ[(summ["period_type"] == "year") & (summ["rule"] == "q1")]
    years = sorted(s["period"].astype(int).unique())
    w = 0.2
    for i, v in enumerate(["IF", "IC", "IM", "IH"]):
        g = s[s["variety"] == v].set_index(s[s["variety"] == v]["period"].astype(int))
        ax.bar(np.array(years) + (i - 1.5) * w, [g["realized_net_cost_ann"].get(y, np.nan) * 100 for y in years],
               width=w, color=col[v], label=v)
    ax.axhline(0, color="#333", lw=0.6)
    ax.set_xticks(years)
    ax.set_ylabel("年化 %(正=成本)")
    ax.set_title("空头当季合约(q1, R=2)已实现年化对冲成本(相对价格指数,含 1bp/边换月成本)", fontsize=11, loc="left")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(REPORTS / "lh_basis_cost_yearly.png", dpi=130)
    plt.close(fig)


def main():
    t0 = time.time()
    log("parse raw")
    panel = LB.parse_raw()
    log(f"panel rows={len(panel)} contracts={panel['symbol'].nunique()} "
        f"{panel['date'].min().date()}..{panel['date'].max().date()}")
    live = LB.load_live_specs()
    expiry = LB.build_expiry(panel, live)
    expiry.to_parquet(LB.P_EXPIRY)
    if LB.P_SPOT.exists():
        LB.P_SPOT.unlink()
    spot = LB.load_spot()
    c = LB.build_contracts(panel, expiry, spot)
    c.to_parquet(LB.P_CONTRACTS)
    log(f"contracts panel saved rows={len(c)}")

    qa_df = qa(panel, expiry, c, spot)

    jobs = [(v, r, R) for v in LB.VARIETIES for r in LB.RULES for R in LB.ROLL_DAYS]
    with ProcessPoolExecutor(6) as ex:
        parts = list(ex.map(_hedge_job, jobs))
    H = pd.concat(parts, ignore_index=True)
    H["held"] = H["held"].astype("string")
    H["held_prev"] = H["held_prev"].astype("string")
    H.to_parquet(LB.P_HEDGE, index=False)
    log(f"hedge long table rows={len(H)} saved {LB.P_HEDGE}")
    # 换月次数核验:整年内 m1/m2 应为 12 次,q1/q2 应为 4 次
    extra = []
    for (v, rule, R), g in H.groupby(["variety", "rule", "roll_days"]):
        g = g[pd.to_datetime(g["date"]).dt.year.between(2011 if v == "IF" else (2023 if v == "IM" else 2016), 2025)]
        cnt = g.groupby(pd.to_datetime(g["date"]).dt.year)["roll"].sum()
        extra.append((v, f"整年换月次数 {rule} R={R} (最小/最大)", f"{int(cnt.min())}/{int(cnt.max())}"))
        extra.append((v, f"持有合约缺失日数 {rule} R={R}", int(g["held"].isna().sum())))
    qa_df = pd.concat([qa_df, pd.DataFrame(extra, columns=qa_df.columns)], ignore_index=True)
    qa_df.to_csv(LEDGER / "lh_basis_qa.csv", index=False)
    log("qa saved")

    summ = build_summary(H, LB.DEFAULT_ROLL)
    summ.to_csv(LEDGER / "lh_basis_summary.csv", index=False, float_format="%.6g")
    sens = pd.concat([build_summary(H, R) for R in LB.ROLL_DAYS])
    sens = sens[sens["period_type"] == "window"][
        ["variety", "rule", "roll_days", "period", "n_days", "basis_ann_mean", "basis_ann_median",
         "dte_held_mean", "realized_cost_ann", "roll_tc_ann", "realized_net_cost_ann", "n_rolls",
         "realized_cost_tr_ann", "cost_vs_cash_ann"]]
    sens.to_csv(LEDGER / "lh_basis_roll_sensitivity.csv", index=False, float_format="%.6g")
    drawdown_cond(H, spot).to_csv(LEDGER / "lh_basis_drawdown_cond.csv", index=False, float_format="%.6g")
    index_years(spot).to_csv(LEDGER / "lh_basis_index_years.csv", index=False, float_format="%.6g")
    leave_one_year_out(H).to_csv(LEDGER / "lh_basis_loo.csv", index=False, float_format="%.6g")
    log("summaries saved")
    plots(H, summ)
    log(f"plots saved; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
