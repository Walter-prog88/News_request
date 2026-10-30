"""INC_T2 判定表与叠加层图(只读账本,不重算收益)。

按预注册 §8 逐条核对:主格(中证1000 官方 ADD_pure)、次级格(中证500 / 沪深300 官方,ADD_800 代 ADD_pure)、
调出腿 H3 / OV-B、信息腿 H4、反转腿 H6;另给 PA1000(代理预测替身)的同口径数字,只报不判。
输出:research/ledger/inc_t2_verdict.csv、reports/inc_t2_overlay.png
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from afactor.config import LEDGER, REPORTS

G = pd.read_csv(LEDGER / "inc_t2_legs_grid.csv", dtype={"index": str})
OV = pd.read_csv(LEDGER / "inc_t2_overlay.csv")
G3 = pd.read_csv(LEDGER / "inc_t2_gate3.csv", dtype={"index": str})
BAL = pd.read_csv(LEDGER / "inc_t2_balance.csv", dtype={"index": str})
ACC = pd.read_csv(LEDGER / "inc_t2_prediction_accuracy.csv", dtype={"index": str})


def g(src, idx, d, et, leg, w, cost="Y1", cap=1e7, col="mean"):
    s = G[(G["source"] == src) & (G["index"] == idx) & (G["direction"] == d) & (G["etype"] == et) & (G["leg"] == leg)
          & (G["window"] == w) & (G["control"] == "matched") & (G["cost"] == cost)]
    if cost != "gross":
        s = s[s["capital"] == cap]
    return float(s[col].iloc[0]) if len(s) else np.nan


def ov(v, uni, x, w, col, cap=1e7):
    s = OV[(OV["variant"] == v) & (OV["universe"] == uni) & (np.isclose(OV["x"], x)) & (OV["window"] == w)
           & (OV["capital"] == cap)]
    return float(s[col].iloc[0]) if len(s) else np.nan


def main_like(src, idx, et, uni, label, judged):
    c1 = dict(dev=g(src, idx, "in", et, "L1", "dev"), test=g(src, idx, "in", et, "L1", "test"),
              t=g(src, idx, "in", et, "L1", "all", col="t_cl"), pos=g(src, idx, "in", et, "L1", "all", col="n_pos_periods"),
              n=g(src, idx, "in", et, "L1", "all", col="n_periods"))
    ok1 = c1["dev"] > 0 and c1["test"] > 0 and c1["t"] >= 2 and c1["pos"] >= 7
    r3 = G3[(G3["source"] == src) & (G3["index"] == idx) & (G3["direction"] == "in") & (G3["etype"] == et)]
    ok2 = bool(r3["gate3_pass"].iloc[0]) if len(r3) else False
    c3 = dict(dev=ov("OVA", uni, 0.10, "dev", "incr_ann_pp"), test=ov("OVA", uni, 0.10, "test", "incr_ann_pp"),
              tpl=ov("OVA", uni, 0.10, "all", "net_pl_t_nw")) if uni else dict(dev=np.nan, test=np.nan, tpl=np.nan)
    ok3 = c3["dev"] >= 0.3 and c3["test"] >= 0.3 and c3["tpl"] >= 2
    c4 = g(src, idx, "in", et, "L1", "all", cap=1e8)
    ok4 = c4 > 0
    no1 = (min(c1["dev"], c1["test"]) <= 0) and c1["t"] <= 0
    no3 = (c3["dev"] <= 0 and c3["test"] <= 0) if uni else False
    b = BAL[(BAL["source"] == src) & (BAL["index"] == idx) & (BAL["etype"] == et) & (BAL["t0_kind"] == "A")
            & (BAL["tol"] == "tol10") & (BAL["window"] == "all")]
    bal_ok = bool(b["pass"].iloc[0]) if len(b) else False
    if ok1 and ok2 and ok3:
        v = "有增量" + ("" if ok4 else "(容量受限)")
    elif no1 or no3:
        v = "无增量"
    else:
        v = "证据不足"
    if not bal_ok and v == "有增量":
        v += "(风格未完全控制)"
    return dict(object=label, judged=judged, c1_dev_net=c1["dev"], c1_test_net=c1["test"], c1_t_cl=c1["t"],
                c1_pos_periods=f"{int(c1['pos'])}/{int(c1['n'])}" if np.isfinite(c1["pos"]) else "",
                c1_pass=ok1, c2_gate3=ok2, c3_ova_dev_pp=c3["dev"], c3_ova_test_pp=c3["test"], c3_net_pl_t=c3["tpl"],
                c3_pass=ok3, c4_net_1e8=c4, c4_pass=ok4, gate2_balance=bal_ok, no_increment_rule=(no1 or no3),
                verdict=v)


rows = [dict(object="主格:中证1000 官方 ADD_pure(L1, Y=1, 1000 万)", judged=True,
             verdict="证据不足(名单)", note="Gate 0:中证1000 官方名单只在公告附件,附件下载须用户同意,本轮未下载;完整期 0 < 8")]
rows.append(main_like("O", "000905", "ADD_800", "O500", "次级:中证500 官方调入(ADD_800 代 ADD_pure)", True))
rows.append(main_like("O", "000300", "ADD_800", None, "次级:沪深300 官方调入(ADD_800 代 ADD_pure;叠加层未单独跑)", True))
rows.append(main_like("PA", "000852", "ADD_pure", "PA1000", "替身:中证1000 代理预测 P^A 调入(只报不判)", False))

# 调出腿 H3 / OV-B
for src, idx, et, uni, lab in (("O", "000905", "DEL_800", "O500", "H3/OV-B:中证500 官方调出临时剔除"),
                               ("O", "000300", "DEL_800", "Oall", "H3/OV-B:沪深300 官方调出(c1 为 300 的 L1;c3 为 300+500 合并篮子 Oall 的 OV-B)"),
                               ("PA", "000852", "DEL_pure", "PA1000", "H3/OV-B 替身:中证1000 代理调出(只报不判)")):
    dev, test = g(src, idx, "out", et, "L1", "dev", cost="gross"), g(src, idx, "out", et, "L1", "test", cost="gross")
    t = g(src, idx, "out", et, "L1", "all", cost="gross", col="t_cl")
    bdev, btest = ov("OVB", uni, 0.0, "dev", "incr_ann_pp"), ov("OVB", uni, 0.0, "test", "incr_ann_pp")
    sup = dev < 0 and test < 0 and t <= -2 and bdev >= 0.1 and btest >= 0.1
    neg = dev >= 0 and test >= 0
    rows.append(dict(object=lab, judged=src == "O", c1_dev_net=dev, c1_test_net=test, c1_t_cl=t,
                     c3_ova_dev_pp=bdev, c3_ova_test_pp=btest, c3_net_pl_t=ov("OVB", uni, 0.0, "all", "net_pl_t_nw"),
                     verdict="支持" if sup else ("否定" if neg else "证据不足"),
                     note="c1_* 为 L1 毛超额(相对对照);c3_* 为 OV-B 年化增量 pp"))

# 信息腿 H4
for idx in ("000905", "000300", "000852"):
    a = ACC[(ACC["source"] == "PC") & (ACC["index"] == idx) & (ACC["direction"] == "in") & ACC["window"].isin(["dev", "test"])]
    rec_min = float(a["recall"].min()) if a["recall"].notna().any() else np.nan
    dev, test = g("PC", idx, "in", "ADD_pure", "L0", "dev"), g("PC", idx, "in", "ADD_pure", "L0", "test")
    t = g("PC", idx, "in", "ADD_pure", "L0", "all", col="t_cl")
    if np.isfinite(rec_min):
        sup = rec_min >= 0.7 and dev > 0 and test > 0 and t >= 2
        neg = rec_min < 0.5 or (dev <= 0 and test <= 0)
        v = "支持" if sup else ("否定" if neg else "证据不足")
    else:
        v = "证据不足(无官方名单,召回率不可算)"
    rows.append(dict(object=f"H4 信息腿:P^C {idx} 调入 L0(C+1→A,Y=1)", judged=True, c1_dev_net=dev, c1_test_net=test,
                     c1_t_cl=t, verdict=v, note=f"P^C 召回率最小值(2021H1-2025H2)={rec_min:.2f}"))

# 反转腿 H6
for src, idx, et, lab in (("O", "000905", "DEL_800", "H6 反转腿:中证500 官方调出 L3_20(Y=1)"),
                          ("O", "000300", "DEL_800", "H6 反转腿:沪深300 官方调出 L3_20(Y=1)"),
                          ("PA", "000852", "DEL_pure", "H6 替身:中证1000 代理调出 L3_20(只报不判)")):
    dev, test = g(src, idx, "out", et, "L3_20", "dev"), g(src, idx, "out", et, "L3_20", "test")
    t = g(src, idx, "out", et, "L3_20", "all", col="t_cl")
    sup = dev > 0 and test > 0 and t >= 2
    neg = dev <= 0 and test <= 0
    rows.append(dict(object=lab, judged=src == "O", c1_dev_net=dev, c1_test_net=test, c1_t_cl=t,
                     verdict="支持" if sup else ("否定" if neg else "证据不足")))
V = pd.DataFrame(rows)
V.to_csv(LEDGER / "inc_t2_verdict.csv", index=False)
pd.set_option("display.width", 250)
print(V.drop(columns=["note"]).round(4).to_string())


# ---------------------------------------------------------------- 图
def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    D = pd.read_parquet(LEDGER / "inc_t2_overlay_daily.parquet")
    D = D[(D["capital"] == 1e7) & (D.index >= "2021-01-01")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    for (v, uni, x), col, ls, lab in ((("OVA", "O500", 0.10), "#c8553d", "-", "OV-A 10% 官方500调入"),
                                     (("PLA", "O500", 0.10), "#c8553d", ":", "PL-A 10% 其匹配对照"),
                                     (("OVA", "PA1000", 0.10), "#3b6ea8", "-", "OV-A 10% 代理1000调入(P^A)"),
                                     (("PLA", "PA1000", 0.10), "#3b6ea8", ":", "PL-A 10% 其匹配对照"),
                                     (("OVB", "O500", 0.0), "#5a9a5a", "-", "OV-B 官方500调出临时剔除"),
                                     (("OVB", "PA1000", 0.0), "#8a6bb0", "-", "OV-B 代理1000调出临时剔除")):
        s = D[(D["variant"] == v) & (D["universe"] == uni) & np.isclose(D["x"], x)]["d"].sort_index()
        ax.plot(s.index, s.cumsum() * 100, color=col, ls=ls, lw=1.5, label=lab)
    ax.axhline(0, color="#999", lw=0.6)
    ax.axvline(pd.Timestamp("2024-01-01"), color="#666", lw=0.8, ls="--")
    ax.set_title("叠加层相对排雷基线 B0 的累计增量(1000 万,Y=1,已扣成本)", fontsize=11)
    ax.set_ylabel("累计增量(百分点,日差求和)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax = axes[1]
    M = pd.read_csv(LEDGER / "inc_t2_mechanism_by_period.csv", dtype={"index": str})
    for (src, idx, d, et), col, lab in ((("O", "000905", "in", "ADD_800"), "#c8553d", "官方500调入"),
                                       (("O", "000905", "out", "DEL_800"), "#5a9a5a", "官方500调出"),
                                       (("O", "000300", "in", "ADD_800"), "#d9a03f", "官方300调入(800外)"),
                                       (("PA", "000852", "in", "ADD_pure"), "#3b6ea8", "代理1000调入(P^A)")):
        s = M[(M["source"] == src) & (M["index"] == idx) & (M["direction"] == d) & (M["etype"] == et)].sort_values("review")
        ax.plot(s["review"], s["lnauc_adv_Em1_diff"], marker="o", color=col, lw=1.4, label=lab)
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_title("机制:E−1 收盘竞价成交额 / ADV20 的对数,事件 − 对照(逐期)", fontsize=11)
    ax.set_ylabel("ln 倍数差(2 ≈ 7.4 倍)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(REPORTS / "inc_t2_overlay.png", dpi=130)
    plt.close(fig)


plot()
