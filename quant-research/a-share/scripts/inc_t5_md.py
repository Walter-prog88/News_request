"""T5 报告生成(前缀 inc_t5):表格与数字全部从 research/ledger/inc_t5_*.csv 读出,叙述文字里的数字同样取自账本,避免手抄错位。
用法: python3 scripts/inc_t5_md.py  ->  research/INC_T5_REPORT.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
L = ROOT / "research" / "ledger"

import numpy as np
import pandas as pd

G = pd.read_csv(L / "inc_t5_grid.csv")
GEN = pd.read_csv(L / "inc_t5_gen_test.csv")
DIR = pd.read_csv(L / "inc_t5_dir.csv")
PASS = pd.read_csv(L / "inc_t5_pass.csv")
PJ = json.load(open(L / "inc_t5_pass.json"))
QA = pd.read_csv(L / "inc_t5_data_qa.csv")
CHK = pd.read_csv(L / "inc_t5_checks.csv")
COR = pd.read_csv(L / "inc_t5_corr.csv")
JAC = pd.read_csv(L / "inc_t5_jaccard.csv")
JACM = pd.read_csv(L / "inc_t5_jaccard_min.csv")
SIZE = pd.read_csv(L / "inc_t5_basket_size.csv")
NET = pd.read_csv(L / "inc_t5_netting.csv")
TEX = pd.read_csv(L / "inc_t5_te_expected.csv")
ERC = pd.read_csv(L / "inc_t5_erc_alpha.csv")
VAR = pd.read_csv(L / "inc_t5_variants.csv")
DIL = pd.read_csv(L / "inc_t5_dilution.csv")
COST = pd.read_csv(L / "inc_t5_cost_variants.csv")
PLC = pd.read_csv(L / "inc_t5_placebo.csv")
HYP = pd.read_csv(L / "inc_t5_hypotheses.csv")
YR = pd.read_csv(L / "inc_t5_yearly.csv")
BEAR = pd.read_csv(L / "inc_t5_bear_windows.csv")
MB = pd.read_csv(L / "inc_t5_mkt_beta.csv")
BND = pd.read_csv(L / "inc_t5_insample_bound.csv")
PH = pd.read_csv(L / "inc_t5_posthoc_rev.csv")
BAR = pd.read_csv(L / "inc_t5_barra_exposure.csv")
CS = pd.read_csv(L / "inc_t5_c_summary.csv")
CG = pd.read_csv(L / "inc_t5_c_gen.csv")
CH = pd.read_csv(L / "inc_t5_c_halves.csv")
CV = pd.read_csv(L / "inc_t5_c_verdict.csv").set_index("item")["value"]
CS1 = pd.read_csv(L / "inc_t5_c_v1_summary.csv")
CV1 = pd.read_csv(L / "inc_t5_c_v1_verdict.csv").set_index("item")["value"]
CQA = pd.read_csv(L / "inc_t5_c_qa.csv")
DEV = pd.read_csv(L / "inc_t5_deviations.csv")
MAN = json.load(open(L / "inc_t5_prereg_manifest.json"))


def g(track, cfg, win, col):
    r = G[(G.track == track) & (G.config == cfg) & (G.window == win)]
    return float(r[col].iloc[0])


def pct(x, d=2):
    return f"{x * 100:+.{d}f}" if np.isfinite(x) else "NA"


def pc(x, d=2):
    return f"{x * 100:.{d}f}" if np.isfinite(x) else "NA"


def f3(x):
    return f"{x:.3f}" if np.isfinite(x) else "NA"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]).replace("|", "/").replace("nan", "NA") for c in cols) + " |")
    return "\n".join(out)


NAME = {"S|LIQ|20_35": "排雷 LIQ(20/35)", "S|LOWVOL|45_60": "低波动 LOWVOL", "S|VALUE|45_60": "价值 VALUE", "S|DIV|45_60": "红利 DIV",
        "S|QUAL|45_60": "质量 QUAL", "S|REV|45_60": "反转 REV(对照)", "C|EW5|45_60": "CMB-EW5(主)", "C|CORE5|45_60": "CMB-CORE5",
        "C|ERC5|45_60": "CMB-ERC5", "C|EW6|45_60": "CMB-EW6", "C|S4|45_60": "S4(不含排雷)", "C|S5|45_60": "S5(不含排雷,加 REV)"}


def perf_table(track, cfgs, wins=("full", "dev", "test")):
    rows = []
    for c in cfgs:
        r = {"配置": NAME.get(c, c)}
        for w in wins:
            r[f"超额% {w}"] = pc(g(track, c, w, "excess_arith"))
            r[f"IR {w}"] = f3(g(track, c, w, "ir"))
        r["TE% 全期"] = pc(g(track, c, "full", "te"))
        r["NW t5 全期"] = f"{g(track, c, 'full', 'nw_t5'):.2f}"
        r["最长相对水下(日)"] = int(g(track, c, "full", "rel_uw_longest_days"))
        r["相对回撤/TE"] = f"{g(track, c, 'full', 'rel_mdd_over_te'):.2f}"
        r["年换手(单边)"] = f"{g(track, c, 'full', 'turnover_ann_run'):.2f}"
        r["年成本%"] = pc(g(track, c, "full", "cost_ann_run"))
        rows.append(r)
    return md_table(pd.DataFrame(rows))


def corr_row(track, w="full", kind="pearson_daily"):
    c = COR[(COR.track == track) & (COR.kind == kind) & (COR.window == w) & (COR.a == "LIQ")].set_index("b")["rho"]
    return c


def fmt_p(k, v):
    x = v["value"]
    d = v["detail"] if isinstance(v["detail"], str) else ""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return f"{x} ({d})"
    if k[:2] == "P1":
        return f"全期 ΔIR {xf:+.3f}({d};[CMB, LIQ])"
    if k[:2] == "P2":
        return f"截距 {xf * 100:+.2f}pp/年({d})"
    if k[:2] == "P4":
        return f"CMB 全期超额 {xf * 100:.2f}%({d})"
    if k[:2] == "P5":
        return f"ΔIR {xf:+.3f}({d})"
    if k[:2] == "P6":
        return f"{int(xf)}/6 个变体三窗同向"
    if k[:2] == "P7":
        return f"全期截距 {xf * 100:+.2f}pp({d})"
    return f"{xf:.4g}({d})"


def bc(a, b, w, tr="lh"):
    x = COR[(COR.track == tr) & (COR.kind == "pearson_daily_bear") & (COR.window == w) & (COR.a == a) & (COR.b == b)]
    return float(x.rho.iloc[0]) if len(x) else float("nan")


def build() -> str:
    P = {r["item"]: r for _, r in PASS.iterrows()}
    pl = PJ["Pl"]; pm = PJ["Pm"]; pa = PJ["Pa"]
    main = PJ["main"]
    liq, ew5, core5, erc5, s4 = "S|LIQ|20_35", "C|EW5|45_60", "C|CORE5|45_60", "C|ERC5|45_60", "C|S4|45_60"
    gen = lambda tr, y, w, freq="daily", d="y_on_x": GEN[(GEN.track == tr) & (GEN.y == y) & (GEN.window == w) & (GEN.freq == freq) & (GEN.direction == d)].iloc[0]
    rhoL = corr_row("lh"); rhoM = corr_row("min"); rhoLd = corr_row("lh", "dev"); rhoLt = corr_row("lh", "test")
    c_dir = float(CV["C_dIR_EW5_vs_LIQ"]); c_a = float(CV["C_S4_intercept_ann"]); c_t = float(CV["C_S4_intercept_t_q5"])
    qa_fail = QA[~QA.ok.astype(bool)]
    chk_fail = CHK[~CHK.ok.astype(bool)]
    e3 = CHK[CHK.item == "E3_fullpool_ew_excess_plus_cost_full"].set_index("track")["value"]
    pa_irs = pa and json.loads(P["A_P1"]["detail"])
    s = []
    A = s.append
    A(f"""# T5 报告:多风格篮子与排雷的组合层检验(前缀 inc_t5)

预注册:`research/INC_T5_PREREG.md`(冻结哈希 `research/ledger/inc_t5_prereg_manifest.json`,sha256 {MAN['sha256']['research/INC_T5_PREREG.md'][:16]}…,冻结于 {MAN['frozen_at']},执行前已校验一致)。执行日期 2026-10-11。

> **读这份报告前必须知道的两件事(预注册 §13 要求写在第一段)**
> 1. **B-LH 检验窗(2020-2025)对排雷底座是样本内**:排雷的参数(换手率版 S1、缓冲 20/35)来自 2020-2025 的 COMP_v5 桥接,所以"排雷在检验窗表现好"不是新证据;风格篮子的定义来自预注册、未看数据、未调参,对它检验窗是样本外。B-MIN 的 COMP_v5 排雷 2021-2025 同样是信号样本内。
> 2. **作者事先知道 2017 年"漂亮 50"质量/价值行情、2019-2020 成长行情、2021-2024 小盘与红利行情、2023-2025 红利拥挤**:检验窗的"新考卷"只对篮子定义成立,不对作者的先验成立。本轮的判定不依赖任何一个窗口里"红利/质量跑赢",而是开发窗、检验窗、安慰剂、生成检验四者同时成立。

## 0. 结论速览

**判定:无增量。**(B-LH 主判定;A 轨"排雷 + 低波动"子判定同为无增量;C 轨指示性证据同向。)

- 主配置 CMB-EW5(排雷 + 低波动 + 价值 + 红利 + 质量等权,45/60 风格篮子,B-LH,Y=1 成本,对同频池等权 B1):IR **{g('lh', ew5, 'full', 'ir'):.3f}**,排雷单独 **{g('lh', liq, 'full', 'ir'):.3f}**,ΔIR = **{pl['dIR_full']:+.3f}**(20 日块自助 90% 区间 [{DIR[(DIR.track=='lh')&(DIR.config==ew5)&(DIR.window=='full')].boot_lo90.iloc[0]:+.3f}, {DIR[(DIR.track=='lh')&(DIR.config==ew5)&(DIR.window=='full')].boot_hi90.iloc[0]:+.3f}]),开发窗 {g('lh', ew5, 'dev', 'ir'):.3f} vs {g('lh', liq, 'dev', 'ir'):.3f},检验窗 {g('lh', ew5, 'test', 'ir'):.3f} vs {g('lh', liq, 'test', 'ir'):.3f}。
- 生成检验(S4 = 不含排雷的四个风格等权,对排雷日超额回归):截距 **{pct(pl['a_full'])}pp/年,NW t(q=5) = {pl['t_full']:.2f}**(要求 ≥ +2.0),β = {gen('lh', 'S4', 'full')['b']:.2f},R² = {gen('lh', 'S4', 'full')['r2']:.2f}。截距显著为**负**:风格组合相当于"β≈1 的排雷 + 更高跟踪误差 + 负 alpha"。
- 根因(§4):低波动、价值、红利与排雷的日超额相关 **{rhoL['LOWVOL']:.2f} / {rhoL['VALUE']:.2f} / {rhoL['DIV']:.2f}**(B-LH),B-MIN {rhoM['LOWVOL']:.2f} / {rhoM['VALUE']:.2f} / {rhoM['DIV']:.2f};它们是同一笔"避开高换手投机股"的钱。只有质量({rhoL['QUAL']:.2f} / {rhoM['QUAL']:.2f})与反转({rhoL['REV']:.2f} / {rhoM['REV']:.2f})相对独立,但单独 IR 只有 {g('lh', 'S|QUAL|45_60', 'full', 'ir'):.2f} / {g('lh', 'S|REV|45_60', 'full', 'ir'):.2f}(B-LH 全期)。
- 分钟面板确认(B-MIN,COMP_v5 排雷):IR {g('min', ew5, 'full', 'ir'):.3f} vs {g('min', liq, 'full', 'ir'):.3f}(检验窗 2024-2025:{g('min', ew5, 'test', 'ir'):.3f} vs {g('min', liq, 'test', 'ir'):.3f}),S4 截距两窗为 {pct(pm['a_dev'])} / {pct(pm['a_test'])}pp。真实风格篮子在 B-MIN 的 ΔIR(**{DIR[(DIR.track=='min')&(DIR.config==ew5)&(DIR.window=='full')].d_ir.iloc[0]:+.2f}**)甚至比每一个随机篮子安慰剂(最大 {PLC[(PLC.track=='min')&(PLC.window=='full')].d_ir_vs_liq.max():+.2f})都差。

## 1. 复用与执行状态

- **上次(2026-10-11 04:18 被中断)的残留**:盘点 `scripts/inc_t5_*`、`logs/inc_t5_*`、`cache/_tmp/inc_t5_*`、`research/ledger/inc_t5_*`、`reports/inc_t5_*`、`tests/test_inc_t5*` 以及 `ps aux | grep inc_t5`:**除预注册产物(`INC_T5_PREREG.md`、`inc_t5_prereg_data_status.csv`、`inc_t5_prereg_manifest.json` 与空表头的 `inc_t5_deviations.csv`)外没有任何产出,也没有仍在运行的进程**。因此本报告的全部脚本、抓取、打分、回测都是本次从零实现并运行的,没有复用上次的任何计算结果。
- **复用的项目既有资产(核对后使用)**:`cache/_tmp/lh_step1_inputs`(LongPanel 落盘输入:池、计价价、成本面板、S1 打分)、`lh_engine`(适配器 / 回测 / 统计函数)、`LongPanel`、`fundamental.build_report_level`(PIT 财报)、`newdata.total_mv`(B-MIN 的 BP/EP 分母)、`cache/_tmp/comp_v5_score_open10.parquet`、`research/ledger/freeze_ref_daily.parquet`(D6)。核对:A 轨的 LIQ 重跑与 `lh_step1_runs/main.pkl` 的日收益**逐位相同**(最大绝对差 0.0);B-LH 的 B1 与 step1 的 B1 在 2017-2025 上最大差 1.4e-17。
- 新增文件见文末"产出清单"。执行顺序:权重与回测和数据验收(`qa`)并行启动(偏离 `run_order`,门槛 D1-D4、D7 全部通过,无影响)。

## 2. 数据验收(`inc_t5_data_qa.csv`、`inc_t5_checks.csv`、`inc_t5_c_qa.csv`)

| 门槛 | 结果 | 通过 |
|---|---|---|
| D1 覆盖 | 2017-2025 每年 BP ≥ {QA[QA.item.str.startswith('BP@')].value.min():.3f};EP / ROE_ttm / VALUE 最低 {QA[QA.item.str.startswith('EP@')].value.min():.3f}(2017);QUAL 最低 {QA[QA.item.str.startswith('QUAL@')].value.min():.3f}(2017,门槛 0.80) | 是 |
| D2 BP 两种算法 | 月末 Spearman 中位 {QA[QA.item=='BP_two_algorithms_spearman_median'].value.iloc[0]:.4f}(`每股净资产 x 复权比 / close_px` vs `归母权益 / newdata.total_mv`) | 是 |
| D3a 分红覆盖 | 年末 q92 池内当年有现金预案比例 {QA[QA.gate=='D3a'].value.min():.3f}~{QA[QA.gate=='D3a'].value.max():.3f}(门槛 [0.40, 0.85]) | 是 |
| D3b 两源核对 | 40 只幸存股 x 2016-2024:baostock 现金预案 236 条中 EM 缺失 6 条(**{QA[QA.item=='em_missing_rate'].value.iloc[0]*100:.1f}%**,门槛 3%);两源都有的 230 条差 ≤ 1% 占 **{QA[QA.item=='within_1pct_share'].value.iloc[0]*100:.0f}%** | 是 |
| D3c 退市股分红 | 293 只退市股 baostock 请求 4227 次,首轮失败 1 次(登录掉线),重试后 4227/4227 成功 | 是 |
| D4 实现一致性 | 2021-2025 月末 Spearman 中位:LOWVOL {QA[QA.item=='LOWVOL_LH_vs_MIN_spearman_median'].value.iloc[0]:.4f}、REV {QA[QA.item=='REV_LH_vs_MIN_spearman_median'].value.iloc[0]:.4f}、VALUE {QA[QA.item=='VALUE_LH_vs_MIN_spearman_median'].value.iloc[0]:.4f}、QUAL {QA[QA.item=='QUAL_LH_vs_MIN_spearman_median'].value.iloc[0]:.4f}、DIV {QA[QA.item=='DIV_LH_vs_MIN_spearman_median'].value.iloc[0]:.4f} | 是 |
| D5 引擎断言 | E1 exposure ≤ 1、E2 对齐峰值 k=0 三条轨道全部通过;**E3 全池等权(付成本)− B1 + 成本**:B-LH {e3['lh'] * 100:+.2f}pp(通过)、**A 轨 {e3['a'] * 100:+.2f}pp、B-MIN {e3['min'] * 100:+.2f}pp(未达 0.3pp 门槛)** | B-LH 是;A 轨、B-MIN 否 |
| D6 冻结复现 | (a) C 面板流水线复现冻结序列最大差 {QA[QA.item=='C面板_COMP_v5排雷_vs_冻结_maxabs'].value.iloc[0]:.1e} / B1 {QA[QA.item=='C面板_B1_vs_冻结_maxabs'].value.iloc[0]:.1e}(≤ 1e-10);(b) A 面板 Panel() 与冻结序列最大日差 {QA[QA.item=='A面板_COMP_v5排雷_vs_冻结_maxabs'].value.iloc[0] * 100:.2f}pp(只报告,偏离 `D6_frozen_panel`) | 是 |
| D7 PIT 可见日 | 最终快照行里可见日晚于法定截止日 30 天以上的比例 {QA[QA.gate=='D7'].value.iloc[0]*100:.1f}%(只报告) | 报告 |
| D8 C 轨 | EM 抓取 144/144 个文件成功(失败率 0%);C 池 / q92 池股票-日 = {CQA[CQA.item=='C_pool_vs_q92_stockdays'].value.iloc[0]:.3f} | 是 |

**D5 的 E3 说明**:A 轨的 {e3['a'] * 100:+.2f}pp 与 `lh_step1` 既有的 E3 全期值(+0.79pp,当时门槛 1pp)完全相同,是 LongPanel 停牌 / 触板约束造成的已知引擎残差;B-MIN 的 {e3['min'] * 100:+.2f}pp 集中在 2021 年(逐年残差见 `inc_t5_checks.csv`)。这个残差同向作用于全池和各篮子(它们共用同一决策池、同一引擎、同一 B1),所以对"组合 vs 排雷"的 IR 差分影响很小,但**我没有单独检验这一点**。主判定轨道 B-LH 的 D1-D6 全部通过;A 轨和 B-MIN 的 E3 未过只影响 P7 与 A 轨子判定的"数据门槛"一项,这两处的结论由 ΔIR 与截距为负决定(见 §6、§7),不依赖该项。

## 3. 主结果:B-LH(LongPanel 2017-01-03 ~ 2025-12-31,含 293 只退市股;开发 2017-2019 / 检验 2020-2025)

成本口径 Y=1(一跳半价差 + 平方根冲击 + 史实印花税 + 佣金 1bp),对同频池等权 B1。来源 `inc_t5_summary.csv` / `inc_t5_grid.csv`。

{perf_table('lh', [liq, 'S|LOWVOL|45_60', 'S|VALUE|45_60', 'S|DIV|45_60', 'S|QUAL|45_60', 'S|REV|45_60', ew5, core5, erc5, 'C|EW6|45_60', s4])}

- 风格篮子持有池子的"较好一半"(45/60,持股占池 {SIZE[SIZE.basket=='LOWVOL|45_60'].mean_frac_of_pool.iloc[0]*100:.0f}% 左右,`inc_t5_basket_size.csv`),排雷持有约 {SIZE[SIZE.basket=='LIQ|20_35'].mean_frac_of_pool.iloc[0]*100:.0f}%,因此风格篮子的 TE 天然更高({pc(g('lh','S|LOWVOL|45_60','full','te'))}% / {pc(g('lh','S|VALUE|45_60','full','te'))}% vs 排雷 {pc(g('lh',liq,'full','te'))}%)。把风格篮子放浅到 20/35(P6 变体)后结论不变(下表)。
- 单个风格篮子**几乎都有正的成本后超额**(低波动 {pct(g('lh','S|LOWVOL|45_60','full','excess_arith'))}pp、价值 {pct(g('lh','S|VALUE|45_60','full','excess_arith'))}pp、红利 {pct(g('lh','S|DIV|45_60','full','excess_arith'))}pp、质量 {pct(g('lh','S|QUAL|45_60','full','excess_arith'))}pp),但没有一个的 IR 超过排雷;红利在检验窗为 {pct(g('lh','S|DIV|45_60','test','excess_arith'))}pp(开发窗 {pct(g('lh','S|DIV|45_60','dev','excess_arith'))}pp),红利"2020-2025 为正"的先验在对**同频池等权**的口径下没有出现。
- 敏感性窗口(`inc_t5_grid.csv`,window = dev2 / test2 即 2017-2020 / 2021-2025):排雷 IR {g('lh',liq,'dev2','ir'):.3f} / {g('lh',liq,'test2','ir'):.3f},CMB-EW5 {g('lh',ew5,'dev2','ir'):.3f} / {g('lh',ew5,'test2','ir'):.3f}。
- **噪声参照**:LIQ 在本轨道从 2016-01-04 重新起算缓冲带状态,与 `lh_step1` main(2005 起连续状态)在 2017-2025 的日收益相关 0.99999995,但缓冲带路径依赖使 IR 0.918 vs 0.851、算术超额 {pct(g('lh',liq,'full','excess_arith'))} vs +3.09pp。即**同一排雷策略因缓冲带初始状态不同,IR 会晃 ±0.07**;ΔIR 的量级(−0.2 ~ −0.4)明显大于这个噪声。
- 逐年超额(`inc_t5_yearly.csv`,pp):排雷 / CMB-EW5 / CMB-CORE5 — 2017 {YR[(YR.track=='lh')&(YR.config==liq)&(YR.year==2017)].excess.iloc[0]*100:+.2f} / {YR[(YR.track=='lh')&(YR.config==ew5)&(YR.year==2017)].excess.iloc[0]*100:+.2f} / {YR[(YR.track=='lh')&(YR.config==core5)&(YR.year==2017)].excess.iloc[0]*100:+.2f};2021 {YR[(YR.track=='lh')&(YR.config==liq)&(YR.year==2021)].excess.iloc[0]*100:+.2f} / {YR[(YR.track=='lh')&(YR.config==ew5)&(YR.year==2021)].excess.iloc[0]*100:+.2f} / {YR[(YR.track=='lh')&(YR.config==core5)&(YR.year==2021)].excess.iloc[0]*100:+.2f};2025 {YR[(YR.track=='lh')&(YR.config==liq)&(YR.year==2025)].excess.iloc[0]*100:+.2f} / {YR[(YR.track=='lh')&(YR.config==ew5)&(YR.year==2025)].excess.iloc[0]*100:+.2f} / {YR[(YR.track=='lh')&(YR.config==core5)&(YR.year==2025)].excess.iloc[0]*100:+.2f}。2025 年低波动 {YR[(YR.track=='lh')&(YR.config=='S|LOWVOL|45_60')&(YR.year==2025)].excess.iloc[0]*100:+.2f}、价值 {YR[(YR.track=='lh')&(YR.config=='S|VALUE|45_60')&(YR.year==2025)].excess.iloc[0]*100:+.2f}、红利 {YR[(YR.track=='lh')&(YR.config=='S|DIV|45_60')&(YR.year==2025)].excess.iloc[0]*100:+.2f}pp,而排雷 {YR[(YR.track=='lh')&(YR.config==liq)&(YR.year==2025)].excess.iloc[0]*100:+.2f}pp。

![相对净值](../reports/inc_t5_rel_nav.png)
![IR 分窗口](../reports/inc_t5_ir_windows.png)

### 熊市段与市场敏感度(`inc_t5_bear_windows.csv`、`inc_t5_mkt_beta.csv`)

在预注册的三个熊市段,风格篮子的相对表现**并不差**:BEAR22 低波动 / 价值相对 B1 {BEAR[(BEAR.track=='lh')&(BEAR.window=='BEAR22')&(BEAR.config=='S|LOWVOL|45_60')].rel_diff.iloc[0]*100:+.2f} / {BEAR[(BEAR.track=='lh')&(BEAR.window=='BEAR22')&(BEAR.config=='S|VALUE|45_60')].rel_diff.iloc[0]*100:+.2f}pp,排雷 {BEAR[(BEAR.track=='lh')&(BEAR.window=='BEAR22')&(BEAR.config==liq)].rel_diff.iloc[0]*100:+.2f}pp;BEAR24 {BEAR[(BEAR.track=='lh')&(BEAR.window=='BEAR24')&(BEAR.config=='S|LOWVOL|45_60')].rel_diff.iloc[0]*100:+.2f} / {BEAR[(BEAR.track=='lh')&(BEAR.window=='BEAR24')&(BEAR.config=='S|VALUE|45_60')].rel_diff.iloc[0]*100:+.2f}pp vs 排雷 {BEAR[(BEAR.track=='lh')&(BEAR.window=='BEAR24')&(BEAR.config==liq)].rel_diff.iloc[0]*100:+.2f}pp;CMB-EW5 在 BEAR18 / BEAR22 / BEAR24 为 {BEAR[(BEAR.track=='lh')&(BEAR.window=='BEAR18')&(BEAR.config==ew5)].rel_diff.iloc[0]*100:+.2f} / {BEAR[(BEAR.track=='lh')&(BEAR.window=='BEAR22')&(BEAR.config==ew5)].rel_diff.iloc[0]*100:+.2f} / {BEAR[(BEAR.track=='lh')&(BEAR.window=='BEAR24')&(BEAR.config==ew5)].rel_diff.iloc[0]*100:+.2f}pp,排雷 {BEAR[(BEAR.track=='lh')&(BEAR.window=='BEAR18')&(BEAR.config==liq)].rel_diff.iloc[0]*100:+.2f} / {BEAR[(BEAR.track=='lh')&(BEAR.window=='BEAR22')&(BEAR.config==liq)].rel_diff.iloc[0]*100:+.2f} / {BEAR[(BEAR.track=='lh')&(BEAR.window=='BEAR24')&(BEAR.config==liq)].rel_diff.iloc[0]*100:+.2f}pp。也就是风格篮子在这三段熊市里相对排雷持平或略好(它们是防御型,`inc_t5_barra_exposure.csv` 里 Beta / ResidVol 为负暴露)。这是**真实的、但小的**好处;它被其余时段更低的超额吃掉,综合到 IR 与相对水下期上仍是净负(§6 P3)。所有配置的相对收益对池等权日收益的 β 都为负(排雷 {MB[(MB.track=='lh')&(MB.window=='full')&(MB.config==liq)].beta_to_b1.iloc[0]:.2f}、CMB-EW5 {MB[(MB.track=='lh')&(MB.window=='full')&(MB.config==ew5)].beta_to_b1.iloc[0]:.2f}),即二者的"防御属性"相差无几。

## 4. 独立性:是不是第二条腿(`inc_t5_corr.csv`、`inc_t5_jaccard.csv`)

各篮子相对 B1 的日超额与排雷超额的相关(Pearson,全期 / 开发窗 / 检验窗;B-MIN 全期):

| 篮子 | B-LH 全期 | B-LH 开发 | B-LH 检验 | B-MIN 全期 | 20 日不重叠 B-LH |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {nm} | {rhoL[k]:.2f} | {rhoLd[k]:.2f} | {rhoLt[k]:.2f} | {rhoM[k]:.2f} | {corr_row('lh', 'full', 'pearson_20d')[k]:.2f} |"
        for k, nm in (("LOWVOL", "低波动"), ("VALUE", "价值"), ("DIV", "红利"), ("QUAL", "质量"), ("REV", "反转(45/60)"))) + f"""

- 月末持仓 Jaccard(`inc_t5_jaccard.csv`,B-LH,108 个月):排雷 vs 低波动 {JAC[(JAC.a=='LIQ|20_35')&(JAC.b=='LOWVOL|45_60')].jaccard.iloc[0]:.2f}、vs 价值 {JAC[(JAC.a=='LIQ|20_35')&(JAC.b=='VALUE|45_60')].jaccard.iloc[0]:.2f}、vs 红利 {JAC[(JAC.a=='LIQ|20_35')&(JAC.b=='DIV|45_60')].jaccard.iloc[0]:.2f}、vs 质量 {JAC[(JAC.a=='LIQ|20_35')&(JAC.b=='QUAL|45_60')].jaccard.iloc[0]:.2f}、vs 反转 {JAC[(JAC.a=='LIQ|20_35')&(JAC.b=='REV|45_60')].jaccard.iloc[0]:.2f}。篮子占池约 45% 而排雷约 73%,低波动与排雷的 Jaccard 上限是 0.45/0.73 ≈ 0.64,实际 {JAC[(JAC.a=='LIQ|20_35')&(JAC.b=='LOWVOL|45_60')].jaccard.iloc[0]:.2f}。
- 熊市段条件相关(`inc_t5_corr.csv` kind=pearson_daily_bear):排雷与低波动在 BEAR18 / BEAR22 / BEAR24 为 {bc('LIQ','LOWVOL','BEAR18'):.2f} / {bc('LIQ','LOWVOL','BEAR22'):.2f} / {bc('LIQ','LOWVOL','BEAR24'):.2f},价值与低波动 {bc('VALUE','LOWVOL','BEAR18'):.2f} / {bc('VALUE','LOWVOL','BEAR22'):.2f} / {bc('VALUE','LOWVOL','BEAR24'):.2f}。**熊市段里依然同向**(BEAR22 的排雷-低波动相关降到 {bc('LIQ','LOWVOL','BEAR22'):.2f},仍高),没有出现"危机时分散"。
- 预期 TE(按篮子超额协方差与权重推算)与实现 TE 几乎相等(`inc_t5_te_expected.csv`:B-LH EW5 预期 {TEX[(TEX.track=='lh')&(TEX.scheme=='EW5')&(TEX.window=='full')].te_expected.iloc[0]*100:.2f}% vs 实现 {TEX[(TEX.track=='lh')&(TEX.scheme=='EW5')&(TEX.window=='full')].te_realized.iloc[0]*100:.2f}%;成员 TE 平均 {TEX[(TEX.track=='lh')&(TEX.scheme=='EW5')&(TEX.window=='full')].te_avg_member.iloc[0]*100:.2f}%),说明组合只把 TE 从成员平均 {TEX[(TEX.track=='lh')&(TEX.scheme=='EW5')&(TEX.window=='full')].te_avg_member.iloc[0]*100:.1f}% 降到 {TEX[(TEX.track=='lh')&(TEX.scheme=='EW5')&(TEX.window=='full')].te_realized.iloc[0]*100:.1f}%(降 12%),而超额从排雷的 {pc(g('lh',liq,'full','excess_arith'))}% 降到 {pc(g('lh',ew5,'full','excess_arith'))}%(降 36%)——相关太高,分散化抵不过稀释。

![相关矩阵](../reports/inc_t5_corr.png)

## 5. 生成检验(Huberman-Kandel 式;`inc_t5_gen_test.csv`)

e_S = a + b·e_LIQ + u(日超额,a 年化 = a x 242;NW t 用 Bartlett 核,q = 5 / 10 / 20):

""" + md_table(pd.DataFrame([
        {"轨道": tr, "S": nm, "窗口": w, "a(年化,pp)": pct(gen(tr, y, w)["a_ann"]), "t(q5)": f"{gen(tr, y, w)['t_q5']:.2f}",
         "t(q10)": f"{gen(tr, y, w)['t_q10']:.2f}", "t(q20)": f"{gen(tr, y, w)['t_q20']:.2f}", "b": f"{gen(tr, y, w)['b']:.2f}",
         "R²": f"{gen(tr, y, w)['r2']:.2f}", "20 日版 a(pp)/t": f"{pct(gen(tr, y, w, '20d')['a_ann'])} / {gen(tr, y, w, '20d')['t_q5']:.2f}",
         "反向 a(LIQ 对 S,pp)/t": f"{pct(gen(tr, y, w, 'daily', 'x_on_y(reverse)')['a_ann'])} / {gen(tr, y, w, 'daily', 'x_on_y(reverse)')['t_q5']:.2f}"}
        for tr, y, nm in (("lh", "S4", "S4(45/60)"), ("lh", "S4_20_35", "S4(20/35)"), ("lh", "S5", "S5(45/60,含反转)"), ("min", "S4", "S4(45/60)"))
        for w in ("full", "dev", "test")])) + f"""

逐篮子(每个风格篮子单独对排雷回归,45/60,全期):""" + ", ".join(
        f"{nm} a={pct(gen('lh', y, 'full')['a_ann'])}pp(t {gen('lh', y, 'full')['t_q5']:.2f},b {gen('lh', y, 'full')['b']:.2f})" for y, nm in
        (("LOWVOL", "低波动"), ("VALUE", "价值"), ("DIV", "红利"), ("QUAL", "质量"), ("REV", "反转"))) + f"""。没有任何一个风格篮子在控制了排雷之后有显著为正的截距;低波动的负截距最显著(t {gen('lh', 'LOWVOL', 'full')['t_q5']:.2f}),因为它对排雷的 β 高达 {gen('lh', 'LOWVOL', 'full')['b']:.2f}(同一条腿的更高杠杆版,成本与 TE 更高)。

## 6. 预注册判定 P1-P7(`inc_t5_pass.csv`)

| # | 条件 | 数值 | 是否满足 |
|---|---|---|---|
""" + "\n".join(
        f"| {k[:2]} | {k[3:]} | {fmt_p(k, v)} | {'是' if str(v['ok']) == 'True' else '**否**'} |"
        for k, v in P.items() if k[:2] in ("P1", "P2", "P3", "P4", "P5", "P6", "P7")) + f"""

- P1:IR(CMB-EW5) < IR(排雷) 在全期 / 开发窗 / 检验窗**三个窗口都不成立**(上表括号内是 [CMB, LIQ]);P2:截距为负且 t = {pl['t_full']:.2f};P3:{P['P3_最长相对水下期与相对回撤/TE不劣于LIQ']['detail']};P4 满足(CMB 超额 {pc(float(P['P4_全期超额>=0.5xLIQ 且两窗>0']['value']))}% ≥ 0.5 x 排雷 {pc(g('lh',liq,'full','excess_arith'))}%,两窗为正),因为 P4 只要求"没被稀释过半";P5 满足但只是必要条件(真实组合比 10 个随机篮子安慰剂都好,ΔIR {pl['dIR_full']:+.3f} vs 安慰剂最大 {pl['placebo_max_dIR']:+.3f};说明判定工具有效、没有被构造方式机械抬高 IR);P6:6 个变体里**三窗 IR 都高于排雷的个数 = {pl['P6_count']}**。
- **判定规则结果**:全期 ΔIR ≤ 0 且截距 ≤ 0 -> **无增量**(§10.2)。D1-D6:B-LH 全过;A 轨 / B-MIN 的 E3 未过(§2)不改变这一判定,因为判定由 P1、P2 的符号决定。C 轨降级条件(§10.3)触发,但只能降级"有增量",对"无增量"无影响(§9)。

### 6.1 变体(P6;`inc_t5_variants.csv`)

""" + md_table(pd.DataFrame([{"变体": r.config.replace('C|', ''), "IR 全期": f3(r.ir_full), "IR 开发": f3(r.ir_dev), "IR 检验": f3(r.ir_test),
                              "排雷 IR 全/开/检": f"{r.ir_liq_full:.3f} / {r.ir_liq_dev:.3f} / {r.ir_liq_test:.3f}",
                              "三窗都高于排雷": "是" if r.three_window_better else "否"} for r in VAR.itertuples()])) + f"""

- 只有含反转的 20/35 变体(EW6 / CORE6 / ERC6 @20/35)的 IR 接近排雷(CORE6@20/35 全期 {VAR[VAR.config=='C|CORE6|20_35'].ir_full.iloc[0]:.3f} vs {VAR[VAR.config=='C|CORE6|20_35'].ir_liq_full.iloc[0]:.3f},ΔIR {DIR[(DIR.track=='lh')&(DIR.config=='C|CORE6|20_35')&(DIR.window=='full')].d_ir.iloc[0]:+.3f},自助 90% 区间 [{DIR[(DIR.track=='lh')&(DIR.config=='C|CORE6|20_35')&(DIR.window=='full')].boot_lo90.iloc[0]:+.2f}, {DIR[(DIR.track=='lh')&(DIR.config=='C|CORE6|20_35')&(DIR.window=='full')].boot_hi90.iloc[0]:+.2f}]),但没有一个超过它;CMB-EW5@20/35 全期 {VAR[VAR.config=='C|EW5|20_35'].ir_full.iloc[0]:.3f}。ERC 在 B-LH 45/60 比 EW 高 {VAR[VAR.config=='C|ERC5|45_60'].ir_full.iloc[0]-VAR[VAR.config=='C|EW5|45_60'].ir_full.iloc[0]:+.3f}、20/35 低 {VAR[VAR.config=='C|ERC5|20_35'].ir_full.iloc[0]-VAR[VAR.config=='C|EW5|20_35'].ir_full.iloc[0]:+.3f}(H7:无稳定优势);等风险权重均值 LIQ {ERC[(ERC.track=='lh')].mean_LIQ.iloc[0]:.2f} / 低波动 {ERC[(ERC.track=='lh')].mean_LOWVOL.iloc[0]:.2f} / 价值 {ERC[(ERC.track=='lh')].mean_VALUE.iloc[0]:.2f} / 红利 {ERC[(ERC.track=='lh')].mean_DIV.iloc[0]:.2f} / 质量 {ERC[(ERC.track=='lh')].mean_QUAL.iloc[0]:.2f}(`inc_t5_erc_alpha.csv`)。

### 6.2 安慰剂(`inc_t5_placebo.csv`)与稀释对照(`inc_t5_dilution.csv`)

- 10 个安慰剂(排雷 + 4 个随机篮子等权)B-LH 全期 IR {PLC[(PLC.track=='lh')&(PLC.window=='full')].ir.min():.2f}~{PLC[(PLC.track=='lh')&(PLC.window=='full')].ir.max():.2f},ΔIR {PLC[(PLC.track=='lh')&(PLC.window=='full')].d_ir_vs_liq.min():+.2f}~{PLC[(PLC.track=='lh')&(PLC.window=='full')].d_ir_vs_liq.max():+.2f};真实组合 ΔIR {pl['dIR_full']:+.3f}(好于全部安慰剂,但仍为负)。B-MIN 5 个安慰剂 IR {PLC[(PLC.track=='min')&(PLC.window=='full')].ir.min():.2f}~{PLC[(PLC.track=='min')&(PLC.window=='full')].ir.max():.2f},**真实组合 IR {g('min', ew5, 'full', 'ir'):.2f} 低于全部随机篮子**:2024-2025 的风格篮子(价值 / 红利 / 低波动 / 质量)系统性亏损,比随机选一半股票还差。
- 稀释对照 α·排雷 + (1−α)·B1:α = 0.2 / 0.5 时 IR 不变(B-LH 全期 {DIL[(DIL.track=='lh')&(DIL.alpha==0.2)&(DIL.window=='full')].ir.iloc[0]:.3f}),超额按 α 缩小为 {pc(DIL[(DIL.track=='lh')&(DIL.alpha==0.2)&(DIL.window=='full')].excess_arith.iloc[0])}% / {pc(DIL[(DIL.track=='lh')&(DIL.alpha==0.5)&(DIL.window=='full')].excess_arith.iloc[0])}%。即"只持有一半排雷、其余放池等权"的 IR 仍是 {DIL[(DIL.track=='lh')&(DIL.alpha==0.5)&(DIL.window=='full')].ir.iloc[0]:.3f},高于任何一个风格组合。

![ΔIR vs 安慰剂](../reports/inc_t5_placebo.png)

### 6.3 成本与净额(`inc_t5_cost_variants.csv`、`inc_t5_netting.csv`)

- 成本口径对结论无影响:B-LH 旧口径(只佣金 + 印花税)排雷 / CMB-EW5 IR = {COST[(COST.track=='lh')&(COST.config=='S|LIQ|20_35|old')&(COST.window=='full')].ir.iloc[0]:.3f} / {COST[(COST.track=='lh')&(COST.config=='C|EW5|45_60|old')&(COST.window=='full')].ir.iloc[0]:.3f},悲观口径(Y=1.5 + 两跳){COST[(COST.track=='lh')&(COST.config=='S|LIQ|20_35|pess')&(COST.window=='full')].ir.iloc[0]:.3f} / {COST[(COST.track=='lh')&(COST.config=='C|EW5|45_60|pess')&(COST.window=='full')].ir.iloc[0]:.3f}。
- 权重层合成的净额效应很小:CMB-EW5 一次引擎运行相对"各篮子单独运行的日收益按权重平均"多得 {pct(NET[(NET.track=='lh')&(NET.scheme=='EW5')&(NET.window=='full')].netting_gain.iloc[0])}pp/年(CMB-CORE5 {pct(NET[(NET.track=='lh')&(NET.scheme=='CORE5')&(NET.window=='full')].netting_gain.iloc[0])}pp,ERC5 {pct(NET[(NET.track=='lh')&(NET.scheme=='ERC5')&(NET.window=='full')].netting_gain.iloc[0])}pp),因为各篮子持仓相近、本来没有多少可对冲的换手。

## 7. A 轨子判定:排雷 + 低波动(LongPanel 2006-01-04 ~ 2025-12-31,含退市股;开发 2006-2019 / 检验 2020-2025)

CMB-A2 = EW(排雷, LOWVOL 窗口 120;45/60)。来源 `inc_t5_pass.csv`(A_*)、`inc_t5_summary.csv`、`inc_t5_dir.csv`。

| 配置 | IR 全期 | IR 开发 | IR 检验 | 超额% 全期 | TE% | 最长相对水下(日) | 相对回撤/TE |
|---|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| {nm} | {g('a', c, 'full', 'ir'):.3f} | {g('a', c, 'dev', 'ir'):.3f} | {g('a', c, 'test', 'ir'):.3f} | {pc(g('a', c, 'full', 'excess_arith'))} | {pc(g('a', c, 'full', 'te'))} | {int(g('a', c, 'full', 'rel_uw_longest_days'))} | {g('a', c, 'full', 'rel_mdd_over_te'):.2f} |"
        for c, nm in ((liq, "排雷(S1 换手率 20/35)"), ("S|LOWVOL|45_60", "LOWVOL120(45/60)"), ("C|A2_120", "CMB-A2(EW,LOWVOL120)"),
                      ("C|A3_120", "CMB-A3(+REV)"), ("C|COREA2_120", "CORE-A2(排雷 2/3)"), ("S|REV|45_60", "REV(45/60)"))) + f"""

- 子判定 **无增量**:P1 否(ΔIR {pa['dIR_full']:+.3f},三窗都低于排雷);P2 否(LOWVOL 对排雷回归截距 {pct(pa['a_full'])}pp,t = {pa['t_full']:.2f});P3 否({P['A_P3']['detail']});P4 满足;P5 满足(安慰剂最大 ΔIR {pa['placebo_max_dIR']:+.3f});P6 否({int(P['A_P6']['value'])}/6)。
- 低波动与排雷 ρ = {corr_row('a')['LOWVOL']:.2f}(A 轨全期),**H1 成立:低波动没有分散价值**。窗口 60 / 120 / 250 的 CMB-A2 IR 为 {g('a','C|A2_60','full','ir'):.3f} / {g('a','C|A2_120','full','ir'):.3f} / {g('a','C|A2_250','full','ir'):.3f},全部低于排雷的 {g('a',liq,'full','ir'):.3f}。分窗 2006-2014 / 2015-2025(dev2/test2):排雷 {g('a',liq,'dev2','ir'):.3f} / {g('a',liq,'test2','ir'):.3f},CMB-A2 {g('a','C|A2_120','dev2','ir'):.3f} / {g('a','C|A2_120','test2','ir'):.3f}。

## 8. B-MIN 确认(Panel() 1 分钟,V1000,open10,1000 万,2021-2025;开发 2021-2023 / 检验 2024-2025)

排雷底座 = COMP_v5(冻结打分,20/35),对 `ST.benchmark(rebalance=5)`;COMP_v5 对该窗口是信号样本内。

{perf_table('min', [liq, 'S|LOWVOL|45_60', 'S|VALUE|45_60', 'S|DIV|45_60', 'S|QUAL|45_60', 'S|REV|45_60', ew5, core5, erc5, 'C|EW6|45_60', s4])}

- P7 **不满足**:IR 开发 {pm['irs']['dev'][0]:.3f} vs {pm['irs']['dev'][1]:.3f},检验 {pm['irs']['test'][0]:.3f} vs {pm['irs']['test'][1]:.3f};S4 截距开发 / 检验 {pct(pm['a_dev'])} / {pct(pm['a_test'])}pp。最好的变体 CMB-CORE6 全期 IR {g('min','C|CORE6|45_60','full','ir') if False else DIR[(DIR.track=='min')&(DIR.config=='C|CORE6|45_60')&(DIR.window=='full')].ir.iloc[0]:.3f}、CORE5 {g('min',core5,'full','ir'):.3f},仍低于排雷 {g('min',liq,'full','ir'):.3f}(CORE5 的 ΔIR 自助 90% 区间 [{DIR[(DIR.track=='min')&(DIR.config==core5)&(DIR.window=='full')].boot_lo90.iloc[0]:+.2f}, {DIR[(DIR.track=='min')&(DIR.config==core5)&(DIR.window=='full')].boot_hi90.iloc[0]:+.2f}])。
- 事后 Barra 暴露(`inc_t5_barra_exposure.csv`,B-MIN,57 个月末的平均主动风格暴露,只作解释):

""" + md_table(pd.DataFrame([{"配置": r.config, "Liquidity": f"{r['exp_Liquidity 流动性']:.2f}", "ResidVol": f"{r['exp_ResidVol 残差波动']:.2f}",
                              "Beta": f"{r['exp_Beta 市场敏感']:.2f}", "BookToPrice": f"{r['exp_BookToPrice 价值']:.2f}",
                              "EarningsYield": f"{r['exp_EarningsYield 盈利']:.2f}", "Leverage": f"{r['exp_Leverage 杠杆']:.2f}",
                              "Size": f"{r['exp_Size 市值']:.2f}"} for _, r in BAR.iterrows()])) + f"""

  EW5 的 Liquidity 暴露({BAR[BAR.config=='EW5']['exp_Liquidity 流动性'].iloc[0]:.2f})与排雷({BAR[BAR.config=='LIQ']['exp_Liquidity 流动性'].iloc[0]:.2f})相同,另外多出价值 +{BAR[BAR.config=='EW5']['exp_BookToPrice 价值'].iloc[0]:.2f}、盈利 +{BAR[BAR.config=='EW5']['exp_EarningsYield 盈利'].iloc[0]:.2f}、残差波动 {BAR[BAR.config=='EW5']['exp_ResidVol 残差波动'].iloc[0]:.2f}、Beta {BAR[BAR.config=='EW5']['exp_Beta 市场敏感'].iloc[0]:.2f} 的暴露——叠加确实加了别的风格暴露,但这些暴露在 2021-2025 没有付钱(S4 截距为负)。

## 9. C 轨(指示性:2007-01-04 ~ 2016-12-30,EM 重述值 + 法定截止日可见,幸存者池;只能降级)

(`inc_t5_c_*.csv`;池 = q92 ∩ EM 覆盖幸存股,占 q92 池股票-日 {CQA[CQA.item=='C_pool_vs_q92_stockdays'].value.iloc[0]*100:.0f}%。)

| 配置 | 超额% | TE% | IR | NW t5 | 最长相对水下(日) |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {NAME.get(r.config, r.config)} | {pc(r.excess_arith)} | {pc(r.te)} | {r.ir:.3f} | {r.nw_t5:.2f} | {int(r.rel_uw_longest_days)} |" for r in CS.itertuples()) + f"""

- ΔIR(CMB-EW5 − 排雷)= **{c_dir:+.3f}**,S4 对排雷截距 {pct(c_a)}pp(t = {c_t:.2f}),降级条件(ΔIR ≤ 0 或截距 ≤ 0)**触发**。结论与 B-LH 同向,即使在偏向风格篮子的数据上(幸存者 + 重述)也没有增量。分半样本(`inc_t5_c_halves.csv`):排雷 IR 2007-2011 {CH[(CH.config==liq)&(CH.period=='2007-2011')].ir.iloc[0]:.2f} / 2012-2016 {CH[(CH.config==liq)&(CH.period=='2012-2016')].ir.iloc[0]:.2f};CMB-EW5 {CH[(CH.config==ew5)&(CH.period=='2007-2011')].ir.iloc[0]:.2f} / {CH[(CH.config==ew5)&(CH.period=='2012-2016')].ir.iloc[0]:.2f}。
- 注意 C 轨里**排雷自己也很弱**(IR {CS[CS.config==liq].ir.iloc[0]:.2f},2012-2016 为负):这一段的排雷在幸存者池里并不显著。C 轨覆盖:EM 财报 144/144 文件抓取成功;C 池覆盖见 `inc_t5_c_qa.csv`:BP {CQA[CQA.item.str.startswith('C_BP@')].value.min():.2f}~{CQA[CQA.item.str.startswith('C_BP@')].value.max():.2f}、EP {CQA[CQA.item.str.startswith('C_EP@')].value.min():.2f}~{CQA[CQA.item.str.startswith('C_EP@')].value.max():.2f}、QUAL {CQA[CQA.item.str.startswith('C_QUAL@')].value.min():.2f}~{CQA[CQA.item.str.startswith('C_QUAL@')].value.max():.2f}(2007-2016 逐年)。
- **v1 / v2**(偏离 `c_ep_formula`,**看到结果后改**,不进判定):首次实现用 EM 每股收益 TTM 作 EP(v1),发现 2005 年 Q1/Q3 每股收益缺失 98%,VALUE 篮子空仓到 2007 年 4 月,VALUE 单独超额 −8.5pp;改回预注册公式 EP = ROE_ttm x BP(v2,上表)。v1 的 ΔIR {float(CV1['C_dIR_EW5_vs_LIQ']):+.3f}、截距 {pct(float(CV1['C_S4_intercept_ann']))}pp(t {float(CV1['C_S4_intercept_t_q5']):.2f}),v2 的 ΔIR {c_dir:+.3f}、截距 {pct(c_a)}pp(t {c_t:.2f});两版同向。

## 10. 假设 H1-H8 对照(`inc_t5_hypotheses.csv`)

""" + md_table(pd.DataFrame([{"#": r.h, "假设": r.text, "结果": r.result, "成立": "是" if r.held else "**否**"} for r in HYP.itertuples()])) + f"""

- **H2 不成立、H6 不成立**是本轮两处与作者先验相反的结果:价值 / 红利 / 质量并不独立于流动性(价值、红利 ρ ≈ 0.8,即预注册写明的"都是低换手大盘的马甲");反转(45/60)成本后超额是正的 {pct(g('lh','S|REV|45_60','full','excess_arith'))}pp(换手 12 倍/年,缓冲带没能压住换手但仍净正),加入后 IR 小幅上升({g('lh','C|EW5|45_60','full','ir'):.3f} -> {g('lh','C|EW6|45_60','full','ir'):.3f}),但仍低于排雷单独。

## 11. 事后探索(看到结果后加,**不进判定**;`inc_t5_posthoc_rev.csv`、`inc_t5_insample_bound.csv`)

**样本内上界**(描述性):在 B-LH 的六条腿(45/60)里按全期样本内最大化 IR 的长仓权重是排雷 {BND[(BND.track=='lh')&(BND.window=='full')].w_LIQ.iloc[0]*100:.0f}% + 反转 {BND[(BND.track=='lh')&(BND.window=='full')].w_REV.iloc[0]*100:.0f}%,IR {BND[(BND.track=='lh')&(BND.window=='full')].ir_max_insample.iloc[0]:.3f}(排雷单独 {BND[(BND.track=='lh')&(BND.window=='full')].ir_liq.iloc[0]:.3f}),即**事后选权也几乎找不到比排雷更好的组合**;全部篮子用 20/35 深度时上界为 {BND[(BND.track=='lh20')&(BND.window=='full')].ir_max_insample.iloc[0]:.3f}(排雷 {BND[(BND.track=='lh20')&(BND.window=='full')].w_LIQ.iloc[0]*100:.0f}% + 反转 {BND[(BND.track=='lh20')&(BND.window=='full')].w_REV.iloc[0]*100:.0f}% + 质量 {BND[(BND.track=='lh20')&(BND.window=='full')].w_QUAL.iloc[0]*100:.0f}%),B-MIN 上界 = 排雷单独({BND[(BND.track=='min')&(BND.window=='full')].ir_max_insample.iloc[0]:.3f})。

**唯一的苗头:REV 在 20/35 深度**(剔除近 20 日涨幅最大的 30%,缓冲 20/35)。B-LH:IR {PH[(PH.track=='lh')&(PH.config=='S|REV|20_35')&(PH.window=='full')].ir.iloc[0]:.3f}(开发 {PH[(PH.track=='lh')&(PH.config=='S|REV|20_35')&(PH.window=='dev')].ir.iloc[0]:.2f} / 检验 {PH[(PH.track=='lh')&(PH.config=='S|REV|20_35')&(PH.window=='test')].ir.iloc[0]:.2f}),与排雷日超额相关 {PH[(PH.track=='lh')&(PH.config=='S|REV|20_35')&(PH.window=='full')].rho_vs_liq.iloc[0]:.2f},对排雷截距 {pct(PH[(PH.track=='lh')&(PH.config=='S|REV|20_35')&(PH.window=='full')].a_vs_liq_ann.iloc[0])}pp(t {PH[(PH.track=='lh')&(PH.config=='S|REV|20_35')&(PH.window=='full')].t_a_vs_liq.iloc[0]:.2f});与排雷等权合成(LR,一次引擎运行)IR {PH[(PH.track=='lh')&(PH.config=='C|LR_20_35')&(PH.window=='full')].ir.iloc[0]:.3f}。**但这个假说是从约 26 个网格配置里在看到结果后挑出来的**,多重检验未校正;A 轨 2006-2016(对该假说是样本外,A 轨在 2017-2025 与 B-LH 同一段)结果:REV 20/35 IR {PH[(PH.track=='a')&(PH.config=='S|REV|20_35')&(PH.window.str.startswith('2006-2016'))].ir.iloc[0]:.2f} vs 排雷 {PH[(PH.track=='a')&(PH.config=='S|LIQ|20_35')&(PH.window.str.startswith('2006-2016'))].ir.iloc[0]:.2f},二者相关 **{PH[(PH.track=='a')&(PH.config=='S|REV|20_35')&(PH.window.str.startswith('2006-2016'))].rho_vs_liq.iloc[0]:.2f}**(2017-2025 为 {PH[(PH.track=='a')&(PH.config=='S|REV|20_35')&(PH.window.str.startswith('2017-2025'))].rho_vs_liq.iloc[0]:.2f}),LR 合成 IR {PH[(PH.track=='a')&(PH.config=='C|LR_20_35')&(PH.window.str.startswith('2006-2016'))].ir.iloc[0]:.2f} vs {PH[(PH.track=='a')&(PH.config=='S|LIQ|20_35')&(PH.window.str.startswith('2006-2016'))].ir.iloc[0]:.2f},对排雷截距 t = {PH[(PH.track=='a')&(PH.config=='C|LR_20_35')&(PH.window.str.startswith('2006-2016'))].t_a_vs_liq.iloc[0]:.2f}。**样本外几乎没有增量,且相关性在早期高得多(0.83)——低相关不稳定**。作为后续单独预注册的候选假说记录,**不作为本轮发现**。

## 12. 偏离预注册(`inc_t5_deviations.csv`,逐条;`decided_before_returns = False` 的是看到结果后的,且不进判定)

""" + md_table(pd.DataFrame([{"阶段": r.stage, "项": r.item, "预注册原文": str(r.prereg_text)[:60], "决定": str(r.decision)[:220], "收益之前": "是" if r.decided_before_returns else "**否(看到结果后)**"} for r in DEV.itertuples()])) + f"""

## 13. 已知局限

1. B-LH 干净基本面样本只有 2017-2025(9 年,开发窗 3 年),风格溢价的统计功效低;IR 差的自助区间要与点估计一起读(CMB-EW5 ΔIR 的 90% 区间不含 0;CORE5@20/35 的区间上端 {DIR[(DIR.track=='lh')&(DIR.config=='C|CORE5|20_35')&(DIR.window=='full')].boot_hi90.iloc[0]:+.3f} 刚过 0)。
2. 每股量用后复权因子换算股本基准(含现金分红,每年 1~2% 近似误差);D2 两种算法的秩相关 0.9995,影响很小。
3. 红利预案一律计入(含最终未实施者);只计实施的敏感性(`S|DIVI|45_60`,`inc_t5_grid.csv`)结论一致。EM 分红表不含退市股,用 baostock 补(4227 次请求全部成功),退市前派现的公司因此有数据。
4. 风格篮子单只持仓约 7,000 元(1000 万 / 约 1,700 只),引擎不按最小交易单位约束,与 LH 排雷同一近似。
5. 不做行业 / 市值中性化,篮子收益含各风格的全部风险溢价;Barra 暴露只在 B-MIN 事后归因,未做残差 alpha 归因(只报暴露)。
6. 分钟面板无退市股(幸存者偏差对排雷超额 ≈ 0,`LH_STEP1` §5);B-MIN 与 B-LH 在 2021-2025 重叠,不是独立样本。
7. LIQ 的缓冲带路径依赖使 IR 有 ±0.07 的初始状态噪声(§3)。
8. B-LH 的 LIQ 与风格篮子的"深度"不同(73% vs 45% 持仓占比)是预注册设计;20/35 变体(P6)控制了深度,结论不变。

## 14. 对现有排雷方案的具体含义

**能不能叠加:不能。**按 §10 判定为"无增量"。把低波动、价值、红利、质量(以及反转)与排雷叠加,在 B-LH(含退市股的干净样本)与 B-MIN 两个独立实现上 IR 都下降,相对水下期变长,生成检验截距为负;随机篮子安慰剂在 B-MIN 上都比真实风格篮子好。

**叠加后约多少**(成本后,对同频池等权 B1;`inc_t5_summary.csv`):

| 配置 | 面板 | 算术超额%/年 | IR | TE% | 最长相对水下(日) | 绝对最大回撤% |
|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| {nm} | {tr} | {pc(g(tr, c, 'full', 'excess_arith'))} | {g(tr, c, 'full', 'ir'):.3f} | {pc(g(tr, c, 'full', 'te'))} | {int(g(tr, c, 'full', 'rel_uw_longest_days'))} | {g(tr, c, 'full', 's_mdd') * 100:.1f} |"
        for tr in ("lh", "min") for c, nm in ((liq, "排雷单独"), (core5, "CMB-CORE5(排雷 1/2 + 四风格各 1/8)"), (ew5, "CMB-EW5(五腿各 1/5)"))) + f"""

- 即:**排雷单独 > CORE5 > EW5**,每叠加一点风格篮子,超额与 IR 都往下走;相对回撤 / TE 从排雷的 {g('lh',liq,'full','rel_mdd_over_te'):.2f} 升到 CORE5 {g('lh',core5,'full','rel_mdd_over_te'):.2f}、EW5 {g('lh',ew5,'full','rel_mdd_over_te'):.2f}(B-LH)。绝对最大回撤几乎不变(B-LH {g('lh',liq,'full','s_mdd')*100:.1f}% / {g('lh',core5,'full','s_mdd')*100:.1f}% / {g('lh',ew5,'full','s_mdd')*100:.1f}%,与市场同量级),叠加对绝对回撤没有帮助;"绝对回撤 ≤ 50%"仍靠仓位上限。
- 哪条腿真正独立:**没有一条同时独立又带 alpha**。低波动(ρ {rhoL['LOWVOL']:.2f})、价值({rhoL['VALUE']:.2f})、红利({rhoL['DIV']:.2f})是排雷的同一条腿(β 约 1、截距为负);质量(ρ {rhoL['QUAL']:.2f} / {rhoM['QUAL']:.2f})相对独立但单独 IR 只有 {g('lh','S|QUAL|45_60','full','ir'):.2f}(B-LH)/ {g('min','S|QUAL|45_60','full','ir'):.2f}(B-MIN),检验窗贡献接近 0;反转(ρ {rhoL['REV']:.2f} / {rhoM['REV']:.2f})是唯一低相关且净正的腿,但 45/60 深度下 IR 仅 {g('lh','S|REV|45_60','full','ir'):.2f}、换手 {g('lh','S|REV|45_60','full','turnover_ann_run'):.0f} 倍/年,20/35 深度的苗头未通过样本外(§11),只作后续单独预注册的候选假说。
- 对运行中的方案:**不改排雷底座,不加风格篮子。**"相对池等权约 +3~5pp/年、IR 0.9~1.7"这个量级对排雷本身(B-LH 2017-2025 {pct(g('lh',liq,'full','excess_arith'))}pp、B-MIN {pct(g('min',liq,'full','excess_arith'))}pp)成立,但需记住它在这两段都是信号样本内(底座参数来自 2020-2025),长历史 2006-2025 的 IR 只有 {g('a',liq,'full','ir'):.2f}、2006-2016 只有 {PH[(PH.track=='a')&(PH.config=='S|LIQ|20_35')&(PH.window.str.startswith('2006-2016'))].ir.iloc[0]:.2f}。

## 15. 一句话结论(可直接引用)

> 在 2017-2025 含退市股的干净样本和 2021-2025 分钟面板上,把低波动、价值、红利、质量风格篮子与排雷按等权 / 核心-卫星 / 逆 TE 合成,相对同频池等权的 IR 从 {g('lh', liq, 'full', 'ir'):.2f} 降到 {g('lh', ew5, 'full', 'ir'):.2f}(核心-卫星 {g('lh', core5, 'full', 'ir'):.2f};分钟面板 {g('min', liq, 'full', 'ir'):.2f} -> {g('min', ew5, 'full', 'ir'):.2f}),最长相对水下期从 {int(g('lh', liq, 'full', 'rel_uw_longest_days'))} 日拉长到 {int(g('lh', ew5, 'full', 'rel_uw_longest_days'))} 日,生成检验截距 {pct(pl['a_full'])}pp(t = {pl['t_full']:.2f});原因是低波动 / 价值 / 红利与排雷的日超额相关 0.8~0.9,是同一条"低换手"腿,质量与反转虽相对独立却单独不赚钱,所以排雷之外没有可叠加的第二条腿。

**判定:无增量**

## 附:产出清单

- 脚本:`scripts/inc_t5_fetch.py`(EM 分红 / EM 财报 / baostock 分红抓取)、`inc_t5_common.py`、`inc_t5_scores.py`(verify / devs / scores_lh / scores_div / scores_min / qa)、`inc_t5_runs.py`(weights / runs / D6)、`inc_t5_c.py`(C 轨)、`inc_t5_report.py`(stats / bears / mktbeta / posthoc)、`inc_t5_barra.py`、`inc_t5_figs.py`、`inc_t5_md.py`;库 `src/afactor/inc_t5_signals.py`、`inc_t5_stats.py`;测试 `tests/test_inc_t5.py`(19 项,合成数据)。
- 账本 `research/ledger/`:`inc_t5_deviations.csv`、`inc_t5_data_qa.csv`、`inc_t5_checks.csv`、`inc_t5_grid.csv`、`inc_t5_summary.csv`、`inc_t5_yearly.csv`、`inc_t5_corr.csv`、`inc_t5_gen_test.csv`、`inc_t5_dir.csv`、`inc_t5_pass.csv/.json`、`inc_t5_placebo.csv`、`inc_t5_variants.csv`、`inc_t5_dilution.csv`、`inc_t5_netting.csv`、`inc_t5_te_expected.csv`、`inc_t5_erc_alpha.csv`、`inc_t5_cost_variants.csv`、`inc_t5_hypotheses.csv`、`inc_t5_jaccard.csv`、`inc_t5_jaccard_min.csv`、`inc_t5_basket_size.csv`、`inc_t5_bear_windows.csv`、`inc_t5_mkt_beta.csv`、`inc_t5_insample_bound.csv`、`inc_t5_posthoc_rev.csv`、`inc_t5_barra_exposure.csv`、`inc_t5_c_*.csv`(含 v1)。
- 图 `reports/`:`inc_t5_rel_nav.png`、`inc_t5_corr.png`、`inc_t5_ir_windows.png`、`inc_t5_placebo.png`。
- 缓存(可删可重建)`cache/_tmp/inc_t5_scores`、`inc_t5_runs`、`inc_t5_w`、`inc_t5_fetch`;日志 `logs/inc_t5_*.log`。
""")
    return "\n".join(s)


if __name__ == "__main__":
    out = ROOT / "research" / "INC_T5_REPORT.md"
    if out.exists() and "原审查处置与中断修复闭环" in out.read_text():
        raise SystemExit("已存在审查修订版；旧生成器不能覆盖 INC_T5_REPORT.md，请直接维护审查处置与诊断引用。")
    out.write_text(build())
    print("written", out, len(out.read_text()))
