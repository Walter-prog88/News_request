# ETF 项目可复用资产与负面结果审计(2026-08-23,4 agent 交叉核对)

来源:工作流 wf_aeb2c30a-614,所有数字已回源文件核对;同 id 两条账本记录时取修正后那条。

## 一、交叉核对:对不上 / 需降级的条目

| # | 报告中的说法 | 源文件核对结果 | 处置 |
|---|---|---|---|
| 1 | 报告1:sleeve_v2 `tsmom_vol(w=252)` 逐窗 0.654 / 1.087 / 1.903,全期 Sharpe 1.093 / 5.12% / MDD −8.48%;`w=63` 0.614/0.500/1.845,全期 0.872 | `trial_ledger.jsonl` 里同一 id 有**两条**记录(append-only,R2 修正前后)。报告 1 引的是**修正前**那条。`results/sleeve_v2.json`、`composite/deflation.json`、`report.html` 用的是修正后:`w=252` 逐窗 **0.499 / 0.794 / 1.860**,全期 **0.955 / 4.56% / −9.11%**;`w=63` **0.384 / 0.395 / 1.812**,全期 0.722 | 报告 1 的 headline(wc 0.50 / full 0.96)是对的,但逐窗数字要换成修正后口径 |
| 2 | 报告1:`month_turn` 逐窗 0.887 / 0.763 / −0.090,全期 0.441 / 4.95% | 同样是修正前条目。修正后(`results/sleeve_v2.json`):**1.133 / 0.715 / −0.089**,全期 **0.518 / 6.02% / MDD −24.51%**;wc 0.715 | 同上 |
| 3 | 报告1、2:"市场状态条件 IC 在 ETF 层独立重现"(D3 表:低波 vs 高波 6 因子全部同向) | `alpha_diagnosis.py:155-158` 的 D3 条件是 **每只 ETF 自身的 `rv_pct` / `dd_now` 分位**(`rv >= 0.7` / `rv <= 0.3`,样本占比 ~6.7%),是**截面子集**,不是市场状态分档。它说的是"低波标的上价格因子更准",不是"市场低波时排序更准" | "跨两个数据域独立重现"这一说法**不成立**,只有风格层 Q2(`style_log.txt:60-72`)是真正的市场级条件 IC |
| 4 | 报告1:"文档汇总口径 低波 0.061/NWt 2.41 vs 高波 0.008/NWt 0.32";报告2:"主题层 IC 0.067 (NWt 1.81)" | 只在 `docs/主动策略系统_研究进度总览_20260814.md:57,72` 出现。`alpha_diag_log.txt:77` 的主题层 IC 是 **nan**,D3 表没有 NWt 列。无法从现存日志复现 | 标为"文档转述、不可追溯",不要转引 |
| 5 | 报告2:主题层冠军的随机对照 p=0.0199 / +2.52σ / 随机 −3.18% / "19 次调仓 t=1.61",来源 `theme_log.txt` | `theme_log.txt` 与 `theme_system_meta.json` 里**没有**这些数字,只有网格表;冠军在日志里是 19.696%(inner=signal),文档写 18.56%(inner=equal 那行) | 来源应写 docs 文件;随机对照数字不可追溯 |
| 6 | 报告2:F3 风险调整 k=2 L=126 +7.94pp,CI [+0.86,+14.85],"全表唯一 CI 过零" | `rotation_excess.csv` 有**两条轨道**。+7.94pp 是 **A 研究版(含 LOF)**。**B 可交易版**同配置只有 **+5.46pp,t 1.39,CI [−1.08, +11.30],未过零**;B 轨最强是 k=3 L=126 +5.65pp,CI [−0.29, +10.67],也未过零 | "唯一 CI 过零"只在研究轨成立;可交易轨全部未过零 |
| 7 | 报告1:smoke 里 onid 的腿 | `smoke_result.json` 键名是 `onid.overnight63.all.h21`(hold=21),而白名单模块 `onid_0a19f3a1ad` 是 hold=5 | 小口径差异,冒烟数字不能直接对应账本模块 |
| 8 | 报告1:参与率 11.72 / 11.17 | 在 docs/*.md 与 README 未找到(8.26 在 `全ETF多引擎策略系统_冻结成果_20260813.md:14` 找到) | 可能在 zoo `report.html`,未复核 |

**核对无误的关键数字**(全部重新读源文件):双趋势 `metrics.json` 全部字段(10.57% / 0.680 / −31.11% / 35 次切换 / HAC α 8.35% p=0.0075 / bootstrap 平均主动 1.27% p=0.79)、README 零现金 8.95% 与 1.62pp;`current-findings.md:585` 五 ETF +2.83 / +0.94 / −2.39 / −4.31 / −10.47pp;`mom_53a47a6d64` 逐窗 0.554/0.291/0.466 全期 0.445;`onid_0a19f3a1ad` 0.733/0.165/0.543 全期 0.518;`sleeve_ix` 13 配置全部数字;`deflation.json` 三组 PSR;`smoke_result.json` 全部;`poolwide_selection.csv` 全表;`rolling_blocks.csv` +3.20 → +1.23 / −0.38 与随机标签 9.77 vs 7.65;`style_log.txt` Q2 全表;`validation_line*.json`、`rolling_D/Dgate.json` 全部;`experiment_ledger.csv` state020-040 阈值曲线、flow_all 逐年、x_uptrend 50.58%;2020-22 有门 9.10%/0.52/−26.25% vs 无门 7.72%/0.45/−29.40%;`milestones.csv` 全表;冻结实验报告 EWMA 3.01→3.59、P1 vs P0 CI [−31.66, −1.11]、17.66pp;`wf_meta.json` −4.03%;`wf_segments.csv` 21/19/17/12/7;`wf_state_switch_summary.csv` 全表;`efl_log.txt`、`efl_add_family.csv`;冻结成果文档因子×块稳定表 6/6/5/5/5;Top5 报告 107.6/36.97/476/81.4/47.7/3.60/0.69;`walkforward.py:209` 白名单。

---

## 二、去重清单

### A. 可直接复用的信号(全部只需日频 OHLC/收益 + 行业标签)

| 信号 | 定义 / 参数 | 源文件 | 样本外证据(核对后口径) | 迁移到板块守门的用法与限定 |
|---|---|---|---|---|
| **A1 价格位置因子 `range_pos`**(及 5 因子组 `path_eff / mom_ens / slope_r2 / dd_now`) | `range_pos = (lv − min_L)/(max_L − min_L)` shift(1),L∈{63,252};其余见 `frozen_core.py:109-144`;逐日截面 z 后等权;`TOPQ=0.30, STEP=5, COST=0.5bp` | `index-project-active-etf-upper-bound/research/frozen_multi_engine_20260813/frozen_core.py:32,109,259,276` | 两条独立线都排第一:因子×块前后半同号 `range_pos` 6/6、`path_eff` 6/6、其余 5/6(冻结成果文档 §2.2);21 段 walk-forward 被选 `range_pos_252` **21/21**、`mom_voladj_252` 19/21、`range_pos_63` 17/21(`wf_segments.csv`)。可交易池不分块持前 10%:10.48% / 0.67 / −16.57%(`poolwide_selection.csv`);成本阶梯 20bp 时 Sharpe 0.37、50bp 为负 | 最稳的纯价格因子。**但**整套 walk-forward 系统是负的(−4.03%);因子稳 ≠ 赚钱。A 股印花税下必须重测成本 |
| **A2 市场状态四变量**(守门候选,条件-IC 口径) | `trend=(m/m.shift(252)−1).shift(1)`;`above_ma200`;`vol_pct = (21日std×√243).rolling(756,min 252).rank(pct)`;`dd = m/rolling_max(252)−1`。阈值 高波>0.7 / 低波<0.3 / 回撤深<−15% / 接近高点>−5% | `.../research/active_system_v1/style_rotation.py:137-148`;实测 `style_log.txt:60-72` | 风格层 2006-2019,n=3120:无条件 IC 0.1411;**252 日趋势上/下 0.1428 vs 0.1390(无分辨)**;MA200 上/下 0.1767 vs 0.0986;**低波 0.2430 (NWt 3.56) vs 高波 0.0101 (0.13)**;**接近高点 0.2665 vs 回撤深 0.0691** | 这是整个 ETF 项目最接近"板块守门员"的证据,但:(1) 它测的是**市场级**状态,板块级未测;(2) 它回答的是"排序可不可信",不是"该不该持有"——项目已为此付过学费(见 C5);(3) 用户直觉的"趋势向上"恰是四个变量里**最无分辨力**的 |
| **A3 板块层面板特征** | `th_breadth_21`(板块内 21 日收益>0 占比)、`th_disp_21`(板块内当日截面 std)、`th_rel_21/63`(板块相对全市场)、`th_lowvol_share`、`th_mom_5/21/63`、`th_rv_21` 等 19 个;板块门 = 可用成分≥2 | `.../active_system_v1/theme_system.py:64-101`;特征清单 `theme_system_meta.json` | 主题层轮动 180 配置冠军 19.70%/Sharpe 1.006,但作者判为不可采信(716 天 × 月频只有 19 次决策),主题层 IC 只有 0.002~0.038 | **定义可抄,结论不能抄**。A 股一个行业几十上百只股、2005 年起可回溯,正好补上 ETF 域样本不足的缺陷 |
| **A4 确认里程碑规则(含误报率)** | breadth_watch 广度≥90% & 等权收益≥1%;confirmation 95%/2%;成交额因果 z252≥1 & 量比≥1.5 / z≥2;等权净值上穿 MA20 / 在 MA60 上方 | `.../research/active_etf_upper_bound/event_diagnostic.py:851-930`(`_row_share_positive`、`_causal_zscore`) | 924 复盘 `milestones.csv`:宽松档事件前 60 日误触发 5/3/3 次;严格档各 1 次且滞后到事件当天;**MA60 上方事件前 0 次,T+1 才转正**;state_risk_on 误触发 41 次 | 把"全池 ETF"换成"板块内成分股"即得板块广度/板块等权 vs MA/板块成交额 z。必须同时报提前量与误报率 |
| **A5 跨类趋势×逆波动 `tsmom_vol(w=252)`** | 每类 `rolling(w).sum()>0` 才进场;权重 = 1/(21日 std×√252) 归一;月频;10bp×换手;信号 t → 收益 t+1(`harness.py:308-320`) | `index project/research/zoo_v1/zoo_axes.py:150-170`;`results/sleeve_v2.json` | 修正后:wc **0.499** / full **0.955**,逐窗 0.499 / 0.794 / 1.860;去偏 PSR 27.4%(N=685)/50.7%(N=99);冒烟 45 日 +0.10%。**纯开关 `tsmom_sign(w=252)` 只有 wc 0.161 / full 0.701**;`risk_parity_all` 单独 full 0.912 | 存活的是"趋势过滤 × 逆波动配权"的合体,**不是二元开关本身**。下沉到指数组层已判死(C1)。可复用形态:整篮子时序开关 + 逆波动权重,且必须允许全空仓 |
| **A6 上证50 双趋势** | `Close>MA200 AND close/close_{t-252}−1>0` → 100%,否则现金;10bp 单边;信号收盘→次日开盘 | `.../strategies/sse50-dual-trend/README.md`、`data/metrics.json` | 2005-02~2026-07:CAGR 10.57% / Sharpe 0.680 / MDD −31.11% / 35 次切换;HAC α 8.35% p=0.0075,但**平均主动收益 α 1.27% p=0.79**(差额来自 β=0.34);零现金 8.95%。**五宽基只有 1/5 成立**(+2.83 / +0.94 / −2.39 / −4.31 / −10.47pp) | 唯一带真实执行账本的择时规则,但跨标的迁移性极差,板块层必须逐板块验收;收益相当部分来自低 β + 现金利息,守门后仍满仓选股时该优势不存在 |
| **A7 12-1 动量截面多空** | `close_{t−21}/close_{t−252}−1`,域内五分位多空,hold 21 | `zoo_axes.py` `AXES["mom"]`;`results/mom.json` id `mom_53a47a6d64` | 逐窗 0.554 / 0.291 / 0.466,全期 0.445 / 3.53% / −13.53%;66 个 mom 配置里唯一双窗同号;冒烟 +5.51% | 只能做"板块排序",不能做守门;**只做多测出来的是 β**(`honest-search/SKILL.md:20`);量级小 |
| **A8 风险调整相对排序 `mom_L/vol_63` 取前 k 类** | k∈{2,3}, L∈{126,252},月频,对静态等权 | `frozen_multi_engine_20260813/rotation.py`、`rotation_excess.csv`、`rotation_log.txt:156-215` | 研究轨 k=2 L=126 +7.94pp(CI 过零);**可交易轨 +5.46pp,CI [−1.08,+11.30] 未过零**;逐年超额 2020 −2.79 / 2021 +15.57 / 2022 +13.91 / 2023 +3.71 / 2024 +6.20 / 2025 +10.93;留一年 +6.26~+9.70 | 7 个类是**跨资产大类**不是 A 股行业。超额集中在 2021-22,与用户"2022-24 只做披露"相反:**相对排序型轮动恰在轮动市最值钱** |
| **A9 EWMA-20 波动目标化** | 目标波动 15%,clip [0.3,1.5] | `reports/冻结实验最终交付_20260807.md` §3-4 | P0 留出段 Calmar 3.01→3.59、MDD −15.40→−14.54(单条路径,272 日);**反向**:全 ETF 线 Sharpe +0.026 且 MDD −40.8→−49.8,32 只线 7 折判负 5/7;zoo `vol_target_dom` wc −1.345 | 跨基座不稳定,必须在自己基座上做 gated/ungated 对照 |
| **A10 指数层状态门 `regime_score≥0.30` 的 A-B 方法** | 冻结 2005-2019 HGB,t 日分数施加到 t+1;外挂 parquet 与选股解耦 | `index project/research/showcase_v1/d_gate.py`;`.../active_etf_upper_bound/validate_factor_minute.py:57-85`;`FROZEN_REGIME.json` | `rolling_D` vs `Dgate`:2020-21 **22.4→15.9%**(门在趋势段净负 6.5pp,MDD 还变差);2022-23 −15.9→−12.3%,MDD −31.6→**−24.8%**;全期 Sharpe 0.79→0.78,MDD −37.8→−32.9%;validation_line 四组 Sharpe 变化 +0.04/0.00/−0.01/+0.03;2024-26 state030 64.35% vs 无门 59.39%,阈值 0.40 掉到 45.07%;2020-22 7.72→9.10%,MDD −29.40→−26.25%;**924 当天分数 0.394 risk-off,T+2 才转正**,前 60 日误触发 41 次 | **方法可搬(把 000985 换成板块指数逐板块产 score),结论偏冷**:守门员的稳定增量只有回撤;验收基线必须是 ungated 同策略 |

### B. 负面结果(ETF 项目已证明没用,A 股项目不要重做)

1. **趋势下沉到行业/指数组层已正式关闭**:`sleeve_ix` 13 配置全灭(`results/sleeve_ix.json`):`tsmom_sign` w=21/63/126/252 wc −1.511/−1.541/−2.009/−1.242;`tsmom_vol` −0.788/−0.830/−1.231/−1.018;逐窗形态 = 2019-21 正、**2022-23 −1.0~−2.0**、2024-26 正。机制(`report.html`):"百余个股票指数组共享同一市场因子,总有组在趋势里,组合永远满仓——熊市窗口 −1.2;趋势的钱来自资产类之间不相关,不来自行业数量"。
2. **最朴素的"板块动量>0 才持有否则转现金"不显著**:`rotation_excess.csv` F1 L=63/126/252 研究轨 −2.09 / −0.01 / +0.46pp(t −0.85 / −0.08 / 0.07),可交易轨 −0.76 / +0.34 / +0.71pp。
3. **"趋势 × 信号"交互是负增量**:`legacy_flow_all_x_uptrend_k4_r5` 50.58%/1.79 < 不加交互 60.37%/2.11(`experiment_ledger.csv`)。
4. **纯价格指数间轮动 10/10 门失败 ×3**(`current-findings.md:575-613`):12-1 IC 0.01224 t 0.22 p 0.81;两簇相对强弱 IC −0.0204;6-1 IC −0.0789。裁定 `PRICE_ROTATION_FAMILY=CLOSE`。
5. **条件 IC 高 ≠ 该持有**(`docs/主动策略系统_全期walkforward_历史探索_20260814.md:77-80`):把"低波 IC 高"实现成"高波减仓"(敞口 0.5+0.5×低波分位)精确砍掉 2024-25 牛市仓位;`wf_state_switch_summary.csv` 所有状态切换变体(5.38~6.39%)均不优于固定 50/50(5.68%/0.70)或纯慢因子(6.09%/0.82)。
6. **主题内个券选择几乎无信号**:`alpha_diag_log.txt:78-80` 主题内 IC 0.0189 t=1.60 vs 个券层 0.0496 t=3.23;`efl_add_family.csv` T_主题相对族加入后 IC −0.0042;"个券精选"已列入否决表。
7. **开发段无选择功能**:P1 vs P0 留出段 −15.76pp/年,90% CI [−31.66, −1.11];"不能在 2022-24 轮动段上挑最好的守门参数再拿到趋势段用"。
8. **测量精度**:32 标的 × 272 日最小可检测效应 17.66pp/年;+4.38pp 需 17.5 年;M_TOP40 CI [−14.27, +22.74]。31 个行业宽度不足以在几年里确认个位数 pp。
9. **分块/聚类前视**:块等权增量 +3.20 → 滚动重建 +1.23 → 可交易池 **−0.38pp**,落在随机标签 20 次对照分布内;随机块标签 E_core 9.77% > 真实 7.65%。
10. **ML 在可交易池全负**(规则 3.94% 零分位 1.00;Ridge/GBM 四变体 −0.81 ~ −2.56pp);四引擎三条 Sharpe 零分位 0.00。
11. **行业 ETF 轮动 v1-v4 无一通过资格门**;Top-5 漂亮曲线是 5G 通信单押 476/585 天,Top5 vs Top10 t=0.69。
12. **zoo 死轴勿重测**:波动率族 48 配置最好 −0.26;微观摩擦 −0.42;星期效应 −1.63~−2.15;尾盘反转执行不可用;三代因果选择器全部跑输拿着不动(trail252_thr0.3 −0.400);669 次搜索后最优组合 PSR 28%/52%。
13. **长历史域指数择时** 修正零假设后与随机无异(0.588 vs 0.567);kmeans/gmm 择时全场最差。
14. **广度/等权均线类确认信号**:宽松档误报 3-5 次/60 日,严格档滞后到当天或 T+1(A4 同表)。

### C. 数据资产(去重)

- `index project/research/zoo_v1/`:`trial_ledger.jsonl`(669 条,**注意同 id 两条时取后一条**)、`results/*.json`(修正后口径)、`composite/{deflation,smoke_result,summary}.json` + `smoke_used.lock`、`module_returns/`(634 个净值)、`harness.py / zoo_axes.py / run_axis.py / walkforward.py:209 / deflate.py / smoke.py`;协议 `index project/.claude/skills/honest-search/SKILL.md`。
- `index-project-active-etf-upper-bound/research/frozen_multi_engine_20260813/`:`frozen_core.py`(路径常量 :20-21 是旧路径)、`poolwide_selection.csv`、`rolling_blocks.csv`、`rotation.py + rotation_grid/excess.csv + rotation_log.txt`、`yearly.csv`、`reproduce.py`。
- `.../research/active_system_v1/`:`style_rotation.py` + `style_log.txt`(Q2 条件 IC)、`alpha_diagnosis.py` + `alpha_diag_log.txt`(D3/D4)、`theme_system.py` + `theme_system.csv/meta.json`、`etf_factor_library.py` + `efl_*.csv` + `efl_log.txt`、`walkforward_system.py` + `wf_meta.json/wf_segments.csv/wf_variants.csv/wf_state_switch_summary.csv`、`freeze_regime.py` + `FROZEN_REGIME.json`、`construction_study.py:50-68`(11 主题关键词映射)。
- `index project/research/active_system_v1/regime_signal.parquet`(sha256 e72e9538…,2005-07-18~2026-05-21;生成时点未归档)。
- `index project/research/showcase_v1/`:`d_gate.py`、`results/validation_line*.json`、`rolling_{A,B,C,D,Dgate}.json`。
- `.../research/results/active_etf_upper_bound/`:`experiment_ledger.csv` + 60 余个 `<配置>/summary.json`(含逐年)、`strategy_explainer_support/state030_vs_selected_equalweight_summary.json`、`event_20240924_core_flow_main/{milestones.csv,event_window.csv,technical_report.md}`;`core_flow_main.npz`(分钟流,**A 股项目无对应数据,不可迁移**)。
- `.../strategies/sse50-dual-trend/{README.md,data/metrics.json,data/chart-ready-curve.csv}`。
- `.../configs/industry_rotation_v{1..4}.yml`(32 只行业 ETF 清单)+ `artifacts/industry_rotation_v2/qa_report.json`(逐只起始日:仅 9 只回到 2019-20,**2020-21 段板块验证必须用行业指数而非 ETF**)。
- `.../reports/{冻结实验最终交付,多因子框架_最终报告,多因子选股择时模型_最终交付}_20260807.md`、`reports/final_model/`、`docs/{全ETF多引擎…冻结成果,研究版与可交易版,主动策略系统_研究进度总览,主动策略系统_全期walkforward}_2026081x.md`、`docs/current-findings.md`。
- `full-etf project2 /full-etf-universe-store/`(日频/分钟/分类版本 `symbol_master.parquet`)。
- 分段口径:ETF 项目干净窗只有 2019-21 与 2022-23;2024-26 全部标记"已污染";留出段 2025-05~2026-06 已被多次触碰;冒烟窗仅 45 日且已锁。

---

## 三、"ETF 强 → 成分股也强"假设:ETF 项目里的证据

**没有直接证据。** 三个子项目没有任何个股/成分股数据(`index project/data/raw/tencent_stock_zh_a_hist_tx_hfq/` 的 5 个"股票"文件全是 ETF 代码;`analyze_candidate_concentration.py` 的 `index_weight` 是按跟踪指数聚合的 ETF 权重;`docs/five-index-next-stage-research.md:11,32` 的 PIT 成分面板只是计划,"真实 Tushare 数据尚未冻结")。ETF 层与板块层之间的全部研究都是 ETF ↔ 主题/指数组,从未触及成分股。

**间接证据都偏反面**:(a) 主题内 IC 0.0189 t=1.60 vs 个券层 0.0496(信号在板块之间,不在板块之内);(b) 指数组趋势下沉 13/13 全灭,2022-23 窗 −1.0~−2.0;(c) 类别级 mom>0 守门三窗不显著;(d) 指数级门在 2020-21 趋势段净负 6.5pp。另外,同期口径下"ETF 强则成分股强"是恒等式(ETF ≈ 成分加权均值),有意义的只有**预测版**:板块过去的趋势状态能否预测 (i) 板块自身未来收益、(ii) 板块内选股因子的未来 IC。

**A 股项目现有数据最快的验证路径**(全部已在本地,不需要分钟数据):

- 输入:`cache/daily/close_px.parquet`、`open_auction_px.parquet`、`ex_ret_day.parquet`(2020-01~2026-08);`data_8:21/stock_basic.csv` 证监会 3 位行业码(5547 只,84 类,340 空;成分≥15 约 51 个行业,`research/r_industry.log:1` 已这样分过);`data_8:21/daily_basic.csv` 真实市值(可做市值加权版板块指数,更接近 ETF)。
- 第一步(板块状态面板,半天工作量):逐行业构造等权与市值加权日净值,按 `style_rotation.py:137-148` 的四个变量 + `theme_system.py:84-86` 的 `breadth_21 / disp_21 / rel_63` 算板块状态;`vol_pct` 的 756 日窗口在 2020 起步数据下改 504/min 252。
- 第二步(测 ii,条件 IC,直接复用 `research/ic_series` 流水线):对现有 COMP_v5 及 `ivol_20 / rev_20 / id_v_ac1_m5` 做**行业内** Rank IC,按板块状态(低波/高波、接近高点/回撤深、MA200 上/下、252 日趋势上/下)分档报 IC 与 NW t。这是对 `style_log.txt` Q2 结构的板块级复现,ETF 项目没做过,是真正增量。预期:若"趋势上/下"分档与 Q2 一样无分辨,用户的守门直觉在条件 IC 口径上就站不住。
- 第三步(测 i,守门员作回撤工具):逐行业 gated vs ungated 成分等权,门用四变量各自单独做,**基线 = 同篮子 ungated**(d_gate 教训),加"同暴露比例随机门"零假设(`rotation.py` N=60 那套),按 2020-21 / 2022-24 / 2025 分段报年化、Sharpe、MDD 三项,只看 MDD 是否稳定改善。门的输出必须是"关闭选股层退回板块等权"与"退回现金"两个版本分开报(C5 教训)。
- 第四步(测"ETF 强 → 成分强"的预测版):板块 t 日状态 → 成分股 t+1..t+21 超额(相对全池)的面板回归/分组,与项目 README §10.7 已有的月度行业动量 IC 表(2020-21、2025-26 为正,2022-24 为负;无条件前 5 行业五年 −28.4pp)对齐。
- 必须写进披露的限制:行业标签是 2026-08 快照(证监会 2024-11 换表,非 PIT);`new_data` 有幸存者偏差;2020-2026 只有一次真状态切换(README §10.7 审查 D1:状态变量 |corr| ≤ 0.11,符号命中 49-58%),所以任何"趋势段 vs 轮动段"的结论 n≈1,要确认必须先补 2005 年起日线(`DATA_WISHLIST.md` 2026-08-23 第 1 项);2026 段已在 §10.7 诊断中使用,不再算干净。