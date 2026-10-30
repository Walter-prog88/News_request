# 阶段二 操作定义：传导弹性检验（所有智能体必须遵守）

阶段一已由用户确认（2026-09-08）。阶段一复盘 §3 提出的设计修改随之生效，本文件把它们落实为可计算的定义。
任何偏离必须在产出 JSON 的 `caveats` 中显式写明理由。**不得对中国恢复年限作判断；不得把 H1–H4 当前提。**

## 0. 输入（全部来自 phase1/data/processed/，不得重新取数除非缺口明确）
- 价格：US `us_price_{ny,sf,la,phx}.csv`（月度，SA），NY 公寓 `us_price_nycondo.csv`，San Jose 备选 `us_price_sanjose_*.csv`；
  JP `jp_tokyo_reins_ward_used_condo_annual.csv`（CY 1992–2025，1991 为导出值）、`jp_tokyo_fkk_new_condo_annual.csv`（1973–2025）、
  `jp_land_koji_tokyo23.csv`（含 `res_chain_index_1975=100`，**地价一律用連鎖指数口径**）、`jp_land_6city_*.csv`、`jp_osaka_*`；
  CN `cn_price_<city>_nbs70.csv`（月度链式水平）。
- 租金收益率：US `us_rent_yield_<city>.csv` 的 `gross_yield_pct`（1987–2026，含 anchor/estimate 标记）；
  JP `jp_tokyo_gross_yield.csv`（2003–2026）、`jp_yield_investor_survey*.csv`（1999–2026，期待利回り＝NOI 口径）、`jp_yield_pre1999_anchors.csv`；
  CN `cn_rent_yield_<city>.csv`。
- 收入：US `us_income_<city>.csv`（BEA PCPI 1969–2024 主；ACS 中位家庭收入 2010–2024 副）；
  JP `jp_income_tokyo.csv`（**主锚 = 家計一人当たり可処分所得 FY2001–2023；副锚 = 1人当たり都民所得 FY1975–2023**）、`jp_income_tokyo_kakei.csv`（生計分析調査 1975–2025，补充）、`jp_income_osaka.csv`；
  CN `cn_income_<city>.csv`（chain 口径）。
- 宏观：`us_macro_annual.csv`、`jp_macro_annual.csv`、`cn_macro_annual.csv`（名义 GDP、实际 GDP 分列，禁止混用）。
- 股指全收益：`us_sp500_tr_monthly.csv`、`jp_topix_tr.csv`（2006-12–）+ `jp_equity_tr_proxy_jsri.csv`（1952–2012，B 级代理）、`cn_csi300_tr_monthly.csv`。
- 阶段一指标：`phase1/output/table1_recovery.csv`（峰、谷、② 日期）。

## 1. 分层与序列选择（阶段一复盘 §3.1）
- **按资产类别分层**：公寓 / 独栋 / 土地 分别做，不混入同一 panel。
- **跨国对比统一到公寓口径**：東京23区 中古マンション（REINS，主）、纽约 NYXRCSA 公寓、中国 70 城二手住宅。
- **美国内部四城对比用独栋**（Case-Shiller 单城，四城同构）。
- **土地口径一律用継続地点連鎖指数**（平均価格只在需要円/㎡水准时引用）。
- 東京 bust 前 10 年窗口（1981–1991）REINS 无数据：用 不動産経済研究所 東京23区 新築マンション 平均価格（1973–）作**替代序列**并显式标注"新築/戸当たり口径"；同时报告 公示地価 連鎖指数 的 bust 前弹性作为土地口径对照。

## 2. 频率与变量
- 频率：**年度**（收入锚是年度）。月度价格取**年平均**（Case-Shiller、NBS）；日本 REINS 为暦年值。
- 因变量（主）：住宅总回报年增速 `g_TR,t = ln(TR_t / TR_{t-1})`，其中 TR 按 DEFINITIONS(阶段一) §2 公式：年度近似 `TR_t = TR_{t-1} × (P_t/P_{t-1}) × (1 + y_t)`，y_t 为该年毛租金收益率（缺失年份 → 该年 TR 增速标缺失，**不得填充**；若某窗口内收益率全缺，则该窗口只报价格口径并标注）。
- 因变量（副）：价格年增速 `g_P,t = ln(P_t/P_{t-1})`。
- 自变量（三者**分别**回归，服务 H2）：
  (a) 城市收入增速 `g_Y,t = ln(Y_t/Y_{t-1})`（主锚）；
  (b) 全国名义 GDP 增速；
  (c) 全国实际 GDP 增速。
- 名义对名义、实际对实际：主回归全部用**名义**（总回报名义 vs 收入名义 / 名义 GDP）；实际 GDP 那一档单独跑（因变量用 CPI 平减后的实际总回报增速）。

## 3. 窗口（阶段一复盘 §3.2，按 episode 对齐）
- **bust 前**：峰值年（含）往前 10 年，即 [t_peak−10, t_peak]，10 个年度增速观测。
- **bust 中**：(t_peak, t_trough]。
- **恢复后（主）**：[t_②, 最新年]，t_② 取阶段一表 1 的名义回前高年。
- **恢复后（稳健性）**：[t_③, 最新]（③ 未达成的城市标"不适用"）；以及 **触底后全程** (t_trough, 最新]（样本更大，作对照，明确标注它混合了"爬坡"和"回前高后"两段）。
- 每个窗口报告 n（观测数）。**n < 6 的窗口只报点估计与 bootstrap 区间，不报 p 值**，并标注"样本过小"。

## 4. 估计与检验
- 基准：OLS `g_TR,t = α + β·g_X,t + ε_t`，按窗口分别估计。n ≥ 8 用 Newey–West（lag 1）标准误；n < 8 用普通 OLS 标准误 + 1000 次 bootstrap 区间。
- **核心检验（恢复后 β 是否显著低于 bust 前 β）**：
  (i) 单城：两窗口 β 之差的 bootstrap 区间；
  (ii) **美国四城 pooled panel**：`g_TR,it = α_i + β_pre·g_X,it + δ·(g_X,it × Post_it) + γ·Post_it + ε_it`，δ 即 β_post − β_pre，报告 δ 的 t 值（城市聚类稳健 SE）。这是主要的显著性证据来源。
- **水平弹性（补充，抗小样本）**：`ln P_t = a + b·ln Y_t + e_t` 在各窗口内估计 b（报告 b 与 DW 统计量；DW < 1 时标注"可能伪回归"，只作描述）。
- 全部结果同时给价格口径与总回报口径。

## 5. 与股指的比值
- 各窗口：住宅总回报 CAGR ÷ 本国股指全收益 CAGR（同窗口、同为名义）。CAGR 用窗口首末年。
- 日本 2006 年前股指用 JSRI 代理并标注 B 级；同时给"配当なし TOPIX"作下界。

## 6. H3（科技城市 vs 非科技城市）
- 美国：旧金山（Case-Shiller，注意不含 Santa Clara）vs 纽约 / 洛杉矶 / 菲尼克斯；补 San Jose 备选序列（Zillow / FHFA，**价格口径**，无收益率）。比较：② 年数、恢复后 CAGR（价格 & 总回报）、恢复后 β、恢复后 β − bust 前 β。
- 中国：深圳 vs 广州（可加杭州）——**只做状态变量对照**：bust 前 10 年价格 CAGR、bust 前收入增速、bust 前 β（2011–2021 可算）、迄今跌幅、当前收益率。不做恢复相关的拟合。
- 日本：無同国科技/非科技对照城市可用（大阪 1990 episode 无同源公寓序列），标"不可检验"。

## 7. 中国状态变量对照（不做硬拟合）
对北上广深杭：bust 前 10 年 β（2011–2021，n=10）、bust 前价格 CAGR、bust 前收入 CAGR、名义/实际 GDP CAGR、GDP 平减指数、当前跌幅区间（三口径）、当前租金收益率、租金收益率 − 国债收益率。与日美各 episode 的同名变量并列成表，标出中国落在日美区间内/外。

## 8. 产出约定
- 数据面板：`phase2/data/panel_<country>_<unit>.csv`（列名统一：year, price, price_growth, yield, tr, tr_growth, real_tr_growth, income, income_growth, ngdp_growth, rgdp_growth, cpi, window_label, yield_flag）。
- 回归结果：`phase2/data/betas_<country>.csv`（episode, city, asset, window, regressor, dep_var, n, beta, se, ci_lo, ci_hi, p, method, dw）。
- 脚本：`phase2/scripts/`，全部 `python3` 可重跑（可 `pip install --user statsmodels`）。
- 笔记：`phase2/notes/<worker_id>.md`。JSON：`phase2/data/<worker_id>.json`。
- 最终文档由主控撰写：`phase2/output/PHASE2_传导弹性检验.md`（含复盘）。
