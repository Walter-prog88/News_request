# 数据截止日纪律:先截断再读,财报版本按 PIT 取

> 2026-10-10,前缀 fixb。起因是上一轮审计的两条发现:`src/afactor/impact.py:121` 的 `ND.raw_close`
> 会把含 2026 行的 `daily_basic.csv` 整表读进来;`FA._real(p, "mv") -> ND.total_mv` 也整表读 daily_basic,
> 而且财报股本按"每个报告期留最晚公告的版本"去重,2026 年的更正公告会改写去重结果。
> 应有的原则是"先截断再读"。同时本轮前向检验需要有合法途径读 2026 年数据。
>
> 代码:`src/afactor/config.py`(截止日与切换接口)、`src/afactor/newdata.py`、`src/afactor/fundamental.py`、
> `src/afactor/factors.py`(只改了缓存键)。
> 测试:`tests/test_data_cutoff.py`(14 项,第二轮新增 5 项;读真实数据,约 2.5 分钟)。另外 `tests/test_barra_risk.py` 现在自己插入 `src`。
> 影响量化:`scripts/fixb_cutoff_impact.py`。账本:`research/ledger/fixb_impact_summary.csv`、
> `fixb_fundamental_columns.csv`、`fixb_cutoff_rowcounts.csv`、`fixb_comp_v5_downstream.csv`、`fixb_pit_dependency_gap.csv`。
> 新缓存:`cache/_tmp/comp_v5_score_open10_fixb.parquet`(md5 e53dca04…)。旧文件 `comp_v5_score_open10.parquet`
> 没动,md5 仍是冻结值 d011468f…
>
> 没有改 strategy.py / index_proxy.py / optimizer.py / impact.py / earnings_events.py。引擎金标准
> `tests/test_engine_regression.py` 28 项全部通过,逐位一致。
>
> **第二轮对抗审查后修订(同日)**:1 条 P1、5 条 P2,处理见文末"审查记录"。要点:新增**源覆盖守卫**(放宽截止日后,
> 源文件覆盖不到的请求日期默认报错,不再静默 ffill / 填 0);版本 PIT 改为 **as-of 快照**,候选行只由 t 时已公告的版本决定;
> 股本 0 占位一律视为未披露;旧脚本的 daily_basic 注入与断言重新生效。新增 `scripts/fixb_rolling_origin.py`(滚动原点检验)、
> `scripts/fixb_review2_checks.py`(逐条复核)和账本 `fixb_rolling_origin*.csv`、`fixb_review2_checks.csv`;
> `scripts/pead_data_acceptance.py` 改了一行(§1.3)。COMP_v5 新打分与上一版逐位相同;真实市值的变化从 11 只增加到 29 只(§3.2)。

---

## 0. 结论先行

| 问题 | 结论 | 出处 |
|---|---|---|
| 改后研究流程还能读到 2026 行吗 | 读不到。10 个源在读入后立即截到 2025-12-31,测试逐个断言 | `tests/test_data_cutoff.py::test_default_loaders_have_no_2026_rows` |
| 引擎会受影响吗 | 不会。金标准逐位一致;ST 面板、未复权价、流通股本、换手率、龙虎榜、宽基指数在研究期**0 格变化** | `fixb_impact_summary.csv` |
| COMP_v5 打分变了吗 | 变了,但幅度很小。决策池内每日截面 Spearman 最低 0.99993;99% 的格子 \|Δ\| ≤ 2.2e-4;\|Δ\| > 0.1 的格子占 0.008%;最大 0.221。第二轮修订后与上一版新打分逐位相同 | `fixb_impact_summary.csv`、`fixb_comp_v5_downstream.csv`、`fixb_review2_checks.csv` |
| 下游结论要重跑吗 | **不需要**。标准排雷的年化收益在两个窗口分别变化 +0.001pp 和 −0.003pp;Top20 和 Top50 变化都在 ±0.1pp 以内 | `fixb_comp_v5_downstream.csv` |
| Barra 暴露 | 截面上普遍有极小的漂移(q99 ≤ 0.015),这是标准化带来的。个别股票日变化大,Growth 最大 6.0(截尾两端互换) | `fixb_impact_summary.csv` |
| 真实市值 | 29 只股票变化(上一版 11 只)。主要是新股:上市到首份正式股本之间,旧口径用"股本 0 占位 → 流通股本"兜底,把市值低估成 1/3~1/20;新口径用招股书版本或上一期股本。其中 11 只在之后的日子里恢复了 `total_mv` 的流通股本外推,随之带上它既有的解禁高估(2,989 格,新/旧中位 1.345) | §3.2、§4 第 8 条 |
| **附带发现:`ND.total_mv` 系统性高估** | 既有问题,与截止日无关,本次不修。流通股本外推只在总股本变化时重新锚定,限售解禁会让它持续累积:研究期 13.6% 的格子是"价 × 总股本"的 1.5 倍以上,1,318 只股票持续一年以上;例 600339.SH 全程 3.60 倍。Barra、中证1000 代理、容量、指增都用它,建议单独立项 | §4 第 8 条,`fixb_review2_checks.csv` |
| 财报 PIT 精确吗 | 取值不前视:21 个滚动原点、14 个映射正确的列,"用到更新报告期 / 更新版本"都是 0,版本处理不再依赖未来修订(0 格)。但会偏陈旧:财报季 0.24%~2.07% 的股票-日比当时可得的信息更旧 | §4 第 2 条,`fixb_rolling_origin_summary.csv` |
| 前向检验怎么读 2026 | `with config.set_data_cutoff("2026-10-09", reason=...)`,只允许文件名以 `fwd`/`test_` 开头的代码放宽。**能放宽不等于有数据**:data_8:21 只到 2026-08 中下旬 | §5 |
| 源文件覆盖不到的日期怎么办 | 默认抛 `SourceCoverageError`(ST 名单只到 07-03、龙虎榜到 08-21、资产负债表到 06-13 等)。要么补数,要么在 `accept_source_gap(源, reason)` 内显式接受 | §5.2,`fixb_review2_checks.csv` |

上一轮审计估计的数值影响是 max\|Δz\| 2.7e-5。本次逐格比较了全部研究期面板,实测影响比这大。原因是除了"2026 修订把行推迟"之外,
还有另外两条机制,都来自"整表读入后取最新":一是部分表修订的选择效应,二是公司类型快照,见 §3.1。以本报告的数字为准。

---

## 1. 改了什么

### 1.1 config.py:截止日与切换接口

| 名字 | 作用 |
|---|---|
| `DATA_CUTOFF = RESEARCH_END` | 默认数据截止日(2025-12-31) |
| `data_cutoff()` | 当前生效的截止日。加载器每次读取时调用,不要在 import 时缓存它 |
| `set_data_cutoff(date, reason)` | 上下文管理器,只供前向检验使用。规则见 §5 |
| `assert_within_cutoff(dates, what)` | 面板日期晚于截止日就报 `PermissionError`。否则截止日之后的行会被静默 ffill 成旧值或留空 |
| `PRICE_ADJ_BASE_END = "2026-08-31"` | 价格面板的前复权基准月末。只供 `build_market_cap` 还原送转使用,见 §1.3 |
| `CUTOFF_LOG` | 每次进入或退出切换都追加一行审计记录:`logs/data_cutoff_switch.log` |
| `record_source_end(源, 日期)` / `source_end(源)` | (第二轮审查 P1 新增)各加载器读入原始文件后、截断前登记该源的原始最晚日期。只是一个日期标量,不进入任何数值 |
| `check_source_coverage(源, dates, what)` | 面板请求日期晚于源原始最晚日期时默认抛 `SourceCoverageError`。研究期所有源都覆盖到 2026 年,恒不触发 |
| `accept_source_gap(源, reason)` | 上下文管理器:显式接受所列源的覆盖缺口,只发 `RuntimeWarning`;进入时打印并写审计行。接受后各加载器的行为见 §5.2 |

### 1.2 newdata.py:每个加载器读入后立即截断,之后才去重、透视、ffill

| 源 | 截断键 | 截断方式 |
|---|---|---|
| daily_basic.csv(`_load_daily_basic`,服务 raw_close / total_mv / circ_mv / turnover_rate / float_share) | trade_date | 丢弃截止日之后的行 |
| forecast.csv / express.csv(`_read_dated_csv`,服务 surprise_panels) | ann_date | 丢弃截止日之后的行,**然后**再按 (股票, 报告期) 保留最晚的版本 |
| top_list.csv(服务 toplist_panels) | trade_date | 丢弃 |
| stock_basic.csv(`_stock_basic`,服务 industry_labels / listing_info) | list_date / delist_date | list_date 在截止日之后的行整行丢弃;delist_date 在截止日之后的置空 |
| 宽基指数快照(`index_daily`) | date | 每个序列先截断,再拼表 |

- 进程内缓存:默认截止日下键就是旧名字(`"db"` / `"ind"` / `"idx"`),其它截止日下键为 `(名字, 截止日)`。
  切换截止日后不会串用另一个截止日的结果,同名只保留一份。
  (第二轮审查 P2 修订。上一版默认键也是 `("db", 截止日)`,于是下面三处旧代码往 `_CACHE["db"]` 注入截断版、
  再断言 `_CACHE["db"]` 最大日期 ≤ 2025-12-31 时,加载器根本不读这个键:断言检查的只是注入的那份,守卫形同虚设,
  daily_basic 还会多占一份约 1GB 内存。默认截止日下没有泄漏,因为加载器自己也截断。)现在默认键沿用旧名字,
  这些注入被加载器真正使用,断言检查的就是加载器用的那份;注入的表若含截止日之后的行,`_load_daily_basic` 会再截一次并写回。
  旧约定的使用方:`index_proxy._sealed_newdata`(注入 / 弹出)、`scripts/pead_build_batches.sealed_daily_basic`(注入)、
  `scripts/pead_run_arms.py:78`(断言)、`scripts/run_index_enhance_pead.seal_newdata / assert_sealed`(注入 / 断言)。
  测试 `test_legacy_daily_basic_injection_is_effective`。
- `_piv` / `surprise_panels` / `toplist_panels` 加了日期守卫。`_piv` 在请求日期超过源文件覆盖范围时会发 RuntimeWarning
  (覆盖之外本来就是 NaN)。
- **源覆盖守卫(第二轮审查 P1)**:`_read_dated_csv` / `_load_daily_basic` 读入后登记原始最晚日期;`earnings_surprise_events` /
  `surprise_panels` / `toplist_panels` 的请求日期晚于源原始最晚日期时默认抛 `SourceCoverageError`。接受缺口后:
  龙虎榜(脉冲型)覆盖之外的行置 NaN,不再 `fillna(0)` 成"全市场未上榜";预告 / 快报(状态型)沿用最后已知值,
  `days_since_surprise` 在覆盖之外置 NaN。
- 新增 `data_coverage()`,返回当前截止日下每个源实际覆盖到哪天,以及原始文件最晚日期 `source_end`(§5.2)。

### 1.3 fundamental.py:财报先截断,版本按 PIT 选

1. **`_load` 读入后立即截断**:只保留可见日 avail = max(ann_date, f_ann_date) ≤ 截止日的版本,
   返回截止日前的**全部版本**,不再在读入时去重。签名 `(name, items)` 与旧版兼容,
   所以 index_proxy 和 pead 脚本对它的包装照常生效。
2. **版本选择改为 PIT**(`build_report_level`):时点 t 用 t 时已经公告的最新版本。
   - 数据源每个 (股票, 报告期) 一般只存最新一版。截止日前只有资产负债表有多版本:377 个报告期、240 只股票,
     最多 4 版(`fixb_cutoff_rowcounts.csv`)。
   - 实现分两部分。最终快照是每期在截止日前的最新版,`pit_vintage = 99`。在不截断的条件下,除股本 ≤ 0 的占位行
     (见下)外,它与旧版报告表逐行一致。
   - **as-of 快照**(第二轮审查 P2 后改写):某个 (股票, 报告期, 表) 的版本 v_k 在 [avail(v_k), avail(v_k+1)) 区间是
     "当时最新、后来被替代"的版本。对落在这种区间里的每个公告日 tau(截止日前 217 只股票、424 个快照日,单股最多 13 个;另 23 只股票的多版本是同日重发,前一版从未"当时最新",不需要快照),
     每个 (报告期, 表) 取 avail ≤ tau 的最新版本,整体派生一遍;tau 时某张表尚无任何版本的报告期取首版,
     其可见日随之推迟,和最终快照一样遵守"三表取最晚"。只追加与最终快照不同的行(截止日前 625 行、193 只股票),
     `pit_vintage` = 快照在本股票内的先后序号。
   - `_pit_daily` 仍取可见日最新的一行;可见日相同的并列行取 `pit_vintage` 最大的(最晚的快照,最终快照为 99)。
     快照行的可见日是它用到的全部版本公告日的最大值,所以可见日 ≤ t 的候选行只用到 avail ≤ t 的版本,**取值不前视**;
     同一报告期的候选行快照越晚可见日越晚,选中的正是"t 之前最后一个公告日的快照",即 t 时的最新版本组合。
     **候选集合只由 avail ≤ t 的版本决定,与 t 之后是否还有修订无关**。
     上一版按"第 k 版"跨报告期拼快照 S_k:两个报告期交错修订时,"旧期新版 + 新期首版"这类组合要等未来某期被修订
     才会出现或消失,取值虽不前视,处理方式却依赖未来。合成用例 `test_asof_snapshots_match_rolling_origin_truth`
     对每个 t 要求"默认报告表在 t 日的取值 = 只用 avail ≤ t 的版本重建的取值":新实现通过;把实现换回 S_k 后该用例失败
     (首个不一致在交错修订处:`total_assets_avg` 1065 vs 真值 1198.75)。真实数据上的滚动原点检验见 §4 第 2 条。
   - **股本 ≤ 0 的占位行一律视为未披露**(第二轮审查 P2 后改写):截止日前最终版有 598 行、425 只,多为上市前的招股书期间;
     被后续版本替代的另有 19 行。`_pit_daily` 跳过它、沿用上一期股本。上一版只在"之后会被修订"的早期版本上这样处理,
     从不修订的 0 行留在最终快照里(经 `total_mv` 的 `replace(0)` 回落到流通股本),t 时用哪种处理取决于 t 之后有没有修订。
     改后在上一版"把 0 当成真实值"会出错的合成用例上也通过(换回旧处理后,首个不一致是 2022-05-06 股本 1e8 vs 真值 0)。
   - 消费方。只需要每期一行的要先筛 `pit_vintage == 99`(`_final_rows`):`build_event_features`、`comp_type_panel`、
     `scripts/pead_data_acceptance.panel_consistency`(第二轮审查 P2:它按 (股票, 报告期) merge 报告表,没筛会让多版本报告期
     重复计数;已加一行 `FU._final_rows(...)`)、`scripts/fixb_cutoff_impact.py`。经 `_pit_daily` 用全部行的:
     `newdata.total_mv`、`factors._fu`、`build_market_cap`、`index_proxy`、`scripts/run_index_proxy.py`。
3. **ST 名单**(`_read_st`)读入后截断。
4. **`build_market_cap` 的唯一例外**:送转倍数用截到前复权基准的股本序列识别(`_adjbase_quarterly_shares`),
   PIT 股本和"t 时可见哪一期"仍用截断后的报告表。原因是 close_px 的前复权基准在 2026-08,研究期价格已经按 2026 年的
   送转下调过,还原必须用同一基准的送转记录。如果也截到 2025-12-31,2026 年送转的 7 只股票(倍数中位 1.20、最大 2.00)
   整个研究期的近似市值都会系统性偏低,而"2026 年会送转"本身与 2025 年的表现相关,这样反而引入与未来相关的误差
   (`fixb_cutoff_rowcounts.csv`,"build_market_cap 送转还原"行)。这些股本只进入送转倍数,不进入其它字段。
   在不截断的条件下,它与旧版逐格相同。
   **这个例外并不完整**(第二轮审查 P2):资产负债表原始文件最晚报告期是 2026Q1(可见日到 2026-06-13),所以只识别
   2026Q1 股本里已经反映的送转,也就是上面那 7 只(股本跳变出现在 2025Q4 → 2026Q1)。年报送转大多在 5~7 月除权,
   2026-04~08 除权的送转已经压低了前复权价,却不在股本序列里,这些股票整个研究期的近似市值仍按送转比例偏低。
   量级用 2021-2025 每年同期的送转频率估计,见 §4 第 5 条。这是既有问题,新旧代码相同;`build_market_cap` 本来就标注为近似。
5. `_pit_daily` / `build_st_panel` / `build_event_features` 加了日期守卫。
   **源覆盖守卫(第二轮审查 P1)**:`_load` / `_read_st` 读入后登记原始最晚日期;`build_st_panel`(ST 名单)、
   `_pit_daily` 与 `build_event_features`(三张财报表)的请求日期晚于源原始最晚日期时默认抛 `SourceCoverageError`。
   `newdata.total_mv`、`build_market_cap`、`factors._fu / _fu_event / _real` 经由它们同样受保护。接受缺口后沿用最后已知值。
6. 诊断开关 `PIT_EARLY_VERSIONS`(默认 True):只供影响量化把"截断"和"版本 PIT"拆开,研究与前向检验不要关。
   关掉时既不追加 as-of 快照,也不做股本 0 占位处理,复现"仅截断"口径。
7. `avail_sue` 改为 sue_bt 的真实依赖可见日:P..P-13 共 14 个季度(上一版 rolling(12) 漏了 P-12 / P-13,
   注释"12 个季度全部已公告"不对);新增 `avail_streak`(earn_streak,P..P-16 共 17 个季度)。两列都只供诊断和将来修复
   §4 第 1 条使用,`_pit_daily` 目前不读它们,所以不改变任何面板数值。

### 1.4 factors.py

`_fu` / `_fu_event` / `_nonfin` / `_nd` / `_real` 的面板缓存键加了截止日后缀。默认截止日下后缀为空串,
键与旧版完全相同。这样同一个面板对象不会串用两个截止日的结果。没有其它改动。

### 1.5 有意没改的

| 位置 | 原因 |
|---|---|
| `impact.py:121`(审计原发现) | 它调用的 `ND.raw_close` 现在读的就是截断后的 daily_basic,问题在源头修掉了。`nd_raw_close` 在研究期 0 格变化 |
| `earnings_events.py`(PEAD) | 已经自己在读入后按 `RESEARCH_END` 截断,并做了版本标注,合规。但截止日是写死的,前向检验如果要用 PEAD 事件,需要另行扩展(§5.3) |
| `index_proxy.py` | 它的 `_sealed_newdata` / `index_close` 自带截断,默认截止日下与本修复一致 |
| `trader_choice_*` / `trader_chain_*` | 独立研究线,有自己的加载器和 source_cutoff 机制 |
| `stock_sw_industry.csv`、stock_basic 的行业字段 | 是当前快照,没有日期,无法截断(已知局限) |

---

## 2. 截掉了多少

`fixb_cutoff_rowcounts.csv`:

| 源 | 总行数 | 截止日后(被截) | 说明 |
|---|---|---|---|
| daily_basic.csv | 7,453,930 | 797,190(10.7%) | |
| forecast.csv | 111,463 | 13,080 | 其中 4 个(股票, 报告期)在截止日前有版本、截止日后又有修正。旧口径会让 2026 年的修正把截止日前的版本挤掉 |
| express.csv | 16,459 | 1,979 | |
| top_list.csv | 121,929 | 13,726 | |
| st_info_latest.csv | 321,993 | 23,492 | |
| stock_basic.csv | 5,547 | 上市 48 / 退市 19 | 截止日后上市的整行丢弃,截止日后的退市日置空 |
| 利润表 | 217,381 | 13,270 | 其中 474 行的报告期 ≤ 2025Q3,多为 2026 年的更正或补报 |
| 现金流量表 | 205,705 | 12,864 | 其中 275 行报告期 ≤ 2025Q3 |
| 资产负债表 | 211,966 | 12,227 | 其中 393 行报告期 ≤ 2025Q3;截止日前有 377 个多版本报告期 |

---

## 3. 影响:研究期(2020-01-02 ~ 2025-12-31)面板逐格对比

"旧"是 git HEAD 的四个文件。"仅截断"是新代码在 `PIT_EARLY_VERSIONS=False` 下的结果。"新"是新代码默认设置下的结果。
三者都在同一个 Panel(1455 × 5565)上计算(`fixb_impact_summary.csv`)。
配方核对:旧代码重算的 COMP_v5 打分与冻结缓存 0 格差异,说明比较基准可靠。

### 3.1 三条机制(都来自旧口径"整表读入后取最新")

1. **2026 修订把截止日前的数据推迟到 2026**。例如 002726.SZ:利润表从 2021Q1 起有 19 期、资产负债表从 2021Q2 起有 13 期,
   都在 2026 年被重新发布(f_ann_date 在 2026)。旧口径只留最新版,这些报告期的可见日都变成 2026,研究期内看不见;
   而"三表取最晚"又让同期另外两张表被一起推迟。新口径截掉 2026 版本后,截止日前的版本恢复可见,
   该股真实市值有 1135 天变化,最大 8%。
   - 部分表修订是同一机制的变体:只有一张表在 2026 被修订时,旧口径会把整期三张表一起推迟,新口径只让那张表缺失。
2. **公司类型快照**。`comp_type_panel` 取每只股票最后一行的 comp_type 作为整段历史的标签。000712.SZ(锦龙股份)、
   600901.SH(江苏金租)、603093.SH(南华期货)在 2026-08 的半年报行里写成了 1(一般企业),
   之前 30~40 期都是 2 或 4(金融)。旧口径用 2026 年的这一行给 2020~2025 全程贴"非金融"标签,新口径恢复为金融。
   因此这 3 只股票的杠杆率、周转率等"仅非金融"因子在研究期全程变成 NaN(`fa_fu_leverage` 有 4365 格从有值变为 NaN)。
3. **版本 PIT**。时点 t 用 t 时已公告的版本。典型的是 2025 年新股:招股书版的 2024 年报在上市前公告,正式年报在 4 月公告。
   旧口径只留 4 月版,上市到 4 月之间看到的是上一期股本为 0 的占位行,真实市值退化为"未复权价 × 流通股本",被严重低估。
   例如 301275.SZ 在 2025-03-11 旧口径是 21.3 亿,新口径是 254.4 亿(总股本 3.80 亿 × 66.92 元)。
   第二轮修订后,**从未被修订的**股本 0 占位行也视为未披露(§1.3),于是同类问题在更多新股上被纠正:例如 688802.SH
   上市后 11 个交易日(2025-12-17 ~ 12-31)旧口径 150.5 亿(流通股本兜底),新口径 2,988 亿(招股书总股本 3.6 亿 × 829.9 元);
   301165.SZ 在 2022-11-21 ~ 2023-03-30 从 21.6 亿(流通股本兜底)改为 187.5 亿(上季总股本 5 亿 × 价;上市后实际 5.68 亿)。
   但同一只股票在 2023-03-31 ~ 2025-08-21,0 股本造成的 NaN 段消失后,`total_mv` 恢复了流通股本外推,
   随之带上它既有的解禁高估(§4 第 8 条):例如 2024-06-03 新口径 219.1 亿,"价 × 总股本"是 185.3 亿。
   这类"恢复外推"的格子有 2,989 个、11 只(`fixb_review2_checks.csv`)。
   另一个例子是 003031.SZ 的 2022 年报资产负债表,2023-03-31 首发(总资产 17.79 亿),2024-04-26 追溯重述(43.42 亿)。
   新口径在 [2023-03-31, 2023Q1 报告) 用首发版,2023 年内不会出现 2024 年的重述值(有测试断言)。

### 3.2 关键对象

| 对象 | 变化格数 | 最大差异 | 截断部分 | 版本 PIT 部分 |
|---|---|---|---|---|
| ST 面板、未复权价、流通市值、流通股本、真实换手率、龙虎榜 10 个面板、宽基指数 | **0** | 0 | 0 | 0 |
| 真实总市值 `ND.total_mv` | 5532 格(占 0.08%),29 只股票(上一版 1864 格、11 只) | 新/旧 = 19.8 倍(688802.SH 新股,见 §3.1 第 3 条) | 3 只,最大相对差 0.226(002726.SZ 等) | 26 只:25 只新股,上市到首份正式股本之间旧口径用流通股本兜底,新/旧最大 3.0~19.8 倍(低估成 1/3~1/20);其中 11 只之后恢复了 `total_mv` 外推(2,989 格,新/旧中位 1.345,见 §3.1 第 3 条);另 1 只 603125.SH 差异 < 5e-4 |
| 近似市值 `build_market_cap` | 1375 格,另有 760 格从 NaN 变为有值(上一版 181);30 只股票 | 相对 0.235 | 4 只 | 26 只(全部是 NaN 变有值,或相对差 < 5e-4) |
| fu_roe / fu_rev_yoy | 11718 / 8226 格,各 51 只 | 0.62 / 3.09 | 40 / 44 只 | 11 / 7 只 |
| 业绩预告面板 | fc_pchange 12 格、days_since_surprise 66 格 | 全部在 2025-11 ~ 12 | 全部 | 0 |
| 报告表 83 列 × `_pit_daily` | 合计 0.131% 的格子变化(或 NaN 翻转);total_share 224 只,其余每列至多 121 只 | 见账本 | 0.092%,每列至多 51 只 | 0.039%;total_share 219 只,其余每列至多 101 只 |

报告表逐列的数字见 `fixb_fundamental_columns.csv`。total_share 的版本 PIT 一步里有 26,479 格从 0 变为 NaN:上市前的股本 0 占位
现在视为未披露(§1.3)。下游 `total_mv` / `build_market_cap` 本来就把 0 当缺失,这部分不改变市值。
total_share 的 max_rel_diff 是 3.9e21,这是旧值为 0 时相对差无意义造成的,不是真实量级。

### 3.3 COMP_v5 打分(open10,24 个成员)

| 步骤 | 变化格数 | 最大 \|Δ\| | q99 \|Δ\| | q99.99 \|Δ\| | \|Δ\| > 0.1 的占比 |
|---|---|---|---|---|---|
| 合计(旧 → 新) | 5,107,164 / 5,889,141 | 0.221 | 2.2e-4 | 0.091 | 0.008% |
| 截断 | 5,106,312 | 0.221 | 2.1e-4 | 0.091 | 0.008% |
| 版本 PIT | 1,980,624 | 0.059 | 4.7e-5 | 1.1e-4 | 0 |

大部分格子都有变化,是因为 3 个基本面交互成员先做截面排序和标准化,少数股票变化会让同日所有股票的 z 值都微移;
真正有意义的变化集中在几十只股票上。打分的 NaN 位置完全不变(非空率都是 0.72732)。

下游(`fixb_comp_v5_downstream.csv`,V1000 open10,旧成本口径,信号样本内):

| 方案 | 2021-2023 年化 旧 → 新 | 2024-2025 年化 旧 → 新 | 日收益差最大绝对值 |
|---|---|---|---|
| 标准排雷 20/35、5 日调仓 | 9.668% → 9.669%(+0.001pp) | 28.266% → 28.263%(−0.003pp) | 2.8e-5 |
| Top20、5 日调仓 | 2.659% → 2.626%(−0.034pp) | 26.965% → 27.057%(+0.092pp) | 7.4e-4 |
| Top50、5 日调仓 | 7.021% → 6.999%(−0.022pp) | 30.215% → 30.141%(−0.074pp) | 5.1e-4 |

决策池内每日截面 Spearman(旧 vs 新)最低 0.99993、中位 0.999995。所有差异都远小于各报告里结论所用的量级(pp 级超额、t 值),
**下游结论不需要重跑**。新打分另存为 `cache/_tmp/comp_v5_score_open10_fixb.parquet`;冻结的旧文件保留,
PEAD 的冻结输入 md5 不受影响。今后新开的研究建议用新文件,并在报告里注明。

### 3.4 Barra 暴露(风险模型 v3 参数,覆盖池)

| 风格 | 最大 \|Δ\| | q99 \|Δ\| | \|Δ\| > 0.1 占比 | 主要来源 |
|---|---|---|---|---|
| Size | 2.19 | 0.0046 | 0.039% | 版本 PIT(新股市值修正) |
| NonlinSize | 4.12 | 0.0142 | 0.032% | 版本 PIT |
| BookToPrice | 1.31 | 0.0011 | 0.020% | 截断 |
| EarningsYield | 5.24 | 0.0027 | 0.054% | 截断 |
| Growth | 6.00 | 0.0058 | 0.102% | 截断(±3 截尾两端互换) |
| Leverage | 2.44 | 0.0039 | 0.022% | 截断(含 3 只金融股全程变为 NaN)+ 版本 PIT |
| Beta / ResidVol / Liquidity / Momentum | ≤ 0.0094 | ≤ 0.0018 | 0 | 市值权重的微小变化 |

(第二轮修订前 Size / NonlinSize 的 q99 是 0.0022 / 0.0027、\|Δ\| > 0.1 占比都是 0.004%;增加的部分来自 §3.2 真实市值多变化的 18 只新股,包括上市窗口的低估纠正和恢复外推两部分。)

`cache/_tmp/risk_open10_*` 风险模型缓存和 `proxy_inputs_*` 指数代理输入缓存都是用旧代码建的,本次没有重建。
如果重建,数值变化就是上表的量级,不改变 RISK_MODEL.md、CSI1000_PROXY.md 的结论。

---

## 4. 已知局限与发现的既有问题(本次未修)

1. **衍生列的映射可见日没有覆盖全部依赖**(既有前视)。`_avail_col` 按后缀映射可见日:`sue_bt` / `sue_ta` / `earn_streak`
   落到 `avail`,只是本期的可见日,`earn_vol_q` / `d_*_q` 落到 `avail_q`。但它们实际依赖 5~17 个季度。

   | 列 | 依赖晚于映射可见日的行数 | 占比 | 中位晚多少天 | 滚动原点:t 日用到 t 之后数据的股票-日(§4 第 2 条) |
   |---|---|---|---|---|
   | sue_bt | 3794 | 2.9% | 302 | 1306 / 96476(1.35%) |
   | sue_ta | 4435 | 2.8% | 337 | 1369 / 105797(1.29%) |
   | earn_vol_q | 3778 | 2.4% | 243 | 821 / 105488(0.78%) |
   | earn_streak | 2674 | 2.4% | 285.5 | 870 / 90888(0.96%) |
   | d_ni_q / d_revenue_q / d_operate_profit_q | 449 | 0.27% | 227 | d_ni_q:70 / 109022(0.06%) |

   (前四列出处 `fixb_pit_dependency_gap.csv`,最后一列出处 `fixb_rolling_origin_summary.csv` 的 lookahead_suspect / n。)
   sue_bt / earn_streak 的真实依赖改用报告表里按季度网格算的 `avail_sue`(P..P-13,14 个季度)/ `avail_streak`
   (P..P-16,17 个季度)。上一版用的 `avail_sue` 是 rolling(12),漏了 P-12 / P-13,行数是 3618 / 2665(第二轮审查 P2);
   现在 avail_sue 的 min_periods 从 12 改为 1,有值的行从 127,433 增加到 131,157,所以两版行数不是同一个分母。
   earn_vol_q 仍按行 rolling 近似。
   **修复建议**(另行立项,会改变因子层 PEAD 族的研究期数值):sue_bt → `avail_sue`,earn_streak → `avail_streak`,
   sue_ta → max(`avail_yoy`, `avail_avg`),earn_vol_q → `avail_q` 的 rolling(8) 最大值,d_*_q → `avail_yoy`。
   影响 `pead_sue_bt`、`pead_sue_ta`、`pead_jump_combo` 等因子层 PEAD 族,不影响 COMP_v5,也不影响 PEAD 正式研究,
   后者走 `earnings_events.py`。
2. **财报取值不前视,但会偏陈旧**(第二轮审查 P2:上一版 §1.3 只说"交错修订时可能用旧版",低估了)。
   滚动原点检验 `scripts/fixb_rolling_origin.py`:在 21 个原点 t(2021~2025 每年 03-31 / 04-30 / 08-31 / 10-31,加 2025-12-31),
   用截止日 = t 重建报告表作真值,与默认构造在 t 日按 `_pit_daily` 的选行规则逐股比较(`fixb_rolling_origin_summary.csv`):

   | 列 | 股票-日 | 默认更新(疑似前视) | 同期快照行取值不同(版本处理依赖未来) | 默认更旧或缺值(偏陈旧) |
   |---|---|---|---|---|
   | total_share | 117,729 | 0 | 0 | 280(0.24%) |
   | total_assets / total_hldr_eqy_exc_min_int | 118,396 | 0 | 0 | 835(0.71%) |
   | n_income_attr_p_ttm / revenue_ttm | 106,521 | 0 | 0 | 851(0.80%) |
   | n_cashflow_act_ttm | 103,400 | 0 | 0 | 1,407(1.36%) |
   | total_hldr_eqy_exc_min_int_avg / total_assets_avg | 108,304 | 0 | 0 | 1,494(1.38%) |
   | n_income_attr_p_q / revenue_q | 114,904 | 0 | 0 | 1,577(1.37%) |
   | revenue_yoy / n_income_attr_p_yoy | 108,901 / 109,021 | 0 | 0 | 1,412(1.30%)/ 1,361(1.25%) |
   | total_assets_yoy / inventories_yoy | 113,700 / 111,041 | 0 | 0 | 2,358(2.07%)/ 2,270(2.04%) |

   14 个映射正确的列:"默认用了更新的报告期"和"同期最终快照行取值不同"都是 0,**没有前视**;"同期 as-of 快照行取值不同"
   修复前审查实测 11 格(total_assets_yoy 6、inventories_yoy 5),修复后 0。剩下的差异全部是保守方向,
   在财报季有 0.24%~2.07% 的股票-日比当时实际可得的信息更旧。三条机制(已写进 `fundamental.py` 模块文档):
   - (a) 数据源用更正版**覆盖**了原行。截止日前 f_ann_date > ann_date 的行:利润表 4,033、资产负债表 3,925、
     现金流量表 2,358(`fixb_cutoff_rowcounts.csv`,与审查实测一致)。原版本的公告日已不可知,整期被推迟到更正日,依赖它的单季 / TTM / 同比随之推迟。
   - (b) 三表取最晚:一张表被更正,整期连同另外两张表一起推迟到更正日;此时下一期往往已经公告,
     该期的行被 `_pit_daily` 的 frontier 永久丢弃。
   - (c) Q1 的单季值等于累计值,并不依赖上一期,但 `avail_q` 仍取 max(本期, 上期)。例:000547.SZ 的 2022 年报利润表
     在 2025-01-21 更正,研究期内 2023-04-29 到 08-31 的 n_income_attr_p_q 一直停在 2022Q3(审查实例)。
   (c) 可以单独放宽,(a)(b) 需要原始版本数据。这些都是既有问题,新旧代码相同,本次不改。
3. `stock_sw_industry.csv` 和 stock_basic 的行业字段是当前快照,截止日只能去掉"截止日时尚未上市"的股票。
   实测 48 只 2026 年上市股里没有独有的行业,所以整数编码与截断前一致。
4. `comp_type_panel` 仍是静态标签(取截止日前最后一期),不随时间变化。
5. **`build_market_cap` 的送转倍数越过了截止日**(§1.3 第 4 条),这是有意保留的例外,**而且并不完整**(第二轮审查 P2)。
   资产负债表原始文件最晚报告期是 2026Q1,只能还原 2026Q1 股本里已经反映的送转(7 只);2026-04~08 除权的送转
   已经压低了前复权价,却不在股本序列里。用研究期同期的频率估计量级(`fixb_cutoff_rowcounts.csv`,
   "build_market_cap 送转还原缺口估计"行,按 `_songzhuan_factor` 同一判据):

   | 年份 | Q1→Q2 送转跳变 | 倍数中位 | Q2→Q3 送转跳变 |
   |---|---|---|---|
   | 2021 | 283 只 | 1.40 | 56 只 |
   | 2022 | 333 只 | 1.40 | 66 只 |
   | 2023 | 334 只 | 1.40 | 73 只 |
   | 2024 | 213 只 | 1.40 | 35 只 |
   | 2025 | 193 只 | 1.40 | 35 只 |

   也就是说,大概率有两三百只股票(约占有股本的股票的 4%~6%)在整个研究期的**近似市值**按送转比例(中位 1.4 倍)偏低。
   上一版说"2026 年送转的 7 只"把例外描述得比实际完整。这是既有问题,新旧代码相同。受影响的是 `p["market_cap"]`
   及其下游(`fu_size`、`fu_bp` / `fu_ep` 等分母);Barra 风格因子和 `rt_size` 用的是 `ND.total_mv`(真实市值),不受影响。
   需要市值的研究和前向检验都应改用 `ND.total_mv`。
6. 预告和快报仍"每期留最晚版本":被修正的原版本在 [原公告, 修正公告) 区间不可见。这是延迟,不是前视,沿用了原口径。
7. Panel 自己的派生缓存(`p["market_cap"]`、`p["is_st"]`)没有按截止日分键。不要把研究期 Panel 对象拿进前向上下文里计算这两项,
   前向检验应新建面板。
8. **真实市值的日频外推会系统性高估(既有问题,第二轮复核时发现,本次不修,严重程度高)**。**【2026-10-10 已单独修复,前缀 mvfix,见 `research/TOTAL_MV_FIX.md`:> 1.5 倍的格子 13.6% → 0.2%,持续一年以上的股票 1,318 → 0;下面是修复前的描述】**`total_mv` 里
   `mult = 流通股本 / "PIT 总股本最近一次数值变化时的流通股本"`,只在 PIT 总股本**数值变化**时重新锚定(首日强制锚定)。
   总股本多年不变、流通股本因限售解禁增长时,mult 一直累积,而解禁并不改变总股本。上一版报告说"新股在首份季报后约高估 6%",
   实际远不止(`fixb_review2_checks.csv`"附带发现"行):研究期流通股本 ≤ 总股本的格子里,total_mv / (未复权价 × PIT 总股本)
   > 1.5 的占 13.6%、> 2 的占 9.4%;1,318 只股票这个比值 > 1.5 持续一年以上。其中一部分是两次财报之间的送转 / 增发(合理外推),
   但典型反例很清楚:600339.SH 总股本与流通股本都是 55.83 亿,total_mv 在整个研究期是"价 × 总股本"的 3.60 倍;600390.SH 是 7.30 倍。
   `ND.total_mv` 被 Barra(Size / NonlinSize / 市值权重)、`rt_size`、中证1000 代理、容量与指增等多处使用,影响面超出本任务,
   建议单独立项(例如每次财报生效都重新锚定、只外推送转形态的跳变),并重检 RISK_MODEL / CSI1000_PROXY / CAPACITY 等报告。
   与本次修复的关系:PIT 股本序列中间出现 NaN 段时,`sh.diff() > 0` 会漏判 NaN 之后的第一次生效,`mult` 退化为 1,
   **意外地**绕开了上面的高估。股本 0 占位一律视为未披露后(§1.3),11 只新股的这种 NaN 段消失,2,989 格恢复了外推
   (新/旧 中位 1.345、最大 3.65,随之带上高估);另有 18 只新股 679 格在上市窗口从"价 × 流通股本"兜底改为招股书 / 上季股本
   (新/旧 中位 3.72、最大 19.8,纠正严重低估)。两部分合计就是 §3.2 真实市值比上一版多出的 18 只。函数本身没改。
9. 幸存者偏差:面板不含已退市股。COMP_v5 的成员在 2020-2025 研究期内选出,§3.3 的组合数字属于信号样本内,只用于新旧对照。

---

## 5. 前向检验如何合法使用 2026 数据

### 5.1 用法

```python
# scripts/fwd_xxx.py —— 文件名必须以 fwd 开头(或 test_),否则放宽会直接报 PermissionError
from afactor import config as C, newdata as ND, factors as FA, fundamental as FU

with C.set_data_cutoff("2026-10-09", reason="前向检验:冻结参数,评估窗口 2026-09-01 ~ 2026-10-09"):
    print(ND.data_coverage())            # 先看各源实际覆盖到哪天(source_end 列 = 原始文件最晚日期)
    p = <前向面板>                        # 新建面板,日期可到 2026-10-09
    p._derived["is_st"] = <新下载名单建的 ST 面板>   # ST 名单只到 07-03:不预先放入,引擎调 build_st_panel 会报错
    with C.accept_source_gap(FU.REPORT_TABLES, reason="财报停在 2026Q1/半年报缺失,前向窗口沿用最后一期"):
        roe = FA._fu(p, "roe")           # 不显式接受,会抛 SourceCoverageError
# 退出后截止日自动恢复为 2025-12-31,研究流程不受影响
```

规则:

- **放宽只限前向检验代码**。调用者文件名必须以 `fwd` 或 `test_` 开头,原因必须写明。收紧截止日不受限制。
  切换可以嵌套,退出时按层恢复。
- 进入和退出都会向 stderr 打印醒目横幅,并向 `logs/data_cutoff_switch.log` 追加 JSON 审计行(时间、新旧截止日、原因、调用者)。
- 面板日期晚于截止日时,`_pit_daily` / `total_mv` / `build_st_panel` / `toplist_panels` 等直接报错。
  这样可以防止在上下文外计算前向面板时,把 2026 的行静默 ffill 成 2025 的旧值。
- **面板日期晚于源文件原始最晚日期时,同样默认报错**(`SourceCoverageError`,第二轮审查 P1)。日期守卫只管
  "请求 ≤ 截止日",管不了"源文件本身只到哪天";上一版在截止日 2026-10-09 下请求 ST 面板到 10-09 时,
  会把 07-03 的 ST 状态静默 ffill 过去,没有任何警告。现在要么补数,要么在 `accept_source_gap(源, reason)` 内显式接受。
- **2026-01-01 ~ 2026-08-30 只能作为特征回看和预热的输入**,不得计算或报告这段时间的策略收益、IC、超额。
  评估只针对预注册的前向窗口。截止日接口只解决"能读到",这一条要由前向检验脚本自己遵守。
- 不碰 `cache/sealed`、`Panel(sealed=True)`、`scripts/run_sealed_*.py`、`research/ledger/smoke_used.lock`。
  封存流程已上锁,本修复不影响它。

### 5.2 放宽不等于有数据:data_8:21 的实际覆盖

在截止日 2026-10-09 下运行 `ND.data_coverage()`:

| 源 | 实际覆盖到 | 对 2026-09 ~ 10 前向窗口的含义 | 请求超出覆盖时(第二轮审查后) | `accept_source_gap` 内的行为 |
|---|---|---|---|---|
| daily_basic.csv | 2026-08-21 | 真实市值、换手、流通股本在 08-21 之后为 NaN,需要另行补数 | `_piv` 发 RuntimeWarning(覆盖之外本来就是 NaN) | 同左 |
| top_list.csv | 2026-08-21 | 龙虎榜 08-21 之后缺失 | `toplist_panels` 报错 | 覆盖之外的行置 NaN(含 20 日滚动量),不再 `fillna(0)` 成"全市场未上榜" |
| forecast.csv / express.csv | 2026-08-19 / 2026-08-22 | 三季报预告(10 月)缺失 | `earnings_surprise_events` / `surprise_panels` 报错 | 状态型面板沿用最后已知值;`days_since_surprise` 覆盖之外置 NaN |
| 利润表 / 现金流量表 | 2026-08-20 | 8 月下旬才披露的半年报缺失(多数公司在 08-31 前披露) | `_pit_daily` / `build_event_features` 报错(三表取最早的那个) | 沿用最后已知的报告期 |
| 资产负债表 | **2026-06-13** | **2026 半年报的资产负债表一行都没有**,股本、权益停在 2026Q1 | 同上;`ND.total_mv`、`FA._fu`、`build_market_cap` 经 `_pit_daily` 受保护 | 同上 |
| st_info_latest.csv | **2026-07-03** | **ST 状态停在 07-03**。引擎的 ST 涨跌幅 5% 判定依赖它,前向窗口必须刷新 | `build_st_panel` 报错;引擎经 `p["is_st"]` 调它,前向面板请预先放入自建的 `p._derived["is_st"]` | 沿用 07-03 的状态(只发警告) |
| stock_basic.csv | 上市日到 2026-08-20 | 之后上市的新股没有记录 | 无守卫(静态快照,不是时间面板) | — |
| 宽基指数快照 | 2026-07-14 | `timing.market_features` 会因快照过期而报错(它原有的检查) | 原有检查 | — |

守卫按各源读入时登记的原始最晚日期判断。前向脚本如果改为包装加载器、把新下载的数据拼进来(例如包装 `FU._read_st`),
包装里要调用 `C.record_source_end(源名, 新的最晚日期)` 更新登记,否则仍按旧文件的最晚日期报错;反过来,包装里额外截断数据
而不更新登记,守卫就检测不到(审查探针 `probe_st_coverage.py` 就是这种写法,复核脚本因此在包装里同时登记了模拟的最晚日期)。

研究期(请求 ≤ 2025-12-31)所有源的原始文件都覆盖到 2026 年,守卫恒不触发:`trunc2` 重跑与上一版 `trunc`
面板逐位一致(§6),全部测试通过。

### 5.3 其它需要前向检验代理自己处理的

- PEAD 事件表(`earnings_events.py`)的截止日写死为 `RESEARCH_END`,不受 `set_data_cutoff` 控制。如果要用,需要扩展。
- `index_proxy.index_close` 等处写死截到 2025-12-31。
- 近似市值 `build_market_cap` 的送转还原依赖 2026-08 前复权基准,与 baostock 前向价格的复权基准不同,前向检验不要用。
  `ND.total_mv` 也有 §4 第 8 条的解禁外推高估,而且 daily_basic 只到 08-21;前向窗口的市值最好直接用新下载的总股本 × 未复权价,
  并在报告里写明口径。

---

## 6. 复现

```bash
python3 scripts/fixb_cutoff_impact.py --make-old-root /tmp/fixb_old_root      # git HEAD 四个文件 + 数据软链
python3 scripts/fixb_cutoff_impact.py --stage panels --tag old --root /tmp/fixb_old_root   # 每步约 1.5 分钟
python3 scripts/fixb_cutoff_impact.py --stage panels --tag trunc --no-early
python3 scripts/fixb_cutoff_impact.py --stage panels --tag new
python3 scripts/fixb_cutoff_impact.py --stage rowcounts
python3 scripts/fixb_cutoff_impact.py --stage compare      # 写 fixb_impact_summary / fixb_comp_v5_downstream 和新打分缓存
python3 scripts/fixb_cutoff_impact.py --stage fundcols     # 约 2 分钟
python3 scripts/fixb_rolling_origin.py                     # 滚动原点 PIT 检验,约 5 分钟 -> fixb_rolling_origin*.csv
python3 scripts/fixb_cutoff_impact.py --stage panels --tag trunc2 --no-early   # 回归:应与 trunc 逐位一致
python3 scripts/fixb_review2_checks.py                     # 第二轮审查逐条复核 -> fixb_review2_checks.csv
PYTHONPATH=src python3 -m pytest tests/test_data_cutoff.py tests/test_barra_risk.py tests/test_engine_regression.py -q
```

`build_report_level` 约 12 秒(与旧版相同)。as-of 快照只对 217 只股票、424 个快照日计算。
`fixb_review2_checks.py` 的"前后对照"需要上一版的 new 面板,存在 `cache/_tmp/fixb_impact/new_r1/`。

---

## 一句话结论(可直接引用)

data_8:21 的 10 个源和宽基指数快照现在都在读入后立即截到 `DATA_CUTOFF`(默认 2025-12-31);财报版本按 as-of 快照取
"时点 t 已公告的最新版本",21 个滚动原点上 14 个映射正确的列没有任何前视,但财报季有 0.24%~2.07% 的股票-日偏陈旧。
修复后研究期的 ST、价格、换手、龙虎榜面板 0 格变化,引擎金标准逐位一致;真实市值有 29 只股票变化(主要是新股上市窗口),
COMP_v5 打分截面秩相关不低于 0.99993、标准排雷年化变化不超过 0.003pp,下游结论不需要重跑。前向检验只能在文件名以 fwd 开头的
脚本里通过 `set_data_cutoff(date, reason)` 读取 2026 数据;源文件覆盖不到的日期(ST 名单只到 07-03、资产负债表只到 06-13、
龙虎榜到 08-21)现在默认报错,要么补数,要么用 `accept_source_gap` 显式接受。另发现一个与截止日无关的既有问题:
`ND.total_mv` 的流通股本外推会把限售解禁当成股本增加,研究期 13.6% 的格子高估 1.5 倍以上,需要单独立项。

---

## 自查记录

| 检查 | 结果 |
|---|---|
| 不截断时,新版最终快照与旧版 `build_report_level` 逐行一致(`check_exact`) | 通过(第一轮);第二轮起股本 ≤ 0 的 598 行改为 NaN,其余逐格不变(与 `PIT_EARLY_VERSIONS=False` 的最终快照对比) |
| 不截断时,`_adjbase_quarterly_shares` 与旧版 `rep.pivot_table(total_share)` 逐格一致 | 通过 |
| 旧代码重算 COMP_v5 与冻结缓存逐格一致(配方核对) | 0 格差异 |
| `tests/test_engine_regression.py` 金标准 | 28 passed |
| `tests/test_data_cutoff.py` | 14 passed:默认无 2026 行;上下文内有,退出后恢复;缓存分键;嵌套;放宽权限;日期守卫;PIT 版本;被包装的 `_load`;第二轮新增:源覆盖守卫(研究期模拟 / 前向上下文)、旧键注入生效、as-of 快照对滚动原点真值、avail_sue 覆盖 14 季 |
| 全部测试(第二轮修复后,`PYTHONPATH=src python3 -m pytest tests`) | 178 passed(`logs/fixb_r2_pytest.log`) |
| 回归:本版代码 `--no-early` 重跑 vs 上一版"仅截断"面板 | 46/46 个面板逐位一致(`fixb_review2_checks.csv`) |
| 反证:合成用例换回上一版 S_k / 只对早期版本置 0 股本 | 两种都失败,新实现通过(`fixb_review2_checks.csv`) |
| 文件替换方式 | 先写同目录临时文件,再用 `mv -f` / `os.replace` 原子替换,没有出现半写状态 |

---

## 审查记录

### 第二轮对抗审查(2026-10-10,审查探针在 scratchpad/review2_fixb_pit/;修复复核 `scripts/fixb_review2_checks.py`)

审查确认的部分(重跑一致,不改):截断行数 daily_basic 797,190 / forecast 13,080 / ST 23,492;冻结打分与新打分每日截面
Spearman 最低 0.99993、NaN 位置相同;向三张财报表注入 3000 条 2026-03-15 公告的伪更正(数值 ×10),研究期 6 列 60 个日期 0 格变化,
阳性对照有检出力;14 个映射正确的列在 20 个滚动原点上没有前视。

| # | 级别 | 发现 | 处理 | 前 → 后 | 出处 |
|---|---|---|---|---|---|
| 1 | P1 | 放宽截止日后,源文件覆盖不到的日期被静默填充:ST 名单 ffill、龙虎榜 `fillna(0)`、财报 / 预告停在最后一期;只有 daily_basic 会警告 | **修复**。`config` 新增源覆盖守卫(`record_source_end` / `check_source_coverage` / `SourceCoverageError` / `accept_source_gap`);接入 `build_st_panel`、`_pit_daily`、`build_event_features`、`earnings_surprise_events` / `surprise_panels`、`toplist_panels`,`total_mv`、`build_market_cap`、`FA._fu / _fu_event / _real / _nd` 经由它们受保护。默认报错;显式接受后:龙虎榜覆盖之外置 NaN,`days_since_surprise` 置 NaN,ST / 财报 / 预告状态沿用最后已知值并发 RuntimeWarning。§5.2 列出每个源的行为 | 审查探针(ST 只到 2025-09-30、请求到 12-31):修复前 0 条警告、静默错 524 格 / 20 只(占真实 ST 格 5.1%)→ 修复后默认 `SourceCoverageError`;显式接受后 1 条 RuntimeWarning、同样 524 格。龙虎榜模拟(只到 11-28):修复前 12-01 之后填 0(真值有 1,403 个上榜格)→ 默认报错,接受后 NaN,12-01 与真值一致。真实场景(截止日 2026-10-09):ST / 龙虎榜 / `_pit_daily` / 预告请求到 10-09 全部报错。研究期:trunc 与 trunc2 共 46 个面板逐位一致 | `fixb_review2_checks.csv`;`test_source_coverage_guard_simulated_in_research_period`、`test_source_coverage_guard_forward_context` |
| 2 | P2 | 版本 PIT 的处理方式依赖未来修订:(a) 股本 0 占位只在"之后会被修订"的早期版本上置 NaN;(b) S_k 按"第 k 版"跨报告期拼快照,交错修订时组合依赖未来 | **修复**。(a) 股本 ≤ 0 一律视为未披露(最终版 598 行 / 425 只 + 被替代版本 19 行);(b) S_k 改为 as-of 快照:在每个"有被替代版本生效"的公告日取当时最新版本整体派生(217 只、424 个快照日、追加 625 行 / 193 只),并列时取最晚快照。§1.3"不会前视"改为"取值不前视,候选行只由 avail ≤ t 的版本决定" | 合成用例(每个 t 要求默认取值 = 截止日 t 重建的取值):旧 S_k + 只对早期版本置 0 → 失败(2022-05-06 total_share 1e8 vs 真值 0);旧 S_k + 0 一视同仁 → 失败(2022-06-15 total_assets_avg 1065 vs 1198.75);新实现通过。真实数据滚动原点"同期快照行取值不同":11 格 → 0 格。研究期影响:COMP_v5 新打分与上一版**逐位相同**(md5 仍 e53dca04);真实市值变化 11 只 → 29 只(679 格新股上市窗口低估被纠正,新/旧中位 3.72;2,989 格 / 11 只恢复了 `total_mv` 外推,中位 1.345,见第 7 行);近似市值 NaN → 有值 181 → 760 格;Barra Size q99 0.0022 → 0.0046 | `fixb_review2_checks.csv`、`fixb_rolling_origin_summary.csv`、`fixb_impact_summary.csv`;`test_asof_snapshots_match_rolling_origin_truth` |
| 3 | P2 | 研究期基本面的陈旧程度被低估:主因是原行被覆盖、三表取最晚、Q1 的 avail_q,而不是"交错修订" | **文档改正**(§1.3、§4 第 2 条、`fundamental.py` 模块文档),新增滚动原点脚本与账本。Q1 的 avail_q 放宽另行立项 | §1.3 原文"多个报告期交错修订时可能用到较旧的版本"→ 三条机制 + 实测:21 个原点、14 列偏陈旧 0.24%~2.07%,前视 0;f_ann_date > ann_date 的行(利润表 4,033 / 资产负债表 3,925 / 现金流量表 2,358)入账 | `fixb_rolling_origin_summary.csv`、`fixb_cutoff_rowcounts.csv` |
| 4 | P2 | 旧脚本往 `_CACHE["db"]` 注入并断言,但加载器的键已改为 `("db", 截止日)`:断言形同虚设、daily_basic 多占一份内存;`pead_data_acceptance.panel_consistency` 按 (股票, 报告期) merge 报告表,没筛最终快照会重复计数 | **修复**。`newdata` 在默认截止日下缓存键恢复为旧名字:注入被加载器真正使用,断言检查的就是加载器用的那份,不再双份内存;注入表含截止日之后的行时加载器再截一次。`scripts/pead_data_acceptance.py` 加一行 `FU._final_rows(...)`(该脚本不属本角色;只改这一行,因为是本修复改变了 `build_report_level` 的形状)。§1.2 / §1.3 列出全部消费方 | 注入后加载器是否使用注入的那份:否 → 是;注入含 2026 行时加载器返回的最大日期 = 2025-12-31。panel_consistency 重跑时多版本报告期:重复计数 → 每期一行(该脚本本次没有重跑,它的账本生成于第一轮修复之前,不受影响) | `test_legacy_daily_basic_injection_is_effective` |
| 5 | P2 | `build_market_cap` 的"同一基准送转还原"例外并不完整:股本只到 2026Q1 | **文档改正**(§1.3 第 4 条、§4 第 5 条、docstring)并估计量级;不修(既有问题,近似市值本来就标注为近似) | "2026 年送转的 7 只"→"只覆盖 2026Q1 股本里的 7 只;按 2021~2025 同期频率,每年 Q1→Q2 有 193~334 只、Q2→Q3 有 35~73 只送转,倍数中位 1.40,2026 年同期这部分未被还原" | `fixb_cutoff_rowcounts.csv`"build_market_cap 送转还原缺口估计"行 |
| 6 | P2 | `avail_sue` 只覆盖 12 个季度,sue_bt 实际依赖 14 个(earn_streak 17 个);§4.1 的修复建议不完整 | **修复**。`avail_sue` 改为 rolling(14)、新增 `avail_streak`(rolling(17)),注释改正;`fixb_pit_dependency_gap.csv` 改用这两列;§4 第 1 条修复建议改写。`_pit_daily` 不读这两列,任何面板数值不变 | sue_bt 依赖晚于映射可见日:3,618 / 127,433 行 → 3,794 / 131,157 行(分母因 min_periods 12 → 1 变化);earn_streak 2,665 → 2,674 | `fixb_pit_dependency_gap.csv`;`test_report_level_avail_sue_covers_14_quarters` |
| 7 | 附带发现(既有,未修) | `ND.total_mv` 的流通股本外推只在总股本数值变化时重新锚定,限售解禁会让外推倍数持续累积 | 不修(超出本任务,影响 Barra / 中证1000 代理 / 容量 / 指增),写入 §4 第 8 条并建议单独立项 | 研究期(流通股本 ≤ 总股本的格子)total_mv / (价 × PIT 总股本) > 1.5 的占 13.6%、> 2 的占 9.4%;1,318 只股票 > 1.5 持续一年以上;600339.SH 全程 3.60 倍、600390.SH 7.30 倍 | `fixb_review2_checks.csv` |

结论是否改变:截止日纪律在研究期成立、下游结论不需要重跑,这两条不变(COMP_v5 新打分与第一轮逐位相同)。改变的是:
真实市值受影响的股票从 11 只增加到 29 只;"版本 PIT 不会前视"细化为"取值不前视、处理方式不依赖未来,但偏陈旧 0.24%~2.07%";
`build_market_cap` 例外的完整性和 `total_mv` 的外推误差都比第一轮描述的严重,已写进 §4。
