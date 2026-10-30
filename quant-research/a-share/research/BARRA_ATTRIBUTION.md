# Barra 式超额归因 —— 排雷策略没有正 alpha

> 2026-09-02 初稿;2026-10-10 两次口径修订:①同频池基准,②真实市值修复(`research/TOTAL_MV_FIX.md`)。
> **正文数字已改为"新市值 + 同频基准"口径,这是今后的推荐口径。** 初稿数字(旧市值 + 每日重置基准)和同频修订数字
> 保留在文末两节修订记录里。
>
> 脚本:`scripts/run_barra_attribution.py`(初稿脚本,未改动,默认仍用每日重置基准)、`scripts/barra2_rerun.py`(本次重跑,
> 复用前者的函数,同时算两种基准和新旧两种市值)。
> 日志:`logs/barra2_run_barra_attribution.log`、`logs/barra2_run_{new,legacy}.log`、`logs/barra2_selfcheck.log`、`logs/barra2_compare.log`、`logs/barra2_bpep.log`;
> 初稿日志 `research/r_barra_attr.log`。
> 账本(`research/ledger/`):
> - `barra2_attribution_summary.csv`:mv × bench × mode 长表,**本文主数字出处**;
> - `barra2_compare.csv`:新旧对照;
> - `barra2_checks.csv`:核对项;
> - `barra2_bpep_truecap.csv`、`barra2_bpep_checks.csv`:审查 P2 的敏感性(BP / EP 分母换真实市值,§7 第 7 条);
> - `barra_attribution_summary.csv`、`barra_factor_returns.csv`:正式账本,已用新市值重跑,前者仍是每日重置基准;
> - 旧版备份在 `research/ledger/_pre_mvfix/`。
>
> 只读研究期面板(`Panel` / `TailPanel`,截至 2025-12-31),**未触碰封存样本**。
> COMP_v5 的成员是在 2020-2025 选出的,所以本文收益数字都属于信号样本内;面板不含已退市股,存在幸存者偏差。
>
> **结论(新市值 + 同频池基准):**
> - COMP_v5 + 排雷缓冲 20/35 在回归样本内的超额:主口径 +4.89pp,尾盘 +5.42pp。
> - 剔除 Barra 十风格与申万一级行业后,残差 alpha:
>   - 主口径 **−0.98pp**,NW t −1.57(q=5;q=0~20 在 −1.57 ~ −1.72 之间);
>   - 尾盘 **−0.54pp**,NW t −0.84(q=5;q=0~20 在 −0.84 ~ −0.90 之间)。
>   - 两个口径都**为负,但不显著**。
> - 流动性一项贡献 +5.26pp,**占总超额的 108%**(尾盘 +5.57pp,103%)。
> - 出处:`barra2_attribution_summary.csv` mv=new、bench=same。
>
> **两次修订的影响**(主口径残差 alpha,出处 `barra2_compare.csv`):
>
> | 口径 | 残差 alpha | NW t(q=5) |
> |---|---|---|
> | 初稿(旧市值 + 每日重置) | −0.31pp | −0.48 |
> | 改用同频基准(当前代码旧市值重跑;fixa 记录为 −1.22pp / −1.94) | −1.21pp | −1.93 |
> | 再改用新市值 | −0.98pp | −1.57 |
>
> 同频修订时写的是"点估计转为边缘显著的负值"(t ≈ −2.0,q0~q20 为 −1.94 ~ −2.15)。市值修复后,这一点不再成立:t 回到 −1.57(q=5),不显著。
> 稳健的结论是 **"剥掉风格后没有正 alpha,超额全部来自流动性暴露"**;"alpha 显著为负"则不稳健。

---

## 1. 为什么要重做一次

上一版归因(`research/ledger/attribution_summary.csv`)用 7 个日频风格 + 8 个行业,
结论是"68% 是风格,剩 1.75pp(主口径)/ 2.45pp(尾盘)残差 alpha,且显著(t 3.9–6.7)"。

那一版有两个已知缺陷:

1. **风格不全。** 价值、盈利、成长、杠杆一族整个漏掉了。
   - COMP_v5 的成员清单 `comp_v5_members.json` 列 25 个,其中恰有三个基本面交互因子:
     `fx_fu_rev_yoy__P__ivol_20`、`fx_fu_roe__P__ivol_20`、`fx_fu_roe__P__id_ret_intraday_m60`。
   - 实际参与打分的是 24 个:`lit_mome_ideal_rev_bigmove` 不在因子注册表,脚本按 `if n in FA.REGISTRY` 静默跳过了它。
     初稿账本和冻结打分缓存都是这样算的,本次重跑沿用,见 `logs/barra2_run_new.log`。
2. **共线性使单个 β 不可解读。**
   - 风格因子两两相关高达 0.72(市值↔流动性)和 0.94(换手↔波动),VIF 达 13–28。
   - 市值 β = +0.235、流动性 β = −0.222,一正一负,贡献 −1.83 与 +2.87 互相抵消。
   - 这是典型的"共线性把同一个真实暴露劈成两个系数"。

---

## 2. 方法:从"rank 多空"改为"纯因子收益"

### 2.1 风格补全到 Barra CNE 的十个

| 风格 | 构造 | 数据来源 |
|---|---|---|
| Size 市值 | log(真实总市值) | `FA._real(p,"mv")` = `ND.total_mv`(2026-10-10 起为修复后口径) |
| Beta 市场敏感 | 对市值加权市场收益的 252 日滚动回归斜率 | 面板 |
| Momentum 动量 | 过去 252 日,剔除最近 21 日 | 面板 |
| ResidVol 残差波动 | Beta 回归残差的 252 日标准差 | 面板 |
| NonlinSize 非线性市值 | z(lncap)³ 对 z(lncap) 正交化后的残差 | 面板 |
| BookToPrice 价值 | `fu_bp` = 归母权益 / 市值 | 注册表;**分母是近似市值 `p["market_cap"]`,不是真实市值**(见 §2.2 末、§7 第 7 条) |
| Liquidity 流动性 | log(真实换手率 21 日均) | `FA._real(p,"turn")` |
| EarningsYield 盈利 | `fu_ep` = TTM 归母净利 / 市值 | 注册表;分母同上,为近似市值 |
| Growth 成长 | `fu_rev_yoy` | 注册表 |
| Leverage 杠杆 | `fu_leverage` | 注册表 |

- 行业:**申万一级,29 个行业因子**(上一版只放 8 个)。
- 暴露度一律做截面标准化:市值加权去均值、除以标准差、±3σ 截尾。
- 函数从 2026-10 起搬进 `src/afactor/barra.py`,逻辑不变(`research/ledger/risk_move_check.csv`)。

> ⚠ 踩过的坑:`fu_bp`、`fu_ep`、`fu_rev_yoy`、`fu_leverage` 是**因子注册表里的名字**,不是报表层的列名。
> 第一版误用 `FA._fu(p,"fu_bp")`,全部 KeyError,四个基本面风格被静默跳过,只跑了 6 个风格。
> 已改走 `FA.REGISTRY[name][1](p)`。

### 2.2 纯因子收益(pure factor returns)

不再用"rank 标准化 + 美元中性多空",改用 Barra 的做法——**每日截面 WLS**:

```
r_i = f_country + Σ_k X_ik · f_k + Σ_j Ind_ij · f_j + u_i
```

- 权重 `sqrt(市值)`。
- 行业用**市值加权和为零**的约束识别,最后一个行业由其余行业线性表出。
- 回归系数 `f` 就是纯因子收益:持有该因子单位暴露、其余因子零暴露的组合收益。

这样造出来的因子收益近似正交,**贡献可以逐项解读**,正好解决 §1 第 2 条。
最后把策略日超额对纯因子收益做时间序列 OLS,截距即 alpha;t 值用 Newey-West(Bartlett 权重,q = 0/5/10/20)。

**市值进入的五个地方**(所以真实市值修复要求重跑):
1. Size / NonlinSize 暴露;
2. 全部风格的市值加权去中心化;
3. Beta / ResidVol 所用的市值加权市场收益;
4. 截面 WLS 的 sqrt(市值) 权重;
5. 行业"市值加权和为零"约束。

**还有一处市值没有随修复换口径:BookToPrice / EarningsYield 的分母。**
- 注册表因子 `fu_bp` / `fu_ep` 除以的是 `p["market_cap"]`(`factors.py` `_build_fundamental` 的 `cap`),
  即 `fundamental.build_market_cap` 的近似市值:前复权价 × PIT 股本 × 送转倍数,含分红泄漏(前复权价扣了历年分红,历史市值偏低)。
  `TOTAL_MV_FIX.md` §6 也写明近似市值不经过 `ND.total_mv`,不受修复影响。
- 所以修复后,模型里并存两种市值口径:上面五处是真实市值,BP / EP 分母是近似市值。
- 池内近似 / 真实市值之比的中位数为 0.985(p10 0.862,p90 1.000;主口径),
  BP 暴露在两种分母下的逐日截面秩相关中位数为 0.981,EP 为 0.995(`barra2_bpep_checks.csv`)。
- 只把这两个分母换成真实市值、其余不变,主口径同频 alpha 从 −0.98pp 变为 −0.93pp(NW t q5 −1.57 → −1.48),
  定性结论不变。详见 §7 第 7 条。

策略打分、决策池、池基准、成本模型都不用 `ND.total_mv`,所以超额序列 y 在新旧市值下逐位相同
(`barra2_checks.csv`"策略与两种基准日收益 new == legacy"两行,最大差 0)。变的只是回归元。

### 2.3 基准:同频再平衡的池等权(2026-10-10 起推荐)

策略每 5 日调仓,其间权重随价格漂移。初稿的池基准却是"每日重置等权",调仓频率不同。
在截面相对收益有正的滞后相关时,"每天卖相对涨的、买相对跌的"会系统性吃亏,这部分与选股无关,却被记成了策略超额。
机制见 `ENGINE_IMPACT_MODEL.md` §3.1 与 `ST.benchmark` 的文档。
本文正文改用 `ST.benchmark(p, "V1000", mode, rebalance=5)`,即同频基准。

### 2.4 样本

- 回归样本:主口径 1,201 天(2021-01-15 ~ 2025-12-29),尾盘 1,202 天(至 2025-12-30)。
  出处:`barra2_attribution_summary.csv` 的 n_days 列;日期区间见 `barra2_checks.csv`。
- 比上一版的 1,453 天少,因为 Beta 与 ResidVol 需要 252 日滚动窗口,2020 年前段被吃掉。
- **这构成一个混淆,已单独排除,见 §4。**
- 两种基准、新旧两种市值用的都是同一组日子。

---

## 3. 结果(新市值 + 同频基准)

### 3.1 总账

出处:`research/ledger/barra2_attribution_summary.csv`,mv=new、bench=same。

| | 主口径 open10 | 尾盘 tail1450 |
|---|---|---|
| 总超额(算术年化) | +4.89pp | +5.42pp |
| 风格合计 | **+5.92pp** | **+5.98pp** |
| 行业合计(29 个) | −0.05pp | −0.02pp |
| **残差 alpha** | **−0.98pp** | **−0.54pp** |
| alpha 的 NW t(q = 0/5/10/20) | −1.68 / −1.57 / −1.59 / −1.72 | −0.88 / −0.84 / −0.85 / −0.90 |
| R² | 0.7886 | 0.7615 |

两个口径的 alpha 点估计都为负,|t| < 2,都不显著。

对照:同样是新市值,换回初稿的每日重置基准(mv=new、bench=d1),结果如下。
- 主口径:总超额 +5.32pp,风格 +5.47pp,alpha −0.04pp(NW t q0 −0.06 / q5 −0.06)。
- 尾盘:总超额 +5.94pp,alpha +0.26pp(NW t q0 +0.43 / q5 +0.42)。

两种基准下总超额相差约 0.43~0.52pp。这部分是再平衡频率差造成的伪超额,不是选股。

### 3.2 逐风格分解(主口径)

出处同上,mode=open10。

| 风格 | β | 纯因子年化 | 贡献 |
|---|---|---|---|
| **Liquidity 流动性** | **−0.465** | **−11.32%** | **+5.26pp** |
| BookToPrice 价值 | +0.112 | +4.68% | +0.52 |
| NonlinSize 非线性市值 | −0.048 | −5.38% | +0.26 |
| ResidVol 残差波动 | −0.103 | −2.49% | +0.26 |
| Growth 成长 | +0.060 | +2.67% | +0.16 |
| Leverage 杠杆 | −0.000 | −3.77% | +0.00 |
| Size 市值 | +0.005 | −8.79% | −0.04 |
| Beta 市场敏感 | −0.015 | +5.49% | −0.08 |
| Momentum 动量 | −0.053 | +3.83% | −0.20 |
| EarningsYield 盈利 | +0.085 | −2.59% | −0.22 |
| 29 个行业合计 | — | — | −0.05(单项绝对值最大 0.11,纺织服饰) |

- **流动性一项 +5.26pp,占总超额 4.89pp 的 108%。** 其余九个风格净贡献 +0.66pp,行业 −0.05pp,残差 −0.98pp。
- 尾盘口径结构相同:Liquidity β −0.447,纯因子年化 −12.46%,贡献 +5.57pp,占总超额 5.42pp 的 103%。
  其余风格中最大的是 BookToPrice +0.37pp 和 NonlinSize +0.30pp(同文件 mode=tail1450)。

---

## 4. ⚠ 排除窗口混淆

新旧两版同时变了**两件事**:风格集,以及样本窗口。必须把两者分开。
做法是在**同一个 1,201 天窗口**上重跑旧版风格集。
下表前三行是初稿(2026-09-02,旧市值 + 每日重置)的分离实验,未重跑;t 为 NW q=0。

| | 主口径 alpha | t | 尾盘 alpha | t |
|---|---|---|---|---|
| 旧风格 / 全窗口 1,453 天 | +1.75pp | 5.25 | +2.45pp | 6.80 |
| 旧风格 / 新窗口 1,201 天 | +1.26pp | 3.63 | +1.83pp | 4.86 |
| 新风格(Barra)/ 新窗口(初稿口径) | −0.31pp | −0.52 | +0.00pp | +0.01 |
| 新风格 / 新窗口,新市值 + 每日重置(`barra2_attribution_summary.csv`) | −0.04pp | −0.06 | +0.26pp | +0.43 |
| 新风格 / 新窗口,**新市值 + 同频**(同上) | **−0.98pp** | **−1.68** | **−0.54pp** | **−0.88** |

- 窗口只解释 1.75 → 1.26 那一段,约 28%。**剩下的全部由方法变更造成。**
- 后两行说明:市值口径和基准口径都不会让 alpha 回到旧版那样的正值。

补充证据(初稿口径):
- 只有 6 个风格时 alpha 为 −0.27pp,补齐到 10 个后是 −0.31pp。**补全风格没有救回 alpha,反而略降。**
- 价值、盈利、成长、杠杆加进去,只是在风格之间重新分配(BookToPrice +0.53、EarningsYield −0.24),总量不变。

---

## 5. 为什么上一版会剩下 1.75pp

是共线性造成的假象(本节数字均为初稿口径)。
- 旧口径下市值 β = +0.235、流动性 β = −0.222,两者相关 0.72,VIF 分别为 23.8 与 18.3。
- 一正一负互相抵消,**系统性低估了流动性的真实贡献**:旧版流动性贡献 +2.87pp,纯因子口径 +5.20pp,差 2.33pp,
  与 alpha 的消失量级吻合。
- 新市值 + 同频口径下,流动性贡献为 +5.26pp(§3.2),这个判断不变。

Barra 的截面 WLS 正是为解决这个问题设计的:纯因子收益会把重叠的部分归到正确的因子上。

---

## 6. 净结论

> **这是一个流动性 smart beta,不是选股 alpha。**
>
> 回归样本(2021-2025)里,对同频池基准的 +4.89pp 超额中,+5.26pp 来自"持有比市场更不流动的股票"这一个暴露。
> 剔除 Barra 十风格与申万一级行业后,选股 alpha 为 −0.98pp(NW t −1.57,q=5,不显著);尾盘口径 −0.54pp(NW t −0.84,q=5)。
> 出处:`barra2_attribution_summary.csv` mv=new、bench=same。

三条推论:

1. **可低成本复制。** 按 20 日换手率排序、剔掉最活跃的 30%、等权持有即可。
   不需要 40GB 分钟数据,也不需要 2,151 个候选因子。
2. **有周期风险,且没有缓冲。**
   - 回归样本内,流动性纯因子年化 −11.32%(冷门票跑赢,`barra2_attribution_summary.csv` factor_ann),是大顺风。
   - 风格反转时,5.3pp 会变成 −5.3pp,而且**没有正 alpha 垫底**。点估计甚至是负的。
3. **该暴露不是主动选择的。**
   - 成员的入选标准只有 `|IC| ≥ 0.025` 加分年方向一致,与风格无关。
   - 策略层对 `industry / log_cap / mv` 零命中。
   - 暴露是从因子里掉出来的,不是决策的结果。这意味着风格不利时,既说不清在赌什么,也没有减仓依据。

### 与因子层旧结论的关系(不矛盾)

README §11 测过:市值、行业、流动性三者正交后,因子 **IC 保留率为 0.635**。
当时的解读是"还剩 63.5%,信息还在"。本次归因说的是:**IC 层面剩下的那部分,落到组合收益上不为正。**

两者不矛盾。IC 衡量的是全区间的排序能力,收益取决于这份排序能不能变成钱。
排雷用法只吃分布最差的那一端,而那一端恰好与流动性高度重合。

---

## 7. 边界(必须主动交代)

1. **归因是线性的。** 它回答的是"超额能否被风格的线性组合复制",不能排除策略在非线性或条件性上有价值。
   不过对一个只做多、等权、5 日调仓的宽分散筛选来说,非线性空间本来就很小。
2. **Beta / ResidVol 用等权 252 日滚动。** Barra 官方用半衰期加权,这里做了简化。
3. **样本少 252 天**(2020 年前段),已在 §4 单独排除这一混淆。
4. **本结论只适用于 A 股这一个策略。** ETF 轮动策略的归因结论完全相反(风格解释 3.9%、alpha 40pp、t = 2.9),
   见 `ETF-Research/index project/research/attribution/`。
5. **alpha 的符号与显著性对口径敏感,定性结论不敏感。**
   - 两次修订(基准频率、市值数据)让主口径 alpha 在 −1.21 ~ −0.04pp 之间移动,NW t(q=5)在 −1.93 ~ −0.06 之间(`barra2_compare.csv`)。
   - 主口径四种口径下,流动性贡献都在 5.13 ~ 5.30pp,alpha 没有一次为显著正值。
6. 纯因子回归是**完整样本**:任一风格缺失的股票整行剔除,金融股因 `fu_leverage` 恒为 NaN 而出局。
   对排雷组合相对池等权的归因影响小;风险模型改用 `risk_factor_regression`(见 `src/afactor/barra.py` 文档)。
7. **BookToPrice / EarningsYield 的分母仍是近似市值。** `fu_bp` / `fu_ep` 除以 `p["market_cap"]`(`build_market_cap`,含分红泄漏),
   不是修复后的真实市值。这是全项目共用的因子定义,改它属于因子定义变更,不在本次口径修正范围内,所以正式口径保持不变。
   敏感性(`barra2_rerun.py --stage bpep`,只换这两个分母,y 与其余 8 个风格暴露逐位不变;账本 `barra2_bpep_truecap.csv`):

   | 口径 | 项 | 近似市值分母(正式) | 真实市值分母 | 变化 |
   |---|---|---|---|---|
   | open10 同频 | 残差 alpha | −0.98pp | −0.93pp | +0.05 |
   | open10 同频 | NW t(q0/q5/q10/q20) | −1.68 / −1.57 / −1.59 / −1.72 | −1.59 / −1.48 / −1.51 / −1.63 | 约 +0.09 |
   | open10 同频 | BookToPrice 贡献 | +0.52pp | +0.44pp | −0.08 |
   | open10 同频 | 风格合计 / 流动性贡献 | +5.92 / +5.26pp | +5.87 / +5.28pp | −0.05 / +0.02 |
   | open10 每日重置 | 残差 alpha(NW t q5) | −0.04pp(−0.06) | +0.01pp(+0.01) | +0.05 |
   | tail1450 同频 | 残差 alpha(NW t q5) | −0.54pp(−0.84) | −0.51pp(−0.81) | +0.02 |
   | tail1450 每日重置 | 残差 alpha(NW t q5) | +0.26pp(+0.42) | +0.26pp(+0.41) | −0.01 |

   alpha 的变化(绝对值 ≤ 0.05pp)远小于基准口径造成的差异(同频与每日重置之间 0.80~0.94pp),四个格子里 alpha 都不显著,
   流动性贡献变动 ≤ 0.02pp。定性结论不受影响。

---

## 8. 复现

```bash
cd "/path/to/user/Desktop/A stock market"
python3 scripts/run_barra_attribution.py                      # 正式账本(新市值、每日重置基准),约 3 分钟
python3 scripts/barra2_rerun.py --stage selfcheck             # 回归函数复现初稿账本(原纯因子收益 + 冻结打分)
python3 scripts/barra2_rerun.py --stage run --mv new          # 新市值,两种基准,约 2.5 分钟
python3 scripts/barra2_rerun.py --stage run --mv legacy       # newdata.TOTAL_SHARE_METHOD="legacy",只差 total_mv
python3 scripts/barra2_rerun.py --stage compare               # 汇总与核对
python3 scripts/barra2_rerun.py --stage bpep                  # 敏感性:BP/EP 分母换真实市值(读 run --mv new 的缓存),约 3 分钟
```

产出:

| 文件 | 内容 |
|---|---|
| `research/ledger/barra2_attribution_summary.csv` | 长表,列为 mv(legacy/new)、bench(d1/same)、mode、term、kind、beta、factor_ann、contrib_ann、n_days。**推荐口径为 mv=new、bench=same** |
| `research/ledger/barra2_compare.csv` | 初稿 / fixa 同频 / 四种口径的对照(总超额、风格、行业、流动性、Size、NonlinSize、alpha、NW t、R²) |
| `research/ledger/barra2_checks.csv`、`barra2_selfcheck.csv` | 核对项(见文末记录) |
| `research/ledger/barra2_bpep_truecap.csv` | BP / EP 分母敏感性长表:列同 summary,外加 bpep_cap(approx = 正式口径,true = 真实市值分母);只有 mv=new |
| `research/ledger/barra2_bpep_checks.csv` | 敏感性的核对:基线复现、其余 8 个风格暴露不变、BP / EP 暴露新旧秩相关与覆盖、纯因子收益新旧相关 |
| `research/ledger/barra_attribution_summary.csv` | `run_barra_attribution.py` 的正式输出:新市值、**每日重置基准**,格式不变 |
| `research/ledger/barra_factor_returns.csv` | 40 个纯因子的逐日收益(两个口径),新市值;列名、日期索引与旧版完全相同 |
| `research/ledger/_pre_mvfix/barra_{attribution_summary,factor_returns}.csv` | 旧版备份,sha256 与原文件一致 |
| `cache/_tmp/barra2_{new,legacy}_{open10,tail1450}_{F,rets}.parquet` | 回归元、策略与两种基准的日收益 |

---

## 修订记录(2026-10-10,同频基准)

> 本节是同频修订当时的记录,**数字是旧市值口径**,仅为留档。新市值口径的数字见下一节。

上一轮组合层审计发现:5 日调仓的策略与"每日重置等权"池基准比较时,基准吃到了截面短期反转的再平衡收益,策略吃不到,
于是超额被系统性抬高(open10 约 +0.4pp/年,vwap1 约 +1.3~2.7pp/年)。
时间序列回归对 y 是线性的,所以换基准只需对基准差额重做同一组回归元的 OLS,不必重跑暴露度。

| 口径 | 项 | 原值(每日重置) | 新值(同频 rebalance=5) | 出处 |
|---|---|---|---|---|
| open10 | 总超额 | +5.32pp | +4.89pp | `research/ledger/fixa_r2_barra_delta.csv` |
| open10 | 风格合计 | +5.65pp | +6.06pp | 同上 |
| open10 | 残差 alpha | −0.31pp(t −0.52,q0) | −1.22pp(NW t −1.94 ~ −2.15,q0~q20) | 同上 |
| tail1450 | 总超额 | +5.94pp | +5.42pp | 同上 |
| tail1450 | 残差 alpha | +0.00pp | −0.78pp(t 未重算) | 同上 |

与之后的组合层结果一致:有约束的中证1000 指增在剥离风格与 20 日反转后,特质 alpha 同样为负,且不显著
(`INDEX_ENHANCEMENT_CSI1000.md`)。

---

## 市值修复重跑记录(2026-10-10)

**改了什么。** `newdata.total_mv` 的日频总股本改为"按报告期末锚定、两次财报之间只外推送转"(`TOTAL_MV_FIX.md` §2)。
旧口径把限售解禁当成股本增加,研究期 13.6% 的格子市值高估 1.5 倍以上。

**怎么重跑。**
- 正式账本由 `run_barra_attribution.py` 原样重跑(未改代码)。
- 四种口径由 `barra2_rerun.py` 在**当前代码**上算。新旧市值只差诊断开关 `newdata.TOTAL_SHARE_METHOD`(legacy / songzhuan),
  所以"旧市值 → 新市值"一列就是市值修复的纯效应。
- "原账本 09-02"一列是初稿。它与当前代码的旧市值重跑之间,还隔着 fixb 的财报 PIT 修复(`DATA_CUTOFF_DISCIPLINE.md`):
  COMP_v5 打分变化日的截面秩相关 ≥ 0.99993(`fixb_comp_v5_downstream.csv`),对归因的影响见下文"复现核对"。

### 四种口径对照

出处:`research/ledger/barra2_compare.csv`;四列的原始行在 `barra2_attribution_summary.csv`。

#### open10(主口径)

| 项 | 原账本 09-02(旧市值+每日重置) | 旧市值+每日重置 | 旧市值+同频 | 新市值+每日重置 | **新市值+同频** |
|---|---|---|---|---|---|
| 总超额 | +5.32pp | +5.32pp | +4.89pp | +5.32pp | **+4.89pp** |
| 风格合计 | +5.65pp | +5.63pp | +6.05pp | +5.47pp | **+5.92pp** |
| 行业合计 | −0.02pp | −0.02pp | +0.05pp | −0.11pp | **−0.05pp** |
| 流动性贡献 | +5.20pp | +5.20pp | +5.13pp | +5.30pp | **+5.26pp** |
| 流动性占总超额 | 97.8% | 97.6% | 104.8% | 99.6% | **107.6%** |
| **残差 alpha** | −0.31pp | −0.30pp | −1.21pp | −0.04pp | **−0.98pp** |
| alpha NW t(q=0/5/10/20) | −0.52 / −0.48 / −0.50 / −0.54 | −0.51 / −0.47 / −0.48 / −0.52 | −2.07 / −1.93 / −1.98 / −2.13 | −0.06 / −0.06 / −0.06 / −0.06 | **−1.68 / −1.57 / −1.59 / −1.72** |
| R² | 0.7927 | 0.7925 | 0.7876 | 0.7892 | 0.7886 |
| Size 贡献 / β / 纯因子年化 | −0.15pp / +0.020 / −7.52% | −0.15pp / +0.020 / −7.51% | +0.04pp / −0.005 / −7.51% | −0.24pp / +0.027 / −8.79% | −0.04pp / +0.005 / −8.79% |
| NonlinSize 贡献 / β / 纯因子年化 | +0.21pp / −0.044 / −4.68% | +0.21pp / −0.045 / −4.68% | +0.51pp / −0.109 / −4.68% | −0.04pp / +0.008 / −5.38% | +0.26pp / −0.048 / −5.38% |

#### tail1450(尾盘)

| 项 | 原账本 09-02(旧市值+每日重置) | 旧市值+每日重置 | 旧市值+同频 | 新市值+每日重置 | **新市值+同频** |
|---|---|---|---|---|---|
| 总超额 | +5.94pp | +5.94pp | +5.42pp | +5.94pp | **+5.42pp** |
| 风格合计 | +5.92pp | +5.91pp | +6.15pp | +5.71pp | **+5.98pp** |
| 行业合计 | +0.02pp | +0.02pp | +0.03pp | −0.03pp | **−0.02pp** |
| 流动性贡献 | +5.52pp | +5.51pp | +5.43pp | +5.64pp | **+5.57pp** |
| 流动性占总超额 | 92.9% | 92.8% | 100.2% | 94.9% | **102.8%** |
| **残差 alpha** | +0.00pp | +0.01pp | −0.77pp | +0.26pp | **−0.54pp** |
| alpha NW t(q=0/5/10/20) | +0.01 / +0.01 / +0.01 / +0.01 | +0.02 / +0.01 / +0.01 / +0.02 | −1.25 / −1.21 / −1.24 / −1.30 | +0.43 / +0.42 / +0.42 / +0.44 | **−0.88 / −0.84 / −0.85 / −0.90** |
| R² | 0.7698 | 0.7697 | 0.7615 | 0.7662 | 0.7615 |
| Size 贡献 / β / 纯因子年化 | −0.03pp / +0.003 / −7.63% | −0.03pp / +0.004 / −7.62% | +0.12pp / −0.016 / −7.62% | −0.15pp / +0.017 / −8.84% | +0.01pp / −0.001 / −8.84% |
| NonlinSize 贡献 / β / 纯因子年化 | +0.31pp / −0.065 / −4.84% | +0.32pp / −0.065 / −4.84% | +0.50pp / −0.103 / −4.84% | +0.10pp / −0.017 / −5.52% | +0.30pp / −0.055 / −5.52% |

### 市值修复的纯效应(旧市值+同频 → 新市值+同频)

出处:`barra2_compare.csv` 的 delta_mv_same 列,以及 `barra2_attribution_summary.csv`。

- **总超额不变。** y 与市值无关,逐位相同。
- **alpha +0.23pp**(两个口径都是),t 提高约 0.36~0.41(q0~q20 各滞后)。来源如下:
  - 风格合计 −0.12pp(尾盘 −0.18pp);
  - 行业合计 −0.11pp(尾盘 −0.06pp)。
- **主要通道是 Size 系。**
  - 纯因子年化:Size −7.51% → −8.79%,NonlinSize −4.68% → −5.38%。
  - 策略对 NonlinSize 的 β 从 −0.109 降到 −0.048,于是 NonlinSize 贡献 +0.51 → +0.26pp,Size 贡献 +0.04 → −0.04pp。
  - 这与 `TOTAL_MV_FIX.md` §4 一致:Size 暴露的逐日截面秩相关中位 0.942,NonlinSize 0.892,其余风格 ≥ 0.999。
- **其余风格的纯因子收益也有平移。** 原因是去中心化权重、市场收益和 WLS 权重都换了市值:
  - Country 5.06% → 3.62%;
  - Momentum +2.62% → +3.83%,贡献 −0.11 → −0.20pp;
  - ResidVol −1.11% → −2.49%,贡献 +0.12 → +0.26pp。
  - 新旧纯因子日收益的序列相关为 0.966 ~ 0.995(核对表列出的 7 个因子)(`barra2_checks.csv`"纯因子收益 new vs 原账本"各行)。
- **流动性几乎不动。** β −0.455 → −0.465,纯因子年化 −11.25% → −11.32%,贡献 +5.13 → +5.26pp。

### 旧值 → 新值(推荐口径的变化)

旧值取同频修订时写进本文的数字(`fixa_r2_barra_delta.csv`,旧市值 + 同频),新值取 `barra2_attribution_summary.csv` mv=new、bench=same。

| 口径 | 项 | 旧值 | 新值 | 变化 |
|---|---|---|---|---|
| open10 | 总超额 | +4.89pp | +4.89pp | 0 |
| open10 | 风格合计 | +6.06pp | +5.92pp | −0.13 |
| open10 | 行业合计 | +0.05pp | −0.05pp | −0.11 |
| open10 | 流动性贡献 | +5.13pp | +5.26pp | +0.13 |
| open10 | 流动性占总超额 | 105.0% | 107.6% | +2.7pt |
| open10 | **残差 alpha** | −1.22pp | **−0.98pp** | +0.24 |
| open10 | alpha NW t(q0/q5/q10/q20) | −2.09 / −1.94 / −2.00 / −2.15 | **−1.68 / −1.57 / −1.59 / −1.72** | 约 +0.4 |
| tail1450 | 总超额 | +5.42pp | +5.42pp | 0(−0.002,fixb 打分) |
| tail1450 | 风格合计 | +6.16pp | +5.98pp | −0.19 |
| tail1450 | 流动性贡献 | +5.43pp | +5.57pp | +0.14 |
| tail1450 | **残差 alpha** | −0.78pp | **−0.54pp** | +0.24 |
| tail1450 | alpha NW t(q0/q5/q10/q20) | 未重算 | −0.88 / −0.84 / −0.85 / −0.90 | 首次计算 |

初稿口径(每日重置基准)的变化(NW t 两个滞后都列出,q0 / q5):
- open10 alpha −0.31pp(t −0.52 / −0.48)→ −0.04pp(t −0.06 / −0.06);
- tail1450 alpha +0.00pp(t +0.01 / +0.01)→ +0.26pp(t +0.43 / +0.42)。

出处:原账本备份 `_pre_mvfix/barra_attribution_summary.csv` → 新正式账本 `barra_attribution_summary.csv`。

### 结论是否变化

- **变了:alpha 的显著性。** 同频修订时的表述"点估计转为边缘显著的负值(NW t ≈ −2.0)"不再成立。
  新市值下主口径 t 为 −1.57 ~ −1.72,尾盘 −0.84 ~ −0.90,都不显著。正文已改为"为负但不显著"。
- **变了:流动性占比。** 从 105% / 100% 升到 108% / 103%(主口径 / 尾盘)。正文已更新。
- **没变:定性结论。** "这是一个流动性 smart beta,不是选股 alpha;剥掉风格后没有正 alpha。"
  四种口径下流动性贡献都在 5.13 ~ 5.64pp,alpha 没有一次为显著正值。
- **没变:BP / EP 分母口径的影响。** 审查指出这两个分母仍是近似市值。换成真实市值后,四个格子的 alpha 变化都在 ±0.05pp 以内,
  主口径同频 t(q5)−1.57 → −1.48,仍不显著(`barra2_bpep_truecap.csv`,§7 第 7 条)。

### 审查处理(2026-10-10)

独立复核重算了全部新数字(暴露、纯因子收益、y、8 组回归),无 P0 / P1;三条 P2 处理如下。

| # | 级别 | 发现 | 处理 | 证据 |
|---|---|---|---|---|
| 1 | P2 | "市值进入的五个地方"漏了 BP / EP:分母仍是近似市值 `p["market_cap"]`(`build_market_cap`,含分红泄漏),修复后模型并存两种市值口径 | **已修**:§2.1 表、§2.2 末、§7 第 7 条注明;新增 `barra2_rerun.py --stage bpep` 实测敏感性。正式口径不改,因为 `fu_bp` / `fu_ep` 是全项目共用的因子定义 | 只换这两个分母:open10 同频 alpha −0.979 → −0.928pp,t q5 −1.57 → −1.48,BP 贡献 +0.523 → +0.438pp;每日重置 −0.036 → +0.009pp;tail1450 同频 −0.537 → −0.515pp(t q5 −0.84 → −0.81),每日重置 +0.264 → +0.258pp。与审查探针报告的 open10 数字一致(到其报告的小数位)。核对:基线纯因子收益与 `barra2_new_*_F.parquet` 最大差 0,基线归因与 `barra2_attribution_new.csv` 最大差 2.2e-16,其余 8 个风格暴露最大差 0 且 NaN 位置一致(`barra2_bpep_checks.csv`) |
| 2 | P2 | 被推翻的"alpha 边际显著为负"和旧 Barra 数字仍被其他报告引用,"未重跑的下游"没列这些报告 | **已修**:"未重跑的下游"新增"引用本文结论的其他报告"表(ENGINE_IMPACT_MODEL §9 第 11 条 / 639 / 645 行、CAPACITY §7.7 / 1218 行、INDEX_ENHANCEMENT_CSI1000 33 / 316 / 473 行、RISK_MODEL 201 行、LONG_SIDE_TEST 91 行、SELECTION_ABILITY_AUDIT 3 / 70 行、TOTAL_MV_FIX §6),交主控分派;本任务未改这些报告 | 行号按 2026-10-10 文件检索 |
| 3 | P2 | 每日重置口径的 t 值在同一句里混用滞后(open10 −0.52 是 q0,tail1450 +0.42 是 q5),§4 表又写 +0.43(q0) | **已修**:§3.1 对照段与"初稿口径的变化"两处改为同时列 q0 / q5;正文其余单个 t 值都标注了 q(摘要、§6、同频修订表);§4 表维持"t 为 NW q=0" | `barra2_attribution_summary.csv` mv=new、bench=d1、tail1450:`__alpha_t_q0` = 0.4336,`__alpha_t_q5` = 0.4161;原账本 `_pre_mvfix/barra_attribution_summary.csv` open10:q0 −0.5247,q5 −0.4848 |

### 复现核对

全部列在 `research/ledger/barra2_checks.csv` 与 `barra2_selfcheck.csv`。

1. **回归函数复现初稿。** 输入为原纯因子收益、冻结打分 `comp_v5_score_open10.parquet`、每日重置基准,
   `barra2_rerun.attribute` 复现初稿 open10 的全部 48 行,最大差 3.4e-14。
2. **正式账本与 barra2 一致。**
   - `run_barra_attribution.py` 重跑的正式账本,与 barra2 的 new + 每日重置逐位一致:summary 最大差 0,纯因子收益最大差 1e-16。
   - 新 open10 纯因子收益与 `cache/_tmp/risk_open10_F_orig.parquet`(风险模型影子新口径,同一库函数)最大差 0。
3. **只有回归元随市值变。**
   - 策略与两种基准的日收益,在新旧市值下最大差 0;open10 打分在新旧市值下最大差 0。
   - 当前打分与 `comp_v5_score_open10_fixb.parquet` 逐位相同。与冻结打分 `comp_v5_score_open10.parquet` 相比,最大差 0.22(fixb 财报 PIT)。
4. **初稿 → 当前代码旧市值重跑的差异小。** 这部分是 fixb 的影响:
   - 纯因子收益日序列相关 ≥ 0.99987;
   - alpha 差:open10 +0.011pp,tail1450 +0.006pp;
   - 风格合计差约 −0.01pp。
   - 同频口径下,与 `fixa_r2_barra_delta.csv` 的 alpha 差 +0.012pp。

### 新账本的格式

`barra_factor_returns.csv` 的列、日期索引都与旧版完全相同:
- 两层列头 (mode, factor),80 列 = 2 个口径 × 40 个因子(Country、10 个风格、29 个行业);
- 1,202 行,2021-01-15 ~ 2025-12-30。

数值换成了新市值。`barra_attribution_summary.csv` 的列、行顺序不变,仍是每日重置基准。

### 未重跑的下游

下列使用者读 `barra_factor_returns.csv`,其结论基于旧纯因子收益,本次没有重跑:
- `BARRA_TOP20.md`(`run_barra_attribution_top20.py`);
- `run_barra_attribution_topfrac.py`;
- `run_comp_split.py`;
- `pead_analyze.py`;
- `stress_analyze.py`(STRESS_TEST,另由压力测试重跑处理)。

另外两处:
- `fixa_r2_barra_delta.csv` 是旧市值下的同频增量,已被本节取代。
- `risk_move_check.csv` 已在市值修复提升后重跑,对新账本最大差 1.0e-16(`RISK_MODEL.md` §1)。

**引用本文结论的其他报告(审查 P2;本任务不改这些报告,清单交主控分派给各自负责人)。**
下列段落引用的是初稿或同频修订时的数字与表述。其中"边际显著为负"已被本次重跑推翻。
新数一律以 `barra2_attribution_summary.csv` mv=new 为准:推荐口径取 bench=same,初稿基准对照取 bench=d1。

| 报告 | 位置(2026-10-10 行号) | 现在写的 | 与本次重跑的关系 | 建议 |
|---|---|---|---|---|
| `ENGINE_IMPACT_MODEL.md` | §9 第 11 条(554–557 行) | 同频后 alpha −0.31 → −1.22pp,NW t −1.94 ~ −2.15;"从与零不可分变成边际显著为负" | **结论被推翻**。新市值 + 同频:alpha −0.98pp,NW t −1.68 / −1.57 / −1.59 / −1.72(q0/5/10/20),不显著 | 改措辞并换数 |
| `ENGINE_IMPACT_MODEL.md` | 审查处理表第 2 行(639 行)、645 行 | 同上;"BARRA_ATTRIBUTION 在同频基准下残差 alpha 边际显著为负" | **结论被推翻**,同上 | 改为"为负但不显著" |
| `CAPACITY_REPORT.md` | §7.7 复核注(997–1002 行)、审查处理表第 2 行(1218 行) | 占比 98% → 105%,alpha −0.31 → −1.22pp(NW t −2.15 ~ −1.94);tail1450 占比 100%、alpha −0.78pp;"BARRA_ATTRIBUTION 正文尚未改" | 数字过时:新为占比 108% / 103%,alpha −0.98 / −0.54pp。"超额几乎全是流动性暴露,读法更强了"仍成立;"正文尚未改"已过时 | 换数 |
| `INDEX_ENHANCEMENT_CSI1000.md` | 33 行、316–317 行;473 行 | "Barra 剥风格后 alpha −0.31pp(t −0.52)";"与 alpha 为零的旧结论一致" | 引的是初稿口径(旧市值 + 每日重置)。推荐口径 −0.98pp(t q0 −1.68);新市值 + 每日重置 −0.04pp(t q0 −0.06)。"没有正 alpha"的一致性不变 | 换数或注明口径 |
| `RISK_MODEL.md` | 201 行 | "时序回归(BARRA_ATTRIBUTION.md 得到 alpha ≈ 0)" | 推荐口径 −0.98pp,不显著。"≈ 0"宜改为"为负但不显著" | 注明口径 |
| `LONG_SIDE_TEST.md` | 91 行(Barra 表"排雷 20/35(对照)"行) | +5.3 / +5.6 / −0.3 / t −0.5 / 流动性 +5.2 | 初稿口径。同表其余行由 `run_barra_attribution_topfrac.py` / `run_comp_split.py` 读旧纯因子收益,见上面的代码清单 | 随代码清单一并重跑 |
| `SELECTION_ABILITY_AUDIT.md` | 3 行、70 行 | "残差 alpha = 0、98% 是流动性" | 09-02 的起因陈述,属于历史记录。现为 alpha −0.98pp(不显著)、流动性占 108% | 加注即可 |
| `TOTAL_MV_FIX.md` | §6(344 行、351 行) | BARRA_ATTRIBUTION 列为"未重检";"近似市值不受影响" | BARRA_ATTRIBUTION 已由本节重跑;BARRA_TOP20 仍未重跑。BP / EP 分母仍用近似市值(§7 第 7 条),可补为既有问题一条 | 更新状态 |

`STRESS_TEST.md` 由压力测试重跑另行处理;`PEAD_REPORT.md` 364 行已注明读入的是新口径 `barra_factor_returns.csv`,无需改。
`README.md`、`FINDINGS.md`、`PLAN.md`、`RESEARCH_LOG.md` 中没有引用本文的数字(按"Barra"检索无命中)。

**处理状态(2026-10-10,主控)**:ENGINE_IMPACT_MODEL(§9 第 11 条、审查表、代理年化)已加注新数;CAPACITY、INDEX_ENHANCEMENT、RISK_MODEL 已由报告更新代理换数;LONG_SIDE_TEST、SELECTION_ABILITY_AUDIT 作为历史记录加注;TOTAL_MV_FIX 已加"合并后更正"。仍未重跑:`run_barra_attribution_top20.py`(BARRA_TOP20.md)、`run_barra_attribution_topfrac.py`、`run_comp_split.py`——它们读旧纯因子收益,结论只作历史参考。
