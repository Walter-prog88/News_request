# 数据补充清单(DATA_WISHLIST)

> 目的:列出当前项目**缺失或近似**的数据,按重要性排序,给出获取途径与我需要的格式。
> 放到 `data_8:21/` 目录即可,文件名按下表。我会自动识别并接入管道。
> 更新:2026-08-22

---

## 一、重要性排序(先拿前两个)

| 优先级 | 数据 | 解决什么问题 | 预期收益 |
|---|---|---|---|
| **P0** | 每日复权因子 `adj_factor` | 根治市值/分红泄漏(当前市值重建秩相关 0.95,它能到 1.00) | 解锁 7 个挂"待确认"的因子;市值中性化结论可最终确认 |
| **P0** | 分析师一致预期 | PEAD 的标准定义是"实际 vs 分析师预期",现在用季节性随机游走替代 | **可能开出全新维度**(预期修正是国际最强异象之一);很可能是 PEAD 偏弱的主因 |
| P1 | 每日基本面 `daily_basic` | 流通市值、真实换手率(股本回推拿不到)、PE/PB 现成 | 真实换手率替代成交额代理,可能改善"交易节奏"维度 |
| P1 | 股票基本信息 `stock_basic` | 申万行业 + 上市/退市日 | 真实行业中性化(现用统计聚类);退市日可部分修复幸存者偏差 |
| P2 | 龙虎榜 / 资金流向 | 第一轮 89 条研报规格中一批卡在这 | 主力资金是独立信息源,但 A 股这类数据噪声大 |
| P2 | 财报公告精确时间(盘中/盘后) | 现在保守假设"T 日公告 T+1 起生效" | 让 PEAD 事件窗口更准,幅度小 |
| P3 | 退市股票的历史行情 | 幸存者偏差(当前多头收益高估 3–6pp) | 修复后所有绝对收益数字更真实,但相对排序影响小 |

---

## 二、逐项说明

### P0-1 每日复权因子 `adj_factor`

**为什么最重要**:我们的价格是前复权(基准 2026-08),`前复权价 × 股本` 算出的市值里带着"未来累计分红"的信息(审查实测该泄漏单独作因子 IC 达 +0.0178)。有了复权因子,`未复权价 = 前复权价 / adj_factor × adj_factor_基准日`,市值问题一次根治,还能反推精确的历史分红。

**获取途径**:
- **tushare**(最直接):`pro.adj_factor(ts_code='', trade_date='20200102')` 按日拉,或按股票 `pro.adj_factor(ts_code='000001.SZ', start_date='20200101', end_date='20260831')`。积分要求低(120 积分)。文档:https://tushare.pro/document/2?doc_id=28
- **baostock**(免费,无需积分):`bs.query_adjust_factor(code='sz.000001', start_date='2020-01-01', end_date='2026-08-31')`,返回 `foreAdjustFactor`/`backAdjustFactor`。文档:http://baostock.com/baostock/index.php/复权因子
- **akshare**:`ak.stock_zh_a_daily(symbol='sz000001', adjust='hfq-factor')` 返回后复权因子序列(新浪源,本机可访问新浪)

**我需要的格式**:`data_8:21/adj_factor.csv`,列 `ts_code, trade_date, adj_factor`,覆盖 2020-01-01 ~ 2026-08-31,全部 A 股。tushare 原始输出直接存即可。

---

### P0-2 分析师一致预期

**为什么重要**:盈余惯性(PEAD)的经典定义 SUE = (实际盈利 − 分析师预期) / 标准差。我们没有预期数据,用"去年同期 + 漂移"替代,这是 PEAD 在本项目中偏弱(IC 0.011)的最可能原因。另外"预期修正"(一致预期的时序变化)本身是独立于 PEAD 的强异象。

**获取途径**:
- **tushare**:`pro.report_rc(ts_code='', report_date='')` 券商盈利预测明细(需 5000 积分);`pro.forecast()` 业绩预告、`pro.express()` 业绩快报(积分低,**这两个是退而求其次的好替代**——公司自己的预告/快报与实际值的差就是一种"意外")。文档:https://tushare.pro/document/2?doc_id=45 / doc_id=46
- **Wind / Choice / 朝阳永续**(若有机构账号):一致预期 EPS、评级、目标价,质量最好
- **东方财富数据中心**(网页,手动):https://data.eastmoney.com/report/ 盈利预测(本机访问东财被拒,但你本地浏览器可以)

**我需要的格式**:
- 首选 `data_8:21/consensus_eps.csv`:`ts_code, date, fy(预测年度), eps_mean, eps_std, n_analysts`
- 次选 `data_8:21/forecast.csv` + `data_8:21/express.csv`:tushare 原始输出直接存

---

### P1-1 每日基本面 `daily_basic`

**解决什么**:流通市值(股本回推拿不到)、真实换手率(现在用成交额/20日均值代理)、PE/PB/PS/股息率现成。

**获取途径**:
- **tushare**:`pro.daily_basic(trade_date='20200102')` 按日拉(120 积分)。字段 `total_mv, circ_mv, turnover_rate, turnover_rate_f, pe_ttm, pb, ps_ttm, dv_ttm`。文档:https://tushare.pro/document/2?doc_id=32
- **baostock**:`bs.query_history_k_data_plus(code, "date,turn,peTTM,pbMRQ,psTTM", ...)` 有换手率和估值,**没有市值**(可用换手率反推流通股本:流通股本 = 成交量/换手率)
- **新浪快照**(已抓,仅当前值,覆盖 3000 只):`data_8:21/sina_snapshot.csv` ← 已有,但只能当校准锚点

**我需要的格式**:`data_8:21/daily_basic.csv`,tushare 原始输出,2020-01-01 ~ 2026-08-31。

---

### P1-2 股票基本信息 `stock_basic`

**解决什么**:申万行业(真实行业中性化,现用收益聚类替代,一致性约 0.3)、上市日、**退市日**(部分修复幸存者偏差)。

**获取途径**:
- **tushare**:`pro.stock_basic(exchange='', list_status='L,D,P', fields='ts_code,name,industry,market,list_date,delist_date')`——注意 `list_status='L,D,P'` 要把退市(D)和暂停(P)的也拉出来。免费。文档:https://tushare.pro/document/2?doc_id=25
- 申万一级行业(更标准):`pro.index_member(index_code='801010.SI')` 逐个申万指数拉成分,或 `pro.sw_daily`。需 2000 积分
- **baostock**(免费):`bs.query_stock_industry()` 证监会行业分类;`bs.query_stock_basic()` 含上市/退市日
- **新浪行业**(已抓 49 个板块,覆盖 3000 只):`data_8:21/sina_snapshot.csv` 的 `industry` 列 ← 已有但不全

**我需要的格式**:`data_8:21/stock_basic.csv`,列 `ts_code, name, industry, list_date, delist_date`(有申万一级更好,另存 `sw_industry.csv`:`ts_code, sw_l1, sw_l2, in_date, out_date`)。

---

### P2-1 龙虎榜 / 资金流向

**解决什么**:主力资金、机构席位是独立信息源;第一轮研报调研中 89 条规格有一批卡在"需要资金流"。

**获取途径**:
- **tushare**:`pro.top_list(trade_date=)` 龙虎榜每日明细(2000 积分);`pro.moneyflow(trade_date=)` 个股资金流向(大单/超大单,2000 积分)。文档 doc_id=106 / 170
- **东方财富**数据中心(网页手动):https://data.eastmoney.com/stock/tradedetail.html

**我需要的格式**:tushare 原始输出,`data_8:21/top_list.csv`、`data_8:21/moneyflow.csv`。

---

### P2-2 财报公告精确时间

**解决什么**:现在只知道公告日期,不知道是盘中还是盘后,保守按"T+1 起生效"。知道精确时间能让 PEAD 事件窗口更准。

**获取途径**:
- **巨潮资讯**(官方披露平台):http://www.cninfo.com.cn/ 公告检索,有精确发布时间;tushare `pro.anns()` 接口也有 `ann_date` 但同样只到日
- 实际上 A 股定期报告绝大多数在**收盘后或非交易日**披露,所以现有假设偏差不大。**优先级最低**。

---

### P3 退市股票历史行情

**解决什么**:数据源只含截至 2026-08 仍上市的股票,已退市的(乐视网、康得新等)完全缺失,多头收益高估 3–6pp。

**获取途径**:
- **tushare**:`pro.stock_basic(list_status='D')` 拿退市名单 → `pro.daily(ts_code=...)` 逐只拉日线(分钟线退市股一般拿不到)
- 注意:只有**日线**可补,分钟特征无法重建,所以这些股票只能进"收益计算"不能进"因子计算"。修复幸存者偏差的主要用途是让基准和多头收益数字更真实。

---

## 三、tushare 一键拉取脚本(参考)

```python
import tushare as ts, pandas as pd
pro = ts.pro_api("你的token")

# P0-1 复权因子(按股票循环,约 5500 次调用)
codes = pro.stock_basic(list_status="L,D,P", fields="ts_code")["ts_code"]
rows = [pro.adj_factor(ts_code=c, start_date="20200101", end_date="20260831") for c in codes]
pd.concat(rows).to_csv("data_8:21/adj_factor.csv", index=False)

# P1-1 每日基本面(按交易日循环,约 1600 次)
cal = pro.trade_cal(start_date="20200101", end_date="20260831", is_open="1")["cal_date"]
rows = [pro.daily_basic(trade_date=d) for d in cal]
pd.concat(rows).to_csv("data_8:21/daily_basic.csv", index=False)

# P1-2 基本信息
pro.stock_basic(exchange="", list_status="L,D,P",
                fields="ts_code,name,industry,market,list_date,delist_date"
                ).to_csv("data_8:21/stock_basic.csv", index=False)

# P0-2 替代品:业绩预告 + 快报
pro.forecast(start_date="20190101", end_date="20260831").to_csv("data_8:21/forecast.csv", index=False)
pro.express(start_date="20190101", end_date="20260831").to_csv("data_8:21/express.csv", index=False)
```

tushare 有调用频率限制(每分钟 200–500 次,按积分),按股票循环拉 adj_factor 约需 15–30 分钟。

---

## 四、已有的数据(不用再补)

| 文件 | 内容 | 状态 |
|---|---|---|
| `new_data/` | 全 A 分钟 open/close/amt,2020-01 ~ 2026-08 | ✅ 已压缩为 10 GB |
| `data_8:21/stock_*_2015_now.pkl` | 财务三表(tushare 格式) | ✅ 已接入,严格 PIT |
| `data_8:21/st_info_latest.csv` | 每日 ST 名单 | ✅ 已接入 |
| `data_8:21/sina_snapshot.csv` | 新浪快照:行业 + 市值(当前值,3000 只) | ✅ 已抓,仅作校准 |
| `data_8:21/momentum_factor_total.pkl` | 外部动量因子 | ✅ 已评估,不并入 |

## 2026-08-23 新增(按优先级)

### 1. 日线历史回溯到 2005(最优先)
**用途**:行业配置层的状态判断。2020 年以来行业动量只有 2021→2022 一次真切换,
状态变量无论 12/6/3/1 月窗口都无预测力(README §10.7)。补 2015 牛市、2017 白马趋势、2019 趋势,
切换次数 → 4–5 次。**行业层只需要日线 OHLCV + 行业分类,不需要分钟数据**,baostock/tushare 2005 年起可得。
**规模**:约 5000 只 × 15 年日线 ≈ 1500 万行,几百 MB。

### 2. 历史行业分类(point-in-time)
当前只有 2026-08-17 的证监会大类快照,回填 2020–2025。证监会 2024-11 换了新版分类表。
需要:申万一/二/三级历史成分(含变更日期)。申万二级才能把 AI/芯片/通信从 C39 里分出来。

### 3. 概念板块 / 主题标签(含历史)
东方财富/同花顺概念成分及变更日。2026 涨幅前 10% 的股票年初时量价特征与后 10% 无差别,
区分信息不在价量里。无此数据,任何"主题"层都做不了。

### 4. 日内分时成交模型所需:逐笔或 1 分钟含买卖方向
集中持仓(≤5 只)评估需要盘中分时成交;现有 10 分钟 TWAP 窗口对 20% 单票权重不适用(§10.7(5))。
