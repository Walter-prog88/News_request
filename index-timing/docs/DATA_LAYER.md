# 2Share 数据层说明

## 为什么不用文档中的私有属性写法

教程建议修改 `pro._DataApi__http_url`。这个属性是 Tushare 客户端的私有实现，
版本变化后可能失效，也不利于保存完整请求回执。本仓库直接使用文档给出的
REST JSON 协议。

此外，教程示例使用 HTTP；本仓库强制 HTTPS，避免 token 明文传输。

## 安全设计

- token 只来自环境变量或隐藏输入；
- token 不进入 QuerySpec、receipt、manifest、异常消息或 `repr`；
- `http://` base URL 会直接拒绝；
- 请求失败时不打印请求体；
- 每个 snapshot 使用新目录，已存在目录不会覆盖；
- 原始响应、标准化 CSV、receipt 和 manifest 都进入 `SHA256SUMS`。

## 数据资格分层

| 数据 | 当前角色 | 正式资格 |
|---|---|---|
| `fund_daily.amount` | W2真实成交额候选 | 需单位与收盘后可得规则 |
| `fund_share` | W2 ETF份额候选 | 缺历史首次已知时点 |
| `margin` | W2交易所融资余额候选 | 缺实际发布时间 |
| `margin_detail` | W2单ETF融资明细候选 | 缺实际发布时间 |
| `eco_cal` | 计划宏观事件候选池 | 缺来源哈希和四家族总体 |
| `index_dailybasic` | PE/PB交叉验证 | 无历史股息率 |
| `shibor` | SHIBOR3M参考值 | 缺官方发布时点与来源哈希 |
| `sf_month/cn_m/cn_*` | 宏观参考值 | 非first-release vintage |

## 快照结构

```text
snapshot/
├── raw/           # API响应，不含请求token
├── normalized/    # CSV表
├── receipts/      # 参数、时点、字段、行数和文件哈希
├── manifest.json
└── SHA256SUMS
```

`index-data verify` 会逐文件检查 SHA-256，并检查 manifest 的凭据策略。

