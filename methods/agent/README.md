# Agent 研究记录工具原型

本目录展示如何把结构化判断、证据状态与研究记录连接起来。它是投研 Agent 的辅助组件原型，尚不构成完整自主投研系统。

## 已实现内容

- `ledger.py`：追加研究条目，使用 SHA256 链检查记录完整性，支持 `append`、`verify`、`show`、`due`。
- `common.py`：可移植的日期和输出目录工具，不读取私有数据湖。
- `examples/synthetic_output.json`：完全虚构的输出示例，展示必需数据缺失时主动弃权。
- `test_ledger.py`：临时目录测试，覆盖追加、链校验、篡改／删行检测、无效字段及弃权输出等情况。

输出保留来源、可得时间、缺失或陈旧状态及 `pass`／`fail`／`insufficient` 判断。部分可选字段的格式异常只产生警告，不会强制拒绝写入。投资者姓氏是兼容原型的字段标识，不表示其本人观点或背书。

## 使用方式

以下命令从上一级 `methods/` 目录执行。先把 `DEMO_LEDGER_DIR` 设置为新建临时目录：

```bash
python agent/ledger.py append --file agent/examples/synthetic_output.json --ledger-dir "$DEMO_LEDGER_DIR"
python agent/ledger.py verify --ledger-dir "$DEMO_LEDGER_DIR"
python -m unittest discover -s agent -p 'test_ledger.py' -v
```

默认输出目录为 `runtime/ledger`，已被 `.gitignore` 排除。测试仅写临时目录。实现使用 Unix `fcntl` 文件锁，暂不支持 Windows。这里不含真实财务、持仓、研究账本或账号连接，也没有 Agent 任务完成率、投资收益等效果承诺。

此前 6 项单元测试通过，见[历史验证记录](../VERIFICATION.md)；本次中文说明维护未重新测试。来源与修改见[署名说明](../ATTRIBUTION.md)。返回[研究方法目录](../README.md)。
