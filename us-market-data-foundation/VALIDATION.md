# 公开版离线验证

验证日期：2026-10-30。

在公开整理目录执行，导入的是本包 `src/` 中的代码；使用已有 Python 环境提供依赖，没有加载生产数据库或下载数据。

```text
PYTHONPATH=src python scripts/verify_offline.py
Ran 155 tests in 1.220s
OK
```

测试期间，`socket.create_connection`、`socket.socket.connect` 和 `socket.socket.connect_ex` 被替换为立即失败的函数；SEC 传输使用模拟响应。文件类测试使用自动清理的临时目录。

覆盖 10 个测试文件：SEC 传输和历史包、财务标准化、会计勾稽、Parquet 验证、公司样本与覆盖率、行业样本汇总、因子输入约束、安全数据湖和外盘检查。

合成例子的输出已验证：

```text
as_of=2024-06-01: revenue=100 USD
as_of=2024-10-01: revenue=110 USD
```

运行环境：Python 3.12.14、NumPy 2.5.3、pandas 3.0.6、PyArrow 25.0.1、DuckDB 1.5.5、Requests 2.34.2。`requirements.txt` 给出兼容范围；其他版本组合尚未逐一验证。

此验证没有访问实际 SEC 接口，没有验证实际磁盘写入、完整生产采集流程或金融策略表现。测试结果不等于任意真实披露都能被自动正确归一化。
