# Agent、持续学习与量化研究方法

这里展示研究工具与小模型实验的精选代码和历史结果。内容分为 Agent 研究工具、LoRA 与持续学习实验、量化因子算子三个部分。

## 内容与完成范围

- **[Agent 研究工具](agent/README.md)：**可移植的追加式研究账本，支持结构化证据、缺失信息状态、明确弃权和完整性验证，并配有模拟测试。这是研究辅助工具原型，尚不能视为完整的自主投研 Agent 或经过验证的交易系统。投资者姓氏仅作为字段标识，输出不代表其本人观点。
- **[LoRA 与持续学习实验](learning/README.md)：**在公开 8×8 手写数字数据上，用小型 MLP 比较 LoRA、全参数更新与连续任务适应。这里保留历史结果表与一份原始指标 CSV；这些结果不能直接当作语言模型或机器人性能。
- **[量化研究方法](quant/README.md)：**精选截面、时间序列因子算子及纯模拟示例，不含真实行情、持仓、券商内部报告或投资收益承诺。

## 轻量检查

需要 Python 3.10+。请在合适的环境安装依赖。以下命令从本目录 `methods/` 执行，仅进行单元测试、接口检查和模拟演示，不训练模型，也不回测交易策略。

```bash
python -m unittest discover -s agent -p 'test_ledger.py' -v
python -m unittest discover -s quant -p 'test_factor_ops.py' -v
python -m unittest discover -s tests -p 'test_learning_interfaces.py' -v
python quant/synthetic_demo.py
```

如果环境没有 PyTorch，相应接口检查会跳过；数据加载检查还需要 scikit-learn。跳过不能计作通过。已完成的检查见[历史验证记录](VERIFICATION.md)。

## 模拟账本演示

先把 `DEMO_LEDGER_DIR` 设置为一个新建的临时目录，再从本目录执行：

```bash
python agent/ledger.py append --file agent/examples/synthetic_output.json --ledger-dir "$DEMO_LEDGER_DIR"
python agent/ledger.py verify --ledger-dir "$DEMO_LEDGER_DIR"
```

JSON 示例完全虚构，并因必需数据缺失而主动弃权。工具只写入本地记录。默认输出目录为已被 `.gitignore` 排除的 `runtime/ledger`；该实现使用 Unix `fcntl` 文件锁，暂不支持 Windows。

## 历史结果与使用边界

`learning/*/results/` 于 2026-10-30 从已有本地实验复制而来，制作展示包时没有重新训练。后续训练必须显式启用，结果写入 `runs/`，不覆盖存档。具体设置和限制见[学习实验说明](learning/README.md)。

本精选包不含模型权重、第三方源码检出目录、原始数据集、真实财务快照、实际研究账本条目、个人文档或凭据，也不连接账号、私有数据湖或执行交易。原始研究项目仍单独保留。可移植性修改与来源哈希见[来源与署名](ATTRIBUTION.md)；本次整理没有对未确认的第三方内容新增统一许可。
