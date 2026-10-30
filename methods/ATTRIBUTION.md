# 来源、署名与文件追溯

本目录是本地研究代码的精选与编辑导出。本次整理没有新增统一许可证，也不主张拥有第三方材料的版权。依赖库与所引用方法的权利仍属于其原作者；没有复制论文官方源码检出目录或原始数据集。

## 方法与数据来源

- Hu et al.（2021），*LoRA: Low-Rank Adaptation of Large Language Models*。本目录在小型 MLP 上实现低秩更新，并非 LoRA 官方仓库的导出。
- Biderman et al.（2024），*LoRA Learns Less and Forgets Less*。该论文提供探索问题，展示实验的设置与实际结果和论文存在差异。
- Lialin et al.（2023），*ReLoRA: High-Rank Training Through Low-Rank Updates*。适配器合并与重置思路与其相关；这里的小型数字分类实验不构成完整 ReLoRA 复现。
- scikit-learn `load_digits`，基于 UCI Optical Recognition of Handwritten Digits 数据集。通过已安装的依赖加载，本目录不重新分发数据集。
- NumPy、pandas、PyTorch、matplotlib 与 scikit-learn 均为外部依赖，没有把其源码打包进本目录。

## 整理时的修改

- **Agent 账本：**用可移植的日期、输出目录工具替代私有数据湖及正式账本依赖；测试仅写临时目录。账本保留原有的宽松校验方式：许多可选字段格式异常只产生警告，不会直接拒绝写入。
- **LoRA 脚本：**用 `load_digits` 替代本机缓存路径，已验证归一化数据一致；增加显式训练开关与独立的新结果目录；修正文档中的缩放公式，使其与实际仅使用 alpha 的实现一致。
- **持续学习脚本：**移除过强的解释性结论，没有把拟议的改进方法写成已完成结果。
- **量化算子：**删除依赖未打包本地模块的市值／行业中性化辅助函数；保留算子仅使用 NumPy／pandas，演示全部采用模拟数值。

## 源文件指纹

以下路径均为原项目内的相对路径，不暴露私人电脑路径。`True` 表示导出时做过修改，`False` 表示未修改；历史指标文件逐字节复制保留。

| 原项目 | 项目内相对路径 | 展示包文件 | 源文件 SHA256 | 是否修改 |
|---|---|---|---|---|
| lens | `ledger.py` | `agent/ledger.py` | `2a62a10d1a94b51445eab075c0b8156b9451615ace725ae776b12e76cb2cc7f1` | True |
| lens | `test_ledger.py` | `agent/test_ledger.py` | `b1e7abef696cb26c75c6b69ed0e9e717a3f711f03c3abd72ea2f5ca5f83420e2` | True |
| dl-research | `05_lora_forgetting/lora_vs_full.py` | `learning/05_lora_forgetting/lora_vs_full.py` | `227dc160fa04279480ee4c4f77601c658349f678bf989b0e171b83e251159c4d` | True |
| dl-research | `05_lora_forgetting/summarize.py` | `learning/05_lora_forgetting/summarize.py` | `caa3c25a50587cea749192d743e55a30b62fd9f7dbb61557f2cb807377b58721` | True |
| dl-research | `05_lora_forgetting/results/summary.md` | `learning/05_lora_forgetting/results/summary.md` | `9d6aa41dde194bfdf17e7e92fdf398e9f1c4127be6a12f1aef821b10cbcfe46a` | False |
| dl-research | `05_lora_forgetting/results/lora_vs_full.csv` | `learning/05_lora_forgetting/results/lora_vs_full.csv` | `ef0963434d3dc2a89bdc1093c97ff43fe1912e64f93bc2350c67de7279a1b9dc` | False |
| dl-research | `06_continual_lora/continual_lora.py` | `learning/06_continual_lora/continual_lora.py` | `f4392ed89974608f45755127a6e6b62e91baa7df73e153539703e2a2272ac243` | True |
| dl-research | `06_continual_lora/results/summary.md` | `learning/06_continual_lora/results/summary.md` | `b272f5fdf4924b3ac07e69fa25d82b146443401f2334e5f722fe9e89936613ca` | False |
| afactor | `ops.py` | `quant/factor_ops.py` | `90e411537f82c8329d125bde21f25ac80569bdf3feccb303b4259127c3769181` | True |
