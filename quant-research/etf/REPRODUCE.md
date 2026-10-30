# 复现手册

三条流水线,全部确定性(固定种子),从原始数据到成品报告。

## 环境

```
PY=/path/to/user/Desktop/ETF-Research/index-project-active-etf-upper-bound/.venv/bin/python   # sklearn+torch,无 matplotlib
PYMPL=/path/to/user/Desktop/ETF-Research/index project/.venv/bin/python                       # matplotlib,画图与出报告用
export INDEX_CACHE_ROOT=~/index-cache        # 缓存根;不设则默认 ~/index-cache(持久,不再回落到会被清理的临时目录)
```

lightgbm/xgboost 缺 libomp 不可用;树模型一律 sklearn HistGradientBoosting,神经网络用 torch(MPS)。

## 外部数据依赖(只读,不入库)

| 依赖 | 路径 | 用途 |
|---|---|---|
| 分钟行情 pivot | `/path/to/user/Desktop/ETF-Research/full-etf project2 /etf_pivot_{amount,close}_qfq.pkl` | 全资产分钟聚合、流因子 |
| 宇宙与分类 | `full-etf-universe-store/`(daily-base-v2、classification-cb3a08d4…;曾位于 Desktop 根,现位于 `full-etf project2 /` 内,代码含双路径回退) | 日频面板、symbol_master |
| 冻结因子库 | `index-project-active-etf-upper-bound/research/{results/active_etf_upper_bound/core_flow_main.npz, active_system_v1/factors_v2.npz}` | 102 流因子 + 210 日频因子 |
| 状态模型分数 | `research/active_system_v1/regime_signal.parquet`(本仓库,已入库) | 状态门 |

## 流水线 A:zoo_v1 诚实搜索(方法论层)

```
cd research/zoo_v1
$PY build_minute_allasset.py        # 一次性:全资产分钟聚合 → $INDEX_CACHE_ROOT/zoo_cache
$PY run_axis.py <axis>              # 机械网格,一轴一命令;结果追加 trial_ledger.jsonl(append-only)
$PY run_axis.py verify --id <mid>   # 滞后+置换验证
$PY walkforward.py                  # 全部合成器代际 + 结构化组合
$PY deflate.py                      # 去偏审判(账本行数为分母)
$PY smoke.py                        # 封存窗一次性冒烟——smoke_used.lock 存在即拒绝重跑
```

红线(写在协议里,勿破):分组只用产品说明书分类;模块一律多空/市场中性;2024-26 永不作入围判据;账本只追加。

## 流水线 B:showcase_v1 展示报告(12 章大报告)

```
cd research/showcase_v1
$PY build_matrix.py                           # 495k×312 训练矩阵 → showcase_cache
$PY sweep_dial.py                             # 正则化档位表(展示口径的旋钮)
$PY finalize.py                               # 三范式+融合+波动目标版+十分位+重要性
$PY explainer_charts.py                       # 方案内等权对照 + 逐ETF贡献(对账断言)
FACTORY_DEV_END=1213 FACTORY_TAG=_pre2024 $PY factor_factory.py   # 干净版因子工厂
FACTORY_TAG=_pre2024 $PY factory_clean.py     # 剔尾盘反转变体
$PY seq_models.py && $PY seq_saliency.py      # LSTM/TFM + 梯度归因
$PYMPL generate_report.py                     # → showcase_report.html
```

## 流水线 C:前向验证汇报页(对外主打)

```
cd research/showcase_v1
$PY validation_report_data.py     # 冻结模型前向回放,四配置,逐ETF贡献对账<1e-9
$PYMPL validation_charts.py       # 四张房屋风格 PNG → results/figs/
$PYMPL generate_validation_report.py   # → validation_report.html
```

## 发布

两份报告是 claude.ai Artifact,更新时**必须传原 URL** 才能保持同链接:
- 汇报页 `validation_report.html` → artifact d4a6d8d7-3e18-4f8a-b547-aa0ca87353d6
- 大报告 `showcase_report.html` → artifact 06a0ff37-ffa5-4cd6-afa2-b7f61fc3960a

## 口径契约(所有流水线共享)

信号 T 日收盘 → T+1 开盘成交 → 开盘-开盘计价;0.5bp/单边;单票≤30%;权重<3%剔除;
流动性门 = 事前 252 日成交额中位数≥500万;干净窗口 = 2019-21 与 2022-23,2024-26 为已污染参考;
展示口径 = 全样本拟合(报告必须写明),验证口径 = 2024-01-02 前冻结、之后纯前向。
