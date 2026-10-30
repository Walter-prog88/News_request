# 强化学习交互实验室

11 个中文 HTML 学习页面，用可拖动参数、概率表、采样实验和逐步计算解释强化学习中的关键机制。适合刚开始读策略梯度、PPO 和 GRPO 的读者。

打开 [课程目录](index.html)，从神经网络更新开始，逐步查看策略、回报、基线、价值估计、重要性采样和策略更新约束。每个页面独立运行，源代码直接放在 HTML 内，便于阅读与修改。

## 页面

| 页面 | 可以观察什么 |
| --- | --- |
| [神经网络怎样被训练](lessons/00-neural-network.html) | 概率、损失、参数更新和优势信号之间的关系 |
| [格子世界与策略](lessons/01-gridworld.html) | 状态、动作、轨迹、回报与策略概率表 |
| [策略梯度](lessons/02-policy-gradient.html) | 为什么用采样回报调整动作概率 |
| [基线与优势](lessons/03-baseline-advantage.html) | 减去状态基线怎样改变学习信号 |
| [状态价值与 Critic](lessons/04-critic.html) | 蒙特卡洛与一步 TD 怎样估计价值 |
| [旧策略数据为什么会过期](lessons/05-stale-data.html) | 采样分布改变后，旧数据怎样影响估计 |
| [重要性采样](lessons/06-importance-sampling.html) | 用概率比重加权，以及有限样本的代价 |
| [概率比与方差](lessons/06b-ratio-variance.html) | 少数大权重怎样造成估计波动 |
| [TRPO 与 PPO 的 clip](lessons/07-ppo-clip.html) | 更新幅度、概率比、正负优势与裁剪目标 |
| [GRPO 的组内比较](lessons/08-grpo.html) | 组均值、标准差与相对优势的作用 |
| [TD 与蒙特卡洛逐步计算](lessons/09-td-monte-carlo.html) | 同一段轨迹中，两种目标怎样逐步更新 |

## 运行

无需 Python 机器学习环境、GPU、模型权重或 API Key。直接用现代浏览器打开 `index.html`。

也可以启动一个本地静态服务器：

```sh
python3 -m http.server 8000
```

然后打开 `http://localhost:8000`。

第 1—5 课用 KaTeX 0.16.9 的 CDN 文件排版数学公式，因此完整公式排版需要联网。离线时正文与原始 LaTeX 公式仍保留。其他六个页面无外部脚本依赖。详见 [依赖说明](DEPENDENCIES.md)。

## 范围与方法来源

这些页面是学习教具：用小规模例子解释机制，包含简化的更新、有限样本实验和手算练习。PPO 页面不包含完整训练系统；GRPO 页面不复现大语言模型的实际训练、token 级目标或完整 KL 处理。浏览器中的演示结果不能作为模型训练效果或算法性能比较。

页面由个人学习过程中积累的材料整理而来，使用了 AI 编程与讲解辅助。公开整理时统一了入口，并移除了个人学习进度、答题记录和会话引用。算法贡献归原作者，见 [方法来源与第三方说明](THIRD_PARTY_NOTICES.md)。

## 内容检查

```sh
python3 tools/check_content.py
python3 tools/check_content.py --node /path/to/node
```

检查包括本地链接、锚点、重复 ID、内联 JavaScript 与事件处理器语法（指定或可找到 Node.js 时）。它不替代浏览器操作检查，也不证明讲解中的每个推导都已经严格审稿。

整理日期：2026-10-30。
