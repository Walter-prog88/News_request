# 方法来源与第三方说明

本项目用中文解释已有强化学习方法，并制作小规模交互例子。个人贡献在于学习材料的组织、例子的实现和展示；不主张发明策略梯度、TD、重要性采样、TRPO、PPO 或 GRPO。

## 方法阅读入口

- Richard S. Sutton 与 Andrew G. Barto，*Reinforcement Learning: An Introduction*，第二版：[作者网站](http://incompleteideas.net/book/the-book-2nd.html)。基础概念、价值估计与策略梯度可结合原书阅读。
- John Schulman 等，*Trust Region Policy Optimization*，2015：[论文](https://arxiv.org/abs/1502.05477)。
- John Schulman 等，*Proximal Policy Optimization Algorithms*，2017：[论文](https://arxiv.org/abs/1707.06347)。
- Zhihong Shao 等，*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*，2024：[论文](https://arxiv.org/abs/2402.03300)。GRPO 的原始方法与真实训练设定以论文为准。

- *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*，2025：[论文](https://arxiv.org/abs/2503.14476)。第 8 课的动态采样术语来自该方法。
- *Understanding R1-Zero-Like Training: A Critical Perspective*，2025：[论文](https://arxiv.org/abs/2503.20783)。第 8 课的 Dr. GRPO 术语阅读入口。

上述链接用于标注方法归属，不表示本项目与原作者存在隶属、合作或认可关系。页面中的文字、绘图与浏览器小例子使用 AI 工具辅助制作，并未随包复制原论文全文或原书。

## KaTeX

部分课程通过 CDN 引用 KaTeX 0.16.9 进行公式排版。KaTeX 采用 MIT License，Copyright (c) 2013–2020 Khan Academy and other contributors。官方许可原文：[KaTeX v0.16.9 LICENSE](https://github.com/KaTeX/KaTeX/blob/v0.16.9/LICENSE)。

本包没有内置 KaTeX 的 JavaScript、CSS 或字体文件。自行离线分发这些文件时，请保留其版权与许可文本。
