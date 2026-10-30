"""指增优化器的"附加 α"接口:把 PEAD 事件信号的 Grinold α 叠加到主信号上(2026-10-10,前缀 iepead)。

## 为什么要有它
上游 afactor.optimizer.IndexEnhancer 只接受**一个**打分 + 一条 IC 预测,α = IC·ω·z。任务要求在 V1 上把 α 改成
α_COMP + α_PEAD,其他参数与 V1 完全相同。两个信号各自 Grinold 再相加,不能先把打分加起来再交给 IndexEnhancer:
它会对合成打分重新做截面 z-score(均值 0、标准差 1、截尾 ±3),两个 α 的相对大小(IC_COMP·z_COMP vs IC_PEAD·z_PEAD)
就被抹掉了。

## 实现方式(以及为什么不改 optimizer.py)
optimizer.py 属于上游 ie 的产物(纪律 9:只写自己负责的文件)。IndexEnhancer.__call__ 在内部调用模块级函数
`optimizer.grinold_alpha(...)` 恰好一次。本类在**单次调仓的调用期间**把这个模块属性换成一个闭包:
    α = grinold_alpha(主打分) + Σ_k grinold_alpha(附加打分_k, IC_k)
返回值里的 z 仍是主打分的 z(上游只在主 IC ≤ 0 时用 z 给候选集排序,COMP_v5 的主 IC 从未为 0)。
调用结束(含异常)在 finally 里还原。这样:
  - V1 的其余代码路径(候选集、成本、约束、求解级联、NaN 约定、诊断)一个字节都不变;
  - 附加 IC 恒为 0 时与 IndexEnhancer 逐位相同 —— scripts/run_index_enhance_pead.py 的作业 Z0 用全期
    相位 0 回测对 ie_V1_ph0 做金标准核对,tests/test_optimizer_pead.py 做合成数据单测。
代价:monkeypatch 不是线程安全的(本项目一个进程只跑一条路径,不并发调用)。

## 附加信号的口径(PEAD,预注册见 scripts/run_index_enhance_pead.py 模块文档)
  - z_k:附加打分在可投资池(V1000 决策池 ∩ 风险模型覆盖)内**非缺失**的股票上做截面 z-score,截尾 ±3;
    非缺失股票 < 20 只时整行 z = 0(沿用 optimizer.grinold_alpha 的门槛),事件池外 α_k = 0。
  - IC_k:附加打分自己的逐日 rank IC(决策池内、非缺失 ≥ 20 只的日子才有值),250 行滚动均值,
    只用 t−6 已实现部分(optimizer.ic_for_decision,lag=6),下限 0,不足 20 个观测记 0。
  - 两个 α 直接相加,不做正交化:等于假设两个信号的预测误差不相关。截面相关性在报告里作为诊断给出。

## 诊断(写进 self.log 的同一条记录,键名带附加信号名)
  ic:{k}         决策日附加 IC 预测(下限 0 后)
  n_z:{k}        附加打分 z ≠ 0 的股票数(= 事件池 ∩ 可投资池,< 20 时为 0)
  alpha_ex_ante:{k}  事前主动 α 中来自附加信号的部分 = α_k'(w − wb)(5 日收益单位)
  aw:{k}         附加信号覆盖股票的主动权重合计 Σ_{z_k≠0}(w − wb)
  exp:{k}        附加 z 的主动暴露 Σ z_k (w − wb)
  wb 用与 IndexEnhancer 相同的口径重建:WB[t] 在风险模型覆盖池内归一 × Σw(覆盖池外卡住的持仓占预算时 Σw < 1)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afactor import optimizer as OPT


def event_rank_ic(score: pd.DataFrame, fwd: pd.DataFrame, universe: pd.DataFrame,
                  min_stocks: int = 20) -> pd.Series:
    """事件型信号的逐日截面 rank IC:只在信号非缺失(事件池)且在决策池内的股票上算,少于 min_stocks 只记 NaN。

    与 optimizer.rank_ic_series 同一实现,只是门槛从 200 降到 20 —— 事件池大多数日子只有 0~几只股票,
    财报季才有成百上千只;门槛与 α 生效的门槛(z 至少 20 只)一致,IC 只在 α 会生效的那类日子上估计。
    索引 = 信号行 s;IC[s] 在 s+6 开盘才实现,决策日用之前必须经 ic_for_decision(lag=6)。"""
    return OPT.rank_ic_series(score, fwd, universe, min_stocks=min_stocks)


class AdditiveAlphaEnhancer(OPT.IndexEnhancer):
    """IndexEnhancer + 附加 Grinold α。add_scores = {名字: (打分 DataFrame, 决策日 IC 预测 Series)}。

    其余参数与 IndexEnhancer 完全相同;附加 IC 恒为 0 时与 IndexEnhancer 逐位相同。"""

    def __init__(self, p, rm, bench_w: pd.DataFrame, score: pd.DataFrame, ic_dec: pd.Series,
                 universe: pd.DataFrame, cfg: OPT.OptConfig = OPT.OptConfig(), tag: str = "",
                 keep_targets: bool = True, extra_exposures: dict | None = None,
                 add_scores: dict | None = None):
        super().__init__(p, rm, bench_w, score, ic_dec, universe, cfg, tag=tag,
                         keep_targets=True, extra_exposures=extra_exposures)
        self._keep_user = keep_targets          # 诊断需要目标权重;用户不要时用完即删
        self.add = {}
        for k, (s, ic) in (add_scores or {}).items():
            self.add[k] = (s.reindex(index=p.dates, columns=p.codes).to_numpy(np.float32),
                           ic.reindex(p.dates).to_numpy(np.float64))

    def __call__(self, t: int, h, ctx: dict):
        orig = OPT.grinold_alpha
        parts: dict = {}

        def combined(score_row, inv, omega, ic, z_clip=3.0):
            a0, z0 = orig(score_row, inv, omega, ic, z_clip)
            tot = a0.copy()
            for k, (S, icv) in self.add.items():
                ick = float(icv[t]) if np.isfinite(icv[t]) else 0.0
                ak, zk = orig(S[t].astype(np.float64), inv, omega, ick, z_clip)
                tot = tot + ak
                parts[k] = (ak, zk, ick)
            return tot, z0

        OPT.grinold_alpha = combined
        try:
            tgt = super().__call__(t, h, ctx)
        finally:
            OPT.grinold_alpha = orig
        if parts:
            self._diag(t, ctx, parts)
        if not self._keep_user:
            self.targets.pop((int(ctx["phase"]), int(t)), None)
        return tgt

    # -------------------------------------------------- 诊断(不影响目标)
    def _diag(self, t: int, ctx: dict, parts: dict) -> None:
        rec = self.log[-1]
        for k, (ak, zk, ick) in parts.items():
            rec[f"ic:{k}"] = ick
            rec[f"n_z:{k}"] = int(np.count_nonzero(zk))
        key = (int(ctx["phase"]), int(t))
        if key not in self.targets or rec.get("status") not in ("optimal", "optimal_inaccurate"):
            return
        ci, w_c = self.targets[key]
        w_c = w_c.astype(np.float64)
        wb_full = self.WB[t].astype(np.float64)
        den = float(wb_full.sum() - rec.get("wb_dropped", 0.0))
        scale = float(w_c.sum())
        if den <= 0:
            return
        # 覆盖池外的成分 α_k 必为 0(z 只在可投资池 ⊂ 覆盖池内非零),所以分子不必知道覆盖掩码
        for k, (ak, zk, ick) in parts.items():
            nz = zk != 0
            wb_k = wb_full * nz / den * scale                     # 覆盖池内归一 × Σw,只取 z≠0 的股票
            w_k = np.zeros(len(wb_full)); w_k[ci] = w_c
            rec[f"alpha_ex_ante:{k}"] = float(np.dot(ak, w_k) - np.dot(ak, wb_full) / den * scale)
            rec[f"aw:{k}"] = float(w_k[nz].sum() - wb_k[nz].sum())
            rec[f"exp:{k}"] = float(np.dot(zk[nz], w_k[nz] - wb_k[nz]))
