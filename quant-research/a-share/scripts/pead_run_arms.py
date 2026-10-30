"""PEAD 第 3 步:各臂 × 复制组合 × 相位,逐个独立账户跑执行引擎。

用法:python3 scripts/pead_run_arms.py <scenario> [<scenario> ...]
场景(预注册第 6 节 + 任务追加的冲击模型):
  base_1m      100 万,基准费率(每边 1bp、印花税、每笔最低 5 元),整手            —— 主情景
  stress_1m    100 万,基准 + 每笔买卖各 +5bp                                     —— 费用压力
  base_10m     1000 万,基准                                                       —— 压力情景
  stress_10m   1000 万,基准 + 5bp
  nomin_1m     100 万,整手、无最低佣金                                            —— 诊断(与项目历史口径衔接)
  nolot_1m     100 万,无整手、无最低佣金(= 项目历史"每边 1bp + 印花税"摩擦)       —— 诊断
  imp_1m / imp_10m / imp_100m  基准费率 + afactor.impact.CostModel(Y=1.0,一跳价差)  —— 随规模变化的冲击
账户:E、F 各 1;E_rand 10;B、C、B_F 各 3 个复制组合;每个 × 5 相位。
产出:cache/_tmp/pead_daily_<scenario>.parquet(逐日长表)、cache/_tmp/pead_stats_<scenario>.parquet;
日志 logs/pead_run_arms_*.log。
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from afactor.config import TMP
from afactor.panel import Panel
from afactor import earnings_events as EE, newdata as ND, pead_engine as PE
from pead_build_batches import sealed_daily_basic

SCENARIOS = {
    "base_1m": dict(capital=1e6, min_commission=5.0, stress_bps=0.0, lots=True, impact=None),
    "stress_1m": dict(capital=1e6, min_commission=5.0, stress_bps=5.0, lots=True, impact=None),
    "base_10m": dict(capital=1e7, min_commission=5.0, stress_bps=0.0, lots=True, impact=None),
    "stress_10m": dict(capital=1e7, min_commission=5.0, stress_bps=5.0, lots=True, impact=None),
    "nomin_1m": dict(capital=1e6, min_commission=0.0, stress_bps=0.0, lots=True, impact=None),
    "nolot_1m": dict(capital=1e6, min_commission=0.0, stress_bps=0.0, lots=False, impact=None),
    "imp_1m": dict(capital=1e6, min_commission=5.0, stress_bps=0.0, lots=True, impact=1.0),
    "imp_10m": dict(capital=1e7, min_commission=5.0, stress_bps=0.0, lots=True, impact=1.0),
    "imp_100m": dict(capital=1e8, min_commission=5.0, stress_bps=0.0, lots=True, impact=1.0),
}
ARMS = ["E", "F", "B", "C", "B_F", "E_rand"]


def sealed_raw_close(dates, codes, db: pd.DataFrame) -> np.ndarray:
    """t_signal 未复权收盘价(整手换算用)。db = 已截到 <= 2025-12-31 的 daily_basic(sealed_daily_basic())。"""
    assert db["trade_date"].max() <= EE.CUTOFF
    return (db.pivot(index="trade_date", columns="ts_code", values="close")
            .reindex(index=dates, columns=codes).to_numpy(np.float64))


def account_batches(B: pd.DataFrame) -> dict:
    """一个账户(臂, 复制, 相位)的批次 -> {T_pos: (j, w)};匹配失败槽位(j<0)转现金。"""
    out = {}
    B = B[B["j"] >= 0].sort_values(["T_pos", "slot"])
    for T_pos, g in B.groupby("T_pos", sort=False):
        out[int(T_pos)] = (g["j"].to_numpy(np.int64), g["w"].to_numpy(np.float64))
    return out


def main(scenarios):
    t0 = time.time()
    # 封存(审查 P2):先把截断后的 daily_basic 塞进 newdata 缓存。CostModel -> impact.tick_floor ->
    # ND.raw_close 走 ND._load_daily_basic,缓存为空时会读入含 2026 行的全表(之后 reindex 到研究期,
    # 数值不进入计算,但 2026 行会进内存)。先占住缓存,2026 行就不会被读入。
    db = sealed_daily_basic()
    p = Panel()
    dates, codes = p.dates, p.codes
    raw_px = sealed_raw_close(dates, codes, db)
    adj_px = p.raw("close_px").to_numpy(np.float64)
    star = PE.star_mask(codes)
    Ball = pd.read_parquet(TMP / "pead_batches.parquet")
    Ball = Ball[Ball["arm"].isin(ARMS)]
    groups = {k: g for k, g in Ball.groupby(["arm", "rep", "phase"])}
    cm = None
    for sc in scenarios:
        cfg = SCENARIOS[sc]
        if cfg["impact"] is not None and cm is None:
            from afactor.impact import CostModel
            cm = CostModel(p, impact_y=cfg["impact"])               # 一跳价差(下界)+ 平方根冲击
            assert ND._CACHE["db"]["trade_date"].max() <= EE.CUTOFF, "newdata 缓存里出现了封存期 daily_basic"
            print(f"  CostModel 构建后 newdata 缓存 daily_basic 最大日期 "
                  f"{ND._CACHE['db']['trade_date'].max().date()},{len(ND._CACHE['db'])} 行", flush=True)
        daily, stats = [], []
        for arm in ARMS:
            reps = sorted(Ball.loc[Ball["arm"] == arm, "rep"].unique())
            for rep in reps:
                for ph in range(5):
                    g = groups.get((arm, rep, ph))
                    batches = account_batches(g) if g is not None else {}
                    out = PE.run_account(p, batches, ph, cfg["capital"], raw_px, adj_px, star,
                                         lots=cfg["lots"], min_commission=cfg["min_commission"],
                                         stress_bps=cfg["stress_bps"],
                                         impact_model=(cm if cfg["impact"] is not None else None))
                    d = out["daily"]
                    d.insert(0, "phase", ph); d.insert(0, "rep", int(rep)); d.insert(0, "arm", arm)
                    daily.append(d.reset_index())
                    stats.append(dict(arm=arm, rep=int(rep), phase=ph, **out["stats"]))
            print(f"  [{sc}] {arm} done [{time.time()-t0:.0f}s]", flush=True)
        D = pd.concat(daily, ignore_index=True)
        for c in D.columns:
            if D[c].dtype == np.float64 and c not in ("ret", "gross_ret", "nav_prev", "aum"):
                D[c] = D[c].astype(np.float32)
        D.to_parquet(TMP / f"pead_daily_{sc}.parquet", index=False)
        pd.DataFrame(stats).to_parquet(TMP / f"pead_stats_{sc}.parquet", index=False)
        print(f"[{sc}] 写 pead_daily_{sc}.parquet {len(D)} 行 [{time.time()-t0:.0f}s]", flush=True)
        del D, daily


if __name__ == "__main__":
    sc = sys.argv[1:] or ["base_1m"]
    for s in sc:
        assert s in SCENARIOS, s
    main(sc)
