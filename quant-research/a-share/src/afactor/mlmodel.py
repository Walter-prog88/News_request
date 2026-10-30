"""阶段3:截面模型 —— 特征集 A/B/C 对比 + 滚动验证 + 因子归因分层。

特征集(用户要求的三组对比):
  A = COMP_v5 的 25 个成员(线性打分的"学习版",对照组)
  B = 143 个日频原料 + 基本面 + 新数据面板(约 250 个,**不经 IC 筛选**)
  C = B + 所有过过一致性关的因子(约 800 个)

核心问题:低 IC 的因子在模型里是否有价值?(非线性/U型/交互关系被 Rank IC 漏掉)
归因分层(用户明确要求):把模型认为重要的特征按**原始 |Rank IC|** 分成
  「线性信号」(原本 IC 就高)与「模型特有信号」(原本 IC 低但模型重用),分开报告。

验证:滚动 3 年训练 -> 测下一年(2023/2024/2025 共 3 个 OOS 年份),
      全样本 2020-2025 训练的模型留给 2026 一次性最终考试。
      **样本内指标不报告** —— 因子是用这六年挑的,再用这六年训练,样本内无信息量。
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from afactor.config import LEDGER
from afactor.panel import Panel

MODEL_LEDGER = LEDGER / "model_trials.jsonl"
MAX_FEATURES = 260      # (F,T,N) float32 内存上限约 8.5GB,留出训练与系统余量


# ---------------- 特征集 ----------------
def feature_set(p: Panel, which: str, screen_names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """构建特征字典。返回 {name: DataFrame}。"""
    from afactor import factors as FA, newdata as ND
    FA.build_all()
    out = {}
    if which == "A":
        man = json.load(open(LEDGER / "comp_v5_members.json"))
        for n in man["members"]:
            if n in FA.REGISTRY:
                out[n] = FA.REGISTRY[n][1](p)
        return out
    # B:原料层(不经 IC 筛选)
    skip = {"twap_open10", "twap_tail8", "close_px", "open_auction_px", "vwap"}  # 绝对价格禁用
    # 尾盘口径追加禁用:14:50 出信号时尚未发生的**当日窗口**。
    # amt_tail8 (14:50-14:57) 尤其危险——它正是标签建仓价所在窗口的成交额,
    # 实测 h1 RankIC 从合规版 -0.0507 虚高到 -0.0675(+33%)。
    # 只禁 amt_tail8:它在 TailPanel.raw 里被特判为**不滞后**。
    # amt_close_auction 走通用分支已被 shift(1) 成 T-1 日,14:50 时点合法,禁掉纯属信息损失。
    if getattr(p, "_is_tail", False):
        skip |= {"amt_tail8"}
    for f in p.features:
        if f in skip or f.startswith("pre_") or f == "statind_labels":
            continue
        out[f] = p.raw(f)
    out["ret_c2c"] = p["ret_c2c"]; out["overnight"] = p["overnight"]
    out["turnover_proxy"] = p["turnover_proxy"]; out["adv20_log"] = np.log1p(p["adv20"])
    # 基本面 / 新数据:**必须复用 factors 里的 _fu/_nd/_real**,它们带口径滞后护栏。
    # (审查 M-3:此前直接调 ND/FU,TailPanel 下全部退回主口径滞后 = 前视)
    for c in ["revenue_ttm", "n_income_attr_p_ttm", "operate_profit_ttm", "n_cashflow_act_ttm",
              "revenue_yoy", "n_income_attr_p_yoy", "total_assets", "total_liab",
              "total_hldr_eqy_exc_min_int", "total_assets_avg", "sue_bt", "sue_ta"]:
        try:
            out["fu_" + c] = FA._fu(p, c)
        except Exception as e:
            print(f"  [警告] 基本面特征 {c} 构建失败: {type(e).__name__}: {e}")
    try:
        out["nd_mv"] = np.log1p(FA._real(p, "mv"))
        out["nd_turn"] = FA._real(p, "turn")
        out["nd_circ"] = np.log1p(FA._real(p, "circ"))
        for k in ["fc_pchange", "ex_yoy_np", "ex_vs_fc", "days_since_surprise"]:
            out["nd_" + k] = FA._nd(p, "surprise", k)
        for k in ["tl_net_ratio", "tl_on_list", "tl_count20", "tl_net20", "tl_updown20"]:
            out["nd_" + k] = FA._nd(p, "toplist", k)
    except Exception as e:
        print(f"  [警告] 新数据特征块整体失败: {type(e).__name__}: {e}")
    if which == "B":
        return out
    # C:在 B 之上叠加过关因子。
    # ⚠ 内存约束:X 是 (F, T, N) float32 = F × 1455 × 5565 × 4B ≈ F × 32MB,
    # 16GB 机器上 F 超过约 300 就会耗尽内存并剧烈 swap(实测 F=516 时 swap 打满 32GB)。
    # 故 C 集对追加因子设上限,并优先取"最差滚动窗口 IC"最高者(信息量优先)。
    budget = max(0, MAX_FEATURES - len(out))
    for n in (screen_names or [])[:budget]:
        if n in FA.REGISTRY and n not in out:
            try:
                out[n] = FA.REGISTRY[n][1](p)
            except Exception:
                pass
    return out


# ---------------- 样本构造 ----------------
def build_xy(feats: dict[str, pd.DataFrame], p: Panel, universe: pd.DataFrame,
             horizon: int = 5, mode: str = "open"):
    """截面标准化后堆成 (样本, 特征) 矩阵。标签 = 前瞻收益的截面百分位排名。"""
    from afactor import ops
    dates, codes = p.dates, p.codes
    names = sorted(feats)
    U = universe.reindex(index=dates, columns=codes).fillna(False).to_numpy(bool)
    y_df = p.fwd_ret(horizon, mode=mode).reindex(index=dates, columns=codes)
    ub = universe.reindex(index=dates, columns=codes).fillna(False).astype(bool)
    Y = y_df.where(ub).rank(axis=1, pct=True).to_numpy(np.float32)
    # U   : 出分池(只用 T 时点信息)
    # Ulab: 训练取样池,额外要求标签存在。二者必须分开——若用 Ulab 出分,等于
    #       "用 T 日之后有没有价格"筛 T 日样本,2025 年末 6 天会整体空仓虚增换手。
    Ulab = U & np.isfinite(Y)
    X = np.full((len(names), len(dates), len(codes)), np.nan, np.float32)
    for i, n in enumerate(names):
        f = feats[n].reindex(index=dates, columns=codes).replace([np.inf, -np.inf], np.nan)
        X[i] = ops.cs_rank(f.where(universe)).to_numpy(np.float32)   # 截面秩,天然抗极值
    return X, Y, U, Ulab, names


def purge_train(train_mask: np.ndarray, horizon: int) -> np.ndarray:
    """purge/embargo:砍掉训练窗末尾 horizon+1 天。

    T 日的标签用 twap[T+1+h]/twap[T+1],最远触及 T+1+h。若不砍,训练窗末尾约
    horizon+1 天的标签会"看到"测试年开头的价格(实测每窗 6 天、占训练样本 0.83%)。
    量级虽小,但属标准的 purge 问题,修掉以杜绝质疑。
    """
    idx = np.flatnonzero(train_mask)
    if len(idx) <= horizon + 1:
        return train_mask
    out = train_mask.copy()
    out[idx[-(horizon + 1):]] = False
    return out


def flatten(X, Y, U, rows: np.ndarray):
    """把选定日期的样本拉平。X (F,T,N)。第三个返回值是每个样本的**日序号**,
    供 fit_ts 做时序验证切分(随机切分会让同日截面同伴泄漏)。"""
    sub = U[rows]
    idx = np.flatnonzero(sub.ravel())
    Xf = X[:, rows].reshape(X.shape[0], -1)[:, idx].T
    Yf = Y[rows].ravel()[idx]
    day = np.repeat(rows.astype(np.int32), sub.shape[1]).ravel()[idx]
    return Xf, Yf, day          # NaN 原样保留:GBT 原生支持,Ridge 在 make_model 里补 imputer


# ---------------- 模型 ----------------
def fit_ts(model, Xtr, Ytr, day_id, val_frac: float = 0.1, purge: int = 6):
    """按时间切验证集再拟合。sklearn 的 validation_fraction 是**随机**切分,
    同日截面同伴会落进验证集,早停因此从未触发(实测 n_iter_ 恒等于 max_iter)。
    改成时序切分并 purge,OOS IC 0.1044 -> 0.1101、耗时减半。"""
    from sklearn.base import clone
    days = np.unique(day_id)
    if len(days) < 50 or not hasattr(model, "staged_predict"):   # Ridge 也有 max_iter,不能用它判据
        model.fit(Xtr, Ytr); return model
    cut = days[int(len(days) * (1 - val_frac))]
    tr = day_id < cut - purge
    va = day_id >= cut
    if tr.sum() < 1000 or va.sum() < 500:
        model.fit(Xtr, Ytr); return model
    m = clone(model); m.set_params(early_stopping=False)
    m.fit(Xtr[tr], Ytr[tr])
    # 用验证集选迭代数,再在全训练集上重训到该迭代数
    errs = np.asarray([float(np.mean((s_ - Ytr[va]) ** 2)) for s_ in m.staged_predict(Xtr[va])])
    # 一倍标准误规则:验证 MSE 曲线在最优点附近几乎平坦,直接 argmin 等于对噪声抽样
    # (实测 n_best 在 55~185 之间乱跳,给 A/B/C 的 IC 对比注入 ±0.001 噪声,
    #  而 A/B/C 的真实差距只有 0.005~0.02)。取"落入最优值 1 个标准误内的最小迭代数",
    # 既降噪又偏保守。
    ei = int(np.argmin(errs))
    thr = errs[ei] + np.std(errs[max(0, ei - 20):ei + 21]) / max(np.sqrt(21), 1.0)
    cand = np.flatnonzero(errs <= thr)
    n_best = int(cand[0]) + 1 if len(cand) else ei + 1
    n_best = max(n_best, 20)
    final = clone(model); final.set_params(early_stopping=False, max_iter=n_best)
    final.fit(Xtr, Ytr)
    final._n_iter_chosen = n_best
    return final


def make_model(kind: str, seed: int = 0):
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        # X 是 cs_rank 后中心化到 [-0.5,0.5],填 0 = 填截面中位数(线性模型的合理默认)
        return make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0),
                             Ridge(alpha=100.0))
    if kind == "gbt":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_depth=4, max_iter=300, learning_rate=0.05, l2_regularization=1.0,
            min_samples_leaf=200, early_stopping=True, validation_fraction=0.1,
            random_state=seed)
    if kind == "gbt_deep":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_depth=6, max_iter=500, learning_rate=0.03, l2_regularization=2.0,
            min_samples_leaf=100, early_stopping=True, validation_fraction=0.1,
            random_state=seed)
    raise ValueError(kind)


def predict_panel(model, X, U, dates, codes) -> pd.DataFrame:
    """对全期做预测,返回打分面板。"""
    F, T, N = X.shape
    out = np.full((T, N), np.nan, np.float32)
    for t in range(T):
        m = U[t]
        if m.sum() < 50:
            continue
        out[t, m] = model.predict(X[:, t, m].T)   # NaN 原样传入,与训练口径一致
    return pd.DataFrame(out, index=dates, columns=codes)


# ---------------- 归因:线性信号 vs 模型特有信号 ----------------
def permutation_importance(model, X, Y, U, rows, names, n_repeat: int = 1,
                           seed: int = 0) -> pd.Series:
    """置换重要性:打乱某特征后模型预测与标签的秩相关下降多少。

    比树模型自带的 split importance 更可信(后者偏向高基数特征)。
    """
    from scipy.stats import rankdata
    rng = np.random.default_rng(seed)
    Xf, Yf, _ = flatten(X, Y, U, rows)
    base = np.corrcoef(rankdata(model.predict(Xf)), rankdata(Yf))[0, 1]
    imp = {}
    for i, n in enumerate(names):
        drops = []
        for _ in range(n_repeat):
            Xp = Xf.copy()
            Xp[:, i] = Xp[rng.permutation(len(Xp)), i]
            s = np.corrcoef(rankdata(model.predict(Xp)), rankdata(Yf))[0, 1]
            drops.append(base - s)
        imp[n] = float(np.mean(drops))
    return pd.Series(imp).sort_values(ascending=False)


def attribute_importance(imp: pd.Series, ic_lookup: dict, ic_hi: float = 0.02) -> pd.DataFrame:
    """按原始 |Rank IC| 把重要特征分成两类(用户明确要求的报告口径)。

    线性信号   : 原始 |IC| >= ic_hi,模型也重用 -> 模型只是复现了已知的线性关系
    模型特有信号: 原始 |IC| <  ic_hi,但模型重用 -> **非线性/U型/交互**,是 B/C 实验的核心产出
    """
    d = pd.DataFrame({"importance": imp})
    d["orig_ic"] = [ic_lookup.get(n, np.nan) for n in d.index]
    d["abs_ic"] = d["orig_ic"].abs()
    d["类别"] = np.where(d["abs_ic"].isna(), "无IC记录(原料层)",
                        np.where(d["abs_ic"] >= ic_hi, "线性信号", "模型特有信号"))
    return d
