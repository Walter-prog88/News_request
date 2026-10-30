"""前向检验脚本的公共件(前缀 fwd):COMP_v5 5 分钟口径打分、日志、账本写入。

COMP_v5 打分 = scripts/m5_bridge.py::_comp_score(panel, members, signs)(冻结代码,原样调用),成员与符号来自
research/ledger/comp_v5_members.json(经 m5_bridge.load_members;注册表里实际 24 个成员)。在前向面板上计算时
必须处于 fwd_panel.forward_context() 内(基本面成员 fu_roe / fu_rev_yoy 经 _fu 读财报,截止日放宽到 2026-10-09,
财报按预注册 §4.3 原规则沿用最后已知报告期)。打分在信号日 t 收盘可知;2026-01 ~ 08 的打分只作特征,不算 IC。
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from afactor.config import TMP, LEDGER

SCORE_FILE = TMP / "fwd_score5.parquet"


def panel_stamp() -> dict:
    """前向面板版本戳:_meta.json 的构建时间 + 统计行业标签文件的修改时间。缓存(打分、风险模型)只在戳相同时复用
    —— 审查 P0 之后面板重建过一次,旧缓存必须失效。"""
    import json
    from afactor.fwd_panel import PANEL_DIR
    meta = json.load(open(PANEL_DIR / "_meta.json", encoding="utf-8"))
    return dict(panel_built_at=meta.get("built_at"),
                labels_mtime_ns=int(os.stat(PANEL_DIR / "statind_labels.parquet").st_mtime_ns))


def stamp_ok(stamp_file) -> bool:
    import json
    try:
        return json.load(open(stamp_file, encoding="utf-8")) == panel_stamp()
    except Exception:
        return False


def write_stamp(stamp_file) -> None:
    import json
    json.dump(panel_stamp(), open(stamp_file, "w", encoding="utf-8"), ensure_ascii=False)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def comp_score(p, cache_file=SCORE_FILE, force: bool = False) -> pd.DataFrame:
    """前向面板上的 COMP_v5 打分(5 分钟口径)。缓存到 cache/_tmp/fwd_score5.parquet(行列与面板对齐、且面板版本戳相同才复用)。"""
    stamp = None if cache_file is None else cache_file.with_suffix(".stamp.json")
    if cache_file is not None and cache_file.exists() and not force and stamp_ok(stamp):
        s = pd.read_parquet(cache_file)
        if s.index.equals(p.dates) and s.columns.equals(p.codes):
            return s
    import m5_bridge as MB
    members, signs, missing = MB.load_members()
    t0 = time.time()
    s = MB._comp_score(p, members, signs).reindex(index=p.dates, columns=p.codes).astype(np.float32)
    log(f"[score] COMP_v5 打分:{len(members)} 个成员(不在注册表:{missing}),{time.time() - t0:.0f}s")
    if cache_file is not None:
        s.to_parquet(cache_file)
        write_stamp(stamp)
    return s


def ledger_append(path, rows: list[dict], key_cols=("section", "item"), drop_where=None) -> pd.DataFrame:
    """按键覆盖式追加到 csv 账本。
    drop_where:可选函数 old_df -> 布尔 Series,为 True 的旧行在写入前删除(审查 P2:条目改名后旧行不会被键覆盖,
    会留下与新结果矛盾的陈旧行 —— 各脚本用它在写入前清掉"本次会重写的整节")。"""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        if len(old) and drop_where is not None:
            old = old[~drop_where(old).to_numpy(bool)]
        if len(old) and all(k in old.columns for k in key_cols):
            k_old = old[list(key_cols)].astype(str).agg("|".join, axis=1)
            k_new = set(new[list(key_cols)].astype(str).agg("|".join, axis=1))
            old = old[~k_old.isin(k_new)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return new
