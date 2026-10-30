"""Controlled Python analysis tool for tabular finance tasks.

This is intentionally narrower than a general REPL. The caller provides a short
analysis script, and the execution environment exposes only pandas/numpy plus
project data-loading helpers. The script must assign its final answer to
`result`.
"""

from __future__ import annotations

import ast
import contextlib
import io
import multiprocessing as mp
from typing import Any


FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
)

FORBIDDEN_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
    "help",
    "globals",
    "locals",
    "vars",
    "dir",
}

FORBIDDEN_ATTRS = {
    "to_csv",
    "to_excel",
    "to_feather",
    "to_hdf",
    "to_json",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "write_text",
    "write_bytes",
    "unlink",
    "rename",
    "replace",
    "mkdir",
    "rmdir",
    "remove",
    "system",
    "popen",
    "spawn",
}


def _validate_code(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise ValueError(f"Unsupported Python syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"Forbidden name in analysis code: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            raise ValueError(f"Forbidden attribute in analysis code: {node.attr}")


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
        import pandas as pd
    except Exception:
        np = None
        pd = None

    if pd is not None and isinstance(value, pd.DataFrame):
        preview = value.head(20).copy()
        return {
            "type": "dataframe",
            "shape": list(value.shape),
            "columns": [str(col) for col in value.columns],
            "records": preview.where(preview.notna(), None).to_dict("records"),
        }
    if pd is not None and isinstance(value, pd.Series):
        preview = value.head(50)
        return {
            "type": "series",
            "name": str(value.name),
            "size": int(value.size),
            "values": preview.where(preview.notna(), None).to_dict(),
        }
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in list(value)[:100]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _worker(code: str, queue: mp.Queue) -> None:
    import numpy as np
    import pandas as pd

    from src.market.data_loader import DEFAULT_TUSHARE_DATA_DIR, load_panel

    stdout = io.StringIO()
    allowed_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    namespace: dict[str, Any] = {
        "__builtins__": allowed_builtins,
        "pd": pd,
        "np": np,
        "load_panel": load_panel,
        "DEFAULT_TUSHARE_DATA_DIR": DEFAULT_TUSHARE_DATA_DIR,
        "result": None,
    }
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, namespace, namespace)
        queue.put({
            "ok": True,
            "stdout": stdout.getvalue()[-4000:],
            "result": _jsonable(namespace.get("result")),
            "error": None,
        })
    except Exception as exc:
        queue.put({
            "ok": False,
            "stdout": stdout.getvalue()[-4000:],
            "result": None,
            "error": f"{type(exc).__name__}: {exc}",
        })


def run_python_analysis(code: str, *, timeout_seconds: int = 8) -> dict[str, Any]:
    """Execute a short pandas/numpy analysis script and return JSON-safe output."""
    try:
        _validate_code(code)
    except Exception as exc:
        return {
            "ok": False,
            "stdout": "",
            "result": None,
            "error": f"ValidationError: {exc}",
        }

    ctx = mp.get_context("fork")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_worker, args=(code, queue))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(1)
        return {
            "ok": False,
            "stdout": "",
            "result": None,
            "error": f"TimeoutError: exceeded {timeout_seconds}s",
        }
    if not queue.empty():
        return queue.get()
    return {
        "ok": False,
        "stdout": "",
        "result": None,
        "error": f"Worker exited with code {process.exitcode}",
    }


def analyze_eps_revision_return_spread(*, timeout_seconds: int = 15) -> dict[str, Any]:
    """Measure whether EPS revisions separate future 5-day returns historically."""
    code = """
panel = load_panel(DEFAULT_TUSHARE_DATA_DIR)
df = panel[
    (panel["is_hs300"] == True)
    & panel["fac_eps_revision_60d"].notna()
    & panel["y_future_5d"].notna()
][["trade_date", "fac_eps_revision_60d", "y_future_5d"]].copy()

rows = []
for trade_date, g in df.groupby("trade_date"):
    if len(g) < 80:
        continue
    low = g["fac_eps_revision_60d"].quantile(0.2)
    high = g["fac_eps_revision_60d"].quantile(0.8)
    down = g[g["fac_eps_revision_60d"] <= low]["y_future_5d"]
    up = g[g["fac_eps_revision_60d"] >= high]["y_future_5d"]
    if len(up) == 0 or len(down) == 0:
        continue
    rows.append({
        "trade_date": trade_date,
        "up_mean": float(up.mean()),
        "down_mean": float(down.mean()),
        "spread": float(up.mean() - down.mean()),
        "sample_size": int(len(g)),
    })

daily = pd.DataFrame(rows)
result = {
    "metric": "fac_eps_revision_60d top20% minus bottom20% future 5d return",
    "start_date": str(daily["trade_date"].min().date()) if not daily.empty else None,
    "end_date": str(daily["trade_date"].max().date()) if not daily.empty else None,
    "n_days": int(len(daily)),
    "avg_top20_future_5d": float(daily["up_mean"].mean()) if not daily.empty else None,
    "avg_bottom20_future_5d": float(daily["down_mean"].mean()) if not daily.empty else None,
    "avg_spread": float(daily["spread"].mean()) if not daily.empty else None,
    "positive_spread_ratio": float((daily["spread"] > 0).mean()) if not daily.empty else None,
}
"""
    return run_python_analysis(code, timeout_seconds=timeout_seconds)

