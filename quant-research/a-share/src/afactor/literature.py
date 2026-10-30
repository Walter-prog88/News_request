"""文献因子:把调研产出的公式规格(受限 DSL)编译成可执行因子。

安全性:公式来自子代理生成的文本,只允许 AST 白名单节点(名称/调用/四则/常量),
禁止属性访问、下标、推导式、赋值、导入。命名空间只含 ops 白名单函数与字段解析器。
"""
from __future__ import annotations
import ast, json
import numpy as np
import pandas as pd

from afactor import ops
from afactor.panel import Panel

_ALLOWED_NODES = (
    ast.Expression, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd, ast.Tuple,
)

_FUNCS = {
    "cs_rank": ops.cs_rank, "cs_zscore": ops.cs_zscore, "cs_winsor": ops.cs_winsor,
    "cs_demean": ops.cs_demean, "cs_neutralize": ops.cs_neutralize_1d,
    "ts_mean": ops.ts_mean, "ts_std": ops.ts_std, "ts_sum": ops.ts_sum,
    "ts_delta": ops.ts_delta, "ts_pct": ops.ts_pct, "ts_rank": ops.ts_rank,
    "ts_zscore": ops.ts_zscore, "ts_corr": ops.ts_corr, "ts_cov": ops.ts_cov,
    "ts_max": ops.ts_max, "ts_min": ops.ts_min, "ts_med": ops.ts_med,
    "ts_skew": ops.ts_skew, "ts_kurt": ops.ts_kurt, "ts_argmax": ops.ts_argmax,
    "ts_argmin": ops.ts_argmin, "ts_decay": ops.ts_decay, "ts_delay": ops.ts_delay,
    "ts_ewm": ops.ts_ewm, "ts_downside_std": ops.ts_downside_std,
    "ts_upside_std": ops.ts_upside_std,
    "safe_div": ops.safe_div, "signed_log": ops.signed_log, "signed_sqrt": ops.signed_sqrt,
    "abs": lambda x: x.abs() if hasattr(x, "abs") else abs(x),
    "sign": lambda x: np.sign(x),
    "log": lambda x: ops.signed_log(x),
    "sqrt": lambda x: ops.signed_sqrt(x),
    "maximum": lambda a, b: (a if not hasattr(a, "clip") else a.clip(lower=b)),
    "minimum": lambda a, b: (a if not hasattr(a, "clip") else a.clip(upper=b)),
    "ln": lambda x: ops.signed_log(x),
    "max": lambda a, b=None: (a.clip(lower=b) if b is not None else a),
    "min": lambda a, b=None: (a.clip(upper=b) if b is not None else a),
    "cs_mean": lambda x: x.sub(x.mean(axis=1), axis=0) * 0 + x.mean(axis=1).to_frame().values,
    "mean": lambda x, w=20: ops.ts_mean(x, w),
    "std": lambda x, w=20: ops.ts_std(x, w),
    "rank": ops.cs_rank,
    "delay": ops.ts_delay,
    "corr": ops.ts_corr,
    "decay_linear": ops.ts_decay,
    "clip": lambda x, lo, hi: x.clip(lo, hi),
}

# 公式里常见的窗口自由变量;规格没给数值网格时用默认网格展开
_WINDOW_VARS = {"w", "n", "N", "W", "d", "k", "window", "lookback", "L", "T", "h", "m"}
_DEFAULT_WINDOWS = (5, 20, 60)


class _Fields(dict):
    """按名字懒加载面板字段;找不到就报错(便于统计不可实现的规格)。"""

    def __init__(self, panel: Panel):
        super().__init__()
        self.p = panel
        from afactor import factors as _FA
        _EV = {"ear_car3", "ear_gap", "days_since_ann"}
        self["fu"] = lambda col: (_FA._fu_event(self.p, col) if col in _EV
                                  else _FA._fu(self.p, col))
        self["fuev"] = lambda col: _FA._fu_event(self.p, col)
        self["p"] = panel

    def __missing__(self, k):
        if k in _FUNCS:
            return _FUNCS[k]
        try:
            v = self.p[k]
        except KeyError:
            raise NameError(f"未知字段/函数: {k}")
        self[k] = v
        return v


def validate(expr: str) -> ast.Expression:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"公式含不允许的语法节点 {type(node).__name__}: {expr[:80]}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("公式含双下划线名称")
    return tree


def compile_formula(expr: str):
    tree = validate(expr)
    code = compile(tree, "<factor>", "eval")

    def fn(p: Panel):
        out = eval(code, {"__builtins__": {}}, _Fields(p))
        if isinstance(out, pd.Series):
            raise TypeError("公式返回 Series,应为 DataFrame")
        return out
    return fn


def load_specs(path) -> list[dict]:
    """读取规格 JSON(允许文件中夹杂说明文字,提取首个 JSON 数组)。"""
    txt = open(path).read()
    i, j = txt.find("["), txt.rfind("]")
    if i < 0 or j < 0:
        return []
    try:
        return json.loads(txt[i:j + 1])
    except json.JSONDecodeError:
        rows, depth, start = [], 0, None
        for k, ch in enumerate(txt[i:j + 1], i):
            if ch == "{":
                if depth == 0:
                    start = k
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        rows.append(json.loads(txt[start:k + 1]))
                    except json.JSONDecodeError:
                        pass
        return rows


import re

_UNIMPL = re.compile(r"需分钟重算|不可实现|需自建|无法实现|需要.*数据")


def clean_formula(raw: str) -> str:
    """规格 formula 归一成单个纯表达式。

    处理:`name = expr` 前缀、`//`/`#` 注释、中文说明,以及**多语句**
    (用 `;` 分隔的中间变量,会被内联展开到最终表达式)。
    `_fu(p,'x')` / `_fu_event(p,'x')` 改写成白名单函数 `fu('x')` / `fuev('x')`
    (DSL 禁止双下划线自由名,且 `_fu` 不在命名空间里)。
    """
    s = raw.strip()
    s = re.sub(r"_fu_event\s*\(\s*p\s*,\s*", "fuev(", s)
    s = re.sub(r"_fu_daily\s*\(\s*p\s*,\s*", "fu(", s)
    s = re.sub(r"_fu\s*\(\s*p\s*,\s*", "fu(", s)
    s = re.sub(r"_safe_ratio\s*\(", "safe_div(", s)
    s = re.sub(r"//[^;\n]*", "", s)
    s = re.sub(r"#[^;\n]*", "", s)
    s = re.split(r"[。]", s)[0]
    parts = [x.strip() for x in re.split(r"[;;]", s) if x.strip()]
    env, expr = {}, None
    for part in parts:
        m = re.match(r"^([A-Za-z\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]*)\s*=\s*(.+)$", part)
        if m and not m.group(2).startswith("="):
            env[m.group(1)] = m.group(2).strip()
            expr = m.group(2).strip()
        else:
            expr = part
    if expr is None:
        return ""
    for _ in range(6):                      # 内联中间变量,防循环
        before = expr
        for k, v in env.items():
            if k in ("factor", "\u56e0\u5b50"):
                continue
            expr = re.sub(rf"\b{re.escape(k)}\b", f"({v})", expr)
        if expr == before:
            break
    return expr.strip().rstrip(",")


def compile_with_params(expr: str, params: dict):
    """把数值参数(如 w=20)作为常量注入命名空间后编译。"""
    tree = validate(expr)
    code = compile(tree, "<factor>", "eval")

    def fn(p: Panel):
        ns = _Fields(p)
        ns.update(params)
        out = eval(code, {"__builtins__": {}}, ns)
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"公式返回 {type(out).__name__},应为 DataFrame")
        return out
    return fn


def register_specs(specs: list[dict], registry: dict, prefix: str = "lit_",
                   max_per_spec: int = 4) -> dict:
    """把规格编译进因子注册表,返回 {name: 失败原因} 供如实记账。"""
    failed = {}
    for s in specs:
        name = str(s.get("name", "")).strip()
        raw = str(s.get("formula", "")).strip()
        if not name or not raw:
            continue
        if _UNIMPL.search(raw):
            failed[name] = "规格自述需分钟重算/不可实现"
            continue
        if not name.startswith(prefix):
            name = prefix + name
        expr = clean_formula(raw)
        if not expr:
            failed[name] = "公式为空"
            continue
        # 数值参数网格
        raw_params = s.get("params") or {}
        if not isinstance(raw_params, dict):
            raw_params = {}
        grids = {k: v for k, v in raw_params.items()
                 if isinstance(v, list) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)}
        # 公式里出现但既非字段也非函数、且形似窗口参数的自由名 -> 补默认网格
        try:
            free = {nd.id for nd in ast.walk(ast.parse(expr, mode="eval"))
                    if isinstance(nd, ast.Name)}
        except SyntaxError:
            free = set()
        for v in free & _WINDOW_VARS:
            grids.setdefault(v, list(_DEFAULT_WINDOWS))
        combos = [{}]
        for k, vs in grids.items():
            combos = [dict(c, **{k: v}) for c in combos for v in vs[:max_per_spec]]
        combos = combos[:8]
        ok_any = False
        for c in combos:
            nm = name + ("_" + "_".join(f"{k}{v}" for k, v in c.items()) if c else "")
            if nm in registry:
                continue
            try:
                fn = compile_with_params(expr, c)
                registry[nm] = ("literature", fn,
                                f"{s.get('source','')} | {s.get('logic','')} | {expr} | {c}")
                ok_any = True
            except Exception as e:
                failed[name] = f"{type(e).__name__}: {e}"
        if ok_any:
            failed.pop(name, None)
    return failed
