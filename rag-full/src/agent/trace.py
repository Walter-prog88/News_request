"""Workflow tracing for research-agent tool calls."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _summarize(value: Any) -> dict[str, Any]:
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is not None and isinstance(value, pd.DataFrame):
        return {
            "type": "dataframe",
            "rows": int(len(value)),
            "columns": [str(col) for col in value.columns[:12]],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "count": len(value),
            "sample_keys": sorted(value[0].keys())[:12] if value and isinstance(value[0], dict) else [],
        }
    if isinstance(value, dict):
        summary = {
            "type": "dict",
            "keys": sorted(str(key) for key in value.keys())[:20],
        }
        if "ok" in value:
            summary["ok"] = bool(value.get("ok"))
        if "error" in value and value.get("error"):
            summary["error"] = str(value.get("error"))[:300]
        return summary
    return {"type": type(value).__name__, "repr": repr(value)[:300]}


class WorkflowTrace:
    """Collect compact timing and status records for a workflow run."""

    def __init__(self, *, stock_code: str, theme: str, question: str) -> None:
        self.run_id = uuid.uuid4().hex[:12]
        self.stock_code = stock_code
        self.theme = theme
        self.question = question
        self.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.events: list[dict[str, Any]] = []

    def call(self, name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a tool, record latency/status, and re-raise failures."""
        started = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            self.events.append({
                "tool": name,
                "ok": False,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        self.events.append({
            "tool": name,
            "ok": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "summary": _summarize(result),
        })
        return result

    def add_event(self, name: str, result: Any, *, ok: bool = True) -> None:
        self.events.append({
            "tool": name,
            "ok": ok,
            "latency_ms": None,
            "summary": _summarize(result),
        })

    def to_dict(self) -> dict[str, Any]:
        total_ms = sum(event.get("latency_ms") or 0 for event in self.events)
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "stock_code": self.stock_code,
            "theme": self.theme,
            "question": self.question,
            "tool_call_count": len(self.events),
            "failed_tool_count": sum(1 for event in self.events if not event.get("ok")),
            "total_recorded_latency_ms": round(total_ms, 2),
            "events": self.events,
        }


def save_trace(trace: dict[str, Any] | WorkflowTrace, *, output_dir: str | Path = PROJECT_ROOT / "reports" / "traces") -> Path:
    """Persist a workflow trace JSON artifact."""
    data = trace.to_dict() if isinstance(trace, WorkflowTrace) else trace
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stock = str(data.get("stock_code") or "unknown").replace(".", "_")
    run_id = str(data.get("run_id") or uuid.uuid4().hex[:12])
    path = output / f"trace_{stock}_{run_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path

