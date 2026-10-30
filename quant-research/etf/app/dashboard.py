"""Local Streamlit dashboard for saved research runs."""

# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from index_timing.artifacts import discover_runs  # noqa: E402
from index_timing.reporting import _preferred_runs  # noqa: E402


st.set_page_config(page_title="指数择时研究", page_icon="📈", layout="wide")
st.title("指数择时：复现、验证与跨市场迁移")
st.caption("只把无前视、含成本的独立样本外结果作为有效性证据。")

runs = _preferred_runs(discover_runs())
if not runs:
    st.info("尚无结果。先在终端运行 `.tools/uv run index-timing demo`。")
    st.stop()

lookup = {run["run_id"]: run for run in runs}
selected = st.sidebar.selectbox("研究运行", list(lookup), index=len(lookup) - 1)
record = lookup[selected]
metrics = record["metrics"]
metadata = record["metadata"]
daily = pd.read_parquet(record["daily_path"])

if not isinstance(daily.index, pd.DatetimeIndex) and "date" in daily:
    daily = daily.set_index(pd.to_datetime(daily.pop("date")))

st.subheader(selected)
st.caption(
    f"{metadata.get('study', 'unknown')} · {metadata.get('asset_id', 'unknown')} · "
    f"{metadata.get('replication_level', 'unclassified')}"
)
st.markdown(f"**结论：** `{metrics.get('status', 'Not evaluated')}`")
if metadata.get("replication_level") == "independent_validation" and not metadata.get(
    "formal_evidence_eligible"
):
    st.warning("该运行是快速/不完整验证，只用于工程与方向观察，不能作为正式有效性证据。")

metric_specs = [
    ("cagr", "年化收益", "percent"),
    ("sharpe", "Sharpe", "number"),
    ("annual_alpha", "对等敞口回归 Alpha", "percent"),
    ("max_drawdown", "最大回撤", "percent"),
]
columns = st.columns(len(metric_specs))
for column, (key, label, kind) in zip(columns, metric_specs, strict=True):
    value = metrics.get(key, "—")
    if isinstance(value, (float, int)) and kind == "percent":
        rendered = f"{value:.2%}"
    elif isinstance(value, (float, int)) and kind == "number":
        rendered = f"{value:.3f}"
    else:
        rendered = str(value)
    column.metric(label, rendered)


def first_column(candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in daily.columns), None)


strategy_col = first_column(
    ("strategy_net_return", "net_return", "strategy_return", "strategy_gross_return")
)
benchmark_col = first_column(("benchmark_return", "asset_return", "execution_return"))
equal_exposure_col = first_column(("equal_exposure_return",))
position_col = first_column(("position", "exposure", "target_position"))

figure = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.72, 0.28],
    vertical_spacing=0.06,
)
if strategy_col:
    equity = (1 + pd.to_numeric(daily[strategy_col], errors="coerce").fillna(0)).cumprod()
    trace = go.Scatter(x=daily.index, y=equity, name="策略", line={"width": 2})
    figure.add_trace(trace, row=1, col=1)
if benchmark_col:
    benchmark = (1 + pd.to_numeric(daily[benchmark_col], errors="coerce").fillna(0)).cumprod()
    figure.add_trace(
        go.Scatter(x=daily.index, y=benchmark, name="买入持有", line={"dash": "dot"}),
        row=1,
        col=1,
    )
if equal_exposure_col:
    equal_exposure = (
        1 + pd.to_numeric(daily[equal_exposure_col], errors="coerce").fillna(0)
    ).cumprod()
    figure.add_trace(
        go.Scatter(
            x=daily.index,
            y=equal_exposure,
            name="等敞口基准",
            line={"dash": "dash"},
        ),
        row=1,
        col=1,
    )
if position_col:
    figure.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily[position_col],
            name="仓位",
            line={"shape": "hv"},
            fill="tozeroy",
        ),
        row=2,
        col=1,
    )
figure.update_yaxes(type="log", title_text="净值", row=1, col=1)
figure.update_yaxes(range=[-0.05, 1.05], tickformat=".0%", title_text="仓位", row=2, col=1)
figure.update_layout(height=620, hovermode="x unified", margin={"l": 30, "r": 20, "t": 20, "b": 20})
st.plotly_chart(figure, width="stretch")

tab_metrics, tab_assumptions, tab_data = st.tabs(["统计证据", "代理假设", "逐日数据"])
with tab_metrics:
    st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "value"}), width="stretch")
with tab_assumptions:
    assumptions = metadata.get("assumptions", [])
    if assumptions:
        for assumption in assumptions:
            st.markdown(f"- {assumption}")
    else:
        st.caption("该运行未记录额外代理假设。")
    st.json(metadata, expanded=False)
with tab_data:
    st.dataframe(daily.tail(500), width="stretch")
