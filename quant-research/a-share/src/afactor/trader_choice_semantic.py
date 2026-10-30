"""Static semantic-theme upper bound for the trader-choice research line.

This module asks a deliberately easier, diagnostic question: *if the relevant
industry / supply-chain classification were already known, would theme
strength and within-theme leadership recover the disclosed choices?*

The stock-to-theme map is therefore explicitly an ``oracle/static`` map.  It
may use text published after a historical decision date, but it may only use
public company metadata: the local SW snapshot, analyst-report industry/title
text, and index names.  Transaction events, disclosed buys, realised returns,
and symbol/name allow-lists are neither accepted nor read.  This makes the
module an upper-bound experiment, not a deployable point-in-time classifier.

Once membership is frozen, every numerical feature for decision session ``T``
is calculated from panel columns whose cutoff is earlier than ``T``.  The
default columns are the T-1 price/amount fields produced by
``trader_choice_data.build_base_panel``.  No T close or T amount is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


STATIC_ORACLE_MAPPING_VERSION = "static-oracle-nine-chain-v3"
STATIC_ORACLE_CLOCK = (
    "oracle/static public-text membership; historical metadata backfill allowed"
)
SEMANTIC_FEATURE_CLOCK = "theme aggregates and leader scores use T-1 fields only"


@dataclass(frozen=True)
class ThemeDefinition:
    """A general text rule; it must never contain a stock code or stock name.

    ``keywords`` are high-precision direct matches.  Each item in
    ``cooccurrence_rules`` is an AND across keyword groups and an OR within a
    group.  For example ``(("散热", "温控"), ("服务器", "数据中心"))`` requires
    at least one thermal term and one compute-infrastructure term in the same
    public metadata row.  This keeps generic words from becoming themes by
    themselves.
    """

    theme_id: str
    theme_name: str
    keywords: tuple[str, ...]
    cooccurrence_rules: tuple[tuple[tuple[str, ...], ...], ...] = ()
    exclude_keywords: tuple[str, ...] = ()
    context_exclude_keywords: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.theme_id or not re.fullmatch(r"[a-z0-9_]+", self.theme_id):
            raise ValueError("theme_id must contain only lower-case letters, digits, and _")
        if not self.theme_name or (not self.keywords and not self.cooccurrence_rules):
            raise ValueError("theme_name and at least one matching rule are required")
        cleaned = tuple(dict.fromkeys(str(value).strip() for value in self.keywords))
        if any(not value for value in cleaned):
            raise ValueError(f"theme {self.theme_id!r} contains an empty keyword")
        cleaned_rules = tuple(
            tuple(
                tuple(dict.fromkeys(str(value).strip() for value in group))
                for group in rule
            )
            for rule in self.cooccurrence_rules
        )
        if any(
            not rule or any(not group or any(not value for value in group) for group in rule)
            for rule in cleaned_rules
        ):
            raise ValueError(f"theme {self.theme_id!r} contains an empty context rule")
        cleaned_excludes = tuple(
            dict.fromkeys(str(value).strip() for value in self.exclude_keywords)
        )
        if any(not value for value in cleaned_excludes):
            raise ValueError(f"theme {self.theme_id!r} contains an empty exclusion")
        cleaned_context_excludes = tuple(
            dict.fromkeys(
                str(value).strip() for value in self.context_exclude_keywords
            )
        )
        if any(not value for value in cleaned_context_excludes):
            raise ValueError(
                f"theme {self.theme_id!r} contains an empty context exclusion"
            )
        all_terms = (
            *cleaned,
            *(value for rule in cleaned_rules for group in rule for value in group),
            *cleaned_excludes,
            *cleaned_context_excludes,
        )
        # Six-digit entries are almost certainly accidental symbol allow-lists.
        if any(
            re.fullmatch(r"\d{6}(?:\.(?:SH|SZ|BJ))?", value, re.I)
            for value in all_terms
        ):
            raise ValueError(f"theme {self.theme_id!r} contains a forbidden stock code")
        object.__setattr__(self, "keywords", cleaned)
        object.__setattr__(self, "cooccurrence_rules", cleaned_rules)
        object.__setattr__(self, "exclude_keywords", cleaned_excludes)
        object.__setattr__(
            self, "context_exclude_keywords", cleaned_context_excludes
        )


# These nine supply chains are frozen and intentionally multi-label.  Terms are
# products, technologies, processes, or industries—not companies or symbols.
# The taxonomy is an explanatory hypothesis proposed after the historical
# cases were already known.  Its rules are nevertheless applied uniformly to
# the full market and contain no company or symbol allow-list; this is why the
# result is an in-sample oracle upper bound, not independent validation.
DEFAULT_STATIC_ORACLE_THEMES: tuple[ThemeDefinition, ...] = (
    ThemeDefinition(
        "pcb",
        "PCB产业链",
        (
            "PCB", "印制电路板", "印刷电路板", "线路板", "覆铜板", "铜箔",
            "HDI", "FPC", "CCL", "IC载板", "封装基板", "高多层板",
            "电子玻纤", "电子布", "玻纤布", "ABF载板", "ABF基板", "BT载板",
            "BT基板", "PCBA", "电子装联",
        ),
        cooccurrence_rules=(
            (("电路板", "电路基板"), ("制造", "材料", "设备", "产能", "订单")),
        ),
    ),
    ThemeDefinition(
        "ai_compute_network",
        "AI算力与高速网络",
        (
            "算力", "智算", "AI服务器", "服务器", "数据中心", "光模块", "CPO",
            "光通信", "高速互联", "交换机", "路由器", "通信设备", "云计算",
            "算力租赁", "算力中心", "GPU", "AI芯片", "加速卡", "800G", "1.6T",
            "5G通信",
        ),
        cooccurrence_rules=(
            (("人工智能", "AI", "大模型", "生成式AI"),
             ("训练", "推理", "服务器", "网络", "集群", "云端", "基础设施")),
            (("光纤", "光器件", "连接器"),
             ("高速", "数据中心", "算力", "800G", "1.6T")),
            (("电源", "UPS", "供配电", "配电"),
             ("算力", "智算", "服务器", "数据中心", "机房", "机柜")),
        ),
    ),
    ThemeDefinition(
        "semiconductor",
        "半导体产业链",
        (
            "半导体", "集成电路", "芯片", "晶圆", "光刻", "刻蚀", "薄膜沉积",
            "封装测试", "封测", "存储器", "GPU", "MCU", "ASIC", "SoC", "IGBT",
            "SiC", "GaN", "功率器件", "科创芯片", "电子特气", "光刻胶",
        ),
        cooccurrence_rules=(
            (("设备", "材料", "零部件"), ("晶圆", "芯片", "半导体")),
        ),
    ),
    ThemeDefinition(
        "liquid_cooling",
        "液冷与算力温控",
        (
            "液冷", "浸没式冷却", "冷却液", "液冷服务器", "液冷机柜", "液冷板",
            "冷板式散热", "相变冷却", "CDU",
        ),
        cooccurrence_rules=(
            (("散热", "温控", "热管理", "制冷", "冷板"),
             ("服务器", "数据中心", "算力", "机房", "芯片", "机柜")),
            (("泵", "泵阀", "阀门", "换热器", "冷却塔", "压缩机"),
             ("服务器", "数据中心", "算力", "机房", "机柜")),
        ),
        exclude_keywords=("冷轧板",),
    ),
    ThemeDefinition(
        "satellite_defense",
        "卫星通信与军工",
        (
            "卫星", "卫星互联网", "商业航天", "低轨", "星链", "北斗", "军工",
            "国防", "航空航天", "航发", "军用", "相控阵雷达", "导弹", "舰船",
            "低空经济", "军贸",
        ),
        cooccurrence_rules=(
            (("雷达", "无人机", "通信"),
             ("军用", "航空", "卫星", "导弹", "防务", "国防")),
        ),
    ),
    ThemeDefinition(
        "robotics_high_end_manufacturing",
        "机器人与高端制造",
        (
            "机器人", "人形机器人", "工业机器人", "工业母机", "数控机床",
            "机床制造", "减速器", "丝杠", "伺服系统", "机器视觉", "自动化设备",
            "高端装备", "智能装备",
        ),
        cooccurrence_rules=(
            (("自动化", "智能制造"),
             ("工业", "生产线", "工厂", "设备", "制造")),
            (("伺服", "执行器", "传感器"),
             ("机器人", "工业控制", "自动化")),
        ),
    ),
    ThemeDefinition(
        "power_grid_electronics",
        "电力电网与电力电子",
        (
            "电网", "特高压", "配电网", "智能电网", "输变电", "变压器",
            "继电保护", "逆变器", "电力电子", "电能质量", "充电桩", "新型电力",
            "柔性直流", "柔直", "储能变流器", "电网设备",
        ),
        cooccurrence_rules=(
            (("电力", "电气"),
             ("设备", "系统", "自动化", "电子", "控制", "变换", "转换", "电源")),
        ),
    ),
    ThemeDefinition(
        "advanced_materials",
        "先进材料",
        (
            "新材料", "先进材料", "聚氨酯", "MDI", "TDI", "TPU", "功能薄膜",
            "光学薄膜", "OCA", "电子胶", "导热材料", "低介电", "超硬材料",
            "人造金刚石", "碳纤维", "复合材料", "陶瓷复材", "靶材", "高温合金",
            "钛合金", "稀土永磁", "稀土功能材料", "电子特气", "光刻胶",
            "半导体材料", "陶瓷材料", "膜材料", "特种材料", "超导材料",
            "化工新材料", "铜箔", "覆铜板",
        ),
        cooccurrence_rules=(
            (("材料", "合金", "纤维", "薄膜"),
             ("高端", "先进", "半导体", "航空航天", "新能源", "电子")),
        ),
    ),
    ThemeDefinition(
        "ai_applications",
        "AI应用",
        (
            "AI应用", "人工智能应用", "大模型应用", "智能客服", "智能营销",
            "AI医疗", "AI教育", "AI办公", "AI软件",
        ),
        cooccurrence_rules=(
            (("人工智能", "生成式AI", "AIGC", "AI", "大模型", "AI Agent",
              "智能体", "多模态", "机器学习", "计算机视觉", "语音识别"),
             ("应用", "平台", "软件", "订单", "收入", "落地", "商业化", "解决方案",
              "赋能", "营销", "广告", "办公", "教育", "医疗", "客服", "内容",
              "游戏", "搜索", "金融", "工业", "政务", "驾驶", "质检")),
        ),
        context_exclude_keywords=(
            "算力", "智算", "服务器", "数据中心", "光模块", "CPO", "交换机",
            "路由器", "通信设备", "液冷", "GPU", "AI芯片", "半导体",
        ),
    ),
)


@dataclass(frozen=True)
class StaticOracleSemanticConfig:
    """Column and minimum-coverage settings for the upper-bound experiment."""

    date_col: str = "decision_date"
    symbol_col: str = "symbol"
    eligible_col: str = "eligible_for_ranking"
    cutoff_col: str = "feature_cutoff_date"
    theme_min_members: int = 3
    theme_min_components: int = 3
    leader_min_components: int = 3
    theme_component_columns: tuple[str, ...] = (
        "mom5",
        "mom20",
        "amt_ratio20",
        "ma20_dist",
    )
    leader_component_columns: tuple[str, ...] = (
        "mom10",
        "mom60",
        "liq20",
        "amt_ratio20",
        "ma20_dist",
        # These are used automatically if a caller has already joined a
        # strictly lagged daily_basic panel.
        "db_turnover_rate_t1",
        "db_volume_ratio_t1",
        "db_circ_mv_t1",
    )

    def __post_init__(self) -> None:
        if self.theme_min_members <= 0:
            raise ValueError("theme_min_members must be positive")
        if self.theme_min_components <= 0 or self.leader_min_components <= 0:
            raise ValueError("component minimums must be positive")
        if not self.theme_component_columns or not self.leader_component_columns:
            raise ValueError("component column lists must not be empty")


_FORBIDDEN_SOURCE_COLUMNS = {
    "event_id",
    "event_type",
    "label",
    "label_confidence",
    "direction_confidence",
    "date_confidence",
    "is_selected",
    "selected",
    "buy_flag",
    "sell_flag",
    "trade_direction",
    "realized_pnl",
    "realised_pnl",
}


def _read_frame(value: pd.DataFrame | str | Path | None, *, name: str) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        result = value.copy()
    else:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        result = pd.read_csv(path)
    normalized = {str(column).strip().lower() for column in result.columns}
    forbidden = sorted(normalized.intersection(_FORBIDDEN_SOURCE_COLUMNS))
    if forbidden:
        raise ValueError(
            f"{name} contains transaction/label columns that this semantic "
            f"module is forbidden to read: {forbidden}"
        )
    return result


def _standard_symbol(value: object) -> str | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    if not text:
        return pd.NA
    if "." in text:
        left, right = text.split(".", 1)
        if left in {"SH", "SZ", "BJ"} and right.isdigit():
            return f"{right.zfill(6)}.{left}"
        if right in {"SH", "SZ", "BJ"} and left.isdigit():
            return f"{left.zfill(6)}.{right}"
    # CSV readers commonly turn 000001 into 1 or 1.0.
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    digits = "".join(character for character in text if character.isdigit())
    if not digits or len(digits) > 6:
        return pd.NA
    digits = digits.zfill(6)
    if digits.startswith(("6", "900")):
        exchange = "SH"
    elif digits.startswith(("0", "3", "200")):
        exchange = "SZ"
    elif digits.startswith(("4", "8", "9")):
        exchange = "BJ"
    else:
        return pd.NA
    return f"{digits}.{exchange}"


def _keyword_pattern(keywords: Sequence[str]) -> str:
    expressions = []
    for keyword in sorted(set(keywords), key=len, reverse=True):
        escaped = re.escape(keyword)
        if keyword.isascii() and keyword.replace("_", "").isalnum():
            expressions.append(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])")
        else:
            expressions.append(escaped)
    return "(?:" + "|".join(expressions) + ")"


def _matching_keywords(text: str, definition: ThemeDefinition) -> str:
    matches = []
    terms = (
        *definition.keywords,
        *(
            value
            for rule in definition.cooccurrence_rules
            for group in rule
            for value in group
        ),
    )
    for keyword in dict.fromkeys(terms):
        if re.search(_keyword_pattern((keyword,)), text, flags=re.I):
            matches.append(keyword)
    return "|".join(matches)


def _definition_match_masks(
    text: pd.Series,
    definition: ThemeDefinition,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return final, direct, and contextual match masks for one rule."""

    direct = pd.Series(False, index=text.index, dtype=bool)
    if definition.keywords:
        direct = text.str.contains(
            _keyword_pattern(definition.keywords), case=False, regex=True, na=False
        )
    contextual = pd.Series(False, index=text.index, dtype=bool)
    for rule in definition.cooccurrence_rules:
        rule_match = pd.Series(True, index=text.index, dtype=bool)
        for group in rule:
            rule_match &= text.str.contains(
                _keyword_pattern(group), case=False, regex=True, na=False
            )
        contextual |= rule_match
    if definition.context_exclude_keywords:
        context_excluded = text.str.contains(
            _keyword_pattern(definition.context_exclude_keywords),
            case=False,
            regex=True,
            na=False,
        )
        contextual &= ~context_excluded
    matched = direct | contextual
    if definition.exclude_keywords:
        excluded = text.str.contains(
            _keyword_pattern(definition.exclude_keywords),
            case=False,
            regex=True,
            na=False,
        )
        matched &= ~excluded
        direct &= ~excluded
        contextual &= ~excluded
    return matched, direct, contextual


def _assert_no_company_name_terms(
    themes: Sequence[ThemeDefinition],
    source_frames: Sequence[pd.DataFrame],
) -> None:
    """Reject exact company-name rules without using names for classification."""

    company_name_columns = (
        "name",
        "stockName",
        "证券简称",
        "成分券名称",
        "constituent_name",
    )
    company_names: set[str] = set()
    for frame in source_frames:
        for column in company_name_columns:
            if column in frame.columns:
                company_names.update(
                    frame[column]
                    .astype("string")
                    .dropna()
                    .str.strip()
                    .str.casefold()
                    .loc[lambda values: values.ne("")]
                    .astype(str)
                )
    if not company_names:
        return
    for definition in themes:
        positive_terms = (
            *definition.keywords,
            *(
                value
                for rule in definition.cooccurrence_rules
                for group in rule
                for value in group
            ),
        )
        theme_label = definition.theme_name.strip().casefold()
        forbidden = sorted(
            {
                term
                for term in positive_terms
                if term.strip().casefold() in company_names
                # A generic taxonomy term can also happen to be a listed
                # company's short name (notably "机器人").  It is admissible
                # only when it is literally part of the frozen theme label;
                # arbitrary company-specific terms remain forbidden.
                and term.strip().casefold() not in theme_label
            }
        )
        if forbidden:
            raise ValueError(
                f"theme {definition.theme_id!r} contains forbidden company-name "
                f"rules: {forbidden}"
            )


def _first_existing(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> str:
    for column in columns:
        if column in frame.columns:
            return column
    raise ValueError(f"{name} needs one of these symbol columns: {list(columns)}")


def _source_theme_matches(
    frame: pd.DataFrame,
    *,
    source_name: str,
    symbol_columns: Sequence[str],
    text_columns: Sequence[str],
    themes: Sequence[ThemeDefinition],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    symbol_column = _first_existing(frame, symbol_columns, name=source_name)
    available_text = [column for column in text_columns if column in frame.columns]
    if not available_text:
        raise ValueError(
            f"{source_name} needs at least one public-text column: {list(text_columns)}"
        )
    symbols = frame[symbol_column].map(_standard_symbol)
    text = frame[available_text[0]].astype("string").fillna("")
    for column in available_text[1:]:
        text = text.str.cat(frame[column].astype("string").fillna(""), sep=" | ")
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()

    outputs: list[pd.DataFrame] = []
    for definition in themes:
        matched, direct, contextual = _definition_match_masks(text, definition)
        mask = symbols.notna() & matched
        if not mask.any():
            continue
        match_rule = pd.Series("context", index=text.index, dtype="string")
        match_rule.loc[direct] = "direct"
        match_rule.loc[direct & contextual] = "direct+context"
        matches = pd.DataFrame(
            {
                "symbol": symbols.loc[mask].astype(str),
                "theme_id": definition.theme_id,
                "theme_name": definition.theme_name,
                "oracle_static_source": source_name,
                "oracle_static_match_rule": match_rule.loc[mask],
                "oracle_static_example_text": text.loc[mask].str.slice(0, 240),
            }
        )
        # One evidence sample per stock/source/theme is enough for the audit
        # ledger and avoids carrying 100k report rows into the research panel.
        matches = matches.drop_duplicates(
            ["symbol", "theme_id", "oracle_static_source"], keep="first"
        )
        matches["oracle_static_matched_keywords"] = matches[
            "oracle_static_example_text"
        ].map(lambda value: _matching_keywords(str(value), definition))
        outputs.append(matches)
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def _join_unique(values: Iterable[object], separator: str = "|") -> str:
    cleaned = sorted(
        {
            str(value).strip()
            for value in values
            if value is not None and not pd.isna(value) and str(value).strip()
        }
    )
    return separator.join(cleaned)


def build_static_oracle_theme_membership(
    stock_sw_industry: pd.DataFrame | str | Path,
    analyst_reports: pd.DataFrame | str | Path | None = None,
    technology_index_constituents: pd.DataFrame | str | Path | None = None,
    *,
    themes: Sequence[ThemeDefinition] = DEFAULT_STATIC_ORACLE_THEMES,
) -> pd.DataFrame:
    """Build an auditable, multi-label stock-to-supply-chain map.

    ``analyst_reports`` may include all locally available publication dates.
    That look-ahead is intentional and is disclosed in every output row.  The
    function accepts no transaction/evidence input and rejects label-like
    columns if they are accidentally supplied as one of the public sources.
    """

    definitions = tuple(themes)
    if not definitions:
        raise ValueError("themes must not be empty")
    ids = [definition.theme_id for definition in definitions]
    if len(ids) != len(set(ids)):
        raise ValueError("theme_id values must be unique")

    sw = _read_frame(stock_sw_industry, name="stock_sw_industry")
    reports = _read_frame(analyst_reports, name="analyst_reports")
    index_constituents = _read_frame(
        technology_index_constituents, name="technology_index_constituents"
    )
    _assert_no_company_name_terms(definitions, (sw, reports, index_constituents))
    source_frames = [
        _source_theme_matches(
            sw,
            source_name="sw_static_snapshot",
            symbol_columns=("ts_code", "code", "symbol"),
            text_columns=("sw_industry", "fine_industry"),
            themes=definitions,
        )
    ]
    if not reports.empty:
        source_frames.append(
            _source_theme_matches(
                reports,
                source_name="analyst_report_static_text",
                symbol_columns=("stockCode", "ts_code", "code", "symbol"),
                text_columns=("indvInduName", "title"),
                themes=definitions,
            )
        )
    if not index_constituents.empty:
        source_frames.append(
            _source_theme_matches(
                index_constituents,
                source_name="technology_index_static_snapshot",
                symbol_columns=("成分券代码", "constituent_code", "ts_code", "symbol"),
                # Deliberately exclude English constituent/company names.  They
                # do not classify a supply chain and create substring noise.
                text_columns=("index_name", "指数名称"),
                themes=definitions,
            )
        )
    evidence = pd.concat(
        [frame for frame in source_frames if not frame.empty],
        ignore_index=True,
    ) if any(not frame.empty for frame in source_frames) else pd.DataFrame()
    columns = [
        "symbol",
        "theme_id",
        "theme_name",
        "oracle_static_sources",
        "oracle_static_source_count",
        "oracle_static_matched_keywords",
        "oracle_static_match_rules",
        "oracle_static_evidence_samples",
        "oracle_static_mapping_version",
        "oracle_static_asof_rule",
        "oracle_static_uses_future_metadata",
    ]
    if evidence.empty:
        return pd.DataFrame(columns=columns)

    grouped = evidence.groupby(["symbol", "theme_id", "theme_name"], sort=True)
    membership = grouped.agg(
        oracle_static_sources=("oracle_static_source", _join_unique),
        oracle_static_source_count=("oracle_static_source", "nunique"),
        oracle_static_matched_keywords=("oracle_static_matched_keywords", _join_unique),
        oracle_static_match_rules=("oracle_static_match_rule", _join_unique),
        oracle_static_evidence_samples=(
            "oracle_static_example_text",
            lambda values: " || ".join(dict.fromkeys(str(value) for value in values))[:1000],
        ),
    ).reset_index()
    membership["oracle_static_mapping_version"] = STATIC_ORACLE_MAPPING_VERSION
    membership["oracle_static_asof_rule"] = STATIC_ORACLE_CLOCK
    membership["oracle_static_uses_future_metadata"] = True
    return membership[columns].sort_values(
        ["symbol", "theme_id"], kind="mergesort"
    ).reset_index(drop=True)


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        return pd.to_numeric(values, errors="coerce").fillna(0).ne(0)
    return (
        values.astype("string")
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
        .fillna(False)
    )


def _prepare_feature_inputs(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
    config: StaticOracleSemanticConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_panel = {config.date_col, config.symbol_col, config.eligible_col}
    missing_panel = sorted(required_panel.difference(panel.columns))
    if missing_panel:
        raise ValueError(f"panel is missing required columns: {missing_panel}")
    required_mapping = {"symbol", "theme_id", "theme_name"}
    missing_mapping = sorted(required_mapping.difference(membership.columns))
    if missing_mapping:
        raise ValueError(f"membership is missing required columns: {missing_mapping}")

    result = panel.copy()
    result[config.date_col] = pd.to_datetime(
        result[config.date_col], errors="coerce"
    ).dt.normalize()
    result[config.symbol_col] = result[config.symbol_col].map(_standard_symbol).astype("string")
    if result[config.date_col].isna().any() or result[config.symbol_col].isna().any():
        raise ValueError("panel contains invalid decision dates or symbols")
    if result.duplicated([config.date_col, config.symbol_col]).any():
        raise ValueError("panel must be unique by decision_date and symbol")
    result[config.eligible_col] = _bool_series(result[config.eligible_col])

    if config.cutoff_col in result.columns:
        cutoff = pd.to_datetime(result[config.cutoff_col], errors="coerce").dt.normalize()
        invalid = cutoff.isna() | cutoff.ge(result[config.date_col])
        if invalid.any():
            raise ValueError(
                f"{config.cutoff_col} must be present and strictly earlier than "
                f"{config.date_col} for every row"
            )

    mapping = membership.copy()
    mapping["symbol"] = mapping["symbol"].map(_standard_symbol).astype("string")
    mapping["theme_id"] = mapping["theme_id"].astype("string").str.strip()
    mapping["theme_name"] = mapping["theme_name"].astype("string").str.strip()
    mapping = mapping.dropna(subset=["symbol", "theme_id", "theme_name"])
    mapping = mapping.drop_duplicates(["symbol", "theme_id"], keep="first")
    if mapping.empty:
        raise ValueError("membership contains no valid stock-theme rows")
    return result, mapping


def _rank_within(
    frame: pd.DataFrame,
    values: str,
    groups: Sequence[str],
    mask: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(frame[values], errors="coerce").where(mask)
    return numeric.groupby([frame[column] for column in groups], sort=False).rank(
        method="average", pct=True, ascending=True
    )


def build_static_oracle_semantic_long(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
    config: StaticOracleSemanticConfig = StaticOracleSemanticConfig(),
) -> pd.DataFrame:
    """Return one row per decision-date / stock / static theme membership."""

    prepared, mapping = _prepare_feature_inputs(panel, membership, config)
    theme_columns = [
        column for column in config.theme_component_columns if column in prepared.columns
    ]
    leader_columns = [
        column for column in config.leader_component_columns if column in prepared.columns
    ]
    if len(theme_columns) < config.theme_min_components:
        raise ValueError(
            "not enough T-1 theme component columns; "
            f"available={theme_columns}, minimum={config.theme_min_components}"
        )
    if len(leader_columns) < config.leader_min_components:
        raise ValueError(
            "not enough T-1 leader component columns; "
            f"available={leader_columns}, minimum={config.leader_min_components}"
        )

    eligible = prepared[config.eligible_col]
    day_pct_columns: list[str] = []
    for column in dict.fromkeys([*theme_columns, *leader_columns]):
        pct_column = f"_oracle_day_pct__{column}"
        prepared[pct_column] = _rank_within(
            prepared, column, [config.date_col], eligible
        )
        day_pct_columns.append(pct_column)

    long = prepared.merge(
        mapping,
        left_on=config.symbol_col,
        right_on="symbol",
        how="inner",
        suffixes=("", "_mapping"),
        validate="many_to_many",
    )
    if config.symbol_col != "symbol":
        long = long.drop(columns=["symbol"]).rename(
            columns={config.symbol_col: "symbol"}
        )
    active = long[config.eligible_col].fillna(False).astype(bool)
    group_columns = [config.date_col, "theme_id", "theme_name"]

    aggregation: dict[str, tuple[str, object]] = {
        "oracle_static_theme_member_count": ("symbol", "nunique"),
    }
    for column in theme_columns:
        aggregation[f"oracle_static_theme_mean_pct__{column}"] = (
            f"_oracle_day_pct__{column}",
            "mean",
        )
    active_long = long.loc[active].copy()
    theme_stats = active_long.groupby(group_columns, sort=True).agg(**aggregation).reset_index()
    if "mom5" in theme_columns:
        breadth = active_long.assign(
            _positive=pd.to_numeric(active_long["mom5"], errors="coerce").gt(0)
        ).groupby(group_columns, sort=True)["_positive"].mean()
        theme_stats = theme_stats.merge(
            breadth.rename("oracle_static_theme_breadth_mom5_positive").reset_index(),
            on=group_columns,
            how="left",
        )
    if "ma20_dist" in theme_columns:
        breadth = active_long.assign(
            _above_ma20=pd.to_numeric(
                active_long["ma20_dist"], errors="coerce"
            ).gt(0)
        ).groupby(group_columns, sort=True)["_above_ma20"].mean()
        theme_stats = theme_stats.merge(
            breadth.rename("oracle_static_theme_breadth_above_ma20").reset_index(),
            on=group_columns,
            how="left",
        )

    strength_components = [
        column
        for column in theme_stats.columns
        if column.startswith("oracle_static_theme_mean_pct__")
        or column.startswith("oracle_static_theme_breadth_")
    ]
    strength_count = theme_stats[strength_components].notna().sum(axis=1)
    theme_stats["oracle_static_theme_component_count"] = strength_count
    theme_stats["oracle_static_theme_strength_raw"] = theme_stats[
        strength_components
    ].mean(axis=1, skipna=True).where(
        strength_count.ge(config.theme_min_components)
        & theme_stats["oracle_static_theme_member_count"].ge(config.theme_min_members)
    )
    theme_stats["oracle_static_theme_strength_pct"] = theme_stats[
        "oracle_static_theme_strength_raw"
    ].groupby(theme_stats[config.date_col], sort=False).rank(
        method="average", pct=True, ascending=True
    )
    long = long.merge(theme_stats, on=group_columns, how="left", validate="many_to_one")

    leader_pct_columns: list[str] = []
    active = long[config.eligible_col].fillna(False).astype(bool)
    valid_theme = long["oracle_static_theme_member_count"].ge(config.theme_min_members)
    for column in leader_columns:
        pct_column = f"oracle_static_leader_pct__{column}"
        long[pct_column] = _rank_within(
            long,
            column,
            [config.date_col, "theme_id"],
            active & valid_theme,
        )
        leader_pct_columns.append(pct_column)
    leader_count = long[leader_pct_columns].notna().sum(axis=1)
    long["oracle_static_leader_component_count"] = leader_count
    long["oracle_static_theme_leader_score"] = long[leader_pct_columns].mean(
        axis=1, skipna=True
    ).where(leader_count.ge(config.leader_min_components))
    long["oracle_static_theme_x_leader"] = (
        long["oracle_static_theme_strength_pct"]
        * long["oracle_static_theme_leader_score"]
    )
    long["oracle_static_mapping_version"] = long.get(
        "oracle_static_mapping_version", STATIC_ORACLE_MAPPING_VERSION
    )
    long["oracle_static_asof_rule"] = STATIC_ORACLE_CLOCK
    long["oracle_static_uses_future_metadata"] = True
    long["oracle_static_feature_clock"] = SEMANTIC_FEATURE_CLOCK
    long["oracle_static_stock_theme_aggregation"] = "max, never sum"
    drop_columns = [column for column in long.columns if column.startswith("_oracle_day_pct__")]
    return long.drop(columns=drop_columns).sort_values(
        [config.date_col, "symbol", "theme_id"], kind="mergesort"
    ).reset_index(drop=True)


def add_static_oracle_semantic_features(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
    config: StaticOracleSemanticConfig = StaticOracleSemanticConfig(),
) -> pd.DataFrame:
    """Attach the best theme interaction to an existing one-row-per-stock panel.

    The primary factor for later ablations is
    ``f7_oracle_static_semantic_score``.  ``best_theme_id`` and the long-form
    builder keep the result interpretable when a stock belongs to several
    supply chains.
    """

    prepared, mapping = _prepare_feature_inputs(panel, membership, config)
    long = build_static_oracle_semantic_long(prepared, mapping, config)

    membership_summary = mapping.groupby("symbol", sort=True).agg(
        oracle_static_theme_count=("theme_id", "nunique"),
        oracle_static_theme_ids=("theme_id", _join_unique),
        oracle_static_theme_names=("theme_name", _join_unique),
    ).reset_index()
    result = prepared.merge(
        membership_summary,
        left_on=config.symbol_col,
        right_on="symbol",
        how="left",
        validate="many_to_one",
        suffixes=("", "_membership"),
    )
    if config.symbol_col != "symbol":
        result = result.drop(columns=["symbol"]).rename(
            columns={config.symbol_col: "symbol"}
        )

    best_columns = [
        config.date_col,
        "symbol",
        "theme_id",
        "theme_name",
        "oracle_static_theme_member_count",
        "oracle_static_theme_strength_raw",
        "oracle_static_theme_strength_pct",
        "oracle_static_theme_leader_score",
        "oracle_static_theme_x_leader",
        "oracle_static_theme_component_count",
        "oracle_static_leader_component_count",
    ]
    best = (
        long.loc[long["oracle_static_theme_x_leader"].notna(), best_columns]
        .sort_values(
            [config.date_col, "symbol", "oracle_static_theme_x_leader", "theme_id"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates([config.date_col, "symbol"], keep="first")
        .rename(
            columns={
                "theme_id": "oracle_static_best_theme_id",
                "theme_name": "oracle_static_best_theme_name",
            }
        )
    )
    result = result.merge(
        best,
        on=[config.date_col, "symbol"],
        how="left",
        validate="one_to_one",
    )
    result["f7_oracle_static_semantic_score"] = result[
        "oracle_static_theme_x_leader"
    ]
    eligible = result[config.eligible_col].fillna(False).astype(bool)
    result["f7_oracle_static_semantic_candidate_pct"] = pd.to_numeric(
        result["f7_oracle_static_semantic_score"], errors="coerce"
    ).where(eligible).groupby(result[config.date_col], sort=False).rank(
        method="average", pct=True, ascending=True
    )
    result["oracle_static_semantic_missing"] = result[
        "f7_oracle_static_semantic_score"
    ].isna()
    result["oracle_static_theme_count"] = result[
        "oracle_static_theme_count"
    ].fillna(0).astype(int)
    result["oracle_static_mapping_version"] = STATIC_ORACLE_MAPPING_VERSION
    result["oracle_static_asof_rule"] = STATIC_ORACLE_CLOCK
    result["oracle_static_uses_future_metadata"] = True
    result["oracle_static_feature_clock"] = SEMANTIC_FEATURE_CLOCK
    result["oracle_static_stock_theme_aggregation"] = (
        "max theme_x_leader; tie break theme_id ascending; never sum"
    )
    return result.sort_values(
        [config.date_col, "symbol"], kind="mergesort"
    ).reset_index(drop=True)


__all__ = [
    "DEFAULT_STATIC_ORACLE_THEMES",
    "SEMANTIC_FEATURE_CLOCK",
    "STATIC_ORACLE_CLOCK",
    "STATIC_ORACLE_MAPPING_VERSION",
    "StaticOracleSemanticConfig",
    "ThemeDefinition",
    "add_static_oracle_semantic_features",
    "build_static_oracle_semantic_long",
    "build_static_oracle_theme_membership",
]
