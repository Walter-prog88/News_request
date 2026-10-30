"""Conservative company-to-sector sample sums; never GDP or value added.

Fiscal period ends are bucketed into calendar years/quarters. Source scopes,
profiles and currencies remain separate. Same-company year-over-year pairs
retain the same legal registrant, source tag, currency and classification.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd
from usmd.lake import PROJECT_ROOT

# Load the default definition input once. aggregate() itself performs no I/O;
# callers can provide an explicit dictionary for reproducible alternative inputs.
_DICTIONARY_BYTES = (PROJECT_ROOT / 'config/financial_metrics.json').read_bytes()
_DICTIONARY = json.loads(_DICTIONARY_BYTES)
_DICTIONARY_SHA256 = hashlib.sha256(_DICTIONARY_BYTES).hexdigest()

DEFAULT_METRICS = ('revenue', 'operating_cash_flow', 'capital_expenditures', 'net_income')
REVENUE_BASE_METRICS = ('revenue', 'revenue_including_assessed_tax',
                        'revenue_from_contracts', 'bank_revenue_net_interest_expense',
                        'insurance_premiums')
CLASS_KEYS = ['sector', 'sector_source_url', 'sector_as_of_date', 'profile']
SCOPE_KEYS = CLASS_KEYS + ['metric_id', 'unit', 'taxonomy', 'source_tag']
PERIOD_POLICY = 'fiscal_period_end_calendar_bucket; not_exact_calendar_period'
POLICIES = {
    'economic_scope': 'selected_listed_company_sample; not_GDP_value_added_or_complete_industry',
    'period_bucketing': PERIOD_POLICY,
    'classification': 'saved_snapshot; not_historical_sector_or_index_membership',
    'coverage_denominator': 'all_selected_unique_company_ids_in_same_classification_source_date_and_profile',
    'concentration': 'absolute_company_value/sum_absolute_values; not_market_cap_or_index_weight',
    'yoy': 'intersection_of_company_and_legal_registrant_in_same_scope; fiscal_dates_within_alignment_tolerance',
    'yoy_revenue_base': 'exclude_company_pairs_with_prior_revenue_le_zero',
    'yoy_revenue_base_metric_ids': list(REVENUE_BASE_METRICS),
    'yoy_other_flow_base': 'signed_company_values_allowed; percentage_requires_positive_matched_aggregate_prior_sum',
    'accounting_basis': 'unverified; equal_tag_and_date_alignment_do_not_prove_unchanged_accounting',
    'missing_values': 'never_zero_filled; count_zero_means_no_rows',
    'ratios': 'not_computed',
}


def _json(values):
    return json.dumps(list(values), ensure_ascii=False, separators=(',', ':'))


def _clean(value):
    return None if value is None or pd.isna(value) else value


def _key(row, columns):
    return tuple(_clean(row[column]) for column in columns)


def _group_id(key, bucket):
    return hashlib.sha256(json.dumps([*key, bucket], ensure_ascii=False, default=str).encode()).hexdigest()[:24]


def _buckets(period, start, end):
    frequency = 'Y' if period == 'annual' else 'Q'
    return pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq=frequency).astype(str).tolist()


def aggregate(frame, *, cohort=None, period=None, start=None, end=None,
              allow_unverified_quarters=False, alignment_tolerance_days=14,
              metric_dictionary=None):
    """Pure DataFrame aggregation returning summary/contributions/exclusions/coverage.

    The selected cohort is required even for missing observations; infer it from
    panel.attrs or supply it explicitly. The definition registry must explicitly
    mark selected metrics as additive duration values with monetary units.
    No implicit summation of ratios, shares or balances.
    """
    period = period or frame.attrs.get('financial_period')
    if period not in ('annual', 'quarter'):
        raise ValueError('Industry aggregation supports annual or quarter only')
    if not isinstance(allow_unverified_quarters, bool):
        raise ValueError('allow_unverified_quarters must be a boolean')
    if isinstance(alignment_tolerance_days, bool) or not isinstance(alignment_tolerance_days, int) or not 0 <= alignment_tolerance_days <= 31:
        raise ValueError('alignment_tolerance_days must be an integer from 0 to 31')
    metadata = cohort if cohort is not None else frame.attrs.get('cohort_records')
    if metadata is None:
        raise ValueError('Selected cohort metadata is required for an honest coverage denominator')
    cohort = pd.DataFrame(metadata).copy()
    for name in ['issuer_id', 'company_id', 'symbol', *CLASS_KEYS]:
        if name not in cohort:
            if not cohort.empty:
                raise ValueError('Cohort metadata missing: ' + name)
            cohort[name] = pd.Series(dtype=object)
    if cohort.issuer_id.duplicated().any():
        raise ValueError('Cohort metadata must be unique by legal registrant')
    if cohort[['issuer_id', 'company_id']].isna().any().any():
        raise ValueError('Cohort identity cannot be missing')
    requested = list(frame.attrs.get('requested_metrics', frame.get('metric_id', pd.Series(dtype=str)).dropna().unique()))
    if not requested:
        requested = list(DEFAULT_METRICS)
    dictionary = metric_dictionary if metric_dictionary is not None else _DICTIONARY
    definitions = {item['metric_id']: item for item in dictionary['metrics']}
    monetary_units = {name: [unit for unit in item.get('units', []) if re.fullmatch(r'[A-Z]{3}', unit)]
                      for name, item in definitions.items()}
    allowed_metrics = {name for name, item in definitions.items()
                       if item.get('period_type') == 'duration' and item.get('additive') is True and monetary_units[name]}
    unknown = set(requested) - allowed_metrics
    if unknown:
        raise ValueError('Unsupported industry sum metrics: ' + ', '.join(sorted(unknown)))
    dictionary_hash = (_DICTIONARY_SHA256 if metric_dictionary is None else
        hashlib.sha256(json.dumps(metric_dictionary, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
    start = start or frame.attrs.get('requested_start')
    end = end or frame.attrs.get('requested_end') or frame.attrs.get('age_reference_date')
    if start is None or end is None:
        raise ValueError('Explicit report-end start/end or panel query date metadata is required')
    if pd.Timestamp(end) < pd.Timestamp(start):
        raise ValueError('end precedes start')
    buckets = _buckets(period, start, end)
    d = frame.copy().reset_index(drop=True)
    d.attrs = {}
    required = ['issuer_id', 'company_id', 'symbol', 'metric_id', 'unit', 'value',
                'value_id', 'source_id', 'source_tag', 'period_start', 'period_end',
                'period_kind', 'row_status', 'tag_conflict', *CLASS_KEYS]
    for name in required:
        if name not in d:
            if not d.empty:
                raise ValueError('Financial panel missing: ' + name)
            d[name] = pd.Series(dtype=object)
    d['_row'] = range(len(d))
    ends = pd.to_datetime(d.period_end, errors='coerce')
    begins = pd.to_datetime(d.period_start, errors='coerce')
    d['bucket'] = ends.dt.to_period('Y' if period == 'annual' else 'Q').astype(str)
    d['duration_days'] = (ends - begins).dt.days + 1
    d['taxonomy'] = d.source_tag.map(lambda tag: tag.split(':', 1)[0] if isinstance(tag, str) and ':' in tag else None)
    d['value'] = pd.to_numeric(d.value, errors='coerce')
    reasons = pd.Series('', index=d.index, dtype=object)

    def reject(mask, reason):
        reasons.loc[mask & reasons.eq('')] = reason

    reject(~d.metric_id.isin(requested), 'metric_not_requested')
    reject(~d.issuer_id.isin(cohort.issuer_id), 'issuer_not_in_selected_cohort')
    reject(~np.isfinite(d.value), 'nonfinite_value')
    reject(~d.row_status.eq('ok') | d.tag_conflict.fillna(True).astype(bool), 'conflicted_or_unusable_row')
    reject(d[['value_id', 'source_id', 'source_tag', 'taxonomy']].isna().any(axis=1), 'missing_financial_provenance')
    reject(d[CLASS_KEYS].isna().any(axis=1), 'missing_classification_or_profile')
    reject(~d.unit.astype(str).str.fullmatch(r'[A-Z]{3}'), 'non_monetary_unit')
    reject(pd.Series([row.unit not in monetary_units.get(row.metric_id, []) for row in d.itertuples()],
                     index=d.index, dtype=bool), 'unit_not_supported_by_metric_definition')
    allowed = ['annual'] if period == 'annual' else ['direct_quarter', 'derived_quarter']
    reject(~d.period_kind.isin(allowed), 'wrong_period_kind')
    duration_ok = d.duration_days.between(350, 380) if period == 'annual' else d.duration_days.between(70, 120)
    reject(~duration_ok | ends.isna() | begins.isna(), 'invalid_duration_or_dates')
    reject(ends.lt(pd.Timestamp(start)) | ends.gt(pd.Timestamp(end)), 'outside_report_end_window')
    if not allow_unverified_quarters:
        basis = d.get('derivation_compatibility', pd.Series(None, index=d.index, dtype=object))
        reject(basis.eq('cross_filing_unverified'), 'cross_filing_quarter_unverified')
        reject(d.period_kind.eq('derived_quarter') & ~basis.eq('same_filing'),
               'derived_quarter_compatibility_unverified')
    registry = {row['issuer_id']: row for row in cohort.to_dict('records')}
    for idx, row in d.iterrows():
        meta = registry.get(row.issuer_id)
        if meta is not None and _key(row, ['company_id', *CLASS_KEYS]) != _key(meta, ['company_id', *CLASS_KEYS]):
            reject(pd.Series(d.index == idx, index=d.index), 'cohort_metadata_mismatch')

    # Retain one exact duplicate, but quarantine conflicting duplicate IDs.
    relevant = ['issuer_id', 'company_id', 'metric_id', 'unit', 'value', 'period_start',
                'period_end', 'source_tag', 'source_id', *CLASS_KEYS]
    valid = d.loc[reasons.eq('')]
    for _, group in valid.loc[valid.value_id.duplicated(False)].groupby('value_id'):
        if len(group[relevant].drop_duplicates()) > 1:
            reject(d.index.isin(group.index), 'conflicting_duplicate_value_id')
        else:
            reject(d.index.isin(group.index[1:]), 'duplicate_identical_value')
    # A company must contribute at most once per metric/end bucket, including
    # across units/tags. Never hide overlapping XOM registrants in separate sums.
    valid = d.loc[reasons.eq('')]
    for _, group in valid.groupby(['company_id', 'metric_id', 'bucket'], dropna=False):
        if group.issuer_id.nunique() > 1:
            reject(d.index.isin(group.index), 'multiple_registrants_in_company_bucket')
        elif len(group) > 1:
            reject(d.index.isin(group.index), 'multiple_periods_or_scopes_in_company_bucket')
    excluded = d.loc[reasons.ne('')].copy()
    excluded['exclusion_stage'] = 'aggregation'
    excluded['exclusion_reason'] = reasons.loc[excluded.index]
    included = d.loc[reasons.eq('')].copy()
    included['period_bucket_policy'] = PERIOD_POLICY
    included['accounting_basis_verified'] = False

    classes = {}
    for row in cohort.to_dict('records'):
        classes.setdefault(_key(row, CLASS_KEYS), set()).add(row['company_id'])
    scoped = {}
    for row in included.to_dict('records'):
        scoped.setdefault((_key(row, SCOPE_KEYS), row['bucket']), []).append(row)
    lookup = {(_key(row, SCOPE_KEYS), row['bucket'], row['company_id'], row['issuer_id']): row
              for row in included.to_dict('records')}
    summaries, contributions, yoy_exclusions = [], [], []
    for (scope, bucket), rows in sorted(scoped.items(), key=lambda item: str(item[0])):
        group_id = _group_id(scope, bucket)
        frequency = 'Y' if period == 'annual' else 'Q'
        prior_bucket = str(pd.Period(bucket, freq=frequency) - (1 if period == 'annual' else 4))
        pairs, pair_status = [], {}
        for row in rows:
            previous = lookup.get((scope, prior_bucket, row['company_id'], row['issuer_id']))
            reason = None
            if previous is None:
                reason = 'no_prior_same_company_registrant_and_scope'
            else:
                begin_shift = abs((pd.Timestamp(row['period_start']) - (pd.Timestamp(previous['period_start']) + pd.DateOffset(years=1))).days)
                end_shift = abs((pd.Timestamp(row['period_end']) - (pd.Timestamp(previous['period_end']) + pd.DateOffset(years=1))).days)
                if max(begin_shift, end_shift, abs(row['duration_days'] - previous['duration_days'])) > alignment_tolerance_days:
                    reason = 'prior_fiscal_period_alignment_mismatch'
                elif row['metric_id'] in REVENUE_BASE_METRICS and previous['value'] <= 0:
                    reason = 'nonpositive_prior_company_revenue'
            pair_status[row['value_id']] = (previous, reason)
            if reason:
                yoy_exclusions.append({**row, 'industry_group_id': group_id,
                    'exclusion_stage': 'yoy_pairing', 'exclusion_reason': reason,
                    'prior_value_id': previous['value_id'] if previous is not None else None})
            else:
                pairs.append((row, previous))
        values = pd.Series([row['value'] for row in rows], dtype=float)
        absolute_total = float(values.abs().sum())
        prior_sum = sum(pair[1]['value'] for pair in pairs) if pairs else None
        current_sum = sum(pair[0]['value'] for pair in pairs) if pairs else None
        growth = 100.0 * (current_sum / prior_sum - 1) if pairs and prior_sum > 0 else None
        growth_status = ('no_matched_companies' if not pairs else
                         'nonpositive_matched_aggregate_prior_sum' if prior_sum <= 0 else
                         'matched_sample_percentage_accounting_basis_unverified')
        denominator = len(classes[scope[:len(CLASS_KEYS)]])
        previous_rows = scoped.get((scope, prior_bucket), [])
        matched_ids = {row['company_id'] for row, _ in pairs}
        group = dict(zip(SCOPE_KEYS, scope))
        starts = [pd.Timestamp(row['period_start']) for row in rows]
        end_dates = [pd.Timestamp(row['period_end']) for row in rows]
        summary = {**group, 'bucket': bucket, 'industry_group_id': group_id,
            'period': period, 'period_bucket_policy': PERIOD_POLICY,
            'sum_value': float(values.sum()), 'sum_absolute_value': absolute_total,
            'included_company_count': len(rows), 'selected_cohort_company_count': denominator,
            'coverage_fraction': len(rows) / denominator if denominator else None,
            'max_absolute_value_share': float(values.abs().max() / absolute_total) if absolute_total > 0 else None,
            'largest_absolute_company_ids': _json(sorted(row['company_id'] for row in rows if absolute_total > 0 and abs(row['value']) == values.abs().max())),
            'negative_value_company_count': int(values.lt(0).sum()),
            'first_period_start': min(starts).date().isoformat(), 'last_period_start': max(starts).date().isoformat(),
            'first_period_end': min(end_dates).date().isoformat(), 'last_period_end': max(end_dates).date().isoformat(),
            'period_start_span_days': (max(starts) - min(starts)).days,
            'period_end_span_days': (max(end_dates) - min(end_dates)).days,
            'minimum_duration_days': int(min(row['duration_days'] for row in rows)),
            'maximum_duration_days': int(max(row['duration_days'] for row in rows)),
            'source_value_ids': _json(row['value_id'] for row in rows),
            'source_ids': _json(sorted({row['source_id'] for row in rows})),
            'source_issuer_ids': _json(sorted({row['issuer_id'] for row in rows})),
            'yoy_prior_bucket': prior_bucket, 'yoy_matched_company_count': len(pairs),
            'yoy_current_unmatched_company_count': len(rows) - len(pairs),
            'yoy_prior_unmatched_company_count': len({row['company_id'] for row in previous_rows} - matched_ids),
            'yoy_matched_current_sum': current_sum, 'yoy_matched_prior_sum': prior_sum,
            'yoy_matched_delta': current_sum - prior_sum if pairs else None,
            'yoy_growth_pct': growth, 'yoy_status': growth_status,
            'yoy_negative_prior_company_count': sum(previous['value'] < 0 for _, previous in pairs),
            'yoy_alignment_tolerance_days': alignment_tolerance_days,
            'yoy_prior_source_value_ids': _json(previous['value_id'] for _, previous in pairs),
            'accounting_basis_verified': False}
        summaries.append(summary)
        for row in rows:
            previous, reason = pair_status[row['value_id']]
            paired = reason is None
            contributions.append({**row, 'industry_group_id': group_id,
                'absolute_value_weight': abs(row['value']) / absolute_total if absolute_total > 0 else None,
                'yoy_pair_status': reason or 'matched',
                'yoy_prior_value_id': previous['value_id'] if previous is not None else None,
                'yoy_prior_value': previous['value'] if paired else None,
                'yoy_delta': row['value'] - previous['value'] if paired else None,
                'yoy_contribution_percentage_points': 100.0 * (row['value'] - previous['value']) / prior_sum if paired and growth is not None else None})

    coverage_rows = []
    for class_key, company_ids in sorted(classes.items(), key=lambda item: str(item[0])):
        class_mask = pd.Series(True, index=included.index)
        exclusion_mask = pd.Series(True, index=excluded.index)
        for column, value in zip(CLASS_KEYS, class_key):
            class_mask &= included[column].isna() if value is None else included[column].eq(value)
            exclusion_mask &= excluded[column].isna() if value is None else excluded[column].eq(value)
        class_values, class_exclusions = included.loc[class_mask], excluded.loc[exclusion_mask]
        for metric in requested:
            for bucket in buckets:
                present = class_values.loc[class_values.metric_id.eq(metric) & class_values.bucket.eq(bucket)]
                dropped = class_exclusions.loc[class_exclusions.metric_id.eq(metric) & class_exclusions.bucket.eq(bucket)]
                present_ids = set(present.company_id)
                coverage_rows.append({**dict(zip(CLASS_KEYS, class_key)), 'metric_id': metric, 'bucket': bucket,
                    'period': period, 'period_bucket_policy': PERIOD_POLICY,
                    'selected_cohort_company_count': len(company_ids), 'included_company_count': len(present_ids),
                    'coverage_fraction': len(present_ids) / len(company_ids) if company_ids else None,
                    'excluded_company_count': len((set(dropped.company_id) & company_ids) - present_ids),
                    'without_included_value_count': len(company_ids - present_ids),
                    'included_company_ids': _json(sorted(present_ids)),
                    'without_included_value_company_ids': _json(sorted(company_ids - present_ids)),
                    'available_source_scopes': _json(sorted(set(zip(present.unit, present.source_tag)))),
                    'status': 'no_qualified_values' if not present_ids else 'complete_selected_sample' if present_ids == company_ids else 'partial_selected_sample'})
    exclusions = pd.concat([excluded, pd.DataFrame(yoy_exclusions)], ignore_index=True, sort=False)
    outputs = {'summary': pd.DataFrame(summaries), 'contributions': pd.DataFrame(contributions),
               'exclusions': exclusions, 'coverage': pd.DataFrame(coverage_rows)}
    empty_schemas = {
        'summary': [*SCOPE_KEYS, 'bucket', 'industry_group_id', 'sum_value',
                    'included_company_count', 'selected_cohort_company_count', 'coverage_fraction',
                    'max_absolute_value_share', 'yoy_growth_pct', 'yoy_matched_delta', 'yoy_status'],
        'contributions': [*required, 'bucket', 'industry_group_id', 'absolute_value_weight',
                          'yoy_pair_status', 'yoy_prior_value_id', 'yoy_delta'],
        'coverage': [*CLASS_KEYS, 'metric_id', 'bucket', 'selected_cohort_company_count',
                     'included_company_count', 'coverage_fraction', 'status'],
    }
    for name, columns in empty_schemas.items():
        if outputs[name].empty:
            outputs[name] = outputs[name].reindex(columns=list(dict.fromkeys(columns)))
    for name, value in outputs.items():
        outputs[name] = value.drop(columns=['_row'], errors='ignore')
    outputs['policies'] = {**POLICIES, 'allow_unverified_quarters': bool(allow_unverified_quarters),
                           'alignment_tolerance_days': alignment_tolerance_days,
                           'metric_definition_sha256': dictionary_hash}
    outputs['metric_definitions'] = {'dictionary_version': dictionary.get('version'),
                                    'source_dictionary_sha256': dictionary_hash,
                                    'metrics': [definitions[name] for name in requested]}
    outputs['query'] = {key: value for key, value in frame.attrs.items() if key != 'cohort_records'}
    outputs['query'].update({'period': period, 'start': str(start), 'end': str(end),
                             'requested_metrics': requested, 'selected_company_count': int(cohort.company_id.nunique()),
                             'selected_registrant_count': len(cohort), 'metric_definition_sha256': dictionary_hash,
                             'aggregation_policy': 'sample_sum_by_source_scope_and_fiscal_period_end_bucket'})
    return outputs


def research(store, *, metrics=None, period='annual', allow_unverified_quarters=False,
             alignment_tolerance_days=14, **filters):
    """One source-preserving financial panel read, then pure in-memory aggregation.

    Include the previous year in start when a first displayed year's YOY is
    needed. No extra hidden history is fetched and no aggregate is persisted.
    """
    from usmd.financial_cohorts import panel
    if 'same_filing_only' in filters:
        raise ValueError('Use allow_unverified_quarters so excluded rows remain auditable')
    frame = panel(store, metrics=metrics if metrics is not None else DEFAULT_METRICS,
                  period=period, same_filing_only=False, **filters)
    return aggregate(frame, period=period, allow_unverified_quarters=allow_unverified_quarters,
                     alignment_tolerance_days=alignment_tolerance_days)
