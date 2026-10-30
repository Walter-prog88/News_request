"""Auditable, same-filing reported-input financial factors; no vintage backfill."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re

import numpy as np
import pandas as pd

from usmd.lake import PROJECT_ROOT

VERSION = '1.0.0'
_CONFIG_BYTES = (PROJECT_ROOT / 'config/financial_factors.json').read_bytes()
_METRIC_BYTES = (PROJECT_ROOT / 'config/financial_metrics.json').read_bytes()
_CONFIG = json.loads(_CONFIG_BYTES)
_METRICS = json.loads(_METRIC_BYTES)
_CONFIG_HASH = hashlib.sha256(_CONFIG_BYTES).hexdigest()
_METRIC_HASH = hashlib.sha256(_METRIC_BYTES).hexdigest()
CLASS_COLUMNS = ['issuer_id', 'company_id', 'symbol', 'profile', 'sector', 'registration_role',
                 'sector_source_url', 'sector_as_of_date']
COVERAGE_COLUMNS = [*CLASS_COLUMNS, 'bucket', 'period_start', 'period_end', 'period',
    'candidate_period_basis', 'factor_id', 'label_zh', 'status', 'reason', 'missing_metrics',
    'input_value_ids', 'input_source_tags', 'input_accessions', 'factor_value_id']
VALUE_COLUMNS = [*COVERAGE_COLUMNS, 'value', 'unit', 'input_unit', 'taxonomy', 'accn',
    'available_at', 'availability_basis', 'input_fact_ids', 'input_metric_ids', 'source_ids',
    'factor_formula', 'factor_version', 'factor_definition_sha256', 'metric_definition_sha256',
    'accounting_scope_verified', 'pit_eligible']


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _definitions(factors, config, metric_dictionary):
    definitions = {item['factor_id']: item for item in config['factors']}
    selected = (list(config.get('default_factors', definitions)) if factors is None else
                [factors] if isinstance(factors, str) else list(factors))
    if selected == ['all']:
        selected = list(definitions)
    elif 'all' in selected:
        raise ValueError('all must be the only factor selection')
    selected = list(dict.fromkeys(selected))
    if not selected or set(selected) - set(definitions):
        raise ValueError('Choose defined, nonempty financial factors')
    metrics = {item['metric_id']: item for item in metric_dictionary['metrics']}
    for factor in (definitions[name] for name in selected):
        if factor['operation'] not in ('ratio_percent', 'difference') or len(factor['inputs']) != 2 or len(set(factor['inputs'])) != 2:
            raise ValueError('Factors require two distinct, explicitly defined operands')
        if factor['operation'] == 'ratio_percent' and factor['inputs'][1] != 'revenue':
            raise ValueError('This factor registry permits reported revenue denominators only')
        if not set(factor.get('nonnegative_inputs', [])) <= set(factor['inputs']):
            raise ValueError('Sign constraints must name actual factor inputs')
        for metric in factor['inputs']:
            entry = metrics.get(metric, {})
            if entry.get('period_type') != 'duration' or entry.get('additive') is not True or not any(re.fullmatch(r'[A-Z]{3}', unit) for unit in entry.get('units', [])):
                raise ValueError('Factor inputs require explicitly additive monetary duration metrics: ' + metric)
    return [definitions[name] for name in selected], metrics


def _array(encoded):
    try:
        result = json.loads(encoded) if isinstance(encoded, str) else encoded
    except (TypeError, ValueError):
        return None
    return result if isinstance(result, list) and result and all(isinstance(x, str) and x for x in result) else None


def _same(left, right):
    return (pd.isna(left) and pd.isna(right)) or left == right


def _input_error(row, metric, issuer, period):
    conflict = row.get('tag_conflict')
    status = row.get('row_status')
    if not isinstance(status, str) or status != 'ok' or not isinstance(conflict, (bool, np.bool_)) or bool(conflict):
        return 'latest_input_conflicted_or_unusable'
    amount = pd.to_numeric(pd.Series([row.get('value')]), errors='coerce').iloc[0]
    if not np.isfinite(amount):
        return 'nonfinite_input'
    unit = row.get('unit')
    if not isinstance(unit, str) or unit not in metric['units'] or not re.fullmatch(r'[A-Z]{3}', unit):
        return 'input_unit_outside_metric_definition'
    tag = row.get('source_tag')
    namespace, _, name = tag.partition(':') if isinstance(tag, str) else ('', '', '')
    cik_value = row.get('cik')
    cik = str(str(issuer['issuer_id']).split(':', 1)[-1] if cik_value is None or pd.isna(cik_value) else cik_value).zfill(10)
    tags = metric.get('tag_overrides_by_cik', {}).get(cik,
        metric.get('tag_overrides_by_profile', {}).get(issuer.get('profile'), metric.get('tags', [])))
    if namespace != metric.get('taxonomy', 'us-gaap') or name not in tags:
        return 'source_tag_outside_metric_definition'
    if row.get('taxonomy') is not None and not pd.isna(row.get('taxonomy')) and row['taxonomy'] != namespace:
        return 'source_taxonomy_mismatch'
    for column in ('value_id', 'source_id', 'accn'):
        if not isinstance(row.get(column), str) or not row[column]:
            return 'missing_input_provenance'
    facts, accessions = _array(row.get('input_fact_ids')), _array(row.get('input_accessions'))
    if facts is None or accessions is None or set(accessions) != {row['accn']}:
        return 'input_single_filing_provenance_unproved'
    kind = row.get('period_kind')
    method = row.get('normalization_method')
    if not isinstance(kind, str) or not isinstance(method, str):
        return 'wrong_input_period_kind'
    if period == 'annual':
        if kind != 'annual' or method != 'reported':
            return 'wrong_input_period_kind'
    elif kind == 'derived_quarter':
        compatibility = row.get('derivation_compatibility')
        if not isinstance(compatibility, str) or compatibility != 'same_filing' or method != 'cumulative_difference' or len(set(facts)) < 2:
            return 'derived_quarter_single_filing_unproved'
    elif kind != 'direct_quarter' or method != 'reported':
        return 'wrong_input_period_kind'
    try:
        begin, end = pd.Timestamp(row['period_start']), pd.Timestamp(row['period_end'])
        duration = (end - begin).days + 1
        if pd.isna(begin) or pd.isna(end) or not ((350 <= duration <= 380) if period == 'annual' else (70 <= duration <= 120)):
            return 'invalid_input_duration'
        available = pd.Timestamp(row['available_at'])
        if pd.isna(available) or available.tzinfo is None:
            return 'missing_conservative_availability'
    except (KeyError, TypeError, ValueError):
        return 'invalid_input_dates_or_availability'
    return None


def derive(frame, *, factors=None, cohort=None, period=None, start=None, end=None,
           config=None, metric_dictionary=None):
    """Pure calculation over a latest-value panel; unavailable pairs stay absent.

    Coverage expands every selected registrant and requested end-date bucket.
    Real observed dates identify candidates; no observed candidate yields a
    bucket-only row with null period dates, never a fabricated fiscal period.
    """
    config = _CONFIG if config is None else config
    dictionary = _METRICS if metric_dictionary is None else metric_dictionary
    definitions, metrics = _definitions(factors, config, dictionary)
    config_hash = _CONFIG_HASH if config is _CONFIG else _hash(_json(config).encode())
    metric_hash = _METRIC_HASH if dictionary is _METRICS else _hash(_json(dictionary).encode())
    period = period or frame.attrs.get('financial_period')
    if period not in ('annual', 'quarter'):
        raise ValueError('Factors require annual or quarter inputs')
    start = start or frame.attrs.get('requested_start')
    end = end or frame.attrs.get('requested_end') or frame.attrs.get('age_reference_date')
    if start is None or end is None or pd.Timestamp(end) < pd.Timestamp(start):
        raise ValueError('Valid requested report-end start/end dates are required')
    raw_cohort = cohort if cohort is not None else frame.attrs.get('cohort_records')
    if raw_cohort is None:
        raise ValueError('Selected cohort metadata is required, including companies with no input values')
    companies = pd.DataFrame(raw_cohort).copy()
    for column in CLASS_COLUMNS:
        if column not in companies:
            if not companies.empty and column in ('issuer_id', 'company_id', 'symbol', 'profile'):
                raise ValueError('Incomplete factor cohort identity: ' + column)
            companies[column] = None
    if companies.issuer_id.duplicated().any() or companies[['issuer_id', 'company_id']].isna().any().any():
        raise ValueError('Factor cohort needs unique, nonmissing registrant identities')
    needed = list(dict.fromkeys(metric for factor in definitions for metric in factor['inputs']))
    inputs = frame.copy()
    inputs.attrs = {}
    required = ['issuer_id', 'company_id', 'symbol', 'metric_id', 'period_start', 'period_end', 'value_id', 'value']
    for column in required:
        if column not in inputs:
            if not inputs.empty:
                raise ValueError('Incomplete financial factor input: ' + column)
            inputs[column] = pd.Series(dtype=object)
    if set(inputs.issuer_id.dropna()) - set(companies.issuer_id):
        raise ValueError('Factor inputs include registrants outside the selected cohort')
    frequency = 'Y' if period == 'annual' else 'Q'
    buckets = pd.period_range(start, end, freq=frequency).astype(str).tolist()
    rows = inputs.loc[inputs.metric_id.isin(needed)].copy()
    period_ends = pd.to_datetime(rows.period_end, errors='coerce')
    rows['_bucket'] = period_ends.dt.to_period(frequency).astype(str)
    rows = rows.loc[period_ends.ge(pd.Timestamp(start)) & period_ends.le(pd.Timestamp(end))]
    values, coverage = [], []
    for issuer in companies.to_dict('records'):
        issuer_rows = rows.loc[rows.issuer_id.eq(issuer['issuer_id'])]
        profile = issuer.get('profile')
        if profile is None or pd.isna(profile):
            profile = None
        sector_value = issuer.get('sector')
        sector = '' if sector_value is None or pd.isna(sector_value) else str(sector_value).casefold()
        applicability = ('excluded_financial_profile_or_sector' if profile in config['excluded_profiles']
                         or sector in {s.casefold() for s in config['excluded_sectors']}
                         else 'profile_not_approved_for_general_company_factors' if profile not in config['allowed_profiles'] else None)
        for bucket in buckets:
            bucket_rows = issuer_rows.loc[issuer_rows._bucket.eq(bucket)]
            candidates = bucket_rows[['period_start', 'period_end']].drop_duplicates().to_dict('records') or [None]
            for candidate in candidates:
                selected = bucket_rows.loc[bucket_rows.period_start.eq(candidate['period_start'])
                    & bucket_rows.period_end.eq(candidate['period_end'])] if candidate is not None else bucket_rows.iloc[:0]
                base = {column: issuer.get(column) for column in CLASS_COLUMNS}
                base.update({'bucket': bucket, 'period': period,
                    'period_start': candidate['period_start'] if candidate else None,
                    'period_end': candidate['period_end'] if candidate else None,
                    'candidate_period_basis': 'observed_exact_input_period' if candidate else 'requested_end_bucket_without_observed_period'})
                for factor in definitions:
                    operand_rows = selected.loc[selected.metric_id.isin(factor['inputs'])]
                    ids = [str(value) for value in operand_rows.value_id.dropna().unique()]
                    source_tags = {metric: sorted(str(tag) for tag in operand_rows.loc[operand_rows.metric_id.eq(metric), 'source_tag'].dropna().unique())
                                   if 'source_tag' in operand_rows else [] for metric in factor['inputs']}
                    accessions = sorted(operand_rows.accn.dropna().astype(str).unique().tolist()) if 'accn' in operand_rows else []
                    result = {**base, 'factor_id': factor['factor_id'], 'label_zh': factor['label_zh'],
                        'status': 'unavailable', 'reason': None, 'missing_metrics': '[]',
                        'input_value_ids': _json(ids), 'input_source_tags': _json(source_tags),
                        'input_accessions': _json(accessions), 'factor_value_id': None}
                    operands, missing = [], []
                    if applicability:
                        result.update(status='not_applicable', reason=applicability)
                    elif candidate is None:
                        result.update(reason='no_candidate_period', missing_metrics=_json(factor['inputs']))
                    else:
                        for metric_id in factor['inputs']:
                            group = operand_rows.loc[operand_rows.metric_id.eq(metric_id)]
                            # Identical alias copies do not become two economic values.
                            group = group.drop_duplicates([column for column in group if column not in ('symbol', '_bucket')])
                            if group.empty:
                                missing.append(metric_id)
                            elif len(group) != 1:
                                result['reason'] = 'ambiguous_latest_input_values'
                            else:
                                operands.append(group.iloc[0].to_dict())
                        if result['reason'] is None and missing:
                            result.update(reason='missing_exact_period_inputs', missing_metrics=_json(missing))
                        if result['reason'] is None:
                            for row in operands:
                                if pd.isna(row['company_id']) or row['company_id'] != issuer['company_id']:
                                    result['reason'] = 'input_company_identity_mismatch'
                                    break
                                error = _input_error(row, metrics[row['metric_id']], issuer, period)
                                if error:
                                    result['reason'] = error
                                    break
                        if result['reason'] is None:
                            for column in ('issuer_id', 'period_start', 'period_end', 'unit', 'accn'):
                                if len({row.get(column) for row in operands}) != 1:
                                    result['reason'] = 'input_' + column + '_mismatch'
                                    break
                            namespaces = {row['source_tag'].split(':', 1)[0] for row in operands}
                            if len(namespaces) != 1:
                                result['reason'] = 'input_taxonomy_mismatch'
                        if result['reason'] is None:
                            amounts = {row['metric_id']: float(row['value']) for row in operands}
                            if any(amounts[metric] < 0 for metric in factor['nonnegative_inputs']):
                                result['reason'] = 'negative_reported_investment_or_expense_input'
                            elif factor['operation'] == 'ratio_percent' and amounts[factor['inputs'][1]] <= 0:
                                result['reason'] = 'nonpositive_reported_revenue_denominator'
                            else:
                                first, second = (amounts[metric] for metric in factor['inputs'])
                                computed = 100.0 * first / second if factor['operation'] == 'ratio_percent' else first - second
                                if not np.isfinite(computed):
                                    result['reason'] = 'nonfinite_computed_factor'
                                else:
                                    input_ids = [row['value_id'] for row in operands]
                                    available = max(pd.Timestamp(row['available_at']).tz_convert('UTC') for row in operands).isoformat()
                                    factor_value_id = 'factor_' + _hash(_json([config_hash, metric_hash, factor['factor_id'],
                                        [(row['value_id'], row['value'], row['source_tag'], row['unit'], row['accn']) for row in operands]]).encode())[:24]
                                    result.update(status='available', reason=None, factor_value_id=factor_value_id,
                                                  input_value_ids=_json(input_ids))
                                    values.append({**result, 'value': computed,
                                        'unit': 'percent' if factor['operation'] == 'ratio_percent' else operands[0]['unit'],
                                        'input_unit': operands[0]['unit'], 'taxonomy': next(iter(namespaces)), 'accn': operands[0]['accn'],
                                        'available_at': available, 'availability_basis': 'maximum_conservative_input_availability',
                                        'input_fact_ids': _json(sorted({fact for row in operands for fact in _array(row['input_fact_ids'])})),
                                        'input_metric_ids': _json(factor['inputs']), 'source_ids': _json(sorted({row['source_id'] for row in operands})),
                                        'factor_formula': factor['formula'], 'factor_version': config['version'],
                                        'factor_definition_sha256': config_hash, 'metric_definition_sha256': metric_hash,
                                        'accounting_scope_verified': False, 'pit_eligible': False})
                    coverage.append(result)
    coverage_frame = pd.DataFrame(coverage, columns=COVERAGE_COLUMNS)
    values_frame = pd.DataFrame(values, columns=VALUE_COLUMNS)
    if not values_frame.empty and values_frame.factor_value_id.duplicated().any():
        raise ValueError('Duplicate derived factor identity')
    metadata = {'schema_version': 1, 'module': 'financial_factors', 'pipeline_version': VERSION,
        'factor_definition_sha256': config_hash, 'metric_definition_sha256': metric_hash,
        'factor_definitions': {**config, 'factors': definitions},
        'input_metric_definitions': {key: metrics[key] for key in needed},
        'query': {**{key: value for key, value in frame.attrs.items() if key != 'cohort_records'},
                  'period': period, 'start': str(start), 'end': str(end), 'factors': [x['factor_id'] for x in definitions]},
        'selected_companies': int(companies.company_id.nunique()), 'selected_registrants': len(companies),
        'factor_status_counts': coverage_frame.status.value_counts().to_dict(),
        'policies': {'input_matching': 'same_issuer_exact_dates_unit_taxonomy_accession',
            'quarter_matching': 'direct_and_proved_same_filing_derived_quarters_may_pair',
            'derived_quarter_proof': 'input_accessions_exactly_equals_top_level_accn; at_least_two_fact_ids; same_filing_compatibility',
            'vintage': 'latest_panel_values; no_older_filing_search_or_fallback',
            'coverage': 'every_selected_registrant_requested_end_bucket_and_factor; absent_period_dates_remain_null',
            'investment_signs': 'negative_reported_R&D_SBC_PPE_productive_asset_inputs_excluded_for_review; never_abs',
            'net_income_scope': 'parent_attributable_only', 'SBC_scope': 'reported_non_cash_amount_not_cash_spending',
            'free_cash_flow_scope': 'CFO_minus_PPE_cash_purchases_proxy_not_owner_earnings_or_maintenance_capex',
            'productive_asset_proxy_scope': 'separate_CFO_minus_broader_productive_asset_purchases; never_PPE_fallback_or_owner_earnings',
            'source_scope': 'reported_tag_definitions_preserved; no_cross_company_accounting_scope_reconciliation',
            'currency_conversion': 'none', 'native_companion_included': False,
            'accounting_scope_verified': False, 'exact_point_in_time': False}}
    return {'values': values_frame, 'coverage': coverage_frame,
            'exclusions': coverage_frame.loc[coverage_frame.status.ne('available')].reset_index(drop=True),
            'inputs': inputs, 'companies': companies, 'metadata': metadata}


def research(store, *, factors=None, period='annual', **filters):
    """Capture one catalog snapshot and one latest financial panel for factors."""
    from usmd.financial_cohorts import panel, _read_snapshot
    if 'metrics' in filters or 'same_filing_only' in filters:
        raise ValueError('Choose factors; input metrics and post-latest eligibility are defined by the factor registry')
    definitions, _ = _definitions(factors, _CONFIG, _METRICS)
    metrics = list(dict.fromkeys(metric for factor in definitions for metric in factor['inputs']))
    with _read_snapshot(store):
        frame = panel(store, metrics=metrics, period=period, same_filing_only=False, **filters)
        catalog = json.loads((store.lake.root / 'catalog.json').read_text())
        result = derive(frame, factors=factors, period=period)
        result['metadata'].update({'generated_at': datetime.now(timezone.utc).isoformat(),
            'catalog_generated_at': catalog.get('generated_at'),
            'source_table_hashes': {name: item['sha256'] for name, item in catalog['tables'].items() if name.startswith('financial_')},
            'verified_query_tables': ['financial_values', 'financial_issuers'],
            'source_hash_policy': 'same_catalog_snapshot_references; query_reverified_values_and_issuers_only',
            'pipeline_code_hashes': {name: _hash((PROJECT_ROOT / 'src/usmd' / name).read_bytes()) for name in
                                     ['financial_factors.py', 'financial_cohorts.py', 'financial_query.py']}})
    return result
