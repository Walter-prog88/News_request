"""Read-only financial views. Daily as-of is conservative, never intraday PIT."""
from __future__ import annotations
import json
from datetime import datetime, time
from zoneinfo import ZoneInfo
import pandas as pd
from usmd.financial_quality import quarantine_values, quarantined_status


def cutoff(value):
    if value is None:
        return None
    text = str(value)
    if len(text) == 10:
        day = datetime.strptime(text, '%Y-%m-%d').date()
        return pd.Timestamp(datetime.combine(day, time.max, ZoneInfo('America/New_York'))).tz_convert('UTC')
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError('as_of timestamp needs an explicit timezone; a date means end of New York day')
    return stamp.tz_convert('UTC')


def select_values(frame, *, metrics=None, period='quarter', start='2015-01-01', end=None,
                  as_of=None, revisions='latest', strict_as_of=False,
                  include_conflicts=False, same_filing_only=False):
    if not isinstance(same_filing_only, bool):
        raise ValueError('same_filing_only must be a boolean')
    if strict_as_of:
        raise ValueError('Exact first-public timestamps are unavailable; use conservative daily as_of explicitly')
    kinds = {'quarter': ['direct_quarter', 'derived_quarter'], 'annual': ['annual'],
             'instant': ['instant'], 'cumulative': ['cumulative_6m', 'cumulative_9m'],
             'reported_12m': ['reported_12m'],
             'all': None}
    if period not in kinds or revisions not in ('latest', 'all'):
        raise ValueError('Choose quarter/annual/instant/cumulative/reported_12m/all and latest/all revisions')
    if metrics is not None:
        if isinstance(metrics, str): metrics = [metrics]
        if not metrics: raise ValueError('metrics cannot be empty')
    if end is not None and end < start: raise ValueError('end precedes start')
    d = quarantine_values(frame)
    if d.empty: return d
    allowed_status = ['ok', 'tag_conflict', 'ambiguous_same_tag_value']
    # Keep quarantined versions through latest-vintage selection. Removing a
    # bad latest disclosure first would resurrect a stale clean disclosure.
    allowed = d.row_status.isin(allowed_status) | d.row_status.map(quarantined_status)
    d = d.loc[allowed & d.period_end.ge(start)]
    if end is not None: d = d.loc[d.period_end.le(end)]
    if kinds[period] is not None: d = d.loc[d.period_kind.isin(kinds[period])]
    if metrics is not None: d = d.loc[d.metric_id.isin(metrics)]
    limit = cutoff(as_of)
    available = pd.to_datetime(d.available_at, utc=True, errors='coerce')
    if limit is not None: d = d.loc[available.notna() & available.le(limit)]
    if d.empty: return d.reset_index(drop=True)
    d['_kind'] = d.period_kind.replace({'direct_quarter': 'quarter', 'derived_quarter': 'quarter'})
    d['_direct'] = d.period_kind.eq('direct_quarter').astype(int)
    d['_available'] = pd.to_datetime(d.available_at, utc=True, errors='coerce')
    d['_acceptance'] = pd.to_datetime(d.get('acceptance_at', pd.Series(index=d.index, dtype=str)), utc=True, errors='coerce')
    keys = ['issuer_id', 'metric_id', 'unit', 'period_start', 'period_end', '_kind']
    # Newer published revisions take priority. Within one availability time prefer
    # a directly reported quarter to a subtraction, with stable id tie-breaking.
    score = ['_available', '_acceptance', '_direct']
    d = d.sort_values(keys + score + ['value_id'], na_position='first')
    if revisions == 'latest':
        best = d.drop_duplicates(keys, keep='last')[keys + score]
        tied = d.merge(best, on=keys + score, how='inner')
        if tied.groupby(keys, dropna=False).value.nunique().gt(1).any():
            raise ValueError('Ambiguous same-time financial revisions; inspect revisions="all"')
        d = d.drop_duplicates(keys, keep='last')
    # Eligibility follows vintage selection. An unverified latest subtraction
    # must not make an older reported/same-filing value appear current. Explicit
    # all-revision and historical as-of queries can still expose older vintages.
    if same_filing_only:
        compatibility = d.get('derivation_compatibility', pd.Series(index=d.index, dtype=str))
        d = d.loc[~d.period_kind.eq('derived_quarter') | compatibility.eq('same_filing').fillna(False)]
    # A conflicted latest disclosure must not silently expose an older clean
    # revision as if it were current. Select its vintage before excluding it.
    if not include_conflicts:
        d = d.loc[d.row_status.eq('ok') & ~d.tag_conflict.fillna(False)]
    # include_conflicts exposes accounting conflicts, never impossible actual
    # periods or inconsistent availability estimates.
    d = d.loc[~d.row_status.map(quarantined_status)]
    return d.drop(columns=['_kind', '_direct', '_available', '_acceptance']).reset_index(drop=True)


def statements(frame, *, period='annual', start='2015-01-01', end=None, as_of=None):
    """Wide values plus currency/unit labels; provenance stays in financials()."""
    if period not in ('annual', 'quarter'): raise ValueError('Statements require annual or quarter')
    flow = select_values(frame, period=period, start=start, end=end, as_of=as_of)
    balances = select_values(frame, period='instant', start=start, end=end, as_of=as_of)
    if flow.empty: return pd.DataFrame()
    # Only balance sheet instants matching an actual selected flow period end.
    dates = flow[['issuer_id', 'period_end']].drop_duplicates()
    balances = balances.merge(dates, on=['issuer_id', 'period_end'], how='inner')
    selected = pd.concat([flow, balances], ignore_index=True)
    selected['metric_unit'] = selected.metric_id + ' [' + selected.unit + ']'
    key = ['symbol', 'issuer_id', 'period_end', 'metric_unit']
    row_key = ['symbol','issuer_id','period_end']
    ambiguous = selected.duplicated(key,keep=False)
    issues = selected.loc[ambiguous].groupby(row_key).metric_unit.agg(lambda x: json.dumps(sorted(set(x))))
    dates = selected[row_key].drop_duplicates()
    clean = selected.loc[~ambiguous]
    wide = clean.pivot(index=row_key, columns='metric_unit', values='value').reset_index()
    wide = dates.merge(wide,on=row_key,how='left',validate='one_to_one')
    # Preserve all metric columns even when every occurrence is quarantined.
    for column in sorted(selected.metric_unit.unique()):
        if column not in wide: wide[column] = float('nan')
    wide = wide.merge(issues.rename('ambiguous_metrics').reset_index(),on=row_key,how='left')
    wide['ambiguous_metrics'] = wide.ambiguous_metrics.fillna('[]')
    wide['statement_status'] = wide.ambiguous_metrics.map(lambda x:'ok' if x == '[]' else 'partial_review')
    wide.columns.name = None
    wide['flow_period'] = period
    identity_columns = [c for c in ('company_id','registration_role','ticker_valid_to') if c in frame]
    if identity_columns:
        identity = frame[['issuer_id',*identity_columns]].drop_duplicates()
        if identity.issuer_id.duplicated().any():
            raise ValueError('Inconsistent issuer registration metadata')
        wide = wide.merge(identity,on='issuer_id',how='left',validate='many_to_one')
    return wide.sort_values(['symbol', 'period_end']).reset_index(drop=True)


def trailing_twelve_months(frame, metric_ids, *, as_of=None, start='2015-01-01', end=None):
    """Sum adjacent additive quarters only when their reported source tags agree.

    Excluded source-scope windows remain inspectable in attrs.excluded_windows.
    Equal source tags do not establish a common restatement/accounting basis.
    """
    from usmd.lake import PROJECT_ROOT
    config = json.loads((PROJECT_ROOT / 'config/financial_metrics.json').read_text())
    definitions = {x.get('metric_id', x.get('id')): x for x in config['metrics']}
    if isinstance(metric_ids, str): metric_ids = [metric_ids]
    if not metric_ids: raise ValueError('metrics cannot be empty')
    for metric in metric_ids:
        if metric not in definitions or not definitions[metric].get('additive', False):
            raise ValueError(f'TTM summation requires an explicitly additive metric: {metric}')
    q = select_values(frame, metrics=metric_ids, period='quarter', start='1900-01-01', end=end, as_of=as_of)
    annual = select_values(frame, metrics=metric_ids, period='annual', start='1900-01-01', end=end, as_of=as_of)
    rows, excluded_windows = [], []
    if q.empty:
        result = pd.DataFrame()
        result.attrs['excluded_windows'] = excluded_windows
        return result
    for (issuer, symbol, metric, unit), group in q.groupby(['issuer_id', 'symbol', 'metric_id', 'unit']):
        group = group.sort_values('period_end').reset_index(drop=True)
        for pos in range(3, len(group)):
            window = group.iloc[pos-3:pos+1]
            begins = pd.to_datetime(window.period_start)
            ends = pd.to_datetime(window.period_end)
            if any((begins.iloc[i] - ends.iloc[i-1]).days != 1 for i in range(1,4)): continue
            duration = (ends.iloc[-1] - begins.iloc[0]).days + 1
            if not 350 <= duration <= 380 or str(window.iloc[-1].period_end) < start: continue
            tags = window.get('source_tag', pd.Series(None, index=window.index, dtype=object))
            source_tags = tags.astype(object).where(tags.notna(), None).tolist()
            known_scope = all(isinstance(tag, str) and bool(tag.strip()) for tag in source_tags)
            if not known_scope or len(set(source_tags)) != 1:
                excluded_windows.append({'issuer_id':issuer, 'symbol':symbol, 'metric_id':metric, 'unit':unit,
                    'period_start':window.iloc[0].period_start, 'period_end':window.iloc[-1].period_end,
                    'reason':'missing_source_scope' if not known_scope else 'source_scope_mismatch',
                    'input_value_ids':window.value_id.tolist(), 'input_source_tags':source_tags})
                continue
            source_tag = source_tags[0]
            value = float(window.value.sum())
            reference = annual.loc[annual.issuer_id.eq(issuer) & annual.metric_id.eq(metric)
                & annual.unit.eq(unit) & annual.period_start.eq(window.iloc[0].period_start)
                & annual.period_end.eq(window.iloc[-1].period_end)]
            annual_value, difference, annual_reference_id = None, None, None
            reconciliation = 'not_available'
            if len(reference) == 1:
                annual_tag = reference.iloc[0].get('source_tag')
                if not isinstance(annual_tag, str) or annual_tag != source_tag:
                    reconciliation = 'source_scope_mismatch'
                else:
                    annual_value = float(reference.iloc[0].value)
                    annual_reference_id = reference.iloc[0].value_id
                    difference = value - annual_value
                    reconciliation = 'match_within_tolerance' if abs(difference) <= max(1.0,abs(annual_value)*1e-4) else 'mismatch'
            elif len(reference) > 1:
                reconciliation = 'ambiguous_annual_period'
            cross_filing = bool(window.accn.nunique() > 1 or window.get('derivation_compatibility', pd.Series(dtype=str)).eq('cross_filing_unverified').any())
            rows.append({'issuer_id': issuer, 'symbol': symbol, 'metric_id': metric, 'unit': unit,
                         'period_start': window.iloc[0].period_start, 'period_end': window.iloc[-1].period_end,
                         'value': value, 'period_kind': 'ttm',
                         'available_at': max(window.available_at), 'availability_basis': 'estimated_filed_next_day',
                         'pit_eligible': False, 'input_value_ids': json.dumps(window.value_id.tolist()),
                         'source_tag': source_tag, 'input_source_tags': json.dumps(source_tags),
                         'source_scope_consistent': True,
                         'cross_filing_unverified': cross_filing,
                         'accounting_basis_verified': False,
                         'annual_reconciliation_status': reconciliation,
                         'annual_reference_value': annual_value, 'difference_to_annual': difference,
                         'annual_reference_value_id': annual_reference_id,
                         'ttm_quality_status': 'review' if reconciliation in ('mismatch','ambiguous_annual_period','source_scope_mismatch') else 'basis_unverified'})
    result = pd.DataFrame(rows)
    result.attrs['excluded_windows'] = excluded_windows
    return result
