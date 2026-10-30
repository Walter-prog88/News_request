"""Coarse company cohorts and source-preserving financial research panels.

Classification snapshots select a research sample, not historical constituents.
No aggregation, currency conversion, fiscal-calendar alignment or missing fill.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile
from zoneinfo import ZoneInfo

import pandas as pd

from usmd.financial_query import cutoff


DEFAULT_METRICS = (
    'revenue', 'operating_income', 'net_income',
    'operating_cash_flow', 'capital_expenditures',
)
COHORT_COLUMNS = (
    'company_id', 'company_name', 'sector', 'sector_source_url',
    'sector_as_of_date', 'sector_mapping_basis', 'classification_status',
    'benchmark_weights', 'benchmark_memberships', 'universe_as_of_date',
    'membership_as_of_date', 'registration_role', 'ticker_valid_to',
    'identity_status', 'symbols',
    'sec_sic', 'sec_sic_description', 'profile', 'profile_basis', 'profile_source_url',
    'unverified_symbols', 'symbol_identity_status', 'symbol_identity_basis',
)


def _list(value, name):
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value)
    if not values:
        raise ValueError(f'{name} cannot be empty')
    return list(dict.fromkeys(str(item) for item in values))


def _decoded(value, expected):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return expected()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, expected):
        raise ValueError(f'Expected serialized {expected.__name__} cohort metadata')
    return value


def _alias(value):
    return str(value).upper().replace('.', '-')


def _read_snapshot(store):
    snapshot = getattr(store, 'read_snapshot', None)
    return snapshot() if callable(snapshot) else store.lake.reader_lock()


def _metric_dictionary_snapshot():
    from usmd.lake import PROJECT_ROOT
    dictionary = json.loads((PROJECT_ROOT / 'config/financial_metrics.json').read_bytes())
    canonical = json.dumps(dictionary, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return dictionary, hashlib.sha256(canonical).hexdigest()


def select_companies(store, *, sectors=None, benchmarks=None, symbols=None,
                     classification_date=None):
    """Return unique published SEC registrants matching AND-combined filters.

    Multiple values within a filter use OR. Unknown symbols fail explicitly.
    An exact classification_date validates the saved snapshot; it is not an
    as-of search through historical sector/index membership.
    """
    sectors = _list(sectors, 'sectors')
    benchmarks = _list(benchmarks, 'benchmarks')
    symbols = _list(symbols, 'symbols')
    frame = store.financial_issuers().copy()
    required = {'issuer_id', 'symbol'}
    if not required.issubset(frame.columns):
        raise ValueError('Published financial issuer metadata is incomplete')
    if frame.issuer_id.duplicated().any():
        raise ValueError('Published issuer metadata must have one row per SEC registrant')
    if 'company_id' not in frame:
        frame['company_id'] = 'US:' + frame.symbol
    if 'registration_role' not in frame:
        frame['registration_role'] = 'current'
    for column in COHORT_COLUMNS:
        if column not in frame:
            frame[column] = None
    known_sectors = set(frame.sector.dropna())
    known_benchmarks = set()
    for value in frame.benchmark_weights:
        known_benchmarks.update(_decoded(value, dict))
    if symbols is not None:
        aliases = {}
        for row in frame.to_dict('records'):
            for name in [row['issuer_id'], row['symbol'], *_decoded(row['symbols'], list)]:
                aliases.setdefault(_alias(name), set()).add(row['issuer_id'])
        wanted = {_alias(value) for value in symbols}
        unknown = wanted - set(aliases)
        if unknown:
            raise ValueError('Financial issuer not collected: ' + ', '.join(sorted(unknown)))
        ids = {issuer for name in wanted for issuer in aliases[name]}
        frame = frame.loc[frame.issuer_id.isin(ids)]
    if sectors is not None:
        unknown = set(sectors) - known_sectors
        if unknown:
            raise ValueError('Unknown published sector: ' + ', '.join(sorted(unknown)))
        frame = frame.loc[frame.sector.isin(sectors)]
    if benchmarks is not None:
        unknown = set(benchmarks) - known_benchmarks
        if unknown:
            raise ValueError('Unknown published benchmark: ' + ', '.join(sorted(unknown)))
        wanted = set(benchmarks)
        # Membership is the presence of a key, not the magnitude of its weight.
        frame = frame.loc[frame.benchmark_weights.map(
            lambda value: bool(wanted.intersection(_decoded(value, dict))))]
    if classification_date is not None:
        requested = str(classification_date)
        datetime.strptime(requested, '%Y-%m-%d')
        if not frame.empty and not frame.sector_as_of_date.eq(requested).all():
            raise ValueError('Requested classification snapshot is unavailable; historical membership is not reconstructed')
    frame = frame.sort_values(['symbol', 'issuer_id']).reset_index(drop=True)
    frame.attrs.update({
        'pool_scope': 'published_registrants',
        'membership_policy': 'saved_current_snapshot_not_historical_membership',
        'company_count': int(frame.company_id.nunique()),
        'registrant_count': len(frame),
        'classification_dates': sorted(frame.sector_as_of_date.dropna().unique().tolist()),
        'membership_dates': sorted(frame.membership_as_of_date.dropna().unique().tolist()),
    })
    return frame


def panel(store, *, sectors=None, benchmarks=None, symbols=None, metrics=None,
          period='annual', start='2015-01-01', end=None, as_of=None,
          classification_date=None, same_filing_only=False,
          batch_size=25, max_rows=1_000_000):
    """Latest eligible values in bounded issuer batches, preserving provenance.

    None metrics selects five common flow metrics, not every stored field.
    as_of affects financial disclosures only. The current saved classification
    and selection dates remain visible and must not be used as historical
    constituent evidence. Missing series are exposed by panel_coverage().
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 100:
        raise ValueError('batch_size must be an integer from 1 to 100')
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError('max_rows must be a positive integer')
    metric_ids = _list(metrics if metrics is not None else DEFAULT_METRICS, 'metrics')
    # Validate even when no cohort rows match.
    if period not in ('annual', 'quarter', 'instant', 'cumulative', 'reported_12m', 'all'):
        raise ValueError('Unsupported financial period')
    if end is not None and end < start:
        raise ValueError('end precedes start')
    limit = cutoff(as_of)
    reference_date = (limit.tz_convert('America/New_York').date().isoformat()
                      if limit is not None else datetime.now(ZoneInfo('America/New_York')).date().isoformat())
    with _read_snapshot(store):
        dictionary, dictionary_hash = _metric_dictionary_snapshot()
        known = {item.get('metric_id', item.get('id')) for item in dictionary['metrics']}
        if set(metric_ids) - known:
            raise ValueError('Unknown financial metrics: ' + ', '.join(sorted(set(metric_ids) - known)))
        cohort = select_companies(store, sectors=sectors, benchmarks=benchmarks,
                                  symbols=symbols, classification_date=classification_date)
        ids = cohort.issuer_id.tolist()
        pieces, total = [], 0
        for offset in range(0, len(ids), batch_size):
            piece = store.financials(ids[offset:offset + batch_size], metrics=metric_ids,
                                    period=period, start=start, end=end, as_of=as_of,
                                    same_filing_only=same_filing_only)
            total += len(piece)
            if total > max_rows:
                raise ValueError('Financial panel exceeds max_rows; narrow dates/metrics/cohort or explicitly raise the limit')
            pieces.append(piece)
        result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
            columns=['issuer_id', 'symbol', 'metric_id', 'period_start', 'period_end',
                     'unit', 'value', 'filed', 'available_at'])
        # Issuer metadata is authoritative for classification. Existing identity
        # columns in values are retained after ensuring they agree where present.
        for column in ('company_id', 'registration_role'):
            if column in result and not result.empty:
                lookup = cohort.set_index('issuer_id')[column]
                expected = result.issuer_id.map(lookup)
                conflict = result[column].notna() & expected.notna() & result[column].ne(expected)
                if conflict.any():
                    raise ValueError('Financial row identity conflicts with cohort metadata: ' + column)
        result = result.drop(columns=[column for column in COHORT_COLUMNS if column in result])
        result = result.merge(cohort[['issuer_id', *COHORT_COLUMNS]], on='issuer_id', how='left', validate='many_to_one')
        for source, target in [('filed', 'filed_age_days'), ('period_end', 'period_age_days')]:
            dates = pd.to_datetime(result[source], errors='coerce')
            result[target] = (pd.Timestamp(reference_date) - dates).dt.days.astype('Int64')
        result['age_reference_date'] = reference_date
        result['classification_policy'] = 'saved_current_snapshot_not_historical_membership'
        result = result.sort_values(['symbol', 'issuer_id', 'metric_id', 'period_end', 'period_start']).reset_index(drop=True)
        result.attrs.update(cohort.attrs)
        result.attrs.update({
            'requested_metrics': metric_ids,
            'financial_as_of': str(as_of) if as_of is not None else None,
            'financial_period': period,
            'requested_start': start,
            'requested_end': end,
            'same_filing_only': same_filing_only,
            'batch_size': batch_size,
            'max_rows': max_rows,
            'metric_dictionary': dictionary,
            'metric_dictionary_sha256': dictionary_hash,
            'metric_dictionary_hash_encoding': 'utf8_sorted_compact_json',
            'metric_dictionary_role': 'query_validation_snapshot; historical_normalization_dictionary_match_not_verified',
            'age_reference_date': reference_date,
            'cohort_records': cohort[['issuer_id', 'symbol', *COHORT_COLUMNS]].to_dict('records'),
            'cohort_issuer_ids_without_values': sorted(set(ids) - set(result.issuer_id)),
            'aggregation_policy': 'none; legal registrants, units and fiscal periods remain separate',
            'exact_point_in_time': False,
        })
    return result


def panel_coverage(frame):
    """Coverage of this exact query, including selected issuer/metric gaps.

    Count zeros mean no returned records, never a financial value of zero.
    Filing age refers to the latest returned value for each metric, not a
    guarantee that Company Facts contains the latest filed financial statement.
    """
    if 'cohort_records' not in frame.attrs or 'requested_metrics' not in frame.attrs:
        raise ValueError('Use the unmodified result of financial_cohorts.panel()')
    cohort = pd.DataFrame(frame.attrs['cohort_records'])
    metrics = pd.DataFrame({'metric_id': frame.attrs['requested_metrics']})
    if cohort.empty:
        return pd.DataFrame(columns=['issuer_id', 'symbol', 'metric_id', 'value_rows',
                                     'period_count', 'coverage_status'])
    result = cohort.merge(metrics, how='cross')
    keys = ['issuer_id', 'metric_id']
    if frame.empty:
        grouped = pd.DataFrame(columns=keys + ['value_rows', 'period_count', 'first_period_end',
                               'latest_period_end', 'latest_value_filed', 'units'])
    else:
        grouped = frame.groupby(keys, dropna=False).agg(
            value_rows=('metric_id', 'size'), period_count=('period_end', 'nunique'),
            first_period_end=('period_end', 'min'), latest_period_end=('period_end', 'max'),
            latest_value_filed=('filed', 'max'),
            units=('unit', lambda values: json.dumps(sorted(set(values.dropna())))),
        ).reset_index()
    result = result.merge(grouped, on=keys, how='left', validate='one_to_one')
    for column in ('value_rows', 'period_count'):
        result[column] = result[column].fillna(0).astype(int)
    result['coverage_status'] = result.value_rows.map(lambda count: 'returned' if count else 'no_selected_values')
    result['age_reference_date'] = frame.attrs['age_reference_date']
    result['latest_value_filed_age_days'] = (
        pd.Timestamp(frame.attrs['age_reference_date']) - pd.to_datetime(result.latest_value_filed, errors='coerce')
    ).dt.days.astype('Int64')
    result['financial_as_of'] = frame.attrs['financial_as_of']
    result['financial_period'] = frame.attrs['financial_period']
    result['coverage_basis'] = 'this_query_only; no_selected_values_is_not_a_financial_zero'
    result.attrs = {key: value for key, value in frame.attrs.items() if key != 'cohort_records'}
    return result.sort_values(['symbol', 'issuer_id', 'metric_id']).reset_index(drop=True)


def export_panel(store, output='exports/financials/research_panel', **filters):
    """Legacy per-file directory export; use export_panel_bundle for atomic sets."""
    import hashlib
    from pathlib import Path
    parts = Path(output).parts
    if Path(output).is_absolute() or not parts or parts[0] != 'exports' or '..' in parts:
        raise ValueError('Financial panel exports must stay under exports/')
    with _read_snapshot(store):
        frame = panel(store, **filters)
        coverage = panel_coverage(frame)
        companies = pd.DataFrame(frame.attrs['cohort_records'])
        catalog = json.loads((store.lake.root/'catalog.json').read_text())
    definition = {**frame.attrs['metric_dictionary'], 'metrics': [m for m in frame.attrs['metric_dictionary']['metrics']
                  if m['metric_id'] in frame.attrs['requested_metrics']]}
    metadata = {key:value for key,value in frame.attrs.items() if key not in ('cohort_records', 'metric_dictionary')}
    manifest = {'schema_version':1, 'query':metadata,
                'table_hashes':{key:item['sha256'] for key,item in catalog['tables'].items() if key.startswith('financial_')},
                'catalog_generated_at':catalog['generated_at'], 'rows':len(frame),
                'registrants':len(companies), 'files':{}}
    with store.lake.writer_lock():
        for filename,data in [('values.csv',frame),('coverage.csv',coverage),('companies.csv',companies)]:
            body=data.to_csv(index=False).encode()
            store.lake.write_bytes(str(Path(output)/filename),body)
            manifest['files'][filename]={'sha256':hashlib.sha256(body).hexdigest(),'rows':len(data),'bytes':len(body)}
        body=(json.dumps(definition,ensure_ascii=False,indent=2)+'\n').encode()
        store.lake.write_bytes(str(Path(output)/'metrics.json'),body)
        manifest['files']['metrics.json']={'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)}
        store.lake.write_json(str(Path(output)/'manifest.json'),manifest)
    return {'output':str(store.lake.root/output),**manifest}


def _panel_code_hashes():
    from usmd.lake import PROJECT_ROOT
    names = ('financial_cohorts.py', 'financial_query.py', 'financial_normalize.py', 'query.py')
    return {name: hashlib.sha256((PROJECT_ROOT / 'src/usmd' / name).read_bytes()).hexdigest() for name in names}


def export_panel_bundle(store, output='exports/financials/research_panel.zip', **filters):
    """Capture one read snapshot and atomically publish its complete research ZIP.

    Definitions are the dictionary actually used by panel(), not a later disk
    read or a claim about the dictionary used in historical normalization.
    """
    output = str(output)
    parts = Path(output).parts
    if Path(output).is_absolute() or len(parts) < 2 or parts[0] != 'exports' or '..' in parts or not output.endswith('.zip'):
        raise ValueError('Financial panel bundles must be .zip files under exports/')
    # Materialize multi-value filters once so the receipt records the same
    # selection even if callers supplied generators.
    filters = dict(filters)
    for name in ('symbols', 'sectors', 'benchmarks', 'metrics'):
        if name in filters and filters[name] is not None:
            filters[name] = _list(filters[name], name)
    with _read_snapshot(store):
        code_hashes = _panel_code_hashes()
        frame = panel(store, **filters)
        coverage = panel_coverage(frame)
        companies = pd.DataFrame(frame.attrs['cohort_records'], columns=['issuer_id', 'symbol', *COHORT_COLUMNS])
        catalog_bytes = (store.lake.root / 'catalog.json').read_bytes()
        catalog = json.loads(catalog_bytes)
        if code_hashes != _panel_code_hashes():
            raise ValueError('Financial panel implementation changed during capture; rerun before exporting')
    definition = frame.attrs['metric_dictionary']
    definition_body = (json.dumps(definition, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    metadata = {key: value for key, value in frame.attrs.items() if key not in ('cohort_records', 'metric_dictionary')}
    manifest = {'schema_version': 1, 'artifact_type': 'financial_panel_bundle',
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'publication_policy': 'single_atomic_zip_from_captured_read_snapshot',
                'query': metadata, 'request': filters,
                'source_table_hashes': {name: catalog['tables'][name]['sha256'] for name in ('financial_values', 'financial_issuers')},
                'source_catalog_sha256': hashlib.sha256(catalog_bytes).hexdigest(),
                'catalog_generated_at': catalog.get('generated_at'),
                'code_sha256': code_hashes, 'code_hash_basis': 'source_files_at_query_capture',
                'rows': len(frame), 'registrants': len(companies), 'files': {}}
    frames = {'values.csv': frame, 'coverage.csv': coverage, 'companies.csv': companies}

    def produce(path):
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for filename, data in frames.items():
                digest, size = hashlib.sha256(), 0
                with archive.open(filename, 'w') as member:
                    # Bound CSV serialization overhead independently of the
                    # caller's panel max_rows. All entries publish together.
                    for offset in range(0, max(len(data), 1), 10_000):
                        body = data.iloc[offset:offset + 10_000].to_csv(index=False, header=offset == 0).encode()
                        member.write(body)
                        digest.update(body)
                        size += len(body)
                manifest['files'][filename] = {'sha256': digest.hexdigest(), 'rows': len(data), 'bytes': size}
            archive.writestr('metric_definitions.json', definition_body)
            manifest['files']['metric_definitions.json'] = {'sha256': hashlib.sha256(definition_body).hexdigest(), 'bytes': len(definition_body)}
            archive.writestr('manifest.json', (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n').encode())

    estimated_size = sum(int(data.memory_usage(index=False, deep=True).sum()) for data in frames.values()) + len(definition_body) + 65536
    with store.lake.writer_lock():
        store.lake.write_generated(output, produce, estimated_size=estimated_size)
    return {'output': str(store.lake.root / output), **manifest}
