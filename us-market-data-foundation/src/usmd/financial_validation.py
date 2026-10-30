"""Bounded-memory validation and catalog metadata for broad financial tables."""
from __future__ import annotations
from datetime import datetime, timezone
import duckdb
import pyarrow.parquet as pq

DISTINCT_PARTITIONS_THRESHOLD = 5_000_000


def _duplicate_count(con, key, rows):
    """Exact bounded identity check; each key belongs to one hash partition."""
    if rows <= DISTINCT_PARTITIONS_THRESHOLD:
        return con.execute(f'SELECT count(*)-count(DISTINCT {key}) FROM financial_rows').fetchone()[0]
    return sum(con.execute(f'SELECT count(*)-count(DISTINCT {key}) FROM financial_rows WHERE hash({key}) % 16 = ?',
                           [partition]).fetchone()[0] for partition in range(16))


def validate_paths(paths, today=None, temp_directory=None):
    errors, warnings, checks = [], [], {}
    con = duckdb.connect(':memory:', config={'memory_limit': '1GB', 'threads': 2})
    try:
        con.execute("SET temp_directory = ''")
        if temp_directory is not None:
            con.execute('SET temp_directory = ?', [str(temp_directory)])
        for name, key in [('financial_facts', 'fact_id'), ('financial_values', 'value_id')]:
            if name not in paths: continue
            con.from_parquet(str(paths[name])).create_view('financial_rows', replace=True)
            columns = set(pq.read_schema(paths[name]).names)
            required = {key, 'issuer_id', 'unit', 'value', 'source_id', 'accn', 'filed', 'available_at', 'row_status'}
            if name == 'financial_values': required |= {'period_end', 'tag_conflict'}
            missing = required - columns
            if missing:
                errors.append({'table': name, 'check': 'required_columns', 'missing': sorted(missing)})
                continue
            rows, issuers, invalid, null_source = con.execute(f'''
                SELECT count(*), count(DISTINCT issuer_id),
                       count(*) FILTER (WHERE try_cast(value AS DOUBLE) IS NULL OR NOT isfinite(try_cast(value AS DOUBLE))),
                       count(*) FILTER (WHERE issuer_id IS NULL OR source_id IS NULL OR unit IS NULL OR accn IS NULL)
                FROM financial_rows''').fetchone()
            duplicates = _duplicate_count(con, key, rows)
            if duplicates or invalid or null_source:
                errors.append({'table': name, 'check': 'fact_integrity', 'duplicates': duplicates,
                               'nonfinite': invalid, 'missing_source': null_source})
            checks[name] = {'rows': rows, 'issuers': issuers, 'duplicates': duplicates, 'source_linked': null_source == 0}
            if name == 'financial_values' and rows:
                future, conflict = con.execute('''SELECT
                    count(*) FILTER (WHERE try_cast(period_end AS DATE) > cast(? AS DATE)),
                    count(*) FILTER (WHERE tag_conflict = true) FROM financial_rows''',
                    [today or datetime.now(timezone.utc).date().isoformat()]).fetchone()
                if future: errors.append({'table': name, 'check': 'future_actual_period', 'count': future})
                period_bad, period_unmarked = con.execute('''SELECT
                    count(*) FILTER (WHERE try_cast(period_end AS DATE) > try_cast(filed AS DATE)),
                    count(*) FILTER (WHERE try_cast(period_end AS DATE) > try_cast(filed AS DATE)
                        AND NOT contains(coalesce(row_status,''),'period_end_after_filing_date'))
                    FROM financial_rows''').fetchone()
                if period_unmarked:
                    errors.append({'table': name, 'check': 'unquarantined_period_after_filing', 'count': period_unmarked})
                if period_bad:
                    warnings.append({'table': name, 'check': 'quarantined_period_after_filing', 'count': period_bad})
                checks[name]['period_after_filing_rows'] = period_bad
                checks[name]['unquarantined_period_after_filing_rows'] = period_unmarked
                if {'acceptance_at', 'availability_basis'} <= columns:
                    early, unmarked = con.execute('''SELECT
                        count(*) FILTER (WHERE starts_with(availability_basis,'estimated_filed_next_day')
                            AND try_cast(available_at AS TIMESTAMPTZ)<try_cast(acceptance_at AS TIMESTAMPTZ)),
                        count(*) FILTER (WHERE starts_with(availability_basis,'estimated_filed_next_day')
                            AND try_cast(available_at AS TIMESTAMPTZ)<try_cast(acceptance_at AS TIMESTAMPTZ)
                            AND NOT contains(coalesce(row_status,''),'availability_precedes_acceptance'))
                        FROM financial_rows''').fetchone()
                    checks[name]['estimated_availability_before_acceptance_rows'] = early
                    checks[name]['unquarantined_availability_rows'] = unmarked
                    if unmarked:
                        errors.append({'table': name, 'check': 'unquarantined_availability_contract', 'count': unmarked})
                    if early:
                        warnings.append({'table': name, 'check': 'quarantined_availability_contract', 'count': early,
                                         'detail': 'Original timestamps retained; acceptance is not first-public evidence'})
                warnings.append({'table': name, 'check': 'conservative_availability',
                                 'detail': 'Daily estimated availability, not exact first-public PIT; raw SEC acceptance retained'})
                if conflict: warnings.append({'table': name, 'check': 'tag_conflicts', 'count': conflict})
    finally:
        con.close()
    return errors, warnings, checks


def parquet_metadata(path):
    file = pq.ParquetFile(path)
    schema = file.schema_arrow
    return {'row_count': file.metadata.num_rows, 'columns': schema.names,
            'dtypes': {field.name: str(field.type) for field in schema},
            'bytes': path.stat().st_size}
