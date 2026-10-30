"""Cohort correctness: aliases, legal boundaries, query coverage and as-of."""
import contextlib
import json
import unittest

import pandas as pd

from usmd.financial_cohorts import panel, panel_coverage, select_companies
from usmd.financial_query import select_values


def issuer(symbol, cik, sector, **changes):
    row = {'issuer_id': f'SEC:{cik}', 'symbol': symbol, 'company_id': 'US:' + symbol,
           'symbols': json.dumps([symbol]), 'sector': sector, 'registration_role': 'current',
           'sector_as_of_date': '2026-09-17', 'membership_as_of_date': '2026-09-17',
           'universe_as_of_date': '2026-09-17', 'sector_source_url': 'https://example.test/holdings',
           'benchmark_weights': json.dumps({'SP500': 1.0}),
           'identity_status': 'verified_sec_submissions'}
    return {**row, **changes}


def value(issuer_row, metric='revenue', **changes):
    row = {'issuer_id': issuer_row['issuer_id'], 'symbol': issuer_row['symbol'],
           'company_id': issuer_row['company_id'], 'registration_role': issuer_row['registration_role'],
           'metric_id': metric, 'unit': 'USD', 'value': 10.0, 'period_start': '2025-01-01',
           'period_end': '2025-12-31', 'period_kind': 'annual', 'filed': '2026-02-01',
           'available_at': '2026-02-02T05:00:00+00:00', 'acceptance_at': '2026-02-01T12:00:00+00:00',
           'row_status': 'ok', 'tag_conflict': False, 'derivation_compatibility': 'direct_reported',
           'value_id': issuer_row['issuer_id'] + metric, 'source_id': 'SEC source', 'accn': 'accession'}
    return {**row, **changes}


class FakeStore:
    lake = type('Lake', (), {'reader_lock': lambda self: contextlib.nullcontext()})()

    def __init__(self):
        rows = [issuer('GOOGL', '1', 'Communication Services', symbols='["GOOGL", "GOOG"]',
                       benchmark_weights='{"SP500": 3.0, "NASDAQ100": 5.0}'),
                issuer('XOM', '2', 'Energy'),
                issuer('XOM', '3', 'Energy', registration_role='predecessor'),
                issuer('PANW', '4', 'Information Technology', benchmark_weights='{"NASDAQ100": 0.0}'),
                issuer('EMPTY', '5', 'Information Technology')]
        self.issuers = pd.DataFrame(rows)
        self.values = pd.DataFrame([value(row) for row in rows[:4]])
        self.calls = []

    def financial_issuers(self):
        return self.issuers.copy()

    def financials(self, symbols, **kwargs):
        self.calls.append((symbols, kwargs))
        return select_values(self.values.loc[self.values.issuer_id.isin(symbols)], **kwargs)


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.store = FakeStore()

    def test_share_classes_are_one_issuer_and_company(self):
        selected = select_companies(self.store, symbols=['GOOG', 'GOOGL'])
        self.assertEqual(selected.issuer_id.tolist(), ['SEC:1'])
        self.assertEqual(selected.attrs['company_count'], 1)

    def test_xom_registrants_are_separate_and_never_summed(self):
        result = panel(self.store, symbols='XOM', metrics='revenue')
        self.assertEqual(set(result.issuer_id), {'SEC:2', 'SEC:3'})
        self.assertEqual(result.attrs['company_count'], 1)
        self.assertEqual(result.attrs['registrant_count'], 2)
        self.assertEqual(result.value.tolist(), [10.0, 10.0])
        direct = select_companies(self.store, symbols='SEC:3')
        self.assertEqual(len(direct), 1)

    def test_filters_are_and_within_filter_or_and_zero_weight_is_member(self):
        selected = select_companies(self.store, sectors='Information Technology', benchmarks='NASDAQ100')
        self.assertEqual(selected.symbol.tolist(), ['PANW'])
        selected = select_companies(self.store, symbols=['GOOG', 'PANW'], sectors='Energy')
        self.assertTrue(selected.empty)

    def test_unknown_filters_and_empty_lists_fail(self):
        for kwargs in ({'symbols': 'UNKNOWN'}, {'symbols': []}, {'sectors': 'Unknown'},
                       {'benchmarks': 'Unknown'}, {'sectors': []}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_companies(self.store, **kwargs)

    def test_snapshot_date_does_not_claim_historical_classification(self):
        selected = select_companies(self.store, classification_date='2026-09-17')
        self.assertEqual(len(selected), 5)
        with self.assertRaisesRegex(ValueError, 'historical membership'):
            select_companies(self.store, classification_date='2020-01-01')

    def test_unknown_classification_stays_null(self):
        self.store.issuers.loc[0, 'sector'] = None
        selected = select_companies(self.store, symbols='GOOG')
        self.assertTrue(pd.isna(selected.iloc[0].sector))

    def test_batched_queries_forward_metric_filters_and_keep_source(self):
        result = panel(self.store, metrics='revenue', batch_size=2, as_of='2026-09-18')
        self.assertEqual([len(call[0]) for call in self.store.calls], [2, 2, 1])
        self.assertEqual(self.store.calls[0][1]['metrics'], ['revenue'])
        self.assertEqual(result.source_id.unique().tolist(), ['SEC source'])
        self.assertTrue(result.sector_source_url.notna().all())
        self.assertEqual(result.attrs['cohort_issuer_ids_without_values'], ['SEC:5'])
        self.assertEqual(result.filed_age_days.unique().tolist(), [229])

    def test_future_disclosures_excluded_without_rewriting_current_membership(self):
        result = panel(self.store, metrics='revenue', as_of='2026-02-01')
        self.assertTrue(result.empty)
        coverage = panel_coverage(result)
        self.assertEqual(len(coverage), 5)
        self.assertTrue(coverage.value_rows.eq(0).all())
        self.assertTrue(coverage.coverage_status.eq('no_selected_values').all())
        self.assertTrue(coverage.latest_value_filed_age_days.isna().all())
        self.assertEqual(result.attrs['classification_dates'], ['2026-09-17'])
        self.assertFalse(result.attrs['exact_point_in_time'])

    def test_coverage_keeps_missing_series_not_zero_values(self):
        result = panel(self.store, symbols=['GOOG', 'EMPTY'], metrics=['revenue', 'net_income'], as_of='2026-09-18')
        coverage = panel_coverage(result)
        self.assertEqual(len(coverage), 4)
        self.assertEqual(coverage.value_rows.sum(), 1)
        self.assertEqual((coverage.coverage_status == 'no_selected_values').sum(), 3)
        self.assertEqual(len(result), 1)
        self.assertEqual(coverage.loc[coverage.value_rows.eq(1), 'latest_value_filed_age_days'].tolist(), [229])

    def test_duplicate_issuer_metadata_and_identity_conflicts_fail(self):
        self.store.issuers = pd.concat([self.store.issuers, self.store.issuers.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'one row per'):
            select_companies(self.store)
        self.store = FakeStore()
        self.store.values.loc[0, 'company_id'] = 'US:WRONG'
        with self.assertRaisesRegex(ValueError, 'identity conflicts'):
            panel(self.store, metrics='revenue')

    def test_resource_limits_fail_instead_of_partial_results(self):
        with self.assertRaisesRegex(ValueError, 'exceeds max_rows'):
            panel(self.store, metrics='revenue', batch_size=2, max_rows=1)
        for kwargs in ({'batch_size': 0}, {'batch_size': 101}, {'max_rows': 0}, {'batch_size': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                panel(self.store, **kwargs)

    def test_metric_and_date_validation_even_for_empty_cohort(self):
        for kwargs in ({'metrics': []}, {'metrics': 'typo'}, {'period': 'month'},
                       {'start': '2025-01-01', 'end': '2024-01-01'}, {'as_of': '2026-01-01T12:00:00'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                panel(self.store, symbols='GOOG', sectors='Energy', **kwargs)

    def test_default_metrics_and_empty_combination(self):
        result = panel(self.store, symbols='GOOG', sectors='Energy')
        self.assertTrue(result.empty)
        self.assertEqual(len(result.attrs['requested_metrics']), 5)
        self.assertTrue(panel_coverage(result).empty)

    def test_coverage_requires_query_metadata(self):
        with self.assertRaisesRegex(ValueError, 'unmodified result'):
            panel_coverage(pd.DataFrame())


if __name__ == '__main__':
    unittest.main()
