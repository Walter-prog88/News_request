"""Industry sample correctness: scope separation, duplicate control and fixed YOY."""
import json
import unittest

import pandas as pd

from usmd.financial_industry import aggregate, _DICTIONARY


def company(symbol, issuer=None, **changes):
    return {'symbol': symbol, 'issuer_id': issuer or 'SEC:' + symbol, 'company_id': 'US:' + symbol,
            'sector': 'Technology', 'sector_source_url': 'https://example.test/holdings',
            'sector_as_of_date': '2026-09-17', 'profile': 'general', **changes}


def value(meta, year, amount, **changes):
    return {**meta, 'metric_id': 'revenue', 'unit': 'USD', 'value': amount,
            'value_id': meta['issuer_id'] + str(year), 'source_id': 'source:' + meta['issuer_id'],
            'source_tag': 'us-gaap:Revenues', 'period_start': f'{year}-01-01', 'period_end': f'{year}-12-31',
            'period_kind': 'annual', 'row_status': 'ok', 'tag_conflict': False,
            'derivation_compatibility': 'direct_reported', **changes}


def panel(companies, rows, **attrs):
    frame = pd.DataFrame(rows)
    frame.attrs = {'cohort_records': companies, 'financial_period': 'annual', 'requested_metrics': ['revenue'],
                   'requested_start': '2024-01-01', 'requested_end': '2025-12-31', **attrs}
    return frame


class IndustryTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.c = [company(symbol) for symbol in ('A', 'B', 'C')]

    def test_unknown_derivation_basis_never_silently_passes_default_policy(self):
        for basis in (None, 'unknown', 'not_applicable'):
            row = value(self.a, 2025, 10, period_kind='derived_quarter', period_end='2025-03-31',
                        derivation_compatibility=basis)
            result = aggregate(panel([self.a], [row], financial_period='quarter'))
            self.assertTrue(result['summary'].empty)
            self.assertIn('derived_quarter_compatibility_unverified', result['exclusions'].exclusion_reason.tolist())
        frame = panel([self.a], [row], financial_period='quarter').drop(columns='derivation_compatibility')
        self.assertTrue(aggregate(frame)['summary'].empty)
        row['derivation_compatibility'] = 'same_filing'
        self.assertEqual(aggregate(panel([self.a], [row], financial_period='quarter'))['summary'].iloc[0].sum_value, 10)

    def test_unverified_flag_requires_a_real_boolean(self):
        for flag in ('false', 0, 1, None):
            with self.assertRaisesRegex(ValueError, 'boolean'):
                aggregate(panel([self.a], [value(self.a, 2025, 10)]), allow_unverified_quarters=flag)

    def test_yoy_uses_same_companies_not_changing_total(self):
        rows = [value(self.a, 2024, 100), value(self.b, 2024, 100),
                value(self.a, 2025, 110), value(self.b, 2025, 120), value(self.c, 2025, 1000)]
        result = aggregate(panel([self.a, self.b, self.c], rows))
        current = result['summary'].loc[lambda d: d.bucket.eq('2025')].iloc[0]
        self.assertEqual(current.sum_value, 1230)
        self.assertAlmostEqual(current.yoy_growth_pct, 15)
        self.assertEqual(current.yoy_matched_company_count, 2)
        self.assertEqual(current.yoy_matched_current_sum, 230)
        self.assertEqual(current.yoy_matched_prior_sum, 200)
        self.assertEqual(current.yoy_current_unmatched_company_count, 1)
        self.assertAlmostEqual(current.max_absolute_value_share, 1000 / 1230)
        contributions = result['contributions'].loc[lambda d: d.bucket.eq('2025')]
        self.assertAlmostEqual(contributions.yoy_contribution_percentage_points.sum(), 15)
        self.assertEqual(len(json.loads(current.source_value_ids)), 3)

    def test_missing_cohort_company_is_in_denominator_without_zero_value(self):
        result = aggregate(panel([self.a, self.b], [value(self.a, 2025, 10)]))
        self.assertEqual(len(result['contributions']), 1)
        coverage = result['coverage'].loc[lambda d: d.bucket.eq('2025')].iloc[0]
        self.assertEqual(coverage.selected_cohort_company_count, 2)
        self.assertEqual(coverage.included_company_count, 1)
        self.assertEqual(coverage.coverage_fraction, .5)
        self.assertEqual(json.loads(coverage.without_included_value_company_ids), ['US:B'])
        self.assertEqual(result['summary'].iloc[0].sum_value, 10)

    def test_currency_tags_profile_and_classification_source_stay_separate(self):
        b = {**self.b, 'profile': 'bank'}
        c = {**self.c, 'sector_source_url': 'https://other.test/holdings'}
        d, e = company('D'), company('E')
        rows = [value(self.a, 2025, 10), value(b, 2025, 20), value(c, 2025, 30),
                value(d, 2025, 40, unit='EUR'), value(e, 2025, 50, source_tag='ifrs-full:Revenue')]
        dictionary = json.loads(json.dumps(_DICTIONARY))
        next(m for m in dictionary['metrics'] if m['metric_id'] == 'revenue')['units'].append('EUR')
        result = aggregate(panel([self.a, b, c, d, e], rows), metric_dictionary=dictionary)
        self.assertEqual(len(result['summary']), 5)
        self.assertEqual(sorted(result['summary'].sum_value.tolist()), [10, 20, 30, 40, 50])

    def test_changed_source_tag_does_not_become_yoy(self):
        rows = [value(self.a, 2024, 100), value(self.a, 2025, 120, source_tag='us-gaap:SalesRevenueNet')]
        result = aggregate(panel([self.a], rows))
        current = result['summary'].loc[lambda d: d.bucket.eq('2025')].iloc[0]
        self.assertEqual(current.yoy_matched_company_count, 0)
        self.assertTrue(pd.isna(current.yoy_growth_pct))

    def test_overlapping_registrants_are_excluded_before_scope_grouping(self):
        old = company('XOM', 'SEC:OLD')
        new = company('XOM', 'SEC:NEW')
        rows = [value(old, 2025, 100), value(new, 2025, 100, source_tag='us-gaap:SalesRevenueNet')]
        result = aggregate(panel([old, new], rows))
        self.assertTrue(result['summary'].empty)
        self.assertIn('bucket', result['summary'])
        self.assertEqual(set(result['exclusions'].exclusion_reason), {'multiple_registrants_in_company_bucket'})
        self.assertTrue(result['coverage'].selected_cohort_company_count.eq(1).all())
        self.assertTrue(result['coverage'].included_company_count.eq(0).all())

    def test_legal_entity_change_cannot_implicitly_stitch_yoy(self):
        old, new = company('XOM', 'SEC:OLD'), company('XOM', 'SEC:NEW')
        result = aggregate(panel([old, new], [value(old, 2024, 100), value(new, 2025, 110)]))
        current = result['summary'].loc[lambda d: d.bucket.eq('2025')].iloc[0]
        self.assertEqual(current.yoy_matched_company_count, 0)

    def test_duplicate_alias_row_contributes_once(self):
        row = value(self.a, 2025, 10)
        alias = {**row, 'symbol': 'A.CLASS'}
        result = aggregate(panel([self.a], [row, alias]))
        self.assertEqual(result['summary'].iloc[0].sum_value, 10)
        self.assertEqual(len(result['contributions']), 1)
        self.assertEqual(result['exclusions'].iloc[0].exclusion_reason, 'duplicate_identical_value')

    def test_conflicting_value_id_quarantines_both(self):
        row = value(self.a, 2025, 10)
        result = aggregate(panel([self.a], [row, {**row, 'value': 20}]))
        self.assertTrue(result['summary'].empty)
        self.assertEqual(set(result['exclusions'].exclusion_reason), {'conflicting_duplicate_value_id'})

    def test_two_annual_ends_in_one_bucket_are_not_summed(self):
        rows = [value(self.a, 2025, 10), value(self.a, 2025, 20, value_id='changed-calendar',
                                            period_start='2024-07-01', period_end='2025-06-30')]
        result = aggregate(panel([self.a], rows))
        self.assertTrue(result['summary'].empty)
        self.assertEqual(set(result['exclusions'].exclusion_reason), {'multiple_periods_or_scopes_in_company_bucket'})

    def test_nonpositive_prior_revenue_is_excluded_only_from_yoy(self):
        rows = [value(self.a, 2024, 0), value(self.a, 2025, 10),
                value(self.b, 2024, 100), value(self.b, 2025, 110)]
        result = aggregate(panel([self.a, self.b], rows))
        current = result['summary'].loc[lambda d: d.bucket.eq('2025')].iloc[0]
        self.assertEqual(current.sum_value, 120)
        self.assertEqual(current.yoy_matched_company_count, 1)
        self.assertAlmostEqual(current.yoy_growth_pct, 10)
        self.assertIn('nonpositive_prior_company_revenue', set(result['exclusions'].exclusion_reason))

    def test_negative_cash_flow_prior_keeps_delta_but_no_percentage(self):
        rows = [value(self.a, 2024, -100, metric_id='operating_cash_flow'),
                value(self.a, 2025, 10, metric_id='operating_cash_flow')]
        result = aggregate(panel([self.a], rows, requested_metrics=['operating_cash_flow']))
        current = result['summary'].loc[lambda d: d.bucket.eq('2025')].iloc[0]
        self.assertEqual(current.yoy_matched_delta, 110)
        self.assertTrue(pd.isna(current.yoy_growth_pct))
        self.assertEqual(current.yoy_status, 'nonpositive_matched_aggregate_prior_sum')

    def test_contract_revenue_obeys_nonpositive_company_base_policy(self):
        rows = [value(self.a, 2024, 0, metric_id='revenue_from_contracts'),
                value(self.a, 2025, 10, metric_id='revenue_from_contracts')]
        result = aggregate(panel([self.a], rows, requested_metrics=['revenue_from_contracts']))
        current = result['summary'].loc[lambda d: d.bucket.eq('2025')].iloc[0]
        self.assertEqual(current.yoy_matched_company_count, 0)
        self.assertEqual(current.sum_value, 10)

    def test_fiscal_alignment_limits_and_date_spans_are_visible(self):
        rows = [value(self.a, 2024, 100), value(self.a, 2025, 110, period_start='2024-04-01', period_end='2025-03-31'),
                value(self.b, 2025, 50)]
        result = aggregate(panel([self.a, self.b], rows))
        current = result['summary'].loc[lambda d: d.bucket.eq('2025')].iloc[0]
        self.assertGreater(current.period_end_span_days, 200)
        self.assertEqual(current.yoy_matched_company_count, 0)
        self.assertIn('prior_fiscal_period_alignment_mismatch', set(result['exclusions'].exclusion_reason))

    def test_quarters_reject_cumulative_and_unverified_difference(self):
        rows = [value(self.a, 2025, 10, period_start='2025-01-01', period_end='2025-03-31',
                      period_kind='derived_quarter', derivation_compatibility='cross_filing_unverified'),
                value(self.b, 2025, 20, period_start='2025-01-01', period_end='2025-06-30', period_kind='cumulative_6m')]
        frame = panel([self.a, self.b], rows, financial_period='quarter')
        result = aggregate(frame)
        self.assertTrue(result['summary'].empty)
        self.assertEqual(set(result['exclusions'].exclusion_reason), {'wrong_period_kind', 'cross_filing_quarter_unverified'})
        permitted = aggregate(frame, allow_unverified_quarters=True)
        self.assertEqual(permitted['summary'].iloc[0].bucket, '2025Q1')
        self.assertFalse(permitted['summary'].iloc[0].accounting_basis_verified)

    def test_empty_values_keep_full_coverage_and_empty_schemas(self):
        result = aggregate(panel([self.a], []))
        self.assertTrue(result['summary'].empty)
        self.assertIn('bucket', result['summary'])
        self.assertEqual(len(result['coverage']), 2)
        self.assertTrue(result['coverage'].included_company_count.eq(0).all())

    def test_missing_classification_and_nonfinite_rows_are_excluded(self):
        missing = {**self.a, 'sector': None}
        result = aggregate(panel([missing, self.b], [value(missing, 2025, 10), value(self.b, 2025, float('inf'))]))
        self.assertTrue(result['summary'].empty)
        self.assertEqual(set(result['exclusions'].exclusion_reason), {'missing_classification_or_profile', 'nonfinite_value'})

    def test_changed_identity_metadata_and_conflict_are_not_aggregated(self):
        wrong = value(self.a, 2025, 10, company_id='US:WRONG')
        conflicted = value(self.b, 2025, 20, row_status='tag_conflict', tag_conflict=True)
        result = aggregate(panel([self.a, self.b], [wrong, conflicted]))
        self.assertTrue(result['summary'].empty)
        self.assertEqual(set(result['exclusions'].exclusion_reason), {'cohort_metadata_mismatch', 'conflicted_or_unusable_row'})

    def test_unsupported_metrics_period_and_missing_cohort_fail(self):
        with self.assertRaises(ValueError):
            aggregate(panel([self.a], [], requested_metrics=['eps_basic']))
        with self.assertRaises(ValueError):
            aggregate(panel([self.a], [], financial_period='cumulative'))
        with self.assertRaises(ValueError):
            aggregate(pd.DataFrame(), period='annual')

    def test_explicit_dictionary_additive_monetary_fields_supported(self):
        rows = [value(self.a, 2025, 10, metric_id='research_development'),
                value(self.b, 2025, 20, metric_id='research_development')]
        result = aggregate(panel([self.a, self.b], rows, requested_metrics=['research_development']))
        self.assertEqual(result['summary'].iloc[0].sum_value, 30)
        self.assertEqual(len(result['query']['metric_definition_sha256']), 64)
        for metric in ['weighted_average_shares_basic', 'total_assets', 'eps_diluted']:
            with self.subTest(metric=metric), self.assertRaises(ValueError):
                aggregate(panel([self.a], [], requested_metrics=[metric]))


if __name__ == '__main__':
    unittest.main()
