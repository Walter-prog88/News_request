"""Financial QA failure modes: scope, vintages, signed values and provenance."""
import json
import unittest

import pandas as pd

from usmd.financial_checks import CHECK_COLUMNS, check_freshness, check_tieouts


def row(metric, value, **extra):
    result = {"issuer_id": "SEC:0000320193", "symbol": "AAPL", "accn": "filing-a",
              "unit": "USD", "period_start": "2025-01-01", "period_end": "2025-03-31",
              "period_kind": "direct_quarter", "metric_id": metric, "value": value,
              "value_id": "value_" + metric, "normalization_method": "reported",
              "tag_conflict": False, "row_status": "ok",
              "input_fact_ids": json.dumps(["fact_" + metric])}
    result.update(extra)
    return result


def balance(assets=100, liabilities=70, equity=30):
    return [row(m, v, period_kind="instant", period_start=None)
            for m, v in [("total_assets", assets), ("total_liabilities", liabilities),
                         ("equity_including_noncontrolling", equity)]]


def gross(revenue=100, cost=70, profit=30):
    return [row(m, v) for m, v in [("revenue", revenue), ("cost_of_revenue", cost), ("gross_profit", profit)]]


def checks(rows, name):
    result = check_tieouts(pd.DataFrame(rows))
    return result.loc[result.check.eq(name)].reset_index(drop=True)


class FinancialTieoutTests(unittest.TestCase):
    def test_consolidated_balance_and_provenance(self):
        result = checks(balance(), "balance_sheet_equation").iloc[0]
        self.assertEqual(result.status, "pass")
        self.assertEqual((result.observed, result.expected, result.difference, result.tolerance), (100, 100, 0, 1))
        self.assertEqual(len(json.loads(result.input_value_ids)), 3)
        self.assertIn("fact_equity_including_noncontrolling", json.loads(result.input_fact_ids))

    def test_parent_equity_is_not_treated_as_consolidated_even_if_it_balances(self):
        rows = balance()
        rows[-1] = row("equity", 30, period_kind="instant", period_start=None)
        result = checks(rows, "balance_sheet_equation").iloc[0]
        self.assertEqual(result.status, "not_tested")
        self.assertIn("equity_including_noncontrolling:missing", result.detail)
        self.assertTrue(pd.isna(result.expected))

    def test_negative_equity_and_cost_signs_are_preserved(self):
        self.assertEqual(checks(balance(100, 120, -20), "balance_sheet_equation").iloc[0].status, "pass")
        self.assertEqual(checks(gross(100, -20, 120), "gross_profit_equation").iloc[0].status, "pass")

    def test_anomaly_is_review_and_input_is_unmodified(self):
        frame = pd.DataFrame(balance(100, 70, 20))
        before = frame.copy(deep=True)
        result = check_tieouts(frame).iloc[0]
        self.assertEqual(result.status, "review")
        self.assertEqual(result.difference, 10)
        pd.testing.assert_frame_equal(frame, before)

    def test_gross_profit_rounding_tolerance_uses_total_revenue(self):
        result = checks(gross(1_000_000, 600_000, 400_050), "gross_profit_equation").iloc[0]
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.tolerance, 100)
        self.assertEqual(result.difference, 50)
        self.assertEqual(checks(gross(1_000_000, 600_000, 400_101), "gross_profit_equation").iloc[0].status, "review")

    def test_missing_value_is_not_zero(self):
        result = checks(gross()[:2], "gross_profit_equation").iloc[0]
        self.assertEqual(result.status, "not_tested")
        self.assertIn("gross_profit:missing", result.detail)

    def test_other_accession_cannot_complete_check(self):
        rows = gross()
        rows[-1]["accn"] = "amended-filing-b"
        result = checks(rows, "gross_profit_equation")
        self.assertEqual(len(result), 2)
        self.assertTrue(result.status.eq("not_tested").all())

    def test_other_issuer_cannot_complete_check(self):
        rows = gross()
        rows[-1]["issuer_id"] = "SEC:0000789019"
        self.assertTrue(checks(rows, "gross_profit_equation").status.eq("not_tested").all())

    def test_actual_start_and_period_kind_prevent_cumulative_quarter_mixing(self):
        for change in [{"period_start": "2024-10-01"}, {"period_kind": "cumulative_6m"}]:
            with self.subTest(change=change):
                rows = gross()
                rows[-1].update(change)
                self.assertTrue(checks(rows, "gross_profit_equation").status.eq("not_tested").all())

    def test_derived_values_do_not_count_as_reported(self):
        rows = gross()
        for r in rows:
            r.update(normalization_method="cumulative_difference", period_kind="derived_quarter")
        self.assertEqual(checks(rows, "gross_profit_equation").iloc[0].status, "not_tested")

    def test_conflicting_tag_and_ambiguous_records_never_pass(self):
        rows = gross()
        rows[-1].update(tag_conflict=True, row_status="tag_conflict")
        result = checks(rows, "gross_profit_equation").iloc[0]
        self.assertEqual(result.status, "not_tested")
        self.assertIn("fact_gross_profit", json.loads(result.input_fact_ids))
        rows = gross() + [row("gross_profit", 30, value_id="different_identity")]
        self.assertEqual(checks(rows, "gross_profit_equation").iloc[0].status, "not_tested")

    def test_exact_duplicate_rows_do_not_create_a_second_economic_input(self):
        rows = gross()
        rows.append(dict(rows[-1]))
        self.assertEqual(checks(rows, "gross_profit_equation").iloc[0].status, "pass")

    def test_currency_and_units_are_never_mixed(self):
        rows = gross()
        rows[-1]["unit"] = "shares"
        self.assertTrue(checks(rows, "gross_profit_equation").status.eq("not_tested").all())
        rows = [dict(r, unit="EUR") for r in gross()]
        self.assertEqual(checks(rows, "gross_profit_equation").iloc[0].status, "not_tested")

    def test_invalid_values_or_provenance_do_not_pass(self):
        for change in [{"value": float("nan")}, {"value": float("inf")}, {"input_fact_ids": "[]"},
                       {"tag_conflict": None}, {"accn": None}]:
            with self.subTest(change=change):
                rows = gross()
                rows[-1].update(change)
                self.assertTrue(checks(rows, "gross_profit_equation").status.eq("not_tested").all())

    def test_cash_cannot_pass_without_fx_and_proven_matching_cash_scope(self):
        rows = [row(m, v) for m, v in [("cash_change", 20), ("operating_cash_flow", 100),
                                     ("investing_cash_flow", -50), ("financing_cash_flow", -30)]]
        result = checks(rows, "cash_flow_equation").iloc[0]
        self.assertEqual(result.status, "not_tested")
        self.assertIn("missing FX is not zero", result.detail)
        self.assertTrue(pd.isna(result.expected))

    def test_empty_has_stable_output_schema_and_bad_schema_is_rejected(self):
        self.assertEqual(list(check_tieouts(pd.DataFrame()).columns), CHECK_COLUMNS)
        with self.assertRaisesRegex(ValueError, "Missing financial tie-out columns"):
            check_tieouts(pd.DataFrame([{"value": 1}]))


class FinancialFreshnessTests(unittest.TestCase):
    issuer = {"issuer_id": "SEC:0000320193", "symbol": "AAPL"}

    def filing(self, **extra):
        result = {**self.issuer, "form": "10-Q", "report_date": "2025-03-31",
                  "filed": "2025-04-20", "accn": "filing-a"}
        result.update(extra)
        return result

    def run_check(self, values, filings, issuers=None):
        return check_freshness(pd.DataFrame(values), pd.DataFrame(filings), issuers or [self.issuer])

    def test_later_share_count_does_not_hide_stale_financials(self):
        values = gross() + [row("shares_outstanding", 1000, unit="shares", period_end="2025-07-15")]
        result = self.run_check(values, [self.filing(report_date="2025-06-30", filed="2025-07-29", accn="filing-b")]).iloc[0]
        self.assertEqual(result.status, "review")
        self.assertEqual(result.latest_normalized_period_end, "2025-03-31")
        self.assertEqual(result.latest_report_date, "2025-06-30")
        self.assertEqual(result.observed, 91)
        self.assertFalse(result.latest_filing_has_values)
        self.assertIn("not an acquisition failure", result.detail)

    def test_matching_accession_and_report_period_pass(self):
        result = self.run_check(gross(), [self.filing()]).iloc[0]
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.observed, 0)
        self.assertTrue(result.latest_filing_has_report_period)
        self.assertEqual(result.latest_filing_accn, "filing-a")

    def test_unprocessed_amendment_does_not_pass_on_old_same_period(self):
        result = self.run_check(gross(), [self.filing(), self.filing(form="10-Q/A", accn="amended", filed="2025-05-01")]).iloc[0]
        self.assertEqual(result.status, "review")
        self.assertEqual(result.observed, 0)
        self.assertEqual(result.latest_filing_accn, "amended")
        self.assertFalse(result.latest_filing_has_values)

    def test_comparative_only_values_from_latest_accession_are_insufficient(self):
        values = gross() + [row("revenue", 60, accn="amended", period_end="2024-03-31")]
        result = self.run_check(values, [self.filing(accn="amended")]).iloc[0]
        self.assertEqual(result.status, "review")
        self.assertTrue(result.latest_filing_has_values)
        self.assertFalse(result.latest_filing_has_report_period)

    def test_missing_either_side_is_not_tested(self):
        for values, filings in [([], [self.filing()]), (gross(), []), ([], [])]:
            with self.subTest(values=len(values), filings=len(filings)):
                self.assertEqual(self.run_check(values, filings).iloc[0].status, "not_tested")

    def test_same_symbol_successor_cannot_fill_predecessor_gap(self):
        old = {"issuer_id": "SEC:0000034088", "symbol": "XOM"}
        new = {"issuer_id": "SEC:0002115436", "symbol": "XOM"}
        values = [row("revenue", 100, **old), row("revenue", 200, **new, period_end="2025-06-30")]
        filings = [self.filing(**r, report_date="2025-06-30") for r in [old, new]]
        result = self.run_check(values, filings, [old, new])
        self.assertEqual(result.status.tolist(), ["review", "pass"])

    def test_latest_report_period_takes_precedence_over_old_amendment_filed_later(self):
        filings = [self.filing(), self.filing(form="10-K/A", report_date="2024-12-31", filed="2025-05-01", accn="old-amendment"),
                   self.filing(form="8-K", report_date="2025-06-30", filed="2025-07-01", accn="8k")]
        self.assertEqual(self.run_check(gross(), filings).iloc[0].status, "pass")

    def test_conflicted_or_invalid_rows_do_not_advance_freshness(self):
        values = gross() + [row("revenue", 200, period_end="2025-06-30", row_status="tag_conflict", tag_conflict=True)]
        result = self.run_check(values, [self.filing(report_date="2025-06-30")]).iloc[0]
        self.assertEqual(result.status, "review")
        self.assertEqual(result.latest_normalized_period_end, "2025-03-31")


if __name__ == "__main__":
    unittest.main()
