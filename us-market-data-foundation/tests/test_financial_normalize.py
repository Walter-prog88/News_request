"""Financial duration, version and source-scope tests; no network or lake writes."""
import json
import unittest

from usmd.financial_normalize import normalize_company, load_metrics, FACT_COLUMNS, VALUE_COLUMNS


ISSUER = {"cik": "123", "issuer_id": "SEC:0000000123", "symbol": "TEST", "companyfacts_source_id": "snapshot:123:one"}
RETRIEVED = "2026-09-19T01:00:00+00:00"


def fact(value, start="2024-01-01", end="2024-03-31", filed="2024-05-01", accn="a1", form="10-Q", **kwargs):
    row = {"val": value, "end": end, "filed": filed, "accn": accn, "form": form, "fy": 2024, "fp": "Q1"}
    if start is not None:
        row["start"] = start
    row.update(kwargs)
    return row


def payload(tags):
    return {"cik": 123, "entityName": "Synthetic Test", "facts": {"us-gaap": tags}}


def concept(rows, unit="USD"):
    return {"units": {unit: rows}}


def normalize(tags, submissions=None):
    return normalize_company(payload(tags), submissions or [], ISSUER, RETRIEVED)


class FinancialNormalizationTests(unittest.TestCase):
    def test_continuing_cashflow_requires_exact_original_statement_evidence(self):
        tag = 'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations'
        original = fact(32231000000, '2013-07-01', '2014-06-30', '2014-07-31',
                        '0001193125-14-289961', form='10-K')
        cases = [original,
                 {**original, 'accn': 'unreviewed-amendment'},
                 {**original, 'val': 32502000000},
                 {**original, 'start': '2014-07-01', 'end': '2015-06-30'}]
        issuer = {'cik':'0000789019','issuer_id':'SEC:0000789019','symbol':'MSFT'}
        body = {'cik':789019,'facts':{'us-gaap':{tag:concept(cases)}}}
        result = normalize_company(body, [], issuer, RETRIEVED, since='2009-01-01')
        values = result['financial_values'].query("metric_id == 'operating_cash_flow'")
        self.assertEqual(values.value.tolist(), [32231000000])
        self.assertEqual(values.source_tag.tolist(), ['us-gaap:' + tag])
        self.assertEqual(len(result['financial_facts']), 4)
        body['cik'] = 123
        unrelated = normalize_company(body, [], ISSUER, RETRIEVED, since='2009-01-01')
        self.assertTrue(unrelated['financial_values'].query("metric_id == 'operating_cash_flow'").empty)

    def test_filing_vintages_and_comparison_fact_dates_are_retained(self):
        a = fact(100, filed="2024-05-01", accn="original", fy=2024, fp="Q1")
        b = fact(110, filed="2025-05-01", accn="comparison", fy=2025, fp="Q1")
        tables = normalize({"RevenueFromContractWithCustomerExcludingAssessedTax": concept([a, b])})
        self.assertEqual(len(tables["financial_facts"]), 2)
        values = tables["financial_values"].query("metric_id == 'revenue'")
        self.assertEqual(values.value.tolist(), [100, 110])
        self.assertEqual(values.period_end.tolist(), ["2024-03-31"] * 2)
        self.assertEqual(values.fiscal_year.tolist(), [2024, 2025])
        self.assertEqual(values.period_kind.tolist(), ["direct_quarter"] * 2)
        self.assertEqual(values.source_id.tolist(), ["snapshot:123:one"] * 2)

    def test_cumulative_durations_not_fp_or_frame_determine_period(self):
        rows = [fact(300, end="2024-06-30", fp="Q2", frame="CY2024Q2"),
                fact(900, end="2024-09-30", accn="a2", fp="Q3", frame="CY2024Q3"),
                fact(1500, end="2024-12-31", accn="a3", form="10-K", fp="FY")]
        values = normalize({"NetCashProvidedByUsedInOperatingActivities": concept(rows)})["financial_values"]
        reported = values[values.normalization_method == "reported"]
        self.assertEqual(reported.period_kind.tolist(), ["cumulative_6m", "cumulative_9m", "annual"])
        self.assertFalse((reported.period_kind == "direct_quarter").any())

    def test_non_calendar_fiscal_cashflow_quarters_and_q4(self):
        rows = [
            fact(100, "2023-10-01", "2023-12-31", "2024-02-01", "q1", fy=2024),
            fact(250, "2023-10-01", "2024-03-31", "2024-05-01", "q2", fy=2024, fp="Q2"),
            fact(400, "2023-10-01", "2024-06-30", "2024-08-01", "q3", fy=2024, fp="Q3"),
            fact(700, "2023-10-01", "2024-09-30", "2024-11-01", "annual", form="10-K", fy=2024, fp="FY"),
        ]
        values = normalize({"NetCashProvidedByUsedInOperatingActivities": concept(rows)})["financial_values"]
        derived = values[values.period_kind == "derived_quarter"]
        self.assertEqual(derived.value.tolist(), [150, 150, 300])
        self.assertEqual(derived.period_start.tolist(), ["2024-01-01", "2024-04-01", "2024-07-01"])
        self.assertTrue(derived.derivation_compatibility.eq("cross_filing_unverified").all())
        self.assertEqual(json.loads(derived.iloc[-1].input_accessions), ["annual", "q3"])

    def test_input_restatement_is_a_new_derived_vintage_not_lookahead(self):
        rows = [
            fact(100, filed="2024-05-01", accn="q1"),
            fact(300, end="2024-06-30", filed="2024-08-01", accn="q2", fp="Q2"),
            fact(110, filed="2024-09-15", accn="q1_amended", form="10-Q/A"),
        ]
        values = normalize({"NetCashProvidedByUsedInOperatingActivities": concept(rows)})["financial_values"]
        derived = values[values.period_kind == "derived_quarter"]
        self.assertEqual(derived.value.tolist(), [200, 190])
        self.assertEqual(derived.available_at.tolist(), ["2024-08-02T04:00:00+00:00", "2024-09-16T04:00:00+00:00"])
        self.assertEqual(json.loads(derived.iloc[0].input_accessions), ["q1", "q2"])
        self.assertEqual(json.loads(derived.iloc[1].input_accessions), ["q1_amended", "q2"])
        self.assertEqual(derived.input_fact_ids.nunique(), 2)

    def test_missing_previous_period_is_not_zero_and_different_fiscal_start_cannot_join(self):
        rows = [fact(100, start="2023-12-31", end="2024-03-30", accn="old_calendar"),
                fact(300, end="2024-06-30", filed="2024-08-01", accn="q2", fp="Q2")]
        values = normalize({"NetCashProvidedByUsedInOperatingActivities": concept(rows)})["financial_values"]
        self.assertEqual(len(values), 2)
        self.assertNotIn("derived_quarter", set(values.period_kind))

    def test_eps_and_weighted_average_shares_are_not_subtracted(self):
        rows = [fact(2), fact(5, end="2024-06-30", accn="q2", filed="2024-08-01")]
        values = normalize({"EarningsPerShareDiluted": concept(rows, "USD/shares"),
                            "WeightedAverageNumberOfDilutedSharesOutstanding": concept(rows, "shares")})["financial_values"]
        self.assertEqual(len(values), 4)
        self.assertNotIn("derived_quarter", set(values.period_kind))

    def test_balance_sheet_is_instant_and_units_are_not_relabelled(self):
        tables = normalize({"Assets": {"units": {
            "USD": [fact(1000, start=None)], "EUR": [fact(900, start=None)],
            "shares": [fact(123, start=None)],
        }}})
        self.assertEqual(len(tables["financial_facts"]), 3)
        values = tables["financial_values"]
        self.assertEqual(len(values), 1)
        self.assertEqual(values.iloc[0].period_kind, "instant")
        self.assertEqual(values.iloc[0].unit, "USD")
        self.assertEqual(values.iloc[0].value, 1000)

    def test_wrong_duration_shape_does_not_become_balance_sheet_value(self):
        tables = normalize({"Assets": concept([fact(1000)])})
        self.assertEqual(len(tables["financial_facts"]), 1)
        self.assertTrue(tables["financial_values"].empty)

    def test_semantically_different_revenue_scope_is_separate(self):
        values = normalize({
            "RevenueFromContractWithCustomerExcludingAssessedTax": concept([fact(100)]),
            "RevenueFromContractWithCustomerIncludingAssessedTax": concept([fact(110)]),
        })["financial_values"]
        self.assertEqual(set(values.metric_id), {"revenue", "revenue_from_contracts", "revenue_including_assessed_tax"})
        self.assertFalse(values.tag_conflict.any())

    def test_tag_precedence_conflict_propagates_to_derived_status(self):
        tables = normalize({
            "RevenueFromContractWithCustomerExcludingAssessedTax": concept([fact(100), fact(300, end="2024-06-30", filed="2024-08-01", accn="q2")]),
            "Revenues": concept([fact(120)]),
        })
        values = tables["financial_values"].query("metric_id == 'revenue'")
        row = values[values.period_end == "2024-03-31"].iloc[0]
        self.assertEqual(row.value, 100)
        self.assertTrue(row.tag_conflict)
        self.assertEqual(row.row_status, "tag_conflict")
        self.assertEqual(len(json.loads(row.conflict_fact_ids)), 2)
        derived = values.query("period_kind == 'derived_quarter'")
        self.assertEqual(len(derived), 1)
        self.assertTrue(derived.tag_conflict.all())
        self.assertTrue(derived.row_status.eq("tag_conflict").all())
        self.assertEqual(len(tables["financial_facts"]), 3)

    def test_identical_aliases_are_collapsed_only_within_same_accession(self):
        tables = normalize({"RevenueFromContractWithCustomerExcludingAssessedTax": concept([fact(100)]),
                            "Revenues": concept([fact(100), fact(100, accn="later", filed="2025-05-01")])})
        self.assertEqual(len(tables["financial_facts"]), 3)
        self.assertEqual(len(tables["financial_values"].query("metric_id == 'revenue'")), 2)
        self.assertFalse(tables["financial_values"].tag_conflict.any())

    def test_same_tag_same_period_ambiguous_values_are_not_arbitrarily_normal(self):
        tables = normalize({"Assets": concept([fact(100, start=None), fact(200, start=None)])})
        self.assertEqual(len(tables["financial_facts"]), 2)
        self.assertEqual(tables["financial_values"].iloc[0].row_status, "ambiguous_same_tag_value")

    def test_conservative_availability_uses_new_york_midnight_with_dst(self):
        rows = [fact(1, filed="2024-05-01", accn="summer"), fact(2, filed="2024-12-01", accn="winter")]
        submissions = [{"filings": {"recent": {
            "accessionNumber": ["summer", "winter"],
            "acceptanceDateTime": ["2024-05-01T23:59:00-04:00", "2024-12-01T20:00:00Z"],
        }}}]
        facts = normalize({"Revenues": concept(rows)}, submissions)["financial_facts"]
        self.assertEqual(facts.available_at.tolist(), ["2024-05-02T04:00:00+00:00", "2024-12-02T05:00:00+00:00"])
        self.assertEqual(facts.acceptance_at.tolist(), ["2024-05-02T03:59:00+00:00", "2024-12-01T20:00:00+00:00"])
        self.assertTrue(facts.availability_basis.eq("estimated_filed_next_day").all())
        self.assertFalse(facts.pit_eligible.any())

    def test_naive_acceptance_is_retained_without_invented_timezone(self):
        submissions = [{"accessionNumber": "a1", "acceptanceDateTime": "2024-05-01T16:30:00"}]
        row = normalize({"Revenues": concept([fact(100)])}, submissions)["financial_facts"].iloc[0]
        self.assertEqual(row.acceptance_raw, "2024-05-01T16:30:00")
        self.assertIsNone(row.acceptance_at)
        self.assertFalse(row.pit_eligible)

    def test_unsupported_custom_tags_and_nonperiodic_forms_remain_raw(self):
        body = payload({"Revenues": concept([fact(100, form="8-K")])})
        body["facts"]["testco"] = {"SpecialRevenue": concept([fact(99)])}
        tables = normalize_company(body, [], ISSUER, RETRIEVED)
        self.assertEqual(len(tables["financial_facts"]), 2)
        self.assertTrue(tables["financial_values"].empty)

    def test_missing_invalid_and_zero_are_distinct(self):
        rows = [fact(None, accn="missing"), fact(float("inf"), accn="infinite"),
                fact(0, accn="zero"), fact(1, filed=None, accn="missing_date")]
        tables = normalize({"Revenues": concept(rows)})
        self.assertEqual(len(tables["financial_facts"]), 4)
        values = tables["financial_values"]
        self.assertEqual(len(values), 1)
        self.assertEqual(values.iloc[0].value, 0)
        self.assertEqual(values.iloc[0].accn, "zero")

    def test_53_week_year_is_annual_and_stub_is_not_quarter(self):
        rows = [fact(100, start="2023-01-01", end="2024-01-06", form="10-K", accn="53weeks"),
                fact(5, start="2024-01-07", end="2024-02-03", accn="stub")]
        values = normalize({"Revenues": concept(rows)})["financial_values"]
        self.assertEqual(values.period_kind.tolist(), ["annual", "reported_other"])
        self.assertEqual(values.duration_days.tolist(), [371, 28])

    def test_quarterly_filing_rolling_twelve_months_is_not_a_fiscal_annual(self):
        rows = [fact(100, start="2025-07-01", end="2026-06-30", filed="2026-07-31", accn="q2", form="10-Q", fp="Q2")]
        values = normalize({"PaymentsToAcquireProductiveAssets": concept(rows)})["financial_values"]
        self.assertEqual(values.iloc[0].period_kind, "reported_12m")
        self.assertEqual(values.iloc[0].metric_id, "productive_asset_purchases")
        self.assertNotIn("capital_expenditures", set(values.metric_id))
        self.assertNotIn("annual", set(values.period_kind))

    def test_retailer_sixteen_week_fourth_quarter_is_derived(self):
        rows = [fact(750, "2024-09-02", "2025-05-11", "2025-06-01", "q3", fp="Q3"),
                fact(1200, "2024-09-02", "2025-08-31", "2025-10-01", "year", form="10-K", fp="FY")]
        values = normalize({"Revenues": concept(rows)})["financial_values"]
        q4 = values.query("period_kind == 'derived_quarter'").iloc[0]
        self.assertEqual(q4.duration_days, 112)
        self.assertEqual(q4.value, 450)
        self.assertEqual(q4.period_start, "2025-05-12")

    def test_insurance_profile_keeps_total_and_contract_revenue_separate(self):
        body = payload({"Revenues": concept([fact(195)]),
                        "RevenueFromContractWithCustomerExcludingAssessedTax": concept([fact(133)])})
        values = normalize_company(body, [], {**ISSUER, "profile": "insurance"}, RETRIEVED)["financial_values"]
        total = values.query("metric_id == 'revenue'").iloc[0]
        contract = values.query("metric_id == 'revenue_from_contracts'").iloc[0]
        self.assertEqual(total.value, 195)
        self.assertEqual(contract.value, 133)
        self.assertFalse(total.tag_conflict)

    def test_explicit_retailer_cik_uses_total_not_contract_only_scope(self):
        body = payload({"Revenues": concept([fact(365)]),
                        "RevenueFromContractWithCustomerExcludingAssessedTax": concept([fact(361)])})
        body["cik"] = 104169
        issuer = {"cik": 104169, "symbol": "WMT", "profile": "general"}
        values = normalize_company(body, [], issuer, RETRIEVED)["financial_values"]
        total = values.query("metric_id == 'revenue'").iloc[0]
        self.assertEqual(total.value, 365)
        self.assertEqual(total.source_tag, "us-gaap:Revenues")
        self.assertFalse(total.tag_conflict)

    def test_bank_net_interest_after_revenue_is_not_spliced_into_total(self):
        body = payload({"Revenues": concept([fact(100, end="2023-03-31", start="2023-01-01")]),
                        "RevenuesNetOfInterestExpense": concept([fact(120)])})
        values = normalize_company(body, [], {**ISSUER, "profile": "bank"}, RETRIEVED)["financial_values"]
        self.assertEqual(values.query("metric_id == 'revenue'").period_end.tolist(), ["2023-03-31"])
        self.assertEqual(values.query("metric_id == 'bank_revenue_net_interest_expense'").period_end.tolist(), ["2024-03-31"])

    def test_conflicted_cumulative_amendment_invalidates_clean_derived_quarter(self):
        from usmd.financial_query import select_values
        rows = [fact(100, filed="2024-05-01", accn="q1"),
                fact(300, end="2024-06-30", filed="2024-08-01", accn="q2"),
                fact(310, end="2024-06-30", filed="2024-09-01", accn="q2_amended"),
                fact(320, end="2024-06-30", filed="2024-09-01", accn="q2_amended")]
        values = normalize({"NetCashProvidedByUsedInOperatingActivities": concept(rows)})["financial_values"]
        before = select_values(values, period="quarter", as_of="2024-08-15")
        self.assertEqual(before.query("period_end == '2024-06-30'").iloc[0].value, 200)
        after = select_values(values, period="quarter", as_of="2024-09-02")
        self.assertTrue(after.query("period_end == '2024-06-30'").empty)
        flagged = values.query("period_kind == 'derived_quarter' and tag_conflict == True")
        self.assertEqual(len(flagged), 1)
        self.assertEqual(len(json.loads(flagged.iloc[0].input_fact_ids)), 3)

    def test_conflicted_prior_cumulative_input_propagates_then_clean_revision_recovers(self):
        from usmd.financial_query import select_values
        rows = [fact(100, filed="2024-05-01", accn="q1"),
                fact(300, end="2024-06-30", filed="2024-08-01", accn="q2"),
                fact(110, filed="2024-09-01", accn="q1_amended"),
                fact(120, filed="2024-09-01", accn="q1_amended"),
                fact(105, filed="2024-10-01", accn="q1_corrected")]
        values = normalize({"NetCashProvidedByUsedInOperatingActivities": concept(rows)})["financial_values"]
        self.assertTrue(select_values(values, period="quarter", as_of="2024-09-02").query("period_end == '2024-06-30'").empty)
        repaired = select_values(values, period="quarter", as_of="2024-10-02").query("period_end == '2024-06-30'")
        self.assertEqual(repaired.iloc[0].value, 195)
        self.assertEqual(json.loads(repaired.iloc[0].input_accessions), ["q1_corrected", "q2"])

    def test_cik_mismatch_rejected_and_empty_tables_keep_schema(self):
        with self.assertRaises(ValueError):
            normalize_company({"cik": 999, "facts": {}}, [], ISSUER, RETRIEVED)
        tables = normalize({})
        self.assertEqual(tables["financial_facts"].columns.tolist(), FACT_COLUMNS)
        self.assertEqual(tables["financial_values"].columns.tolist(), VALUE_COLUMNS)

    def test_metric_dictionary_has_unique_ids_and_nonadditive_share_metrics(self):
        metrics = load_metrics()
        self.assertGreaterEqual(len(metrics), 50)
        self.assertEqual(len({row["metric_id"] for row in metrics}), len(metrics))
        for row in metrics:
            if row["period_type"] == "instant" or row["units"] != ["USD"]:
                self.assertFalse(row["additive"])


if __name__ == "__main__":
    unittest.main()
