"""Strict reported-input factor contracts, missing coverage and vintage isolation."""
import copy
import json
import unittest

import pandas as pd

from usmd.financial_factors import derive, _CONFIG, _METRICS

TAGS = {'revenue':'Revenues', 'operating_income':'OperatingIncomeLoss', 'net_income':'NetIncomeLoss',
        'operating_cash_flow':'NetCashProvidedByUsedInOperatingActivities',
        'research_development':'ResearchAndDevelopmentExpense', 'stock_based_compensation':'ShareBasedCompensation',
        'capital_expenditures':'PaymentsToAcquirePropertyPlantAndEquipment',
        'productive_asset_purchases':'PaymentsToAcquireProductiveAssets'}


def issuer(symbol='A', **changes):
    return {'issuer_id':'SEC:'+symbol, 'company_id':'US:'+symbol, 'symbol':symbol, 'profile':'general',
            'sector':'Information Technology', 'registration_role':'current', **changes}


def value(metric, amount, company=None, **changes):
    company = company or issuer()
    return {**company, 'metric_id':metric, 'value':amount, 'value_id':company['symbol']+'-'+metric,
        'unit':'USD', 'source_id':'source:A', 'source_tag':'us-gaap:'+TAGS[metric], 'accn':'same-filing',
        'period_start':'2025-01-01','period_end':'2025-12-31','period_kind':'annual',
        'row_status':'ok','tag_conflict':False,'normalization_method':'reported','derivation_compatibility':'not_applicable',
        'input_fact_ids':json.dumps([company['symbol']+'-fact-'+metric]),'input_accessions':'["same-filing"]',
        'available_at':'2026-02-02T05:00:00Z', **changes}


def panel(rows, companies=None, **attrs):
    frame=pd.DataFrame(rows)
    frame.attrs={'cohort_records':companies or [issuer()], 'financial_period':'annual',
                 'requested_start':'2025-01-01', 'requested_end':'2025-12-31', **attrs}
    return frame


class FactorTests(unittest.TestCase):
    def inputs(self):
        return [value(metric, amount) for metric,amount in [('revenue',100),('operating_income',20),
            ('net_income',-5),('operating_cash_flow',-10),('research_development',8),
            ('stock_based_compensation',3),('capital_expenditures',12)]]

    def test_seven_defined_factors_preserve_negative_profit_and_cashflow(self):
        result=derive(panel(self.inputs()))
        output=result['values'].set_index('factor_id')
        self.assertEqual(len(output),7)
        self.assertEqual(output.loc['operating_margin_pct','value'],20)
        self.assertEqual(output.loc['parent_net_margin_pct','value'],-5)
        self.assertEqual(output.loc['operating_cash_flow_margin_pct','value'],-10)
        self.assertEqual(output.loc['free_cash_flow_ppe_proxy','value'],-22)
        self.assertEqual(output.loc['free_cash_flow_ppe_proxy','unit'],'USD')
        self.assertEqual(output.loc['ppe_capex_to_revenue_pct','unit'],'percent')
        self.assertFalse(output.accounting_scope_verified.any())
        self.assertFalse(output.pit_eligible.any())
        self.assertEqual(len(result['metadata']['factor_definition_sha256']),64)
        self.assertTrue(result['exclusions'].empty)

    def test_all_missing_company_and_missing_year_keep_full_coverage(self):
        companies=[issuer(),issuer('B')]
        result=derive(panel([value('revenue',100),value('operating_income',20)], companies,
                            requested_start='2024-01-01'), factors='operating_margin_pct')
        self.assertEqual(len(result['coverage']),4)
        self.assertEqual(len(result['values']),1)
        b=result['coverage'].loc[lambda d:d.symbol.eq('B')]
        self.assertTrue(b.reason.eq('no_candidate_period').all())
        self.assertTrue(b.period_end.isna().all())
        self.assertTrue(result['values'].value.eq(20).all())

    def test_missing_operand_never_becomes_zero(self):
        result=derive(panel([value('revenue',100)]),factors='research_development_to_revenue_pct')
        self.assertTrue(result['values'].empty)
        self.assertEqual(json.loads(result['coverage'].iloc[0].missing_metrics),['research_development'])

    def test_different_period_starts_do_not_pair(self):
        rows=[value('revenue',100),value('operating_income',20,period_start='2025-01-02')]
        result=derive(panel(rows),factors='operating_margin_pct')
        self.assertTrue(result['values'].empty)
        self.assertEqual(len(result['coverage']),2)
        self.assertTrue(result['coverage'].reason.eq('missing_exact_period_inputs').all())

    def test_different_latest_filings_are_not_backfilled(self):
        rows=[value('revenue',100),value('operating_income',20,accn='new-filing',input_accessions='["new-filing"]')]
        result=derive(panel(rows),factors='operating_margin_pct')
        self.assertTrue(result['values'].empty)
        self.assertEqual(result['coverage'].iloc[0].reason,'input_accn_mismatch')
        # Supplying an older compatible row as well still cannot silently choose it.
        rows.append(value('operating_income',10,value_id='old-op'))
        result=derive(panel(rows),factors='operating_margin_pct')
        self.assertTrue(result['values'].empty)
        self.assertEqual(result['coverage'].iloc[0].reason,'ambiguous_latest_input_values')

    def test_conflict_no_fallback_to_earlier_clean_input(self):
        rows=[value('revenue',100),value('operating_income',20,tag_conflict=True,row_status='tag_conflict')]
        result=derive(panel(rows),factors='operating_margin_pct')
        self.assertEqual(result['coverage'].iloc[0].reason,'latest_input_conflicted_or_unusable')
        self.assertTrue(result['values'].empty)

    def test_no_nonpositive_revenue_ratio(self):
        for denominator in (0,-1):
            with self.subTest(denominator=denominator):
                result=derive(panel([value('revenue',denominator),value('net_income',10)]),factors='parent_net_margin_pct')
                self.assertTrue(result['values'].empty)
                self.assertEqual(result['coverage'].iloc[0].reason,'nonpositive_reported_revenue_denominator')

    def test_negative_investment_is_reviewed_without_abs_or_cashflow_inflation(self):
        rows=[value('revenue',100),value('operating_cash_flow',30),value('capital_expenditures',-10)]
        result=derive(panel(rows),factors=['ppe_capex_to_revenue_pct','free_cash_flow_ppe_proxy'])
        self.assertTrue(result['values'].empty)
        self.assertTrue(result['coverage'].reason.eq('negative_reported_investment_or_expense_input').all())

    def test_zero_reported_investment_is_valid_not_missing(self):
        result=derive(panel([value('operating_cash_flow',30),value('capital_expenditures',0)]),factors='free_cash_flow_ppe_proxy')
        self.assertEqual(result['values'].iloc[0].value,30)

    def test_bank_insurance_financial_sector_and_unknown_profile_are_not_applicable(self):
        for changes in [{'profile':'bank'},{'profile':'insurance'},{'sector':'Financials'},{'profile':None}]:
            company=issuer(**changes)
            result=derive(panel([value('revenue',100,company),value('operating_income',20,company)],[company]),factors='operating_margin_pct')
            self.assertTrue(result['values'].empty)
            self.assertTrue(result['coverage'].status.eq('not_applicable').all())

    def test_currency_mismatch_even_if_both_units_defined(self):
        dictionary=copy.deepcopy(_METRICS)
        for item in dictionary['metrics']:
            if item['metric_id']=='operating_income':item['units'].append('EUR')
        result=derive(panel([value('revenue',100),value('operating_income',20,unit='EUR')]),
                      factors='operating_margin_pct',metric_dictionary=dictionary)
        self.assertTrue(result['values'].empty)
        self.assertEqual(result['coverage'].iloc[0].reason,'input_unit_mismatch')

    def test_source_tags_must_be_in_frozen_metric_definition(self):
        result=derive(panel([value('revenue',100,source_tag='us-gaap:UnmappedGrossSales'),value('operating_income',20)]),factors='operating_margin_pct')
        self.assertEqual(result['coverage'].iloc[0].reason,'source_tag_outside_metric_definition')

    def test_parent_profit_cannot_be_replaced_with_other_income_scope(self):
        row=value('net_income',20,metric_id='net_income_consolidated')
        result=derive(panel([value('revenue',100),row]),factors='parent_net_margin_pct')
        self.assertTrue(result['values'].empty)
        self.assertEqual(json.loads(result['coverage'].iloc[0].missing_metrics),['net_income'])

    def test_broader_asset_purchases_cannot_replace_ppe(self):
        row=value('capital_expenditures',10,metric_id='productive_asset_purchases')
        result=derive(panel([value('operating_cash_flow',30),row]),factors='free_cash_flow_ppe_proxy')
        self.assertTrue(result['values'].empty)

    def test_direct_and_proved_same_filing_derived_quarters_can_pair(self):
        common={'period_start':'2025-01-01','period_end':'2025-03-31','period_kind':'direct_quarter'}
        revenue=value('revenue',100,**common)
        cash=value('operating_cash_flow',20,**{**common,'period_kind':'derived_quarter'},
            normalization_method='cumulative_difference',derivation_compatibility='same_filing',input_fact_ids='["current","previous"]')
        result=derive(panel([revenue,cash],financial_period='quarter'),factors='operating_cash_flow_margin_pct')
        self.assertEqual(result['values'].iloc[0].value,20)

    def test_top_level_matching_accn_does_not_prove_derived_quarter(self):
        common={'period_start':'2025-01-01','period_end':'2025-03-31','period_kind':'direct_quarter'}
        revenue=value('revenue',100,**common)
        for changes in [dict(input_accessions='["same-filing","older"]'),dict(derivation_compatibility='cross_filing_unverified'),dict(input_fact_ids='["one"]')]:
            cash=value('operating_cash_flow',20,**{**common,'period_kind':'derived_quarter',
                'normalization_method':'cumulative_difference','derivation_compatibility':'same_filing','input_fact_ids':'["current","previous"]',**changes})
            result=derive(panel([revenue,cash],financial_period='quarter'),factors='operating_cash_flow_margin_pct')
            self.assertTrue(result['values'].empty)

    def test_available_time_is_maximum_of_conservative_inputs(self):
        rows=[value('revenue',100),value('operating_income',20,available_at='2026-02-03T05:00:00Z')]
        result=derive(panel(rows),factors='operating_margin_pct')
        self.assertEqual(result['values'].iloc[0].available_at,'2026-02-03T05:00:00+00:00')
        self.assertEqual(len(json.loads(result['values'].iloc[0].input_value_ids)),2)
        self.assertEqual(len(json.loads(result['values'].iloc[0].input_fact_ids)),2)

    def test_share_class_duplicate_does_not_duplicate_factor(self):
        revenue=value('revenue',100)
        result=derive(panel([revenue,{**revenue,'symbol':'A.CLASS'},value('operating_income',20)]),factors='operating_margin_pct')
        self.assertEqual(len(result['values']),1)

    def test_empty_panel_still_has_selected_factor_coverage(self):
        result=derive(panel([]))
        self.assertEqual(len(result['coverage']),7)
        self.assertTrue(result['values'].empty)
        self.assertIn('input_value_ids',result['values'])

    def test_input_registry_rejects_stock_or_nonadditive_metric(self):
        config=copy.deepcopy(_CONFIG)
        config['factors'][0]['inputs'][0]='total_assets'
        with self.assertRaisesRegex(ValueError,'monetary duration'):
            derive(panel([]),factors='operating_margin_pct',config=config)
        with self.assertRaises(ValueError):derive(panel([]),factors='invented')
        with self.assertRaises(ValueError):derive(panel([]),period='all')

    def test_nullable_flags_and_input_identity_exclude_without_aborting_panel(self):
        for changes in [{'tag_conflict':pd.NA}, {'row_status':pd.NA}, {'company_id':pd.NA},
                        {'accn':pd.NA}, {'source_id':None}, {'unit':pd.NA}, {'period_kind':pd.NA}]:
            with self.subTest(changes=changes):
                result=derive(panel([value('revenue',100),value('operating_income',20,**changes)]),factors='operating_margin_pct')
                self.assertTrue(result['values'].empty)
                self.assertEqual(result['coverage'].iloc[0].status,'unavailable')

    def test_nullable_profile_not_applicable_and_cik_uses_known_issuer_identity(self):
        company=issuer(profile=pd.NA)
        result=derive(panel([value('revenue',100,company),value('operating_income',20,company)],[company]),factors='operating_margin_pct')
        self.assertTrue(result['values'].empty)
        self.assertEqual(result['coverage'].iloc[0].status,'not_applicable')
        result=derive(panel([value('revenue',100,cik=pd.NA),value('operating_income',20,cik=None)]),factors='operating_margin_pct')
        self.assertEqual(result['values'].iloc[0].value,20)

    def test_all_nine_add_broad_asset_proxies_without_replacing_ppe(self):
        rows=self.inputs()+[value('productive_asset_purchases',15)]
        result=derive(panel(rows),factors='all')
        values=result['values'].set_index('factor_id')
        self.assertEqual(len(values),9)
        self.assertEqual(values.loc['productive_asset_purchases_to_revenue_pct','value'],15)
        self.assertEqual(values.loc['free_cash_flow_productive_assets_proxy','value'],-25)
        self.assertEqual(values.loc['free_cash_flow_ppe_proxy','value'],-22)
        no_ppe=[row for row in rows if row['metric_id']!='capital_expenditures']
        missing=derive(panel(no_ppe),factors='all')
        self.assertIn('free_cash_flow_productive_assets_proxy',set(missing['values'].factor_id))
        self.assertNotIn('free_cash_flow_ppe_proxy',set(missing['values'].factor_id))
        self.assertEqual(len(derive(panel(rows))['values']),7)
        self.assertEqual(len(derive(panel(rows),factors=['all'])['values']),9)
        with self.assertRaisesRegex(ValueError,'only factor'):
            derive(panel(rows),factors=['all','parent_net_margin_pct'])

    def test_negative_broad_asset_purchase_is_not_fcf_benefit(self):
        rows=[value('operating_cash_flow',30),value('productive_asset_purchases',-10)]
        result=derive(panel(rows),factors='free_cash_flow_productive_assets_proxy')
        self.assertTrue(result['values'].empty)
        self.assertEqual(result['coverage'].iloc[0].reason,'negative_reported_investment_or_expense_input')


if __name__=='__main__':unittest.main()
