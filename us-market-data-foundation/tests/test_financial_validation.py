import tempfile
import unittest
from pathlib import Path
import pandas as pd
from usmd.financial_validation import validate_paths, parquet_metadata

class StreamValidationTests(unittest.TestCase):
    def test_published_validation_rejects_bad_facts_without_dataframe_materialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'facts.parquet'
            row=dict(fact_id='same',issuer_id='SEC:1',unit='USD',value=1.,source_id='source',accn='accn',filed='2025-01-01',available_at='2025-01-02',row_status='ok')
            pd.DataFrame([row,{**row,'value':float('inf'),'source_id':None}]).to_parquet(path)
            errors,warnings,checks=validate_paths({'financial_facts':path},'2026-01-01')
            self.assertEqual(errors[0]['duplicates'],1)
            self.assertEqual(errors[0]['nonfinite'],1)
            self.assertEqual(errors[0]['missing_source'],1)
            self.assertEqual(parquet_metadata(path)['row_count'],2)

    def test_future_actuals_and_conflicts_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'values.parquet'
            pd.DataFrame([dict(value_id='v',issuer_id='SEC:1',unit='USD',value=1.,source_id='s',accn='a',filed='2025-01-01',available_at='2025-01-02',row_status='ok',period_end='2027-01-01',tag_conflict=True)]).to_parquet(path)
            errors,warnings,_=validate_paths({'financial_values':path},'2026-01-01')
            self.assertEqual(errors[0]['check'],'future_actual_period')
            self.assertTrue(any(w['check']=='tag_conflicts' for w in warnings))
