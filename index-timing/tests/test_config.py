from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from index_timing_data.config import load_query_specs


class ConfigTests(unittest.TestCase):
    def test_loads_specs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "queries": [
                            {
                                "dataset_id": "sample",
                                "api_name": "daily",
                                "params": {"ts_code": "000001.SZ"},
                                "fields": ["ts_code", "trade_date"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            specs = load_query_specs(path)
            self.assertEqual(specs[0].fields, ("ts_code", "trade_date"))

    def test_duplicate_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "queries": [
                            {"dataset_id": "same", "api_name": "daily"},
                            {"dataset_id": "same", "api_name": "shibor"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_query_specs(path)


if __name__ == "__main__":
    unittest.main()

