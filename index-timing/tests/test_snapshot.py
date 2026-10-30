from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from index_timing_data.models import ApiTable, QuerySpec
from index_timing_data.snapshot import SnapshotError, SnapshotWriter, verify_snapshot


class FakeClient:
    base_url = "https://jiaoch.site"

    def query(self, spec: QuerySpec) -> ApiTable:
        return ApiTable(
            spec=spec,
            fields=("trade_date", "amount"),
            items=(("20260105", 123.5),),
            response_code=0,
            message="success",
            http_status=200,
            received_at_utc="2026-07-21T00:00:00+00:00",
        )


class SnapshotTests(unittest.TestCase):
    def test_capture_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "snapshot"
            writer = SnapshotWriter(FakeClient())  # type: ignore[arg-type]
            writer.capture([QuerySpec("fund_daily_sample", "fund_daily")], output)
            result = verify_snapshot(output)

            self.assertTrue(result["verified"])
            self.assertEqual(result["dataset_count"], 1)
            combined = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output.rglob("*")
                if path.is_file() and path.suffix in {".json", ".csv"}
            )
            self.assertNotIn("token", combined.lower())

    def test_capture_is_create_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "snapshot"
            output.mkdir()
            with self.assertRaises(SnapshotError):
                SnapshotWriter(FakeClient()).capture(  # type: ignore[arg-type]
                    [QuerySpec("sample", "daily")],
                    output,
                )

    def test_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "snapshot"
            writer = SnapshotWriter(FakeClient())  # type: ignore[arg-type]
            writer.capture([QuerySpec("sample", "daily")], output)
            (output / "normalized" / "sample.csv").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(SnapshotError, "mismatch"):
                verify_snapshot(output)


if __name__ == "__main__":
    unittest.main()

