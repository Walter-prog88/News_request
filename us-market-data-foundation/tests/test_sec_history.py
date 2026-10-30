import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from usmd.sec_history import build_manifest, parse_fsds_zip


def member(header, rows):
    lines = [header, *rows]
    return ("\n".join("\t".join(row) if isinstance(row, list) else row for row in lines) + "\n").encode()


class SecHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="usmd-sec-history-")
        self.path = Path(self.tmp.name) / "2020q1.zip"
        files = {
            "sub.txt": member(
                ["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed", "accepted"],
                [["0001-20-000001", "1", "Example Co", "1000", "10-K", "20191231", "2019", "FY", "20200201", "20200201T12:00:00Z"],
                 ["0002-20-000002", "2", "Later Co", "2000", "10-Q", "20200331", "2020", "Q1", "20200501", "20200501T12:00:00Z"]]),
            "num.txt": member(
                ["adsh", "tag", "version", "ddate", "qtrs", "uom", "coreg", "footnote", "frame", "value", "segments"],
                [["0001-20-000001", "NetIncomeLoss", "us-gaap/2020", "20191231", "4", "USD", "", "", "CY2019", "10", ""],
                 ["0001-20-000001", "NetCashProvidedByUsedInOperatingActivities", "us-gaap/2020", "20191231", "4", "USD", "", "", "CY2019", "20", "dim=North"],
                 ["0001-20-000001", "Assets", "us-gaap/2020", "20210331", "0", "USD", "", "", "", "30", ""],
                 ["0001-20-000001", "RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap/2020", "20191231", "4", "USD", "", "", "", "99", ""],
                 ["0002-20-000002", "Assets", "us-gaap/2020", "20200331", "0", "USD", "", "", "", "40", ""]]),
            "tag.txt": member(["tag", "version", "custom", "abstract", "datatype", "iord", "crdr", "tlabel", "doc"],
                               [["NetIncomeLoss", "us-gaap/2020", "0", "0", "xbrli:monetaryItemType", "", "", "Net income", ""],
                                ["Assets", "us-gaap/2020", "0", "0", "xbrli:monetaryItemType", "", "D", "Assets", ""]]),
            "pre.txt": member(["adsh", "report", "line", "stmt", "inpth", "rfile", "stprf", "negating", "plabel", "tag", "version"],
                               [["0001-20-000001", "R1", "1", "IS", "1", "F", "", "0", "Net income", "NetIncomeLoss", "us-gaap/2020"]]),
        }
        with zipfile.ZipFile(self.path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in files.items():
                zf.writestr(name, data)
        self.addCleanup(self.tmp.cleanup)

    def test_parse_keeps_as_filed_identity_and_segments_and_filters_filed_cutoff(self):
        result = parse_fsds_zip(self.path, cutoff="2020-03-01", version="2020q1")
        self.assertEqual(len(result["filings"]), 1)
        filing = result["filings"][0]
        self.assertEqual({filing[k] for k in ("adsh", "cik", "name", "form", "period", "filed", "accepted", "sic")},
                         {"0001-20-000001", "1", "Example Co", "10-K", "20191231", "20200201", "20200201T12:00:00Z", "1000"})
        self.assertEqual(len(result["facts"]), 3)
        cfo = next(row for row in result["facts"] if row["tag"] == "NetCashProvidedByUsedInOperatingActivities")
        self.assertEqual(cfo["segments"], "dim=North")
        self.assertTrue(next(row for row in result["facts"] if row["tag"] == "Assets")["observation_after_cutoff"])
        self.assertEqual(result["manifest"]["observation_after_cutoff_count"], 1)
        self.assertNotIn("RevenueFromContractWithCustomerExcludingAssessedTax", {row["tag"] for row in result["facts"]})
        self.assertEqual(result["manifest"]["version"], "2020q1")
        self.assertEqual(result["manifest"]["source_zip_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_manifest_is_json_ready_and_hashes_without_ticker_lookup(self):
        manifest = build_manifest([self.path], cutoff="2020-03-01", version="2020q1")
        self.assertEqual(manifest["package_count"], 1)
        self.assertEqual(manifest["packages"][0]["source_bytes"], self.path.stat().st_size)

    def test_duplicate_filing_is_rejected(self):
        duplicate_path = Path(self.tmp.name) / "duplicate.zip"
        with zipfile.ZipFile(self.path) as source, zipfile.ZipFile(duplicate_path, "w", zipfile.ZIP_DEFLATED) as target:
            for info in source.infolist():
                if info.filename != "sub.txt":
                    target.writestr(info.filename, source.read(info.filename))
            target.writestr("sub.txt", member(
                ["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed", "accepted"],
                [["0001-20-000001", "1", "Example Co", "1000", "10-K", "20191231", "2019", "FY", "20200201", "20200201T12:00:00Z"],
                 ["0001-20-000001", "1", "Example Co", "1000", "10-K", "20191231", "2019", "FY", "20200201", "20200201T12:00:00Z"]]))
        with self.assertRaisesRegex(ValueError, "duplicate accession"):
            parse_fsds_zip(duplicate_path)


if __name__ == "__main__":
    unittest.main()
