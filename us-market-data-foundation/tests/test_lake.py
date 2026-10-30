"""Lake failure tests never instantiate or write to the configured real volume."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from usmd.lake import Lake


class LakeWriteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="usmd-lake-test-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "external" / "lake"
        self.root.mkdir(parents=True)
        self.config = {
            "external_data_root": str(self.root),
            "external_phase1_budget_gib": 1,
            "external_allocation_bytes_observed": 4096,
        }
        self.preflight_patch = patch("usmd.lake.check_storage", return_value={"status": "ok"})
        self.preflight = self.preflight_patch.start()
        self.addCleanup(self.preflight_patch.stop)
        self.lake = Lake(self.config)
        self.preflight.reset_mock()
        self.published = self.root / "normalized" / "sample.parquet"
        self.published.parent.mkdir()
        self.published.write_bytes(b"known-good-published-data")
        self.original = self.published.read_bytes()

    def assert_old_preserved(self):
        self.assertEqual(self.published.read_bytes(), self.original)
        self.assertEqual(list(self.root.rglob("*.tmp")), [])

    def test_absolute_and_parent_traversal_paths_are_rejected(self):
        paths = [str(self.base / "outside.dat"), "../outside.dat", "normalized/../../outside.dat", ""]
        for path in paths:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    self.lake.write_bytes(path, b"bad")
        self.assertFalse((self.base / "outside.dat").exists())
        self.assert_old_preserved()

    def test_symlink_parent_cannot_redirect_writes(self):
        outside = self.base / "internal-cache"
        outside.mkdir()
        (self.root / "redirect").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(OSError):
            self.lake.write_bytes("redirect/nested/file.dat", b"must-not-escape")
        self.assertEqual(list(outside.iterdir()), [])
        self.assert_old_preserved()

    def test_symlink_target_cannot_overwrite_outside_file(self):
        target = self.base / "local-file"
        target.write_bytes(b"local-original")
        (self.root / "normalized" / "link.dat").symlink_to(target)
        with self.assertRaises(ValueError):
            self.lake.write_bytes("normalized/link.dat", b"replacement")
        self.assertEqual(target.read_bytes(), b"local-original")
        self.assertTrue((self.root / "normalized" / "link.dat").is_symlink())
        self.assert_old_preserved()

    def test_failed_initial_preflight_preserves_old_file(self):
        self.preflight.side_effect = ValueError("simulated detached volume")
        with self.assertRaises(ValueError):
            self.lake.write_bytes("normalized/sample.parquet", b"replacement")
        self.assert_old_preserved()

    def test_failed_publication_preflight_preserves_old_file(self):
        self.preflight.side_effect = [{"status": "ok"}, ValueError("simulated changed volume identity")]
        with self.assertRaises(ValueError):
            self.lake.write_bytes("normalized/sample.parquet", b"replacement")
        self.assertEqual(self.preflight.call_count, 2)
        self.assert_old_preserved()

    def test_missing_root_is_never_recreated_even_after_stale_preflight(self):
        detached = self.base / "detached-lake"
        self.root.rename(detached)
        # Simulate a previously successful preflight followed by a missing mount/root.
        with self.assertRaises(FileNotFoundError):
            self.lake.write_bytes("normalized/sample.parquet", b"replacement")
        self.assertFalse(self.root.exists())
        self.assertEqual((detached / "normalized" / "sample.parquet").read_bytes(), self.original)

    def test_budget_failure_preserves_old_file(self):
        self.config["external_phase1_budget_gib"] = 1 / (1024 ** 3)
        with self.assertRaises(ValueError):
            self.lake.write_bytes("normalized/sample.parquet", b"replacement")
        self.assert_old_preserved()

    def test_replace_failure_preserves_old_file_and_cleans_temp(self):
        with patch("usmd.lake.os.replace", side_effect=OSError("simulated I/O failure")):
            with self.assertRaises(OSError):
                self.lake.write_bytes("normalized/sample.parquet", b"replacement")
        self.assert_old_preserved()

    def test_success_replaces_atomically_without_temporary_residue(self):
        result = self.lake.write_bytes("normalized/sample.parquet", b"new-published-data")
        self.assertEqual(Path(result), self.published)
        self.assertEqual(self.published.read_bytes(), b"new-published-data")
        self.assertEqual(list(self.root.rglob("*.tmp")), [])
        self.assertEqual(self.preflight.call_count, 2)

    def test_nonfinite_json_is_rejected_before_publication(self):
        old_manifest = self.root / "manifest.json"
        old_manifest.write_bytes(b'{"status":"good"}')
        with self.assertRaises(ValueError):
            self.lake.write_json("manifest.json", {"value": float("nan")})
        self.assertEqual(old_manifest.read_bytes(), b'{"status":"good"}')
        self.preflight.assert_not_called()

    def test_second_writer_cannot_acquire_same_lock(self):
        with self.lake.writer_lock():
            with self.assertRaises(BlockingIOError):
                with self.lake.writer_lock():
                    self.fail("second writer unexpectedly acquired lock")

    def test_generated_output_failure_preserves_previous_file(self):
        def fail(path):
            path.write_bytes(b'partial-stream')
            raise RuntimeError('interrupted producer')
        with self.assertRaises(RuntimeError):
            self.lake.write_generated('normalized/sample.parquet', fail)
        self.assert_old_preserved()

    def test_generated_budget_overrun_cannot_publish(self):
        with patch.object(self.lake, '_check_budget', side_effect=[None, ValueError('budget')]):
            with self.assertRaises(ValueError):
                self.lake.write_generated('normalized/sample.parquet', lambda p:p.write_bytes(b'large-output'))
        self.assert_old_preserved()

    def test_generated_success_and_work_dir_symlink_guard(self):
        self.lake.write_generated('normalized/sample.parquet', lambda p:p.write_bytes(b'streamed'))
        self.assertEqual(self.published.read_bytes(), b'streamed')
        self.assertEqual(self.lake.safe_work_dir('staging/financials'), self.root/'staging/financials')
        (self.root/'alias').symlink_to(self.base, target_is_directory=True)
        with self.assertRaises(OSError):
            self.lake.safe_work_dir('alias/escape')
        self.assertFalse((self.base/'escape').exists())


if __name__ == "__main__":
    unittest.main()
