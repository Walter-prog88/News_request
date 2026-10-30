"""Storage failure tests use synthetic directories and never touch a real volume."""
import importlib.util
import json
from pathlib import Path
import plistlib
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_storage.py"
SPEC = importlib.util.spec_from_file_location("storage_preflight", SCRIPT)
storage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(storage)


class StoragePreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="usmd-storage-test-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.mount = self.base / "external"
        self.root = self.mount / "lake"
        self.root.mkdir(parents=True)
        self.config = {
            "external_mount": str(self.mount),
            "external_data_root": str(self.root),
            "external_volume_uuid": "TEST-UUID",
            "project_id": "test-lake",
            "identity_file": "_lake_identity.json",
            "external_minimum_free_gib": 1,
        }
        self.marker = self.root / "_lake_identity.json"
        self.marker.write_text(json.dumps({
            "project_id": "test-lake", "volume_uuid": "TEST-UUID"
        }), encoding="utf-8")

    def info(self, **changes):
        result = {
            "VolumeUUID": "TEST-UUID", "Internal": False,
            "MountPoint": str(self.mount), "FilesystemType": "exfat",
        }
        result.update(changes)
        return SimpleNamespace(stdout=plistlib.dumps(result))

    def snapshot(self):
        return {str(p.relative_to(self.base)): p.read_bytes()
                for p in self.base.rglob("*") if p.is_file()}

    def check_with_disk(self, **changes):
        with patch.object(storage.os.path, "ismount", return_value=True), \
             patch.object(storage.subprocess, "run", return_value=self.info(**changes)), \
             patch.object(storage.shutil, "disk_usage", return_value=SimpleNamespace(free=8 * 1024 ** 3)):
            return storage.check(self.config)

    def test_unmounted_path_does_not_create_fallback(self):
        missing = self.base / "missing-volume"
        self.config["external_mount"] = str(missing)
        self.config["external_data_root"] = str(missing / "lake")
        before = self.snapshot()
        with patch.object(storage.subprocess, "run") as diskutil:
            with self.assertRaises(ValueError):
                storage.check(self.config)
        diskutil.assert_not_called()
        self.assertFalse(missing.exists())
        self.assertEqual(before, self.snapshot())

    def test_regular_directory_cannot_impersonate_mount(self):
        with patch.object(storage.os.path, "ismount", return_value=False), \
             patch.object(storage.subprocess, "run") as diskutil:
            with self.assertRaises(ValueError):
                storage.check(self.config)
        diskutil.assert_not_called()

    def test_wrong_volume_uuid_is_rejected_without_changes(self):
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.check_with_disk(VolumeUUID="OTHER-UUID")
        self.assertEqual(before, self.snapshot())

    def test_internal_volume_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check_with_disk(Internal=True)

    def test_ancestor_symlink_cannot_escape_external_root(self):
        outside = self.base / "local-data"
        outside.mkdir()
        (outside / "lake").mkdir()
        (self.mount / "redirect").symlink_to(outside, target_is_directory=True)
        self.config["external_data_root"] = str(self.mount / "redirect" / "lake")
        with self.assertRaises(ValueError):
            self.check_with_disk()

    def test_wrong_project_marker_is_rejected(self):
        self.marker.write_text(json.dumps({
            "project_id": "other-project", "volume_uuid": "TEST-UUID"
        }), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.check_with_disk()

    def test_space_floor_blocks_check(self):
        with patch.object(storage.os.path, "ismount", return_value=True), \
             patch.object(storage.subprocess, "run", return_value=self.info()), \
             patch.object(storage.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
            with self.assertRaises(ValueError):
                storage.check(self.config)

    def test_valid_check_is_read_only(self):
        before = self.snapshot()
        result = self.check_with_disk()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["read_only_check"])
        self.assertFalse(result["fallback_to_internal_disk"])
        self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
