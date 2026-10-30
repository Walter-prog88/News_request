#!/usr/bin/env python3
"""Read-only macOS storage preflight. Never creates directories or downloads data."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys


def check(config):
    mount = Path(config["external_mount"])
    root = Path(config["external_data_root"])
    if not mount.is_dir() or not os.path.ismount(mount):
        raise ValueError("外置盘未挂载；停止。不会回退到本机。")
    result = subprocess.run(
        ["/usr/sbin/diskutil", "info", "-plist", str(mount)],
        check=True, capture_output=True, timeout=15,
    )
    info = plistlib.loads(result.stdout)
    if info.get("VolumeUUID", "").upper() != config["external_volume_uuid"].upper():
        raise ValueError("外置卷 UUID 不符；停止。")
    if info.get("Internal") is not False:
        raise ValueError("目标不是已确认的外置卷；停止。")
    if Path(info.get("MountPoint", "")).resolve() != mount.resolve():
        raise ValueError("挂载路径不符；停止。")
    if not root.is_dir() or root.is_symlink():
        raise ValueError("数据目录不存在或为软链接；停止，不自动创建。")
    root.resolve().relative_to(mount.resolve())
    if root.stat().st_dev != mount.stat().st_dev:
        raise ValueError("数据目录不在目标卷上；停止。")
    marker = root / config["identity_file"]
    if marker.is_symlink():
        raise ValueError("数据目录标记不能是软链接；停止。")
    identity = json.loads(marker.read_text(encoding="utf-8"))
    if identity.get("project_id") != config["project_id"]:
        raise ValueError("数据目录项目标记不符；停止。")
    if identity.get("volume_uuid", "").upper() != config["external_volume_uuid"].upper():
        raise ValueError("数据目录卷标记不符；停止。")
    free = shutil.disk_usage(root).free
    minimum = config["external_minimum_free_gib"] * 1024 ** 3
    if free < minimum:
        raise ValueError("外置盘剩余容量低于规划安全余量；停止。")
    return {
        "status": "ok",
        "read_only_check": True,
        "external_root": str(root),
        "volume_uuid": info["VolumeUUID"],
        "filesystem": info.get("FilesystemType"),
        "free_gib": round(free / 1024 ** 3, 2),
        "fallback_to_internal_disk": False,
        "note": "只读检查通过；不代表采集器、容量配额或备份已经启用。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/storage.json")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = check(config)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, plistlib.InvalidFileException) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
