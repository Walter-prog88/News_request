from __future__ import annotations

import argparse
import getpass
import json
import os
from collections.abc import Sequence
from pathlib import Path

from .client import TwoShareClient
from .config import load_query_specs
from .snapshot import SnapshotWriter, verify_snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="index-data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="capture a create-exclusive snapshot")
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument(
        "--base-url",
        default=os.getenv("TWOSHARE_BASE_URL", "https://jiaoch.site"),
    )
    capture.add_argument("--token-env", default="TWOSHARE_TOKEN")
    capture.add_argument("--timeout-seconds", type=float, default=35.0)

    verify = subparsers.add_parser("verify", help="verify all files in a snapshot")
    verify.add_argument("snapshot", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        print(json.dumps(verify_snapshot(args.snapshot), ensure_ascii=False, indent=2))
        return

    token = os.getenv(args.token_env, "")
    if not token:
        token = getpass.getpass(f"{args.token_env} (hidden): ")
    specs = load_query_specs(args.config)
    client = TwoShareClient(
        token,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    path = SnapshotWriter(client).capture(specs, args.output)
    result = verify_snapshot(path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
