#!/usr/bin/env python3
"""Copy whole trajectories into a smaller HDF5 with reproducible disjoint splits.

The source is opened read-only. No resizing, normalization, relabeling, or action
conversion is performed. All per-episode XML/state/action/image data are kept.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from audit_dataset import episode_key, json_value


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--valid", type=int, default=50)
    parser.add_argument("--test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--source-revision", default="181967e10c6277a653e7c9761f3978312af3646a")
    parser.add_argument("--source-repo", default="MimicGen/dexmimicgen_datasets")
    parser.add_argument("--source-filename", default="generated/two_arm_transport.hdf5")
    parser.add_argument("--skip-source-hash", action="store_true")
    parser.add_argument("--source-integrity", type=Path,
                        help="Completed downloader integrity JSON; validated against this source hash.")
    parser.add_argument("--exclude-episodes", nargs="*", default=[])
    parser.add_argument("--exclusion-reason", default="explicit pre-split exclusion")
    args = parser.parse_args()
    if min(args.train, args.valid, args.test) < 0:
        raise ValueError("Split counts cannot be negative")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.output.with_suffix(".manifest.json")
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".partial")
    if temp.exists():
        raise FileExistsError(temp)
    manifest = {"source_dataset": str(args.dataset.resolve()),
                "source_repo": args.source_repo, "source_revision": args.source_revision,
                "source_filename": args.source_filename,
                "source_bytes": args.dataset.stat().st_size,
                "source_revision_verified_by_this_script": False,
                "seed": args.seed, "selection": "random whole-episode selection; not OOD split",
                "source_sha256": None if args.skip_source_hash else sha256_file(args.dataset),
                "excluded_episodes": sorted(set(args.exclude_episodes), key=episode_key),
                "exclusion_reason": args.exclusion_reason if args.exclude_episodes else None}
    if args.source_integrity:
        evidence = json.loads(args.source_integrity.read_text())
        if (not evidence.get("verified") or evidence.get("sha256") != manifest["source_sha256"]
                or evidence.get("bytes") != manifest["source_bytes"]
                or f"/resolve/{args.source_revision}/{args.source_filename}" not in evidence.get("url", "")):
            raise ValueError("Completed source integrity evidence does not match this file/revision")
        manifest["source_revision_verified_by_this_script"] = True
        manifest["source_integrity_evidence"] = evidence
    with h5py.File(args.dataset, "r") as src, h5py.File(temp, "w") as dst:
        if src.attrs.get("partial_range_probe", False):
            raise ValueError("Partial HTTP Range preflight artifact cannot be a formal training source")
        names = sorted([k for k in src["data"] if isinstance(src["data"][k], h5py.Group)], key=episode_key)
        source_count = len(names)
        unknown = set(args.exclude_episodes) - set(names)
        if unknown:
            raise ValueError(f"Excluded episodes not found: {sorted(unknown)}")
        names = [name for name in names if name not in args.exclude_episodes]
        total = args.train + args.valid + args.test
        if total > len(names):
            raise ValueError(f"Requested {total} episodes from {len(names)}")
        selected = np.random.default_rng(args.seed).permutation(names)[:total].tolist()
        splits = {"train": selected[:args.train],
                  "valid": selected[args.train:args.train + args.valid],
                  "test": selected[args.train + args.valid:]}
        manifest.update(source_episode_count=source_count, eligible_episode_count=len(names), splits=splits)
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        data = dst.create_group("data")
        for key, value in src["data"].attrs.items():
            data.attrs[key] = value
        frames = 0
        for index, name in enumerate(selected, 1):
            src.copy(src["data"][name], data, name=name)
            frames += data[name]["actions"].shape[0]
            if index % 10 == 0 or index == total:
                print(f"Copied {index}/{total} whole episodes", flush=True)
        data.attrs["total"] = frames
        mask = dst.create_group("mask")
        for key, episode_names in splits.items():
            mask.create_dataset(key, data=np.asarray(episode_names, dtype="S"))
        dst.attrs["transport_subset_manifest"] = json.dumps(manifest)
        manifest["subset_total_frames"] = frames
    os.replace(temp, args.output)
    manifest.update(subset_path=str(args.output.resolve()), subset_bytes=args.output.stat().st_size,
                    subset_sha256=sha256_file(args.output))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "bytes": manifest["subset_bytes"],
                      "split_sizes": {k: len(v) for k, v in splits.items()},
                      "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
