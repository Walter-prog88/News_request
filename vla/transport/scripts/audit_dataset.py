#!/usr/bin/env python3
"""Read-only inventory of a DexMimicGen HDF5; never imports a simulator.

Example: python audit_dataset.py --dataset /path/to/two_arm_transport.hdf5 \
    --output /path/to/new/data_audit.json --scan-episodes 5
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def episode_key(name):
    suffix = name.rsplit("_", 1)[-1]
    return (int(suffix) if suffix.isdigit() else -1, name)


def dataset_schema(group):
    result = {}
    def visit(name, item):
        if isinstance(item, h5py.Dataset):
            result[name] = {"shape": list(item.shape), "dtype": str(item.dtype),
                            "compression": item.compression}
    group.visititems(visit)
    return result


def array_stats(dataset):
    arr = np.asarray(dataset)
    if not np.issubdtype(arr.dtype, np.number):
        return {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    finite = np.isfinite(arr)
    result = {"shape": list(arr.shape), "dtype": str(arr.dtype),
              "all_finite": bool(finite.all())}
    if arr.size and finite.all():
        axis = 0 if arr.ndim > 1 else None
        result.update(min=np.min(arr, axis=axis).tolist(),
                      max=np.max(arr, axis=axis).tolist(),
                      mean=np.mean(arr, axis=axis).tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-episodes", type=int, default=5,
                        help="Read numeric arrays for this many episodes; 0 scans none, -1 scans all.")
    parser.add_argument("--sha256", action="store_true", help="Stream the entire file to hash it.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    report = {"dataset": str(args.dataset.resolve()), "bytes": args.dataset.stat().st_size,
              "mode": "read_only_hdf5_audit", "errors": []}
    if args.sha256:
        digest = hashlib.sha256()
        with args.dataset.open("rb") as src:
            for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        report["sha256"] = digest.hexdigest()
    with h5py.File(args.dataset, "r") as src:
        data = src["data"]
        report["root_attrs"] = {k: json_value(v) for k, v in src.attrs.items()}
        report["data_attrs"] = {k: json_value(v) for k, v in data.attrs.items()}
        raw_meta = report["data_attrs"].get("env_args")
        report["env_metadata"] = json.loads(raw_meta) if raw_meta else None
        names = sorted([k for k in data if isinstance(data[k], h5py.Group)], key=episode_key)
        report["episode_count"] = len(names)
        report["episode_names"] = names
        report["masks"] = {}
        if "mask" in src:
            for key, values in src["mask"].items():
                report["masks"][key] = [json_value(x) for x in values[:]]
        inventory = []
        for name in names:
            group = data[name]
            shape = list(group["actions"].shape) if "actions" in group else None
            state_shape = list(group["states"].shape) if "states" in group else None
            item = {"episode": name, "actions_shape": shape, "states_shape": state_shape,
                    "num_samples_attr": json_value(group.attrs.get("num_samples")),
                    "has_model_file": "model_file" in group.attrs,
                    "has_ep_meta": "ep_meta" in group.attrs}
            inventory.append(item)
            if shape is None or state_shape is None:
                report["errors"].append(f"{name}: missing actions or states")
            elif shape[0] != state_shape[0]:
                report["errors"].append(f"{name}: action/state lengths differ")
        report["inventory"] = inventory
        report["total_action_frames"] = sum(x["actions_shape"][0] for x in inventory if x["actions_shape"])
        report["sample_schemas"] = {}
        report["numeric_scans"] = {}
        scan_names = names if args.scan_episodes < 0 else names[:args.scan_episodes]
        for name in scan_names:
            group = data[name]
            report["sample_schemas"][name] = dataset_schema(group)
            scans = {}
            if "actions" in group:
                scans["actions"] = array_stats(group["actions"])
            if "action_dict" in group:
                for key, values in group["action_dict"].items():
                    if isinstance(values, h5py.Dataset):
                        scans[f"action_dict/{key}"] = array_stats(values)
            if "obs" in group:
                for key, values in group["obs"].items():
                    if isinstance(values, h5py.Dataset) and values.ndim <= 2:
                        scans[f"obs/{key}"] = array_stats(values)
            report["numeric_scans"][name] = scans
            for key, stats in scans.items():
                if stats.get("all_finite") is False:
                    report["errors"].append(f"{name}/{key}: nonfinite numeric values")
        report["checks_passed"] = not report["errors"] and bool(names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ["dataset", "bytes", "episode_count", "total_action_frames", "checks_passed", "errors"]}, indent=2))
    print(f"Full report: {args.output}")
    if not report["checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
