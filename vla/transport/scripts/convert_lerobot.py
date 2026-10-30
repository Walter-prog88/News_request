#!/usr/bin/env python3
"""Stream whole HDF5 episodes into local LeRobot v2.0 image Parquet files.

No torch/LeRobot import is needed while exporting. Embedded PNG images preserve
the source pixels; actions retain the exact env controller input convention.
This exporter does not fit normalization statistics or contact any Hub.

Example:
  python convert_lerobot.py --dataset transport_subset.hdf5 --split train \
    --output /path/to/lerobot/local/transport_train
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from audit_dataset import episode_key, json_value


CAMERAS = ("agentview", "robot0_eye_in_hand", "robot1_eye_in_hand")
STATE_FIELDS = tuple((f"robot{robot}_{field}", dimension)
                     for robot in (0, 1)
                     for field, dimension in (("eef_pos", 3), ("eef_quat", 4), ("gripper_qpos", 2)))
TASK = "Remove the lid from the source bin, clear the target bin into the trash bin, and transfer the hammer to the target bin using both arms."


def png_bytes(frame):
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected uint8 HWC RGB, got {frame.shape} {frame.dtype}")
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return buffer.getvalue()


def feature_definitions(image_shapes):
    features = {
        "observation.state": {"dtype": "float32", "shape": [18], "names": ["state"]},
        "action": {"dtype": "float32", "shape": [14], "names": ["action"]},
    }
    hf = {
        "observation.state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 18, "_type": "Sequence"},
        "action": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 14, "_type": "Sequence"},
    }
    columns = [pa.field("observation.state", pa.list_(pa.float32(), 18)),
               pa.field("action", pa.list_(pa.float32(), 14))]
    for camera in CAMERAS:
        key = f"observation.images.{camera}"
        features[key] = {"dtype": "image", "shape": image_shapes[camera],
                         "names": ["height", "width", "channel"]}
        hf[key] = {"_type": "Image"}
        columns.append(pa.field(key, pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])))
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        dtype = "float32" if key == "timestamp" else "int64"
        features[key] = {"dtype": dtype, "shape": [1], "names": None}
        hf[key] = {"dtype": dtype, "_type": "Value"}
        columns.append(pa.field(key, pa.float32() if dtype == "float32" else pa.int64()))
    metadata = {b"huggingface": json.dumps({"info": {"features": hf}}).encode()}
    return features, pa.schema(columns, metadata=metadata)


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))


def check_episode(group, image_shapes):
    actions = np.asarray(group["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite raw actions (T,14), got {actions.shape}")
    length = len(actions)
    pieces = []
    for key, size in STATE_FIELDS:
        values = np.asarray(group[f"obs/{key}"], dtype=np.float32)
        if values.shape != (length, size) or not np.isfinite(values).all():
            raise ValueError(f"{group.name}/obs/{key}: expected finite {(length, size)}, got {values.shape}")
        pieces.append(values)
    state = np.concatenate(pieces, axis=1)
    for camera in CAMERAS:
        images = group[f"obs/{camera}_image"]
        if images.shape != (length, *image_shapes[camera]) or images.dtype != np.uint8:
            raise ValueError(f"{group.name}: camera {camera} shape/dtype mismatch: {images.shape} {images.dtype}")
    return state, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--max-episodes", type=int, default=0, help="0 exports the whole selected mask.")
    parser.add_argument("--row-group-size", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite or append: {args.output}")
    if args.row_group_size < 1 or args.max_episodes < 0:
        raise ValueError("Invalid row group size / max episodes")
    mapping, episode_info = [], []
    frame_offset = 0
    with h5py.File(args.dataset, "r") as src:
        mask_path = f"mask/{args.split}"
        if mask_path not in src:
            raise ValueError(f"Missing {mask_path}; use extract_subset.py to create explicit disjoint splits")
        names = [json_value(value) for value in src[mask_path][:]]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate episode in selected split")
        # Check all existing masks for train/validation/test overlap before exporting.
        split_sets = {key: {json_value(value) for value in src[f"mask/{key}"][:]}
                      for key in ("train", "valid", "test") if f"mask/{key}" in src}
        for left in split_sets:
            for right in split_sets:
                if left < right and split_sets[left] & split_sets[right]:
                    raise ValueError(f"Episode overlap between {left} and {right}")
        if args.max_episodes:
            names = names[:args.max_episodes]
        if not names:
            raise ValueError("Cannot export an empty split")
        env_meta = json.loads(json_value(src["data"].attrs["env_args"]))
        if env_meta["env_name"] != "TwoArmTransport":
            raise ValueError(f"Unexpected environment {env_meta['env_name']}")
        fps = env_meta["env_kwargs"].get("control_freq")
        if fps is None or int(fps) != fps or fps <= 0:
            raise ValueError(f"Missing or nonintegral env control_freq: {fps}")
        fps = int(fps)
        first = src["data"][names[0]]
        image_shapes = {camera: list(first[f"obs/{camera}_image"].shape[1:]) for camera in CAMERAS}
        features, schema = feature_definitions(image_shapes)
        # Validate the full first episode before creating a destination.
        check_episode(first, image_shapes)
        args.output.mkdir(parents=True)
        meta_dir = args.output / "meta"
        meta_dir.mkdir()
        manifest = {"source_dataset": str(args.dataset.resolve()), "source_bytes": args.dataset.stat().st_size,
                    "source_split": args.split, "source_episodes": names,
                    "source_subset_manifest": json.loads(json_value(src.attrs["transport_subset_manifest"])) if "transport_subset_manifest" in src.attrs else None,
                    "env_metadata": env_meta, "fps": fps, "action_dim": 14, "state_dim": 18,
                    "state_fields": [{"hdf5_key": f"obs/{key}", "dimension": size} for key, size in STATE_FIELDS],
                    "state_quaternion_convention": "robosuite eef_quat: xyzw; stored unchanged",
                    "image_shapes_hwc": image_shapes, "images_flipped": False,
                    "images_resized": False, "image_encoding": "lossless PNG embedded in Parquet",
                    "action_source": "data/<episode>/actions; cast float32, no scaling or delta conversion",
                    "controller_config_source": "env_metadata.env_kwargs.controller_configs",
                    "action_representation": "relative_eef_controller_command",
                    "action_semantics_note": "OSC controller input numbers; physical units/frame/scaling must be checked against saved env controller config before policy rollout",
                    "task": args.task, "task_is_constant": True,
                    "statistics": "none fitted here; fit openpi normalization on train export only",
                    "version": "LeRobot v2.0", "complete": False}
        manifest_path = meta_dir / "conversion_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        for local_id, name in enumerate(names):
            group = src["data"][name]
            state, actions = check_episode(group, image_shapes)
            length = len(actions)
            path = args.output / f"data/chunk-{local_id // 1000:03d}/episode_{local_id:06d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            with pq.ParquetWriter(path, schema=schema, compression="zstd") as writer:
                for start in range(0, length, args.row_group_size):
                    stop = min(length, start + args.row_group_size)
                    indices = np.arange(start, stop, dtype=np.int64)
                    block = {"observation.state": state[start:stop].tolist(), "action": actions[start:stop].tolist(),
                             "timestamp": (indices / fps).astype(np.float32).tolist(),
                             "frame_index": indices.tolist(), "episode_index": [local_id] * len(indices),
                             "index": (indices + frame_offset).tolist(), "task_index": [0] * len(indices)}
                    for camera in CAMERAS:
                        key = f"observation.images.{camera}"
                        images = group[f"obs/{camera}_image"][start:stop]
                        block[key] = [{"bytes": png_bytes(frame), "path": None} for frame in images]
                    writer.write_table(pa.Table.from_pydict(block, schema=schema))
            # Numeric and image spot checks after serialization, not just file existence.
            table = pq.read_table(path)
            for index in sorted({0, length // 2, length - 1}):
                if not np.array_equal(np.asarray(table["action"][index].as_py(), dtype=np.float32), actions[index]):
                    raise AssertionError(f"Serialized action differs: {name} frame {index}")
                if not np.array_equal(np.asarray(table["observation.state"][index].as_py(), dtype=np.float32), state[index]):
                    raise AssertionError(f"Serialized state differs: {name} frame {index}")
                for camera in CAMERAS:
                    encoded = table[f"observation.images.{camera}"][index].as_py()["bytes"]
                    decoded = np.asarray(Image.open(io.BytesIO(encoded)))
                    if not np.array_equal(decoded, group[f"obs/{camera}_image"][index]):
                        raise AssertionError(f"Serialized RGB differs: {name} {camera} frame {index}")
            del table
            episode_info.append({"episode_index": local_id, "tasks": [args.task], "length": length})
            mapping.append({"episode_index": local_id, "source_episode": name, "length": length,
                            "from": frame_offset, "to": frame_offset + length,
                            "actions_float32_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                            "serialization_spot_checks_passed": True})
            frame_offset += length
            print(f"Exported {args.split} {local_id + 1}/{len(names)}: {name}, {length} frames", flush=True)
        info = {"codebase_version": "v2.0", "robot_type": "dual_panda_parallel_grippers",
                "total_episodes": len(names), "total_frames": frame_offset, "total_tasks": 1,
                "total_videos": 0, "total_chunks": math.ceil(len(names) / 1000), "chunks_size": 1000,
                "fps": fps, "splits": {"train": f"0:{len(names)}"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": None, "features": features}
        (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")
        write_jsonl(meta_dir / "episodes.jsonl", episode_info)
        write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": args.task}])
        (meta_dir / "stats.json").write_text("{}\n")
        (meta_dir / "source_episode_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
        manifest.update(complete=True, total_episodes=len(names), total_frames=frame_offset,
                        note="Each output is a single split in a separate local repo; LeRobot internal split name is train for reader compatibility")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "source_split": args.split,
                      "episodes": len(names), "frames": frame_offset, "fps": fps,
                      "complete": True}, indent=2))


if __name__ == "__main__":
    main()
