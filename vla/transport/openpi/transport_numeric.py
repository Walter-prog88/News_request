"""Read only numeric Parquet columns, with train-split provenance checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from transport_policy import ACTION_DIM


def read_train_numeric(root, schema):
    """Each recorded row contributes once; no images or action chunks are decoded."""
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    manifest_path = root / "meta/conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mapping = json.loads((root / "meta/source_episode_mapping.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    if manifest.get("complete") is not True or manifest.get("source_split") != "train":
        raise ValueError("Normalization requires a completed train-only conversion manifest.")
    if manifest.get("action_representation") != "relative_eef_controller_command" or manifest.get("action_dim") != ACTION_DIM:
        raise ValueError("Converted action representation differs from the adapter contract.")
    if not np.isclose(float(info["fps"]), schema.fps) or not np.isclose(float(manifest["fps"]), schema.fps):
        raise ValueError("Metadata, conversion manifest, and schema fps must agree.")
    source_episodes = manifest["source_episodes"]
    splits = (manifest.get("source_subset_manifest") or {}).get("splits")
    if not splits or set(splits) != {"train", "valid", "test"}:
        raise ValueError("Source subset must record explicit train/valid/test episode masks.")
    split_sets = {key: set(value) for key, value in splits.items()}
    if any(len(split_sets[key]) != len(splits[key]) for key in splits):
        raise ValueError("Duplicate source episode in a split mask.")
    if any(split_sets[a] & split_sets[b] for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("Source split provenance contains episode leakage.")
    if len(set(source_episodes)) != len(source_episodes) or not set(source_episodes).issubset(split_sets["train"]):
        raise ValueError("Converted source episodes are duplicated or not exclusively training episodes.")
    total_frames, total_episodes = int(info["total_frames"]), int(info["total_episodes"])
    if total_frames < 2 or total_episodes < 1:
        raise ValueError("Require a nonempty demonstration dataset.")
    if manifest.get("total_frames") != total_frames or manifest.get("total_episodes") != total_episodes:
        raise ValueError("Conversion manifest and dataset frame/episode totals differ.")
    if len(mapping) != total_episodes or len(episodes) != total_episodes or len(source_episodes) != total_episodes:
        raise ValueError("Episode metadata and source mapping counts differ.")
    if [item["episode_index"] for item in mapping] != list(range(total_episodes)) or [item["episode_index"] for item in episodes] != list(range(total_episodes)):
        raise ValueError("Require contiguous, ordered local episode indices.")
    if [item["source_episode"] for item in mapping] != source_episodes:
        raise ValueError("Source episode mapping differs from the conversion manifest.")
    keys = list(dict.fromkeys([field.key for field in schema.state_fields]))
    features = info["features"]
    if features[schema.action_key]["shape"] != [ACTION_DIM] or features[schema.action_key]["dtype"] != "float32":
        raise ValueError("Action feature must be float32 with width 14.")
    for key in keys:
        if features[key]["dtype"] != "float32" or len(features[key]["shape"]) != 1:
            raise ValueError(f"Expected a float32 state vector feature: {key}")
    columns = list(dict.fromkeys([*keys, schema.action_key, "episode_index", "frame_index", "index"]))
    states = np.empty((total_frames, schema.state_dim), dtype=np.float64)
    actions = np.empty((total_frames, ACTION_DIM), dtype=np.float64)
    offset, paths, digests = 0, set(), []
    for episode, provenance in zip(episodes, mapping):
        episode_id, length = int(episode["episode_index"]), int(episode["length"])
        if length <= 0 or provenance["length"] != length or provenance["from"] != offset or provenance["to"] != offset + length:
            raise ValueError(f"Episode {episode_id} frame offsets differ from conversion provenance.")
        if offset + length > total_frames:
            raise ValueError("Episode lengths exceed the metadata frame total.")
        path = (root / info["data_path"].format(episode_chunk=episode_id // info["chunks_size"], episode_index=episode_id)).resolve()
        if not path.is_relative_to(root) or path in paths:
            raise ValueError("Numeric data paths must be unique and inside the selected repository.")
        paths.add(path)
        table = pq.read_table(path, columns=columns)
        if table.num_rows != length:
            raise ValueError(f"Episode {episode_id} Parquet row count differs from metadata.")
        numeric = table.to_pydict()  # The selected columns contain no image bytes.
        for key, expected in (("episode_index", np.full(length, episode_id)),
                              ("frame_index", np.arange(length)),
                              ("index", np.arange(offset, offset + length))):
            if not np.array_equal(np.asarray(numeric[key]), expected):
                raise ValueError(f"Episode {episode_id} has repeated, missing, or incorrect {key} values.")
        raw_actions = np.asarray(numeric[schema.action_key], dtype=np.float32)
        if raw_actions.shape != (length, ACTION_DIM) or not np.isfinite(raw_actions).all():
            raise ValueError(f"Episode {episode_id} has invalid raw actions.")
        action_digest = hashlib.sha256(raw_actions.tobytes()).hexdigest()
        if action_digest != provenance["actions_float32_sha256"]:
            raise ValueError(f"Episode {episode_id} action hash differs from the source conversion.")
        for key in keys:
            shape = (length, *features[key]["shape"])
            if np.asarray(numeric[key]).shape != shape:
                raise ValueError(f"Episode {episode_id} state feature {key} has a wrong shape.")
        for row in range(length):
            sample = {key: numeric[key][row] for key in keys}
            states[offset + row] = np.concatenate([field.extract(sample) for field in schema.state_fields])
        actions[offset:offset + length] = raw_actions
        offset += length
        digests.append(action_digest)
        print(f"NORMALIZATION_EPISODE {episode_id + 1}/{total_episodes} FRAMES {offset}/{total_frames}", flush=True)
    if offset != total_frames:
        raise ValueError("Sum of episode rows differs from the metadata frame count.")
    report = {"frames": total_frames, "episode_indices": list(range(total_episodes)),
              "source_episodes": source_episodes, "conversion_manifest": str(manifest_path),
              "conversion_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "source_action_hashes_verified": digests, "parquet_columns_read": columns,
              "image_columns_decoded": 0, "each_frame_counted_once": True,
              "train_split_provenance_verified": True}
    return {"state": states, "actions": actions}, report
