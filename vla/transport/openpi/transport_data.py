"""Train-only normalization and observable proof that both arms reach pi05 tokens."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from openpi import transforms
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import normalize
from openpi.training import data_loader

from transport_config import add_common_arguments, config_from_args, record_run, results_directory
from transport_numeric import read_train_numeric
from transport_policy import ACTION_DIM, CAMERA_SLOTS, TransportInputs


def load_real_dataset(config, schema, *, horizon=None):
    metadata = LeRobotDatasetMetadata(config.data.repo_id)
    if not np.isclose(float(metadata.fps), schema.fps):
        raise ValueError(f"Schema fps={schema.fps} differs from dataset fps={metadata.fps}")
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(
        data_config, config.model.action_horizon if horizon is None else horizon, config.model)
    if len(dataset) < 2:
        raise ValueError("Require a nonempty real demonstration dataset.")
    return dataset, data_config


def compute_norms(config, schema, *, output_dir: Path):
    output_path = config.assets_dirs / config.data.repo_id
    if (output_path / "norm_stats.json").exists():
        raise FileExistsError(f"Norm stats already exist: {output_path}; use a fresh experiment root.")
    metadata = LeRobotDatasetMetadata(config.data.repo_id)
    if not np.isclose(float(metadata.fps), schema.fps):
        raise ValueError("Schema and LeRobot metadata fps differ.")
    # No LeRobotDataset.__getitem__: it would decode three PNGs for every row.
    # Read numeric columns and provenance once per episode, with no action chunking.
    values, provenance = read_train_numeric(metadata.root, schema)
    # Exact statistics on small demonstration sets, including every final partial batch.
    stats = {key: normalize.NormStats(mean=value.mean(axis=0), std=value.std(axis=0),
                                    q01=np.quantile(value, .01, axis=0),
                                    q99=np.quantile(value, .99, axis=0))
             for key, value in values.items()}
    normalize.save(output_path, stats)
    report = {**provenance, "repo_id": config.data.repo_id,
              "normalization": "exact train-repo per-frame q01/q99 from numeric Parquet columns",
              "norm_stats": str(output_path / "norm_stats.json"), "schema": schema.summary(),
              "constant_dimensions": {key: np.flatnonzero(value.std(axis=0) < 1e-8).tolist()
                                      for key, value in values.items()},
              "scope": "Only this explicitly supplied train repository was read; split provenance is recorded by the converter."}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "normalization_audit.json").write_text(json.dumps(report, indent=2))
    return report


def audit_model_input(config, schema, *, sample_index=0):
    if not config.model.discrete_state_input:
        raise ValueError("Proprioception is disabled.")
    dataset, data_config = load_real_dataset(config, schema)
    if data_config.norm_stats is None:
        raise ValueError("Train-only normalization must exist before input audit or training.")
    for key, dimension in (("state", schema.state_dim), ("actions", ACTION_DIM)):
        stats = data_config.norm_stats[key]
        if stats.q01 is None or stats.q99 is None or stats.mean.shape != (dimension,):
            raise ValueError(f"Incompatible norm stats for {key}")
    raw = dataset[sample_index]
    parsed = TransportInputs(schema)(raw)
    normalized = transforms.Normalize(data_config.norm_stats, use_quantiles=True)(copy.deepcopy(parsed))
    model_transform = transforms.compose(data_config.model_transforms.inputs)
    model_input = model_transform(copy.deepcopy(normalized))
    if model_input["state"].shape != (32,):
        raise ValueError("State must be padded to the base checkpoint's 32-D interface.")
    if model_input["actions"].shape != (config.model.action_horizon, 32):
        raise ValueError("Unexpected padded action shape.")
    if not all(bool(model_input["image_mask"][key]) for key in CAMERA_SLOTS):
        raise ValueError("All three real cameras must be active.")
    # Detect silent loss of late proprioceptive fields to tokenizer truncation.
    full_tokens, full_mask = PaligemmaTokenizer(4096).tokenize(normalized["prompt"], normalized["state"])
    full_length = int(full_mask.sum())
    if full_length > config.model.max_token_len:
        raise ValueError(f"Task + both-arm state uses {full_length} tokens; exceeds {config.model.max_token_len}.")
    changes = {}
    for arm in schema.arm_order:
        variant = copy.deepcopy(normalized)
        indices = schema.arm_state_indices(arm)
        variant["state"][indices] = np.where(variant["state"][indices] > 0, -.75, .75)
        modified = model_transform(variant)
        changed = int(np.count_nonzero(model_input["tokenized_prompt"] != modified["tokenized_prompt"]))
        if not changed:
            raise ValueError(f"Changing {arm} proprioception did not reach model language tokens.")
        changes[arm] = changed
    digest = hashlib.sha256(parsed["state"].tobytes() + parsed["actions"].tobytes()).hexdigest()
    episode_index = int(np.asarray(raw["episode_index"]).item())
    frame_index = int(np.asarray(raw["frame_index"]).item())
    metadata = LeRobotDatasetMetadata(config.data.repo_id)
    mapping = json.loads((metadata.root / "meta/source_episode_mapping.json").read_text())
    source_episode = next(item["source_episode"] for item in mapping if item["episode_index"] == episode_index)
    return {"repo_id": config.data.repo_id, "sample_index": sample_index,
            "episode_index": episode_index, "frame_index": frame_index, "source_episode": source_episode,
            "state_dim_before_padding": schema.state_dim, "state_dim_after_padding": 32,
            "action_shape_before_padding": list(parsed["actions"].shape),
            "action_shape_after_padding": list(model_input["actions"].shape),
            "three_real_cameras_active": True, "discrete_state_input": True,
            "untruncated_task_state_tokens": full_length, "proprio_token_changes_by_arm": changes,
            "state_action_sha256": digest,
            "interpretation": "A structural counterfactual verifies both arms reach conditioning tokens; it does not prove learned physical sensitivity."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--mode", choices=("normalize", "audit"), default="audit")
    args = parser.parse_args()
    config, schema = config_from_args(args)
    output_dir = results_directory(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "normalize":
        print(json.dumps(compute_norms(config, schema, output_dir=output_dir), indent=2), flush=True)
    report = audit_model_input(config, schema)
    record_run(args, schema, output_dir / "data_run_config.json")
    (output_dir / "model_input_audit.json").write_text(json.dumps(report, indent=2))
    print("MODEL_INPUT_AUDIT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
