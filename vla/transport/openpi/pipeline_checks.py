"""CPU-only checks of split integrity, numeric normalization, and rollout semantics.

Uses tiny synthetic Parquet files and a fake simulator/server. These checks do not
establish that a learned policy solves Transport in a real physics environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from transport_numeric import read_train_numeric
from transport_policy import CAMERA_SLOTS, StateField, TransportSchema
from transport_rollout import initial_orientation_check, policy_observation, rollout_one, runtime_bindings, validation_episodes

CAMERAS = ("agentview", "robot0_eye_in_hand", "robot1_eye_in_hand")


def contract():
    schema = TransportSchema(
        cameras={slot: f"observation.images.{name}" for slot, name in zip(CAMERA_SLOTS, CAMERAS)},
        state_fields=tuple(StateField("observation.state", 9, f"robot{i}", "synthetic pose and gripper", tuple(range(i * 9, (i + 1) * 9))) for i in range(2)),
        action_key="action", arm_order=("robot0", "robot1"), action_description="Synthetic raw controller command",
        fps=20, audit_source="CPU checks only; not real demonstration evidence")
    schema.validate()
    fields = [{"hdf5_key": f"obs/robot{i}_{key}", "dimension": size}
              for i in range(2) for key, size in (("eef_pos", 3), ("eef_quat", 4), ("gripper_qpos", 2))]
    manifest = {"task": "synthetic transport", "state_fields": fields,
                "image_shapes_hwc": {camera: [16, 16, 3] for camera in CAMERAS}}
    return schema, manifest


def expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("Expected integrity rejection")


def check_numeric(root, schema):
    (root / "meta").mkdir()
    (root / "data").mkdir()
    info = {"fps": 20, "total_episodes": 2, "total_frames": 5, "chunks_size": 1000,
            "data_path": "data/episode_{episode_index:06d}.parquet",
            "features": {"observation.state": {"dtype": "float32", "shape": [18]},
                         "action": {"dtype": "float32", "shape": [14]}}}
    manifest = {"complete": True, "source_split": "train", "action_dim": 14,
                "action_representation": "relative_eef_controller_command", "fps": 20,
                "source_episodes": ["demo_0", "demo_1"], "total_episodes": 2, "total_frames": 5,
                "source_subset_manifest": {"splits": {"train": ["demo_0", "demo_1"], "valid": ["demo_2"], "test": ["demo_3"]}}}
    episodes, mapping, all_states, all_actions = [], [], [], []
    offset = 0
    for i, length in enumerate((2, 3)):
        state = np.arange(offset * 18, (offset + length) * 18, dtype=np.float32).reshape(length, 18)
        action = np.arange(offset * 14, (offset + length) * 14, dtype=np.float32).reshape(length, 14) / 100
        block = {"observation.state": state.tolist(), "action": action.tolist(),
                 "episode_index": [i] * length, "frame_index": list(range(length)), "index": list(range(offset, offset + length)),
                 "observation.images.agentview": [b"deliberately not a decodable PNG"] * length}
        pq.write_table(pa.Table.from_pydict(block), root / f"data/episode_{i:06d}.parquet")
        episodes.append({"episode_index": i, "length": length})
        mapping.append({"episode_index": i, "source_episode": f"demo_{i}", "length": length,
                        "from": offset, "to": offset + length, "actions_float32_sha256": hashlib.sha256(action.tobytes()).hexdigest()})
        offset += length
        all_states.append(state)
        all_actions.append(action)
    for name, value in (("info.json", info), ("conversion_manifest.json", manifest), ("source_episode_mapping.json", mapping)):
        (root / "meta" / name).write_text(json.dumps(value))
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(item) + "\n" for item in episodes))
    with patch.object(pq, "read_table", wraps=pq.read_table) as reader:
        values, report = read_train_numeric(root, schema)
        assert len(reader.call_args_list) == 2
        assert all(not any("images" in key for key in call.kwargs["columns"]) for call in reader.call_args_list)
    np.testing.assert_array_equal(values["state"], np.concatenate(all_states))
    np.testing.assert_array_equal(values["actions"], np.concatenate(all_actions))
    assert report["frames"] == 5 and report["each_frame_counted_once"] and report["image_columns_decoded"] == 0
    manifest["source_split"] = "valid"
    (root / "meta/conversion_manifest.json").write_text(json.dumps(manifest))
    expect_value_error(lambda: read_train_numeric(root, schema))
    manifest["source_split"] = "train"
    (root / "meta/conversion_manifest.json").write_text(json.dumps(manifest))
    mapping[0]["actions_float32_sha256"] = "incorrect"
    (root / "meta/source_episode_mapping.json").write_text(json.dumps(mapping))
    expect_value_error(lambda: read_train_numeric(root, schema))


class FakeEnv:
    def __init__(self, manifest, success_at=None, done_at=None):
        self.step_count, self.reset_count, self.actions = 0, 0, []
        self.success_at, self.done_at = success_at, done_at
        self.action_spec = (-np.ones(14), np.ones(14))
        frame = np.broadcast_to(np.arange(16, dtype=np.uint8)[:, None, None] * 15, (16, 16, 3)).copy()
        self.obs = {f"{camera}_image": frame.copy() for camera in CAMERAS}
        self.obs.update({item["hdf5_key"][4:]: np.arange(item["dimension"], dtype=np.float32) for item in manifest["state_fields"]})

    def _get_observations(self, force_update):
        assert force_update is True
        return self.obs

    def step(self, action):
        self.actions.append(action.copy())
        self.step_count += 1
        return self.obs, 0, self.step_count == self.done_at, {}


class FakeReplay:
    @staticmethod
    def reset_episode(env, group):
        env.reset_count += 1
        return {}

    @staticmethod
    def success_flags(env):
        return {"task": env.step_count == env.success_at,
                "trash_in_trash_bin": env.step_count >= 3, "payload_in_target_bin": env.step_count >= 7}


class FakeClient:
    def __init__(self):
        self.inputs = []

    def infer(self, observation):
        chunk = np.zeros((10, 14), dtype=np.float32)
        chunk[:, 0] = (len(self.inputs) * 10 + np.arange(10)) / 100
        chunk[:, -1] = 1.5
        self.inputs.append(dict(observation))
        return {"actions": chunk, "policy_timing": {"infer_ms": 1}}


def check_rollout(root, schema, manifest):
    masks = {"mask/train": np.array([b"demo_0"]), "mask/valid": np.array([b"demo_1", b"demo_2"]), "mask/test": np.array([b"demo_3"])}
    assert validation_episodes(masks, None, 2) == ["demo_1", "demo_2"]
    expect_value_error(lambda: validation_episodes(masks, ["demo_0"], 1))
    expect_value_error(lambda: validation_episodes(masks, ["demo_3"], 1))
    expect_value_error(lambda: validation_episodes(masks, None, 6))
    bindings = runtime_bindings(schema, {}, manifest)
    args = argparse.Namespace(output_dir=root, image_transform="none", video_skip=5, execution_horizon=5,
                              observer_resolution=0)
    imageio = types.ModuleType("imageio")
    imageio.v2 = types.ModuleType("imageio.v2")
    imageio.v2.get_writer = lambda *a, **kw: types.SimpleNamespace(append_data=lambda frame: None, close=lambda: None)
    with patch.dict(sys.modules, {"imageio": imageio, "imageio.v2": imageio.v2}):
        for success_at, done_at, expected_steps, expected_success in ((None, None, 12, False), (7, None, 7, True), (None, 2, 2, False)):
            args.output_dir = root / f"rollout_{success_at}_{done_at}"
            args.output_dir.mkdir()
            env, client, record = FakeEnv(manifest, success_at, done_at), FakeClient(), {"policy_seed": 101}
            group = {f"obs/{camera}_image": np.stack([env.obs[f"{camera}_image"]]) for camera in CAMERAS}
            rollout_one(env, group, "demo_1", client, FakeReplay, schema, manifest, bindings, args, 12, record)
            assert env.reset_count == 1 and record["completed"]
            assert record["steps"] == expected_steps and record["success"] is expected_success
            expected = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21])[:expected_steps] / 100
            np.testing.assert_allclose(np.asarray(env.actions)[:, 0], expected)
            assert np.all(np.asarray(env.actions)[:, -1] == 1)
            with np.load(record['action_trace']['path'], allow_pickle=False) as trace:
                np.testing.assert_array_equal(trace['executed_actions'], np.asarray(env.actions))
                assert trace['state_before_action'].shape == (expected_steps, schema.state_dim)
                assert np.all(trace['proposed_actions'][:, -1] == 1.5)
                assert trace['predicted_chunks'].shape == (len(client.inputs), 10, 14)
                np.testing.assert_array_equal(trace['inference_steps'], np.arange(0, expected_steps, 5))
            assert np.isclose(record["clipped_component_ratio"], 1 / 14) and record["clipped_step_ratio"] == 1
            assert "_episode_policy_seed" in client.inputs[0] and all("_episode_policy_seed" not in item for item in client.inputs[1:])
            if done_at:
                assert record["termination_reason"] == "environment_done_without_success"
            observation = policy_observation(env.obs, schema, manifest, bindings, "flipud")
            orientation = initial_orientation_check(observation, group, schema, bindings)
            assert all(item["strong_orientation_mismatch"] for item in orientation.values())
        for execution_horizon in (1, 10):
            args.output_dir = root / f"rollout_h{execution_horizon}"
            args.output_dir.mkdir()
            args.execution_horizon = execution_horizon
            env, client, record = FakeEnv(manifest), FakeClient(), {"policy_seed": 101}
            group = {f"obs/{camera}_image": np.stack([env.obs[f"{camera}_image"]]) for camera in CAMERAS}
            rollout_one(env, group, "demo_1", client, FakeReplay, schema, manifest, bindings, args, 12, record)
            expected = np.array([(i // execution_horizon) * 10 + i % execution_horizon for i in range(12)]) / 100
            np.testing.assert_allclose(np.asarray(env.actions)[:, 0], np.clip(expected, -1, 1))
            assert len(client.inputs) == int(np.ceil(12 / execution_horizon))
            assert record["execution_horizon"] == execution_horizon and env.reset_count == 1
            assert record["control_timing_comparison"]["executed_chunk_simulated_duration_ms"] == execution_horizon / 20 * 1000


def main():
    schema, manifest = contract()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        check_numeric(root, schema)
        check_rollout(root, schema, manifest)
    print("PASS: numeric-only Parquet reads count every frame once, train provenance and action hashes are enforced; validation-only rollouts reset once, execute 1/5/10 of 10 actions, distinguish done from success, clip/report actions, and detect image flips. Synthetic checks only.")


if __name__ == "__main__":
    main()
