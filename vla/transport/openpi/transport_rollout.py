#!/usr/bin/env python3
"""At most five validation-only closed-loop Transport development rollouts.

The source episode supplies XML and state once through replay_dataset.reset_episode.
After that, only policy actions drive the simulator. Recorded future actions/states
are never replayed. Dataset images at t=0 are used only for an orientation diagnostic.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from transport_policy import ACTION_DIM, CAMERA_SLOTS, TransportInputs, TransportSchema, parse_rgb

ACTION_HORIZON = 10
DEFAULT_EXECUTION_HORIZON = 5


def decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def validation_episodes(source, requested, count):
    if not 1 <= count <= 5:
        raise ValueError("Development evaluation is limited to 1..5 validation episodes.")
    splits = {}
    for split in ("train", "valid", "test"):
        key = f"mask/{split}"
        if key not in source:
            raise ValueError(f"Missing explicit split mask {key}; do not use an unsplit source.")
        splits[split] = [decode(value) for value in source[key][:]]
        if len(set(splits[split])) != len(splits[split]):
            raise ValueError(f"Duplicate episode in {split} mask")
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if set(splits[left]) & set(splits[right]):
            raise ValueError(f"Episode leakage between {left} and {right}")
    names = list(requested) if requested else splits["valid"][:count]
    if not 1 <= len(names) <= 5 or len(names) != len(set(names)):
        raise ValueError("Choose one to five distinct validation episodes.")
    if not set(names).issubset(splits["valid"]):
        raise ValueError("Only mask/valid episode IDs may be evaluated by this development script.")
    return names


def transform_image(image, operation):
    image = parse_rgb(image)
    if operation == "flipud":
        image = image[::-1]
    elif operation == "rot180":
        image = image[::-1, ::-1]
    elif operation != "none":
        raise ValueError(f"Unsupported image transform {operation}")
    return np.ascontiguousarray(image)


def runtime_bindings(schema, schema_json, manifest):
    runtime = schema_json.get("runtime", {})
    image_keys = runtime.get("cameras", {})
    cameras = {}
    for slot in CAMERA_SLOTS:
        feature = schema.cameras[slot]
        name = image_keys.get(feature)
        if name is None:
            matches = [camera for camera in manifest["image_shapes_hwc"]
                       if feature == f"observation.images.{camera}"]
            if len(matches) != 1:
                raise ValueError(f"Set schema runtime.cameras[{feature!r}] to its simulator camera name.")
            name = matches[0]
        if name not in manifest["image_shapes_hwc"]:
            raise ValueError(f"Camera {name!r} is absent from the converted-data audit.")
        cameras[slot] = name
    if len(set(cameras.values())) != 3:
        raise ValueError("Three distinct simulator cameras are required.")
    return {"cameras": cameras,
            "combined_state_key": runtime.get("combined_state_key", "observation.state"),
            "state_keys": runtime.get("state_keys", {})}


def policy_observation(obs, schema, manifest, bindings, image_transform):
    values = {schema.prompt_key: manifest["task"]}
    for slot in CAMERA_SLOTS:
        camera = bindings["cameras"][slot]
        values[schema.cameras[slot]] = transform_image(obs[f"{camera}_image"], image_transform)
    pieces = []
    for item in manifest["state_fields"]:
        source_key = item["hdf5_key"]
        if not source_key.startswith("obs/"):
            raise ValueError(f"Unrecognized converted state source {source_key}")
        state = np.asarray(obs[source_key[4:]], dtype=np.float32)
        if state.shape != (item["dimension"],):
            raise ValueError(f"Runtime state shape mismatch: {source_key}: {state.shape}")
        pieces.append(state)
    values[bindings["combined_state_key"]] = np.concatenate(pieces)
    for key, obs_key in bindings["state_keys"].items():
        values[key] = np.asarray(obs[obs_key], dtype=np.float32)
    # Reuse the same input contract as training, but send unnormalized feature values.
    TransportInputs(schema)(values)
    return values


def initial_orientation_check(observation, group, schema, bindings):
    """Measure, do not silently rotate images to select a more favorable policy input."""
    report = {}
    for slot in CAMERA_SLOTS:
        camera = bindings["cameras"][slot]
        reference = np.asarray(group[f"obs/{camera}_image"][0])
        actual = np.asarray(observation[schema.cameras[slot]])
        if actual.shape != reference.shape:
            raise ValueError(f"Initial image shape mismatch for {camera}: {actual.shape} vs {reference.shape}")
        variants = {"none": actual, "flipud": actual[::-1], "rot180": actual[::-1, ::-1]}
        errors = {name: float(np.sqrt(np.mean((frame.astype(np.float64) - reference) ** 2)))
                  for name, frame in variants.items()}
        best = min(errors, key=errors.get)
        # A large improvement under flipping indicates a convention mismatch. Mild
        # renderer differences alone are recorded and do not trigger an auto-transform.
        mismatch = best != "none" and errors["none"] > 5 and errors[best] < .5 * errors["none"]
        report[camera] = {"rmse_uint8": errors, "best_orientation_after_configured_transform": best,
                          "strong_orientation_mismatch": mismatch,
                          "direct_pixels_equal": bool(np.array_equal(actual, reference))}
    return report


def patch_websocket_headers():
    import websockets.sync.client
    connect = websockets.sync.client.connect
    if "additional_headers" not in inspect.signature(connect).parameters:
        def compatible_connect(uri, *, additional_headers=None, **kwargs):
            return connect(uri, extra_headers=additional_headers, **kwargs)
        websockets.sync.client.connect = compatible_connect


def load_replay_module(path):
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("transport_replay_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def video_panel(observation, schema):
    frames = [observation[schema.cameras[slot]] for slot in CAMERA_SLOTS]
    if len({frame.shape[0] for frame in frames}) != 1:
        raise ValueError("Video panel requires equal camera heights; keep source camera sizes consistent.")
    return np.concatenate(frames, axis=1)


def latency_summary(values):
    if not values:
        return {"count": 0, "first_ms": None, "steady_p50_ms": None, "steady_p95_ms": None}
    return {"count": len(values), "first_ms": values[0],
            "steady_p50_ms": float(np.median(values[1:])) if len(values) > 1 else None,
            "steady_p95_ms": float(np.percentile(values[1:], 95)) if len(values) > 1 else None}


def update_diagnostics(env, obs, step, record):
    """Read-only contact/distance evidence; never used for policy inputs or success."""
    diagnostic = record.setdefault("diagnostics", {
        "payload_finger_contacts": {}, "minimum_object_eef_distance_m": {}, "diagnostic_unavailable": {},
        "interpretation": "Grasp predicate means finger-pad contacts; distances and contacts do not prove stable holding, release, or handover."})
    for arm in (0, 1):
        key = f"robot{arm}"
        if key in diagnostic["diagnostic_unavailable"]:
            continue
        try:
            active = bool(env._check_grasp(env.robots[arm].gripper, env.transport.payload))
            item = diagnostic["payload_finger_contacts"].setdefault(key, {
                "first_step": None, "last_step": None, "positive_observations": 0,
                "contact_onsets": [], "total_onsets": 0, "previous_active": False})
            if active:
                if item["first_step"] is None:
                    item["first_step"] = step
                item["last_step"] = step
                item["positive_observations"] += 1
                if not item["previous_active"]:
                    item["total_onsets"] += 1
                    if len(item["contact_onsets"]) < 50:
                        item["contact_onsets"].append(step)
            item["previous_active"] = active
        except Exception as error:
            diagnostic["payload_finger_contacts"][key] = None
            diagnostic["diagnostic_unavailable"][key] = f"{type(error).__name__}: {str(error)[:200]}"
    for key in ("gripper0_to_payload", "gripper1_to_payload", "gripper0_to_lid_handle", "gripper1_to_trash"):
        if key in diagnostic["diagnostic_unavailable"]:
            continue
        try:
            vector = np.asarray(obs[key], dtype=np.float64)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError("Expected finite 3-D world-frame object-minus-EEF vector.")
            distance = float(np.linalg.norm(vector))
            previous = diagnostic["minimum_object_eef_distance_m"].get(key)
            if previous is None or distance < previous["distance"]:
                diagnostic["minimum_object_eef_distance_m"][key] = {"distance": distance, "step": step}
        except Exception as error:
            diagnostic["minimum_object_eef_distance_m"][key] = None
            diagnostic["diagnostic_unavailable"][key] = f"{type(error).__name__}: {str(error)[:200]}"
    # Object world poses are privileged simulation diagnostics only. They never
    # enter policy_observation and never replace the official success predicate.
    physical = diagnostic.setdefault('object_progress', {
        'objects': {}, 'unavailable': {},
        'interpretation': 'Body-origin displacement and finger-pad contacts only; lift/contact alone does not certify opening, stable grasp, transfer, or release.'})
    for name in ('lid', 'payload', 'trash'):
        if name in physical['unavailable']:
            continue
        try:
            obj = getattr(env.transport, name)
            body_id = env.sim.model.body_name2id(obj.root_body)
            xyz = np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64).copy()
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError('Expected finite world XYZ body position')
            item = physical['objects'].setdefault(name, {
                'initial_world_xyz_m': xyz.tolist(), 'final_world_xyz_m': xyz.tolist(),
                'maximum_height_gain_m': 0.0, 'maximum_displacement_m': 0.0,
                'finger_contacts': {f'robot{arm}': {'positive_observations': 0, 'first_step': None}
                                    for arm in (0, 1)}})
            delta = xyz - np.asarray(item['initial_world_xyz_m'])
            item['final_world_xyz_m'] = xyz.tolist()
            item['maximum_height_gain_m'] = max(item['maximum_height_gain_m'], float(delta[2]))
            item['maximum_displacement_m'] = max(item['maximum_displacement_m'], float(np.linalg.norm(delta)))
            for arm in (0, 1):
                if env._check_grasp(env.robots[arm].gripper, obj):
                    contact = item['finger_contacts'][f'robot{arm}']
                    contact['positive_observations'] += 1
                    if contact['first_step'] is None:
                        contact['first_step'] = step
        except Exception as error:
            physical['unavailable'][name] = f'{type(error).__name__}: {str(error)[:200]}'


def rollout_one(env, group, name, client, replay, schema, manifest, bindings, args, horizon, record):
    import imageio.v2 as imageio
    if args.execution_horizon not in (1, 5, 10):
        raise ValueError("Execution horizon must be 1, 5, or 10; policy output remains 10 actions.")
    started = time.monotonic()
    writer = None
    observer_writer = None
    network_ms, server_ms = [], []
    component_clips = np.zeros(ACTION_DIM, dtype=np.int64)
    clipped_steps = 0
    # Debug evidence only: these arrays never feed back into the controller.
    trace = {key: [] for key in ('proposed_actions', 'executed_actions',
                                'state_before_action', 'inference_steps', 'predicted_chunks')}
    record.update(episode=name, split="valid",
                  evaluation_kind="observer_diagnostic" if args.observer_resolution else "development", completed=False,
                  success=False, steps=0, first_success_step=None, first_stage_steps={},
                  done_seen=False, termination_reason=None, image_transform=args.image_transform,
                  action_horizon=ACTION_HORIZON, execution_horizon=args.execution_horizon)
    try:
        # The only source state initialization. No set_state/reset occurs inside the policy loop.
        record["ep_meta"] = replay.reset_episode(env, group)
        obs = env._get_observations(force_update=True)
        initial = replay.success_flags(env)
        record["initial_flags"] = initial
        update_diagnostics(env, obs, 0, record)
        if initial["task"]:
            raise ValueError(f"Episode {name} starts already successful; invalid development initial state.")
        observation = policy_observation(obs, schema, manifest, bindings, args.image_transform)
        record["initial_image_alignment"] = initial_orientation_check(observation, group, schema, bindings)
        if any(value["strong_orientation_mismatch"] for value in record["initial_image_alignment"].values()):
            raise ValueError("Image convention differs from converted demonstrations; inspect initial_image_alignment before choosing an explicit --image-transform.")
        record["orientation_evidence"] = "Initial live images compared with stored frame 0; no strong flip mismatch detected. See RMSE values; this is not an exact simulator-state reproduction claim."
        for flag, value in initial.items():
            if value:
                record["first_stage_steps"][flag] = 0
        low, high = (np.asarray(value, dtype=np.float32) for value in env.action_spec)
        if low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,):
            raise ValueError(f"Expected 14-D simulator action space, got {low.shape}/{high.shape}")
        if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low >= high):
            raise ValueError("Simulator action bounds must be finite and ordered.")
        record["action_bounds"] = {"low": low.tolist(), "high": high.tolist()}
        video_path = args.output_dir / f"{name}_policy_seed_{record['policy_seed']}.mp4"
        if video_path.exists():
            raise FileExistsError(video_path)
        writer = imageio.get_writer(video_path, fps=max(1, round(schema.fps / args.video_skip)),
                                    codec="libx264", quality=7, macro_block_size=1)
        record["video_path"] = str(video_path.resolve())
        writer.append_data(video_panel(observation, schema))
        if args.observer_resolution:
            observer_path = args.output_dir / f"{name}_observer_seed_{record['policy_seed']}.mp4"
            if observer_path.exists():
                raise FileExistsError(observer_path)
            observer_writer = imageio.get_writer(observer_path, fps=max(1, round(schema.fps / args.video_skip)),
                                                 codec="libx264", quality=8)
            record["observer_video"] = {"path": str(observer_path.resolve()), "camera": "agentview",
                                        "resolution": args.observer_resolution,
                                        "used_as_policy_input": False,
                                        "purpose": "Additional diagnostic recording; rendering changes wall time, not the observation supplied to the policy."}
            observer_writer.append_data(np.ascontiguousarray(np.flipud(env.sim.render(
                camera_name="agentview", height=args.observer_resolution, width=args.observer_resolution))))
        queued = collections.deque()
        for step in range(horizon):
            if not queued:
                observation = policy_observation(obs, schema, manifest, bindings, args.image_transform)
                if not network_ms:
                    observation["_episode_policy_seed"] = record["policy_seed"]
                tick = time.perf_counter()
                prediction = client.infer(observation)
                network_ms.append((time.perf_counter() - tick) * 1000)
                timing = prediction.get("policy_timing", {}).get("infer_ms")
                if timing is not None:
                    server_ms.append(float(timing))
                chunk = np.asarray(prediction["actions"], dtype=np.float32)
                if chunk.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(chunk).all():
                    raise ValueError(f"Require finite (10,14) policy chunk, got {chunk.shape}")
                trace['inference_steps'].append(step)
                trace['predicted_chunks'].append(chunk.copy())
                queued.extend(chunk[:args.execution_horizon])
            proposed = queued.popleft()
            clipped = (proposed < low) | (proposed > high)
            action = np.clip(proposed, low, high)
            state_before = np.concatenate([np.asarray(obs[item['hdf5_key'][4:]], dtype=np.float32)
                                           for item in manifest['state_fields']])
            obs, _, done, _ = env.step(action)
            trace['proposed_actions'].append(proposed.copy())
            trace['executed_actions'].append(action.copy())
            trace['state_before_action'].append(state_before)
            component_clips += clipped
            clipped_steps += int(clipped.any())
            record["steps"] = step + 1
            record["done_seen"] = bool(record["done_seen"] or done)
            flags = replay.success_flags(env)
            record["final_flags"] = flags
            update_diagnostics(env, obs, step + 1, record)
            for flag, value in flags.items():
                if value and flag not in record["first_stage_steps"]:
                    record["first_stage_steps"][flag] = step + 1
            if flags["task"]:
                record["success"] = True
                record["first_success_step"] = step + 1
                record["termination_reason"] = "environment_success_predicate"
            elif done:
                record["termination_reason"] = "environment_done_without_success"
            elif step + 1 == horizon:
                record["termination_reason"] = "development_horizon"
            if (step + 1) % args.video_skip == 0 or record["termination_reason"]:
                observation = policy_observation(obs, schema, manifest, bindings, args.image_transform)
                writer.append_data(video_panel(observation, schema))
                if observer_writer is not None:
                    observer_writer.append_data(np.ascontiguousarray(np.flipud(env.sim.render(
                        camera_name="agentview", height=args.observer_resolution, width=args.observer_resolution))))
            if record["termination_reason"]:
                break
        record["completed"] = True
    except Exception:
        record["error"] = traceback.format_exc()
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - started
        record["roundtrip_latency"] = latency_summary(network_ms)
        record["reported_server_latency"] = latency_summary(server_ms)
        control_budget_ms = args.execution_horizon / schema.fps * 1000
        p95_ms = record["roundtrip_latency"]["steady_p95_ms"]
        record["control_timing_comparison"] = {
            "control_frequency_hz": schema.fps,
            "executed_chunk_simulated_duration_ms": control_budget_ms,
            "steady_roundtrip_p95_to_chunk_duration_ratio": p95_ms / control_budget_ms if p95_ms is not None else None,
            "interpretation": "Offline synchronous simulation pauses for inference. Comparing latency with executed-chunk simulated duration does not establish real-time deployment or an asynchronous control deadline."}
        record["clipped_action_components_by_dimension"] = component_clips.tolist()
        record["clipped_component_ratio"] = float(component_clips.sum() / (record["steps"] * ACTION_DIM)) if record["steps"] else None
        record["clipped_step_ratio"] = float(clipped_steps / record["steps"]) if record["steps"] else None
        if trace['executed_actions']:
            trace_path = args.output_dir / f"{name}_trace_seed_{record['policy_seed']}.npz"
            with trace_path.open('xb') as trace_file:
                np.savez_compressed(trace_file, **{key: np.asarray(value) for key, value in trace.items()})
            record['action_trace'] = {
                'path': str(trace_path.resolve()), 'completed_action_steps': len(trace['executed_actions']),
                'inference_count': len(trace['inference_steps']), 'step_index_base': 0,
                'state_timing': 'state_before_action precedes the corresponding executed action',
                'action_units': '14 normalized OSC_POSE/gripper commands; not joint angles or measured motion',
                'purpose': 'Numerical diagnostic evidence, including unused predicted actions; not policy input'}
        close_error = None
        for video_writer, error_key in ((writer, "video_close_error"),
                                        (observer_writer, "observer_video_close_error")):
            if video_writer is not None:
                try:
                    video_writer.close()
                except Exception as error:
                    record[error_key] = traceback.format_exc()
                    record["completed"] = False
                    close_error = error
        if close_error is not None and "error" not in record:
            raise close_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Split HDF5 retaining per-episode XML and states")
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-module", type=Path,
                        default=Path(__file__).resolve().parents[1] / "scripts" / "replay_dataset.py")
    parser.add_argument("--episodes", nargs="+")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--horizon", type=int, help="Default: saved env horizon, otherwise 1200")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model-label", default="transport_smoke")
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument("--observer-resolution", type=int, choices=(0, 384), default=0,
                        help="Optional extra agentview recording for a separately labelled diagnostic rerun; policy cameras remain unchanged")
    parser.add_argument("--execution-horizon", type=int, choices=(1, 5, 10), default=DEFAULT_EXECUTION_HORIZON,
                        help="Actions executed before observing again; keep 5 for paired model comparisons, use fresh outputs for ablations")
    parser.add_argument("--image-transform", choices=("none", "flipud", "rot180"),
                        help="Use audited schema runtime.image_transform, otherwise none; never infer orientation automatically")
    args = parser.parse_args()
    if args.video_skip < 1 or (args.horizon is not None and args.horizon < 1):
        raise ValueError("Video skip and horizon must be positive.")
    if args.output_dir.exists():
        raise FileExistsError(f"Use a fresh development output directory: {args.output_dir}")
    schema = TransportSchema.load(args.schema)
    schema_json = json.loads(args.schema.read_text())
    audited_transform = schema_json.get("runtime", {}).get("image_transform")
    if audited_transform is not None and args.image_transform is not None and args.image_transform != audited_transform:
        raise ValueError("CLI image transform differs from the audited schema runtime convention.")
    args.image_transform = args.image_transform or audited_transform or "none"
    if args.image_transform not in ("none", "flipud", "rot180"):
        raise ValueError("Unknown audited runtime image transform.")
    manifest = json.loads(args.conversion_manifest.read_text())
    if not manifest.get("complete") or manifest.get("action_representation") != "relative_eef_controller_command":
        raise ValueError("Require a completed converted-data/controller audit.")
    if not np.isclose(float(manifest["fps"]), schema.fps):
        raise ValueError("Schema and converted-data control frequency differ.")
    if manifest.get("action_dim") != ACTION_DIM or sum(item["dimension"] for item in manifest["state_fields"]) != manifest.get("state_dim"):
        raise ValueError("Converted numeric dimensions are inconsistent.")
    if manifest.get("images_flipped") is not False or manifest.get("images_resized") is not False:
        raise ValueError("This runtime expects the converter's unflipped, unresized source RGB.")
    bindings = runtime_bindings(schema, schema_json, manifest)
    os.environ.setdefault("MUJOCO_GL", "egl")
    replay = load_replay_module(args.replay_module)
    import h5py
    import robosuite
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    patch_websocket_headers()
    args.output_dir.mkdir(parents=True)
    report = {"model_label": args.model_label, "dataset": str(args.dataset.resolve()),
              "evaluation_kind": "observer_diagnostic" if args.observer_resolution else "development",
              "conversion_manifest": str(args.conversion_manifest.resolve()), "schema": schema.summary(),
              "runtime_bindings": bindings, "versions": replay.versions(), "episodes": [],
              "action_horizon": ACTION_HORIZON, "execution_horizon": args.execution_horizon,
              "observer_resolution": args.observer_resolution,
              "observer_policy_input": False,
              "image_transform": args.image_transform, "success_source": "env._check_success(); done is not success",
              "diagnostic_observation_boundary": schema_json.get("runtime", {}).get("diagnostics_boundary", "Optional diagnostic observations may be absent; missing values remain null and never change task success."),
              "success_boundary": "Official Transport success is payload contact with target-bin bottom AND trash contact with trash-bin bottom; it does not certify stable release or the handover sequence.",
              "claim": "Validation-only exploratory closed-loop policy evaluation; not a final test or Sim2Real result."}
    env = client = None
    try:
        with h5py.File(args.dataset, "r") as source:
            names = validation_episodes(source, args.episodes, args.n)
            metadata = json.loads(decode(source["data"].attrs["env_args"]))
            if metadata != manifest["env_metadata"] or metadata["env_name"] != "TwoArmTransport":
                raise ValueError("Evaluation environment metadata differs from converted data.")
            kwargs = dict(metadata["env_kwargs"])
            kwargs.pop("env_lang", None)
            horizon = args.horizon or int(kwargs.get("horizon") or 1200)
            if horizon <= 0:
                raise ValueError("Invalid stored horizon; specify --horizon explicitly.")
            names_camera = [bindings["cameras"][slot] for slot in CAMERA_SLOTS]
            shapes = [manifest["image_shapes_hwc"][name] for name in names_camera]
            kwargs.update(env_name=metadata["env_name"], has_renderer=False, has_offscreen_renderer=True,
                          use_camera_obs=True, camera_names=names_camera,
                          camera_heights=[shape[0] for shape in shapes], camera_widths=[shape[1] for shape in shapes],
                          camera_depths=False, renderer="mjviewer", ignore_done=True, horizon=horizon)
            report.update(horizon=horizon, effective_env_kwargs=kwargs, selected_validation_episodes=names)
            np.random.seed(args.seed)
            env = robosuite.make(**kwargs)
            report["robosuite_image_convention"] = robosuite.macros.IMAGE_CONVENTION
            if not np.isclose(float(env.control_freq), schema.fps):
                raise ValueError("Actual simulator control frequency differs from training.")
            client = WebsocketClientPolicy("127.0.0.1", args.port)
            server = client.get_server_metadata()
            report["server_metadata"] = server
            for key, expected in (("action_dim", ACTION_DIM), ("state_dim", schema.state_dim),
                                  ("discrete_state_input", True), ("cameras", schema.cameras)):
                if server.get(key) != expected:
                    raise ValueError(f"Policy server {key} does not match evaluation schema.")
            if tuple(server.get("arm_order", ())) != schema.arm_order:
                raise ValueError("Policy server arm order differs from evaluation.")
            for key in ("state_fields", "action_description", "fps", "audit_source"):
                if json.dumps(server.get(key), sort_keys=True) != json.dumps(schema.summary()[key], sort_keys=True):
                    raise ValueError(f"Policy server {key} differs from the audited evaluation schema.")
            for index, name in enumerate(names):
                record = {"policy_seed": args.seed + index}
                report["episodes"].append(record)
                rollout_one(env, source["data"][name], name, client, replay, schema,
                            manifest, bindings, args, horizon, record)
                (args.output_dir / f"{name}.json").write_text(json.dumps(record, indent=2))
                print("TRANSPORT_ROLLOUT " + json.dumps(record), flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        for resource in ([client._ws] if client is not None and hasattr(client, "_ws") else []) + ([env] if env is not None else []):
            try:
                resource.close()
            except Exception:
                report.setdefault("cleanup_errors", []).append(traceback.format_exc())
        completed = [record for record in report["episodes"] if record.get("completed")]
        report.update(attempted_count=len(report["episodes"]), completed_count=len(completed),
                      successes=sum(record["success"] for record in completed))
        report["success_rate"] = report["successes"] / len(completed) if completed else None
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print("TRANSPORT_ROLLOUT_SUMMARY " + json.dumps({key: report[key] for key in
              ("attempted_count", "completed_count", "successes", "success_rate")}), flush=True)


if __name__ == "__main__":
    main()
