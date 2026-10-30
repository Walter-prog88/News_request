#!/usr/bin/env python3
"""Verify a demo by resetting ONCE and stepping its recorded control actions.

action mode is a control replay; states mode is a diagnostic reconstruction and
NEVER counts as evidence that a controller or policy can complete the task.
No policy inference is performed by this script.
"""
import argparse
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import imageio.v2 as imageio
import numpy as np
import robosuite
import dexmimicgen  # noqa: F401 - registers the benchmark environments

from audit_dataset import episode_key, json_value


def versions():
    result = {}
    for name in ("mujoco", "robosuite", "dexmimicgen", "numpy", "h5py", "numba"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def success_flags(env):
    flags = {"task": bool(env._check_success())}
    if hasattr(env, "transport"):
        flags["payload_in_target_bin"] = bool(env.transport.payload_in_target_bin)
        flags["trash_in_trash_bin"] = bool(env.transport.trash_in_trash_bin)
    return flags


def controller_audit(env):
    """Capture the actual initialized controller, including config defaults."""
    report = {"control_freq": env.control_freq, "robots": []}
    offset = 0
    for robot in env.robots:
        item = {"name": robot.name, "idn": robot.idn,
                "action_range_global": [offset, offset + robot.action_dim],
                "parts": {}, "grippers": {}}
        split = robot.composite_controller._action_split_indexes
        for name, controller in robot.part_controllers.items():
            local_range = split.get(name)
            part = {"class": f"{type(controller).__module__}.{type(controller).__name__}",
                    "action_range_local": list(local_range) if local_range is not None else None}
            for field in ("input_type", "input_ref_frame", "control_dim", "input_min", "input_max",
                          "output_min", "output_max", "impedance_mode", "_goal_update_mode"):
                if hasattr(controller, field):
                    part[field] = json_value(getattr(controller, field))
            item["parts"][name] = part
        for name, gripper in robot.gripper.items():
            item["grippers"][name] = {
                "class": f"{type(gripper).__module__}.{type(gripper).__name__}",
                "dof": gripper.dof,
                "format_action_source": inspect.getsource(gripper.format_action)}
        report["robots"].append(item)
        offset += robot.action_dim
    report["action_dim"] = offset
    return report


def reset_episode(env, group):
    """Match official reset_to: load episode XML and physics state once."""
    xml = json_value(group.attrs["model_file"])
    ep_meta_raw = json_value(group.attrs.get("ep_meta", "{}"))
    ep_meta = json.loads(ep_meta_raw) if isinstance(ep_meta_raw, str) else ep_meta_raw
    if hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)
    elif hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(xml))
    env.sim.reset()
    env.sim.set_state_from_flattened(group["states"][0])
    env.sim.forward()
    for callback in ("update_sites", "update_state"):
        if hasattr(env, callback):
            getattr(env, callback)()
    return ep_meta


def render_panel(env, cameras, size):
    # Raw MuJoCo offscreen buffers use the OpenGL origin (bottom-left).
    frames = [env.sim.render(width=size, height=size, camera_name=name)[::-1]
              for name in cameras]
    return np.concatenate(frames, axis=1)


def replay_one(env, group, name, args):
    start = time.monotonic()
    ep_meta = reset_episode(env, group)
    actions = np.asarray(group["actions"])
    states = group["states"]
    low, high = map(np.asarray, env.action_spec)
    if actions.ndim != 2 or actions.shape[1:] != low.shape:
        raise ValueError(f"{name}: actions {actions.shape} vs env.action_spec {low.shape}")
    if not np.isfinite(actions).all():
        raise ValueError(f"{name}: nonfinite actions")
    if states.shape[0] != len(actions):
        raise ValueError(f"{name}: states and actions length mismatch")
    steps = min(len(actions), args.max_steps) if args.max_steps else len(actions)
    report = {"episode": name, "mode": args.mode, "ep_meta": ep_meta,
              "total_recorded_steps": len(actions), "steps_executed": 0,
              "truncated": steps < len(actions), "action_dim": len(low),
              "actions_outside_env_bounds": int(np.sum(np.any((actions < low - 1e-5) | (actions > high + 1e-5), axis=1))),
              "initial_flags": success_flags(env), "first_success_step": None,
              "first_stage_steps": {}, "state_error_checks": [],
              "control_replay_verified": False}
    writer = None
    try:
        if args.video_dir:
            args.video_dir.mkdir(parents=True, exist_ok=True)
            path = args.video_dir / f"{name}_{args.mode}.mp4"
            if path.exists() and not args.overwrite:
                raise FileExistsError(path)
            writer = imageio.get_writer(path, fps=max(1, round(env.control_freq / args.video_skip)),
                                        codec="libx264", quality=7)
            report["video_path"] = str(path.resolve())
            writer.append_data(render_panel(env, args.cameras, args.image_size))
        for step in range(steps):
            if args.mode == "action":
                # Only this path tests whether recorded controls reproduce a task.
                env.step(actions[step])
            else:
                # Deliberately separate diagnostic state reconstruction.
                env.sim.set_state_from_flattened(states[step])
                env.sim.forward()
            report["steps_executed"] = step + 1
            flags = success_flags(env)
            for key, value in flags.items():
                if value and key not in report["first_stage_steps"]:
                    report["first_stage_steps"][key] = step + 1
            if flags["task"] and report["first_success_step"] is None:
                report["first_success_step"] = step + 1
            if args.mode == "action" and step + 1 < len(states) and (step < 5 or (step + 1) % args.compare_every == 0):
                actual = np.asarray(env.sim.get_state().flatten())
                expected = np.asarray(states[step + 1])
                if actual.shape != expected.shape:
                    raise ValueError(f"State shape differs: {actual.shape} vs {expected.shape}")
                diff = actual - expected
                report["state_error_checks"].append({"after_action": step,
                    "l2": float(np.linalg.norm(diff)), "linf": float(np.max(np.abs(diff)))})
            if writer is not None and (step + 1) % args.video_skip == 0:
                writer.append_data(render_panel(env, args.cameras, args.image_size))
        report["final_flags"] = success_flags(env)
        report["success_ever"] = report["first_success_step"] is not None
        report["control_replay_verified"] = bool(args.mode == "action" and not report["truncated"] and report["success_ever"])
    finally:
        if writer is not None:
            writer.close()
        report["wall_seconds"] = round(time.monotonic() - start, 3)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("action", "states"), default="action")
    parser.add_argument("--episodes", nargs="+")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--compare-every", type=int, default=50)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--cameras", nargs="+", default=["agentview", "robot0_eye_in_hand", "robot1_eye_in_hand"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.compare_every, args.video_skip, args.image_size) < 1 or args.max_steps < 0 or args.n < 1:
        raise ValueError("Invalid positive/range arguments")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    report = {"dataset": str(args.dataset.resolve()), "mode": args.mode,
              "versions": versions(), "episodes": [],
              "claim": "recorded action control replay" if args.mode == "action" else "state reconstruction ONLY; not control validation",
              "action_source": "raw HDF5 data/<episode>/actions, unchanged; no policy inference"}
    env = None
    try:
        with h5py.File(args.dataset, "r") as src:
            metadata = json.loads(json_value(src["data"].attrs["env_args"]))
            report["original_env_metadata"] = metadata
            if metadata["env_name"] != "TwoArmTransport":
                raise ValueError(f"Expected TwoArmTransport, got {metadata['env_name']}")
            kwargs = dict(metadata["env_kwargs"])
            kwargs.pop("env_lang", None)
            kwargs.update(env_name=metadata["env_name"], has_renderer=False,
                          has_offscreen_renderer=bool(args.video_dir), use_camera_obs=False,
                          renderer="mjviewer", ignore_done=True)
            report["effective_env_kwargs"] = kwargs
            env = robosuite.make(**kwargs)
            report["initialized_controller_audit"] = controller_audit(env)
            names = args.episodes or sorted(src["data"].keys(), key=episode_key)[:args.n]
            for name in names:
                result = replay_one(env, src["data"][name], name, args)
                report["episodes"].append(result)
                print(json.dumps({"episode": name, "mode": args.mode,
                                  "verified": result["control_replay_verified"],
                                  "success_ever": result["success_ever"],
                                  "final_flags": result["final_flags"],
                                  "wall_seconds": result["wall_seconds"]}), flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        if env is not None:
            env.close()
        report["verified_count"] = sum(x["control_replay_verified"] for x in report["episodes"])
        report["episode_count"] = len(report["episodes"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"Report: {args.output}", flush=True)
    if args.mode == "action" and report["verified_count"] != report["episode_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
