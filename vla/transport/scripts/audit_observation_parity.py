#!/usr/bin/env python3
"""Compare recorded initial RGB/proprioception to the simulator observation path."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import imageio.v2 as imageio
import numpy as np
import robosuite

from replay_dataset import controller_audit, reset_episode
from audit_dataset import json_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.dataset, "r") as source:
        group = source["data"][args.episode]
        meta = json.loads(json_value(source["data"].attrs["env_args"]))
        kwargs = dict(meta["env_kwargs"])
        kwargs.pop("env_lang", None)
        kwargs.update(env_name=meta["env_name"], has_renderer=False, has_offscreen_renderer=True,
                      use_camera_obs=True, renderer="mjviewer")
        env = robosuite.make(**kwargs)
        try:
            reset_episode(env, group)
            obs = env._get_observations(force_update=True)
            report = {"dataset": str(args.dataset), "episode": args.episode,
                      "source_partial_range_probe": bool(source.attrs.get("partial_range_probe", False)),
                      "controller_audit": controller_audit(env), "images": {}, "state": {}, "actions": {}}
            for camera in kwargs["camera_names"]:
                recorded = group[f"obs/{camera}_image"][0]
                h, w, _ = recorded.shape
                raw = env.sim.render(width=w, height=h, camera_name=camera)
                current = np.asarray(obs[f"{camera}_image"])
                def errors(array):
                    delta = array.astype(np.float32) - recorded.astype(np.float32)
                    return {"mae": float(np.abs(delta).mean()), "exact_fraction": float((delta == 0).mean())}
                report["images"][camera] = {"shape": list(recorded.shape), "raw_render": errors(raw),
                                           "raw_render_flipped": errors(raw[::-1]), "env_observation": errors(current)}
                imageio.imwrite(args.output_dir / f"{camera}_source_raw_flip_obs.png",
                                np.concatenate([recorded, raw, raw[::-1], current], axis=1))
            for robot in (0, 1):
                for field in ("eef_pos", "eef_quat", "gripper_qpos"):
                    key = f"robot{robot}_{field}"
                    delta = np.asarray(obs[key]) - group[f"obs/{key}"][0]
                    report["state"][key] = {"max_abs_error": float(np.abs(delta).max())}
            raw_actions = group["actions"][:]
            for side in ("right", "left"):
                actions = np.concatenate([group[f"action_dict/{side}_rel_pos"][:],
                                          group[f"action_dict/{side}_rel_rot_axis_angle"][:],
                                          group[f"action_dict/{side}_gripper"][:]], axis=1)
                report["actions"][side] = {"vs_robot0_max_abs": float(np.abs(actions - raw_actions[:, :7]).max()),
                                           "vs_robot1_max_abs": float(np.abs(actions - raw_actions[:, 7:]).max())}
            path = args.output_dir / "observation_parity.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"images": report["images"], "state": report["state"], "actions": report["actions"]}, indent=2))
        finally:
            env.close()


if __name__ == "__main__": main()
