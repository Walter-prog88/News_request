"""Load Transport base/LoRA policy; optionally verify reloaded action inference."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import jax
import numpy as np
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import normalize

from transport_config import add_common_arguments, config_from_args, results_directory
from transport_data import audit_model_input, load_real_dataset
from transport_policy import ACTION_DIM


class EpisodeSeedPolicy:
    def __init__(self, policy):
        self.policy = policy

    def infer(self, observation):
        observation = dict(observation)
        seed = observation.pop("_episode_policy_seed", None)
        if seed is not None:
            self.policy._rng = jax.random.key(int(seed))
        return self.policy.infer(observation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--model", choices=("base", "lora"), default="lora")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    config, schema = config_from_args(args, lora=args.model == "lora")
    if args.model == "base":
        checkpoint = args.base_checkpoint
        # Base has no Transport stats. Use exactly the train-only stats used for adaptation.
        norm = normalize.load(config.assets_dirs / args.repo_id)
    else:
        checkpoint = args.checkpoint or config.checkpoint_dir / str(config.num_train_steps - 1)
        if not (checkpoint / "transport_run.json").exists():
            raise FileNotFoundError("Transport checkpoint lacks its recorded run/schema sidecar.")
        recorded = json.loads((checkpoint / "transport_run.json").read_text())
        if recorded["schema_file"] != json.loads(args.schema.read_text()):
            raise ValueError("Serving schema differs from the training checkpoint schema.")
        if recorded["arguments"]["horizon"] != args.horizon or recorded["arguments"]["repo_id"] != args.repo_id:
            raise ValueError("Serving action horizon or normalization repository differs from training.")
        norm = None  # Load the normalization snapshot saved inside this checkpoint.
    policy = policy_config.create_trained_policy(config, checkpoint, norm_stats=norm)
    if args.verify_only:
        audit = audit_model_input(config, schema)
        dataset, _ = load_real_dataset(config, schema)
        sample = dataset[0]
        # Supervision is not provided to inference.
        sample = dict(sample)
        sample.pop(schema.action_key, None)
        policy._rng = jax.random.key(args.seed)
        result = policy.infer(sample)
        actions = np.asarray(result["actions"])
        if actions.shape != (args.horizon, ACTION_DIM) or not np.isfinite(actions).all():
            raise ValueError(f"Reloaded inference failed: {actions.shape}")
        report = {"checkpoint": str(checkpoint), "model": args.model,
                  "checkpoint_reloaded": args.model == "lora", "finite_action_chunk": True,
                  "action_shape": list(actions.shape), "policy_timing": result.get("policy_timing"),
                  "input_audit": audit,
                  "interpretation": "Reload and one real-observation inference only; no closed-loop task-success claim."}
        output_dir = results_directory(args)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{args.model}_reload_verification.json").write_text(json.dumps(report, indent=2))
        print("TRANSPORT_RELOAD_VERIFIED " + json.dumps(report), flush=True)
        return
    logging.info("Transport policy loaded: %s", checkpoint)
    WebsocketPolicyServer(EpisodeSeedPolicy(policy), host="127.0.0.1", port=args.port,
                          metadata={**schema.summary(), "model": args.model, "checkpoint": str(checkpoint),
                                    "episode_seed_reset": True}).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
