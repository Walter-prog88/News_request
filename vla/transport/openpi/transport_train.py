"""Real pi05 LoRA smoke training, with parameter-change and checkpoint evidence."""
from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import flax.traverse_util as traverse
import jax
import numpy as np
from openpi.training import checkpoints, data_loader, sharding

from transport_config import add_common_arguments, config_from_args, record_run, results_directory
from transport_data import audit_model_input


def training_invariants(config):
    norm_path = config.assets_dirs / config.data.repo_id / "norm_stats.json"
    return {"model": {key: getattr(config.model, key) for key in
                      ("pi05", "action_dim", "action_horizon", "discrete_state_input", "max_token_len",
                       "paligemma_variant", "action_expert_variant", "image_augmentation")},
            "batch_size": config.batch_size, "seed": config.seed,
            "lr_schedule": dataclasses.asdict(config.lr_schedule), "optimizer": dataclasses.asdict(config.optimizer),
            "norm_stats_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
            "frozen_vision_encoder": True, "ema_decay": config.ema_decay}


def validate_resume_source(source, args, config, invariants):
    if not source.name.isdigit() or not (source / "params").is_dir() or not (source / "train_state").is_dir():
        raise ValueError("Resume source must be a numbered full checkpoint containing params and train_state.")
    previous = json.loads((source / "transport_run.json").read_text())
    if previous["schema_file"] != json.loads(args.schema.read_text()):
        raise ValueError("Resume schema differs from the saved checkpoint.")
    if previous.get("training_invariants") != invariants or previous["arguments"]["repo_id"] != args.repo_id:
        raise ValueError("Resume must preserve the model, optimizer, schedule, batch size, seed, repository and normalization.")
    saved_norm = source / "assets" / args.repo_id / "norm_stats.json"
    if hashlib.sha256(saved_norm.read_bytes()).hexdigest() != invariants["norm_stats_sha256"]:
        raise ValueError("Checkpoint normalization snapshot differs from the current training statistics.")
    expected_step = int(source.name) + 1
    if config.num_train_steps <= expected_step:
        raise ValueError("Target steps must exceed the saved checkpoint's completed updates.")
    return expected_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--upstream-train", type=Path, required=True, help="Pinned openpi/scripts/train.py")
    parser.add_argument("--resume", action="store_true", help="Continue to --steps total completed updates")
    parser.add_argument("--resume-from-checkpoint", type=Path,
                        help="Restore full Adam, model and step from an archived checkpoint into a NEW experiment")
    args = parser.parse_args()
    config, schema = config_from_args(args)
    if args.resume and args.resume_from_checkpoint:
        raise ValueError("Choose same-run --resume or external --resume-from-checkpoint, not both.")
    if args.resume and not args.phase:
        raise ValueError("Same-run resume requires a fresh --phase to retain earlier reports.")
    output_dir = results_directory(args)
    if (output_dir / "train_run_config.json").exists():
        raise FileExistsError("Training invocation reports already exist; choose a fresh experiment or phase.")
    invariants = training_invariants(config)
    source = args.resume_from_checkpoint
    if args.resume:
        candidates = [path for path in config.checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit()]
        if not candidates:
            raise ValueError("No completed checkpoint is available for same-run resume.")
        source = max(candidates, key=lambda path: int(path.name))
    expected_step = validate_resume_source(source, args, config, invariants) if source else None
    output_dir.mkdir(parents=True, exist_ok=True)
    input_audit = audit_model_input(config, schema)
    (output_dir / "model_input_audit.json").write_text(json.dumps(input_audit, indent=2))
    record_run(args, schema, output_dir / "train_run_config.json", training_invariants=invariants)
    spec = importlib.util.spec_from_file_location("transport_upstream_train", args.upstream_train)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    jax.config.update("jax_compilation_cache_dir", str(args.root / "jax-cache"))
    if config.batch_size % jax.device_count():
        raise ValueError("Batch size must be divisible by JAX device count.")
    train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=args.resume)
    try:
        loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
        iterator = iter(loader)
        batch = next(iterator)
        state, state_sharding = upstream.init_train_state(config, init_rng, mesh, resume=resuming or source is not None)
        if resuming:
            state = checkpoints.restore_state(manager, state, loader)
        elif source is not None:
            source_manager, source_exists = checkpoints.initialize_checkpoint_dir(
                source.parent, keep_period=None, overwrite=False, resume=True)
            try:
                if not source_exists:
                    raise ValueError("External full checkpoint is not recognized by Orbax.")
                state = checkpoints.restore_state(source_manager, state, loader, step=int(source.name))
            finally:
                source_manager.close()
        jax.block_until_ready(state)
        start_step = int(state.step)
        optimizer_counters = {jax.tree_util.keystr(path): int(np.asarray(value))
                              for path, value in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]
                              if getattr(value, "shape", None) == () and np.issubdtype(value.dtype, np.integer)}
        if source is not None and (start_step != expected_step or not optimizer_counters or any(value != start_step for value in optimizer_counters.values())):
            raise ValueError(f"Full optimizer/step restoration failed: step={start_step}, counters={optimizer_counters}")
        if start_step >= config.num_train_steps:
            raise ValueError(f"Already at update {start_step}; --steps must be larger to continue.")
        trainable = traverse.flatten_dict(state.params.filter(config.trainable_filter).to_pure_dict())
        all_params = traverse.flatten_dict(state.params.to_pure_dict())
        if any("img" in "/".join(key) for key in trainable):
            raise AssertionError("Vision encoder must remain frozen.")
        if any("llm" in "/".join(key) and "lora" not in "/".join(key) for key in trainable):
            raise AssertionError("Unexpected unfrozen VLM/action-expert backbone parameter.")
        lora_keys = [key for key in trainable if "lora" in "/".join(key)][:8]
        if not lora_keys:
            raise AssertionError("No trainable LoRA parameters.")
        before = {key: np.asarray(trainable[key]).copy() for key in lora_keys}
        frozen_keys = [next(key for key in all_params if "img" in "/".join(key)),
                       next(key for key in all_params if "llm" in "/".join(key) and key not in trainable)]
        frozen_before = {key: np.asarray(all_params[key].reshape(-1)[:4096]).copy() for key in frozen_keys}
        leaves = [{"path": "/".join(key), "shape": list(value.shape), "count": int(value.size)}
                  for key, value in trainable.items()]
        parameter_report = {"trainable_count": sum(row["count"] for row in leaves),
                            "total_count": sum(int(value.size) for value in all_params.values()), "leaves": leaves}
        (output_dir / "trainable_parameters.json").write_text(json.dumps(parameter_report, indent=2))
        step_fn = jax.jit(functools.partial(upstream.train_step, config),
                          in_shardings=(replicated, state_sharding, data_sharding),
                          out_shardings=(state_sharding, replicated), donate_argnums=(1,))
        metrics = []
        with (output_dir / "training_metrics.jsonl").open("a") as log:
            for step in range(start_step, config.num_train_steps):
                started = time.perf_counter()
                with sharding.set_mesh(mesh):
                    state, info = step_fn(train_rng, state, batch)
                values = {key: float(value) for key, value in jax.device_get(info).items()}
                if not all(np.isfinite(value) for value in values.values()) or values["grad_norm"] <= 0:
                    raise FloatingPointError(f"Invalid training metrics at {step + 1}: {values}")
                row = {"completed_updates": step + 1, "update_seconds": time.perf_counter() - started, **values}
                metrics.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                if step == start_step or (step + 1) % config.log_interval == 0:
                    print("TRANSPORT_TRAIN_STEP " + json.dumps(row), flush=True)
                if (step + 1) % config.save_interval == 0 or step == config.num_train_steps - 1:
                    checkpoints.save_state(manager, state, loader, step)
                if step < config.num_train_steps - 1:
                    batch = next(iterator)
        manager.wait_until_finished()
        after = traverse.flatten_dict(state.params.to_pure_dict())
        lora_delta = {"/".join(key): float(np.linalg.norm(np.asarray(after[key], dtype=np.float32) -
                                                       np.asarray(before[key], dtype=np.float32))) for key in lora_keys}
        frozen_unchanged = {"/".join(key): bool(np.array_equal(frozen_before[key],
                                          np.asarray(after[key].reshape(-1)[:4096]))) for key in frozen_keys}
        if not any(value > 0 for value in lora_delta.values()) or not all(frozen_unchanged.values()):
            raise AssertionError("Parameter-change/freeze audit failed.")
        checkpoint = config.checkpoint_dir / str(config.num_train_steps - 1)
        record_run(args, schema, checkpoint / "transport_run.json", training_invariants=invariants)
        report = {"completed_updates": int(state.step), "updates_this_invocation": len(metrics),
                  "restored_from_checkpoint": str(source) if source else None,
                  "starting_completed_updates": start_step, "starting_optimizer_counters": optimizer_counters,
                  "training_invariants": invariants,
                  "sampler_resume_boundary": "Each invocation restarts the seeded shuffled loader. Model, Adam state, global step and schedule are restored; this is not bitwise uninterrupted training.",
                  "checkpoint": str(checkpoint), "first_update_seconds": metrics[0]["update_seconds"],
                  "steady_update_median_seconds": float(np.median([row["update_seconds"] for row in metrics[1:]])) if len(metrics) > 1 else None,
                  "finite_loss_gradients": True, "lora_sample_delta_l2": lora_delta,
                  "frozen_parameter_samples_unchanged": frozen_unchanged,
                  "jax_memory_stats": jax.devices()[0].memory_stats(), "warmup_steps": 300,
                  "reload_verified": False,
                  "interpretation": "Training mechanics only. Run transport_serve.py --verify-only separately; closed-loop task success is a separate evaluation."}
        (output_dir / "training_smoke_summary.json").write_text(json.dumps(report, indent=2))
        print("TRANSPORT_TRAINING_COMPLETE " + json.dumps(report), flush=True)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
