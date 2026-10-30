"""External Transport configs: imported directly, without editing upstream registries."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import flax.nnx as nnx

from openpi.models.pi0_config import Pi0Config
from openpi.shared.nnx_utils import PathRegex
from openpi.training import config as c
from openpi.training import optimizer as o
from openpi.training.weight_loaders import CheckpointWeightLoader
from openpi import transforms
from transport_policy import TransportInputs, TransportOutputs, TransportSchema

CONFIG_NAME = "pi05_transport_lora"


@dataclasses.dataclass(frozen=True)
class TransportDataConfig(c.DataConfigFactory):
    schema: TransportSchema | None = None

    def create(self, assets_dirs, model_config):
        if self.schema is None or not model_config.pi05 or not model_config.discrete_state_input:
            raise ValueError("Transport requires audited schema and active pi05 proprioception.")
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=transforms.Group(),
            data_transforms=transforms.Group(
                inputs=[TransportInputs(self.schema)], outputs=[TransportOutputs()]),
            model_transforms=c.ModelTransformFactory()(model_config),
            action_sequence_keys=(self.schema.action_key,), prompt_from_task=True,
        )


def make_config(*, schema: TransportSchema, root: str | Path, repo_id: str,
                base_checkpoint: str | Path, exp_name="day1_smoke", steps=100,
                batch_size=4, num_workers=2, horizon=10, lora=True, seed=42,
                augmentation="none", resume=False):
    schema.validate()
    if repo_id in ("fake", "") or steps <= 0 or horizon <= 0 or batch_size <= 0:
        raise ValueError("A real training dataset and positive run dimensions are required.")
    root = Path(root).expanduser().resolve()
    if Path(exp_name).name != exp_name or exp_name in (".", ".."):
        raise ValueError("Experiment name must be a single directory name.")
    model = Pi0Config(
        pi05=True, action_dim=32, action_horizon=horizon, discrete_state_input=True,
        paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
        image_augmentation=augmentation,
    )
    return c.TrainConfig(
        name=CONFIG_NAME, exp_name=exp_name, model=model,
        data=TransportDataConfig(repo_id=repo_id, schema=schema,
                                base_config=c.DataConfig(prompt_from_task=True)),
        weight_loader=CheckpointWeightLoader(str(Path(base_checkpoint) / "params")),
        freeze_filter=nnx.Any(model.get_freeze_filter(), PathRegex(".*img.*")) if lora else nnx.Nothing,
        assets_base_dir=str(root / "assets"), checkpoint_base_dir=str(root / "checkpoints"),
        batch_size=batch_size, num_workers=num_workers, seed=seed, num_train_steps=steps,
        lr_schedule=o.CosineDecaySchedule(warmup_steps=300, peak_lr=5e-5,
                                         decay_steps=max(6000, steps), decay_lr=1e-5),
        optimizer=o.AdamW(clip_gradient_norm=1.0), ema_decay=None,
        log_interval=10, save_interval=steps, keep_period=None, wandb_enabled=False,
        resume=resume, overwrite=False,
        policy_metadata={"task": "Transport", **schema.summary()},
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="New isolated Transport experiment root")
    parser.add_argument("--repo-id", required=True, help="Already split, train-only LeRobot repository")
    parser.add_argument("--base-checkpoint", type=Path, required=True, help="pi05_base directory containing params/")
    parser.add_argument("--exp-name", default="day1_smoke")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", help="Optional distinct results subdirectory; required for same-run resume")
    parser.add_argument("--checkpoint-base-dir", type=Path,
                        help="Override checkpoint storage only; assets, reports and compilation cache remain under --root")


def results_directory(args) -> Path:
    directory = args.root / "results" / args.exp_name
    if args.phase:
        if Path(args.phase).name != args.phase or args.phase in (".", ".."):
            raise ValueError("Phase must be a single directory name.")
        directory = directory / "phases" / args.phase
    return directory


def config_from_args(args, *, lora=True):
    schema = TransportSchema.load(args.schema)
    config = make_config(schema=schema, root=args.root, repo_id=args.repo_id,
                         base_checkpoint=args.base_checkpoint, exp_name=args.exp_name,
                         steps=args.steps, batch_size=args.batch_size, num_workers=args.num_workers,
                         horizon=args.horizon, seed=args.seed, lora=lora,
                         resume=getattr(args, "resume", False))
    if getattr(args, "checkpoint_base_dir", None) is not None:
        config = dataclasses.replace(config, checkpoint_base_dir=str(args.checkpoint_base_dir.expanduser().resolve()))
    return config, schema


def record_run(args, schema: TransportSchema, path: Path, *, training_invariants=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    content = {"arguments": values, "schema": schema.summary(), "schema_file": json.loads(args.schema.read_text())}
    if training_invariants is not None:
        content["training_invariants"] = training_invariants
    path.write_text(json.dumps(content, indent=2))
