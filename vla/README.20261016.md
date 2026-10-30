# VLA Bimanual Manipulation Research

[中文说明](README.zh-CN.md)

An ongoing simulation research project on how vision-language-action policies learn and fail during long-horizon robot manipulation.

Two Panda arms must remove a lid, clear an obstruction, pick up a hammer, hand it over, and place it in a target bin. The work combines **π0.5 adaptation, a BC-RNN baseline, and closed-loop failure analysis**. The focus is on connecting model predictions to what the robot actually does.

## Demonstrations

These are selected examples, not success-rate estimates. Each video retains its original model and evaluation label.

| Video | What it shows | Evaluation scope |
| --- | --- | --- |
| [π0.5 autonomous rollout](assets/pi05_autonomous_development.mp4) | The B16000 policy reaches the environment's original task-success condition | Previously used development initial state; no expert prefix |
| [BC-RNN autonomous rollout](assets/bcrnn_autonomous_training_state.mp4) | The fixed 50k baseline reaches the original success condition | Training initial state; no expert prefix |
| [Failure despite a close command](assets/pi05_contact_loss_failure.mp4) | Contact is lost while the gripper is commanded to close | Small-set continuation diagnostic; full task fails |

Rollouts stop at the first original task-success signal. A success signal does not establish long-term placement stability or robust handover.

## Implemented work

- Prepared **200 training demonstrations and 82,036 frames**, using trajectory-level splits and normalization fitted on training data only.
- Connected camera images, an **18-dimensional robot state**, and task text to π0.5. The policy predicts **10-step chunks of 14-dimensional controller commands**; these are not 14 physical joint angles.
- Ran LoRA adaptation and continuation experiments, including uniform and grasp/handover-focused sampling, and retained sample-consumption and checkpoint records.
- Built evaluation and expert-prefix diagnostics that separate grasping, lifting, handover, and placement from complete-task success.
- Compared one versus five executed actions before observing again in a bounded development experiment.
- Completed a five-camera **BC-RNN baseline with a predetermined 50,000 training updates**.
- Used videos, gripper commands, per-finger contact records, and object motion to distinguish unsuccessful grasping, commanded release, loss of contact under a close command, and receiver misalignment.

## Selected results

This table reports historical **development diagnostics**, not a formal generalization benchmark. Full per-case extracts are provided in [evidence/selected_results.json](evidence/selected_results.json).

| Checkpoint | Evaluation group | Original task successes |
| --- | --- | ---: |
| π0.5 A, 16,000 updates | 20 development initial states × 3 executions | 5/60 |
| π0.5 B, 16,000 updates | Same 20 states × 3 executions | 4/60 |
| π0.5 weighted, 11,000 updates | Same 20 states × 3 executions | 2/60 |
| BC-RNN, fixed 50,000 updates | Three predetermined old-validation states | 0/3 |
| Same BC-RNN checkpoint | Three predetermined training states | 1/3 |

The π0.5 checkpoints have different training histories; the table does not isolate a sampling effect. The BC-RNN groups are bounded development checks and are too small to support a stable success-rate claim. The BC-RNN training-state success first occurs at control step 423.

A key finding is that **lower training error does not guarantee completion of a closed-loop task**. A single-demonstration continuation reduced its training loss but still failed its fixed complete-task checks. Other runs failed through different physical sequences, so “premature release” cannot explain every failed handover.

## Current work

The public snapshot is dated **16 October 2026**.

- Five-camera π0.5 training and the paired comparison are being extended. This snapshot reports no completed formal five-camera comparison.
- The separately reserved initial states are not used to select the historical checkpoints shown above.
- Action-dimension loss masking and controlled short-term-memory ablations are **planned work**, not reported completed results.

## Code and reproduction

| Location | Purpose |
| --- | --- |
| `transport/openpi/` | Observation/action adaptation, data loading, training, serving, and rollout collection |
| `transport/scripts/` | Dataset audit, subset extraction, conversion, and expert-action replay |
| `transport/day2/` | Phase events and controlled sampling |
| `transport/day3/training/` | Additional grasp/phase sampling |
| `transport/day4/` | Five-camera extension and BC-RNN baseline helpers |
| `configs/` and `patches/` | Interface schema and the patch required by the pinned openpi checkout |
| `evidence/` and `assets/` | Selected result extracts and explicitly labeled original videos |

Start with [reproduction instructions](docs/REPRODUCTION.md) and [third-party notices](THIRD_PARTY_NOTICES.md). CPU interface checks can be run without a GPU, a policy checkpoint, or the task dataset:

```bash
python -m pip install -r requirements/review.txt
python transport/openpi/adapter_checks.py
python transport/openpi/pipeline_checks.py
python scripts/check_evidence.py
```

The first two commands use synthetic inputs to check interfaces and rollout bookkeeping. Passing them is not evidence that a trained policy completes the task. Full training requires the separately acquired upstream code, datasets, and model weights. This export is not claimed to reproduce every historical run bit for bit.

## Scope and attribution

- Simulation only, one task instruction, and limited training seeds.
- Repeated policy executions are not independent training seeds. Results must be interpreted by initial state and by evaluation split.
- Expert-prefix diagnostics are assisted experiments and are kept separate from autonomous rollouts.
- No claim of stable improvement, arbitrary-language generalization, out-of-distribution robustness, or Sim2Real transfer.
- Upstream model implementations and datasets are credited to their authors. This repository contains task adaptation and experiment tooling; it does not claim authorship of π0.5, BC-RNN, or the simulators.

No model weights, raw datasets, cloud credentials, or personal resume files are included. Third-party components retain their original licenses, including the applicable noncommercial research/evaluation restrictions on DexMimicGen.
