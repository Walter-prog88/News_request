# Reproduction and review guide

This is a source release of a VLA bimanual-manipulation study. The runnable
minimal route is the **three-camera Transport adaptation**: audit and split an
official HDF5, convert train episodes, normalize train-only numeric features,
fine-tune a pi0.5 LoRA policy, reload it, and evaluate development episodes in a
separate simulator process. Full weights, demonstrations, private infrastructure
scripts and the original cloud environment are not included.

Commands below are instructions; publishing this repository does not execute
them. Full training and physics reproduction have not been rerun from this
export. The small CPU checks are described separately from experiment outcomes.

## 1. Quick CPU review

From the repository root, in an existing Python environment with NumPy and
PyArrow (or a new virtual environment):

```bash
python -m pip install -r requirements/review.txt
python transport/openpi/adapter_checks.py
python transport/openpi/pipeline_checks.py
```

The first script checks camera/state/action adaptation. The second uses tiny
synthetic Parquet and a fake environment/client to check train-only provenance,
frame hashes, single-reset rollouts, action clipping and the distinction between
`done` and actual task success. It intentionally needs no model weights, real
physics, demonstration download or GPU. Passing these checks is not evidence of
learned task completion.

The dependency file is a minimal check environment, not the full historical
cloud lockfile. `requirements/simulation.txt` contains the historical simulator's
direct package pins; training uses the fixed openpi `uv.lock`.

## 2. External sources and separate environments

Use Linux with an NVIDIA GPU for the full training route; pinned openpi documents
Ubuntu 22.04 as its tested platform. Keep a Python 3.11 openpi training environment
and a Python 3.10 simulator environment separate, communicating through the local
WebSocket policy server. Do not install the old simulator's NumPy/Numba pins into
the openpi environment.

Set task-specific paths to new local directories. These placeholders are not
credentials or original machine paths:

```bash
export VLA_PORTFOLIO_DIR="$PWD"
export VLA_EXTERNAL_DIR=/absolute/path/to/new/external_sources
export VLA_WORK_DIR=/absolute/path/to/new/transport_experiment
export VLA_SOURCE_HDF5=/absolute/path/to/two_arm_transport.hdf5
export VLA_BASE_CHECKPOINT=/absolute/path/to/pi05_base
```

Obtain the following sources at the exact commits, keeping their licenses:

| Directory under `$VLA_EXTERNAL_DIR` | Repository | Commit |
| --- | --- | --- |
| `openpi` | `https://github.com/Physical-Intelligence/openpi.git` | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| `robosuite` | `https://github.com/ARISE-Initiative/robosuite.git` | `51cc01785bab80ffeed20da15e67d7dd4140e76a` |
| `dexmimicgen` | `https://github.com/NVlabs/dexmimicgen.git` | `940e8a1b3ad70eb1925ada6b364b197de6bb2af9` |
| `robomimic` (optional BC baseline) | `https://github.com/ARISE-Initiative/robomimic.git` | `9ce065180dbcc38ddf8fa4218daa2e75b5fdc28a` |

In the openpi checkout, initialize its submodules at the pinned state, then apply
and check the included patch before installing the pinned Python environment:

```bash
git -C "$VLA_EXTERNAL_DIR/openpi" submodule update --init --recursive
git -C "$VLA_EXTERNAL_DIR/openpi" apply --check "$VLA_PORTFOLIO_DIR/patches/openpi-image-augmentation.patch"
git -C "$VLA_EXTERNAL_DIR/openpi" apply "$VLA_PORTFOLIO_DIR/patches/openpi-image-augmentation.patch"
cd "$VLA_EXTERNAL_DIR/openpi"
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd "$VLA_PORTFOLIO_DIR"
```

LeRobot is pinned by upstream openpi to
`0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`. openpi uses JAX0.5.3,
Flax0.10.2 and PyTorch2.7.1 at this revision. Follow its GPU installation
requirements; GPU architecture/driver compatibility must be checked locally.
The Transport-specific `image_augmentation` argument requires the patch above;
unpatched upstream is not a compatible runtime.

For the isolated simulator environment:

```bash
python3.10 -m venv "$VLA_WORK_DIR/.venv-sim"
"$VLA_WORK_DIR/.venv-sim/bin/python" -m pip install 'pip<26' 'setuptools==69.5.1' wheel
"$VLA_WORK_DIR/.venv-sim/bin/python" -m pip install -r requirements/simulation.txt
"$VLA_WORK_DIR/.venv-sim/bin/python" -m pip install --no-deps -e "$VLA_EXTERNAL_DIR/robosuite" -e "$VLA_EXTERNAL_DIR/dexmimicgen"
"$VLA_WORK_DIR/.venv-sim/bin/python" -m pip install -e "$VLA_EXTERNAL_DIR/openpi/packages/openpi-client"
```

Also ensure PyArrow and h5py are available for conversion. The policy-side
commands use the openpi environment's Python; the simulator-side commands use
the Python3.10 environment. The examples below use `python` to mean the correct
active environment, not a single merged environment.

Obtain official pi0.5 base weights separately using the fixed openpi project's
instructions (`gs://openpi-assets/checkpoints/pi05_base`). The path supplied via
`--base-checkpoint` must contain `params/`. Model/tokenizer assets may require
separate downloads and retain their own source terms.

## 3. Data source, audit and split

Official dataset: `MimicGen/dexmimicgen_datasets`, revision
`181967e10c6277a653e7c9761f3978312af3646a`, file
`generated/two_arm_transport.hdf5` (1,029 source episodes). The audited original
file has 15,177,869,792 bytes and SHA-256
`8eea65271eb21eabe6de1310cbc04b15f8f21343b59efc26b1f5cb4d54795f57`.
Obtain it from the upstream dataset page; no data download is bundled.

Use the conversion/review environment for these commands:

```bash
python transport/scripts/audit_dataset.py --dataset "$VLA_SOURCE_HDF5" --output "$VLA_WORK_DIR/source_audit.json" --scan-episodes 5 --sha256
python transport/scripts/extract_subset.py --dataset "$VLA_SOURCE_HDF5" --output "$VLA_WORK_DIR/transport_subset.hdf5" --train 200 --valid 50 --test 50 --seed 20261014 --exclude-episodes demo_0 --exclusion-reason 'Control-replay inspection before split construction'
export HF_LEROBOT_HOME="$VLA_WORK_DIR/lerobot"
python transport/scripts/convert_lerobot.py --dataset "$VLA_WORK_DIR/transport_subset.hdf5" --split train --output "$HF_LEROBOT_HOME/local/transport_train"
```

Check the source audit's file size and hash against the fixed values before
training. `extract_subset.py` hashes its input by default but does not itself
verify the revision unless matching `--source-integrity` evidence is supplied.
The split is seeded, whole-episode and disjoint: 200 train / 50 valid / 50 test;
`demo_0` is excluded because it was inspected before the original split. This is
an in-distribution initial-state holdout, not an OOD benchmark. The original
train subset contained 82,036 frames; match this count before comparing runs.

The converter retains raw float32 controls, raw 84px RGB encoded losslessly as PNG
inside Parquet, and both-arm 18D state. It does not subtract state from the raw
controller commands, rescale them into physical units, or normalize using held
out data.

`configs/transport_schema.json` records the audited **original** controller and
camera contract: 3 real views, 18D state, 14D controller commands at 20Hz, quaternion
order xyzw and live-image vertical flip. It includes historical evidence hashes
but does not bundle all original audit files. Re-audit any changed source,
controller or simulator; do not turn `audited` on merely to bypass checks.

Before model training, use the simulator environment to verify recorded-action
replay and source/live-image orientation. The original demo 0 is outside all
splits, so it can be used for this engineering check:

```bash
MUJOCO_GL=egl python transport/scripts/replay_dataset.py --dataset "$VLA_SOURCE_HDF5" --episodes demo_0 --mode action --output "$VLA_WORK_DIR/replay_demo0.json"
MUJOCO_GL=egl python transport/scripts/audit_observation_parity.py --dataset "$VLA_SOURCE_HDF5" --episode demo_0 --output-dir "$VLA_WORK_DIR/parity_demo0"
```

Inspect those reports against the schema. Recorded-action replay is an
engineering control; it does not measure a learned policy's success.

## 4. Train-only normalization and minimal LoRA run

Activate the openpi environment, preserving the same task paths. Its current
working directory can remain this repository root:

```bash
export HF_LEROBOT_HOME="$VLA_WORK_DIR/lerobot"
export PYTHONPATH="$VLA_PORTFOLIO_DIR/transport/openpi${PYTHONPATH:+:$PYTHONPATH}"
python transport/openpi/transport_data.py --schema configs/transport_schema.json --root "$VLA_WORK_DIR/experiment" --repo-id local/transport_train --base-checkpoint "$VLA_BASE_CHECKPOINT" --exp-name smoke100 --mode normalize
python transport/openpi/transport_train.py --schema configs/transport_schema.json --root "$VLA_WORK_DIR/experiment" --repo-id local/transport_train --base-checkpoint "$VLA_BASE_CHECKPOINT" --exp-name smoke100 --steps 100 --upstream-train "$VLA_EXTERNAL_DIR/openpi/scripts/train.py"
python transport/openpi/transport_serve.py --schema configs/transport_schema.json --root "$VLA_WORK_DIR/experiment" --repo-id local/transport_train --base-checkpoint "$VLA_BASE_CHECKPOINT" --exp-name smoke100 --steps 100 --verify-only
```

Normalization reads numeric Parquet columns once per train frame and uses exact
q01/q99; source masks, row identity and raw action hashes are checked. A token
audit checks that both arms reach the model input. The run uses LoRA in the VLM
and action expert with a frozen image encoder, batch 4, predicted horizon 10,
seed 42, 300-update warmup and a 6000-step decay schedule. The model pads actions to
32 slots internally; 14 are real controls. Its raw training loss is not a
14-dimensional physical action error metric.

A 100-update run is a mechanics check inside warmup, not evidence of convergence.
It still requires the full model and suitable GPU resources. Upstream checkpoint
labels are zero-based (`99/` for 100 completed updates). `--verify-only` reloads
the saved model in a separate process and checks a real-observation 14D action
chunk; it does not run the simulator. A new invocation must use new experiment/
phase names as required by each CLI; existing reports are protected.

## 5. Development rollout

In the openpi environment start the policy server (it binds to 127.0.0.1):

```bash
python transport/openpi/transport_serve.py --schema configs/transport_schema.json --root "$VLA_WORK_DIR/experiment" --repo-id local/transport_train --base-checkpoint "$VLA_BASE_CHECKPOINT" --exp-name smoke100 --steps 100 --port 8001
```

In another process using the Python3.10 simulator environment:

```bash
MUJOCO_GL=egl python transport/openpi/transport_rollout.py --dataset "$VLA_WORK_DIR/transport_subset.hdf5" --schema configs/transport_schema.json --conversion-manifest "$VLA_WORK_DIR/lerobot/local/transport_train/meta/conversion_manifest.json" --output-dir "$VLA_WORK_DIR/development_smoke100" --model-label smoke100 --port 8001 --n 5 --seed 101 --execution-horizon 5 --image-transform flipud
```

This published collector accepts 1–5 validation episodes and rejects train/test
episodes. It resets the source XML and initial state once, then policy actions
alone drive physics. It predicts 10 actions and executes 5 before observing
again; `--execution-horizon 1|5|10` supports a separately named ablation. Use the
same validation cases, seed and horizon when comparing base/adapted models.
Serving `--model base` uses the same input mapping and train-only statistics.
The development collector is not the private later formal holdout runner.

Reports distinguish actual environment task success, intermediate stages,
clipping and latency; videos record the three policy views. Optional contact
indicators do not prove force closure or completed handover. Inference pauses
the synchronous simulator; measured latency is not proof of real-time robot
control. The single constant instruction does not test language grounding.

## 6. Reviewable extensions and historical limits

- `transport/day2/phase_events.py` and
  `transport/day3/training/grasp_events.py` implement numeric contact/height/
  command rules for train-only event windows. These are contact proxies, not
  measured grasp forces. `sampling.py`, `sampling_loader.py` and
  `sampling_day3.py` implement frozen, checkpoint-step-addressed schedules with
  actual source/frame identity records. Their historical CLIs require original
  labels, run hashes and source checkpoints, which are not distributed. They
  are reviewable methods, not a one-command replay of the private runs.
- `transport/day4/pi05/five_camera.py` extends the model to five independent
  views through isolated variants of pinned upstream methods. The included
  converter/schema remain the runnable **three-camera** route. The original
  five-view loader, source-B16000 proof and frozen formal run are not bundled;
  this release does not offer an end-to-end five-view training command.
- `transport/day4/bcrnn/` includes an official-source config generator, CPU
  preflight, a local trainer wrapper and validation-I/O helper. Obtain the
  pinned robomimic/DexMimicGen checkouts separately and pass `--upstream` to the
  generator/preflight/trainer. `source_pins.json` verifies required upstream
  files. The baseline recipe is controlled 200-demo training, not reproduction
  of the full published benchmark or equal architecture/history/compute with
  pi0.5. It requires a separately prepared BC-compatible HDF5 and environment.
- The historical BC v4 run changed validation workers 1→4 after equivalence
  checks. The clean included `train.py` exposes the earlier general wrapper;
  `validation_io.py` documents the additional I/O change. The private v4
  freeze/proofs and orchestration are omitted, so the public package does not
  claim byte-for-byte reproduction of that run. The upstream resume also does
  not fully restore RNG/DataLoader state.
- No real robot, Sim2Real transfer, OOD improvement or stable bimanual
  coordination is established merely by this source release. Numerical
  results must be read alongside the dataset split, run status and original
  evidence scope; planned experiments are not completed results.

See `THIRD_PARTY_NOTICES.md` before obtaining upstream sources or assets.
