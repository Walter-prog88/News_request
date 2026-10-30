# Third-party sources and licenses

This repository publishes task adapters, experiment utilities and a small patch;
it does not claim authorship of the underlying foundation model, simulator,
benchmark task, baseline architecture or demonstration dataset. No blanket MIT
license is applied to the combined project. Upstream software and assets retain
their own licenses; model weights and datasets are not distributed here.

| Component | Fixed source | Attribution and license |
| --- | --- | --- |
| openpi | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479) | Apache-2.0; full upstream license retained in `licenses/openpi-Apache-2.0.txt`. |
| robosuite | [ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite/tree/51cc01785bab80ffeed20da15e67d7dd4140e76a) | Referenced dependency; source is not vendored. Retain the upstream license when obtaining it. |
| DexMimicGen | [NVlabs/dexmimicgen](https://github.com/NVlabs/dexmimicgen/tree/940e8a1b3ad70eb1925ada6b364b197de6bb2af9) | NVIDIA License at this revision, including the non-commercial research/evaluation use limitation in section 3.3. Source is not vendored. |
| robomimic | [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic/tree/9ce065180dbcc38ddf8fa4218daa2e75b5fdc28a) | MIT, copyright Stanford Vision and Learning Lab. Referenced dependency; source is not vendored. |
| LeRobot | [huggingface/lerobot](https://github.com/huggingface/lerobot/tree/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5) | Referenced through the pinned openpi dependency configuration; source is not vendored. |
| Demonstrations | [MimicGen/dexmimicgen_datasets](https://huggingface.co/datasets/MimicGen/dexmimicgen_datasets/tree/181967e10c6277a653e7c9761f3978312af3646a) | `generated/two_arm_transport.hdf5`; obtain separately under the source terms. No demonstrations are redistributed. |

## Included openpi modification

`patches/openpi-image-augmentation.patch` modifies three upstream Apache-2.0
files: `src/openpi/models/model.py`, `pi0.py`, and `pi0_config.py`. It adds an
explicit image augmentation option (`default`, `none`, `mild_rgb`) and propagates
that option to training preprocessing. Upstream-default behavior is retained
when `default` is selected; the Transport configuration explicitly selects
`none`. This is an experimental preprocessing change, not a newly developed
foundation model. The patch must be applied to the fixed openpi revision.

`transport/day4/pi05/five_camera.py` builds isolated method variants from the
installed pinned openpi methods at runtime. It extends the image-key list while
reusing the original implementation and shared image encoder. It does not
redistribute a full copy of those upstream methods or replace their attribution.

## Referenced baseline and task

`transport/day4/bcrnn/generate_configs.py` reads the official pinned DexMimicGen
configuration generator and robomimic template; the upstream source files are
obtained separately. `source_pins.json` records their revisions and file hashes.
The training wrapper calls the upstream robomimic trainer and does not claim a
new BC-RNN architecture. The two-arm task and physics implementation come from
DexMimicGen/robosuite, not this repository.
