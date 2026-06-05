# External Sources and Version Pins

These version pins are used by `scripts/setup/clone_external.sh`.

| Component | Repository | Commit |
| --- | --- | --- |
| 2D Gaussian Splatting | https://github.com/hbb1/2d-gaussian-splatting | `335ad612f2e783a4e57b9cbc4d1e167bd599fc98` |
| threestudio | https://github.com/threestudio-project/threestudio | `28d9d80d9d00f308244adfcf3be8b17ca0cb6465` |
| Magic123 | https://github.com/guochengqian/Magic123 | `c2eb289f0b9e03e5cf39cf1417f05ca33e9eb0a5` |
| LeRobot | https://github.com/huggingface/lerobot | `8fff0fde7c79f23a93d845d1a50e985de01f8b8a` |
| CALVIN | https://github.com/mees/calvin | `fa03f01f19c65920e18cf37398a9ce859274af76` |

Additional public data used for smoke tests:

- Full Mip-NeRF 360 official archive: `https://storage.googleapis.com/gresearch/refraw360/360_v2.zip`; reachable but too slow from this machine during the checked run.
- NerfBaselines Mip-NeRF 360 sparse garden-n24 files: `https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data/tree/main/mipnerf360-sparse`.

Official command assumptions:

- 2DGS trains with `python train.py -s <dataset> -m <output>` and extracts meshes/videos with `python render.py`.
- threestudio text-to-3D uses `python launch.py --config configs/dreamfusion-sd.yaml --train ...` and mesh export uses `--export system.exporter_type=mesh-exporter`.
- Magic123 uses `preprocess_image.py` and a two-stage `main.py` coarse/fine workflow.
- LeRobot trains with `lerobot-train --policy.type=act` and evaluates policies with `select_action` / `forward`.
- CALVIN official challenge evaluation calls `model.reset()` and `model.step(obs, goal)` for a custom model.

Local reproducibility notes:

- `scripts/setup/clone_external.sh` applies a small 2DGS compatibility patch after checkout, replacing `np.byte` with `np.uint8` in the NeRF synthetic image loader so current Pillow accepts RGB arrays.
- `scripts/setup/clone_external.sh` also patches the pinned 2DGS unbounded extractor to convert current `trimesh.Trimesh` outputs to Open3D explicitly.
- `scripts/setup/install_2dgs_env.sh` installs `mediapy` and `trimesh`, which are required by the pinned 2DGS `render.py` mesh extraction path.
