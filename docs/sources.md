# External Sources and Version Pins

These version pins are used by `scripts/setup/clone_external.sh`.

| Component | Repository | Commit |
| --- | --- | --- |
| 2D Gaussian Splatting | https://github.com/hbb1/2d-gaussian-splatting | `335ad612f2e783a4e57b9cbc4d1e167bd599fc98` |
| threestudio | https://github.com/threestudio-project/threestudio | `28d9d80d9d00f308244adfcf3be8b17ca0cb6465` |
| Magic123 | https://github.com/guochengqian/Magic123 | `c2eb289f0b9e03e5cf39cf1417f05ca33e9eb0a5` |
| LeRobot | https://github.com/huggingface/lerobot | `8fff0fde7c79f23a93d845d1a50e985de01f8b8a` |
| CALVIN | https://github.com/mees/calvin | `fa03f01f19c65920e18cf37398a9ce859274af76` |
| tiny-cuda-nn | https://github.com/NVlabs/tiny-cuda-nn | `749dd70c5afc5a9dadb85e5652ed65d55e0ba187` |
| tiny-cuda-nn cutlass submodule | https://github.com/NVIDIA/cutlass | `82f5075946e2569589439d500733b700a3141374` |
| tiny-cuda-nn fmt submodule | https://github.com/fmtlib/fmt | `fa2eb2d2e3ec5c21629f8ccd88ae05ec40b963fa` |
| tiny-cuda-nn cmrc submodule | https://github.com/vector-of-bool/cmrc | `952ffddba731fc110bd50409e8d2b8a06abbd237` |
| nvdiffrast | https://github.com/NVlabs/nvdiffrast | `main` archive used locally |
| envlight | https://github.com/ashawkey/envlight | `main` archive used locally |
| CLIP | https://github.com/openai/CLIP | `main` archive used locally |
| cubvh | https://github.com/ashawkey/cubvh | `main` archive used locally |

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
- `scripts/setup/clone_external.sh` downloads local archive copies of AIGC dependencies that upstream requirements otherwise install through nested pip/Git URLs.
- `scripts/setup/clone_external.sh` patches Magic123 CUDA extension build flags from C++14 to C++17 for current PyTorch headers and adds a `rembg` fallback when optional carvekit background removal is unavailable.
- `scripts/setup/install_2dgs_env.sh` installs `mediapy` and `trimesh`, which are required by the pinned 2DGS `render.py` mesh extraction path.
- `scripts/setup/install_threestudio_env.sh` skips optional Gradio and xformers by default; set `INSTALL_THREESTUDIO_UI=1` or `INSTALL_THREESTUDIO_XFORMERS=1` to include them.
- `scripts/setup/install_magic123_env.sh` skips optional Shape-E by default; set `INSTALL_MAGIC123_SHAPE=1` to include it.
