# Completion Audit

This audit records the current local state against `HW3_计算机视觉.pdf`.

## Task 1

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Object A from phone multi-view/video + COLMAP + 2DGS | Uploaded video staged at `data/raw/object_A/object_a.mp4`; 80 frames extracted; COLMAP `SIMPLE_PINHOLE` exhaustive run registers 80/80 frames with 2496 sparse points; alpha-masked 2DGS was continued to 20000 steps; final mask/COLMAP visual-hull mesh is `results/task1/2dgs/object_A/train/ours_latest/fuse_post.ply` with 41115 vertices / 82258 faces, and the official 20000-step TSDF mesh is preserved as `fuse_post_2dgs_tsdf.ply` | Complete for the checked real object-A run |
| Object B text-to-3D with threestudio/SDS | `results/task1/aigc/object_B_text3d/export/model.obj` is from `results/task1/aigc/object_B_text3d/toycar_sd_fp16_12000_resume@20260620-150907`, a threestudio DreamFusion/SDS run resumed from 8000 to 12000 steps with local Stable Diffusion v1.5 fp16 weights and prompt `a small toy car, smooth colorful plastic, simple rounded shape, single object, centered`; mesh has 37934 vertices / 75872 faces | Complete for the checked text-to-3D run |
| Object C single image to 3D with Magic123 | Uploaded image staged at `data/raw/object_C/object_c.png`; RGBA foreground is `results/task1/aigc/object_C_image3d/data/rgba.png`; Magic123/Zero123 completed 5000 coarse NeRF + 5000 DMTet iterations; final mesh is `results/task1/aigc/object_C_image3d/model.obj` with 107662 vertices / 215328 faces | Complete for the checked real object-C run |
| Mip-NeRF 360 background reconstructed with 2DGS | Public NerfBaselines Mip-NeRF 360 sparse garden-n24 data are converted at `data/task1/mipnerf360_sparse_garden_n24/2dgs_dataset`; final 30000-step unbounded mesh is `results/task1/2dgs/background_garden/train/ours_latest/fuse_unbounded_post.ply` with 1238366 vertices / 2476489 faces | Complete for the checked public sparse Mip-NeRF 360 background; full-resolution archive rerun is optional quality improvement |
| Fused scene preview and scene file | `results/task1/fusion/fusion_preview.png` was regenerated with Blender after the 12000-step Object B update; `results/task1/fusion/fusion_scene.blend` exists from the scene setup; `fusion_walkthrough.mp4` has not been regenerated in this preview-only step | Preview complete; walkthrough video pending final packaging if required |
| Representation unification explanation | `report/HW3_report.ipynb` is the Chinese notebook report and explains mesh export, Blender fusion, provenance, results, limitations, and reproduction commands; `report/main.tex` / PDF are retained as backup | Present |

## Task 2

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Train ACT on environment B | `data/task2/lerobot_v30/calvin_B_inferred`, `results/task2/act_env_B_inferred/checkpoints/000100/pretrained_model`, `results/task2/act_env_B_inferred_500/checkpoints/000500/pretrained_model` | Complete for checked small runs; B labels are inferred from HF episode order |
| Train ACT on A/B/C mixed | `data/task2/lerobot_v30/calvin_ABC_small`, `results/task2/act_env_ABC_small/checkpoints/000100/pretrained_model`, `results/task2/act_env_ABC_small_500/checkpoints/000500/pretrained_model`; larger ABC 0:40 download attempt stalled after 2 episodes and is logged in `results/task2/download_abc_40_full_attempt.json` | Complete for checked small runs |
| Zero-shot evaluation on D | `data/task2/lerobot_v30/calvin_D_eval`, `results/task2/eval_B_on_D.json`, `results/task2/eval_ABC_on_D.json`, `results/task2/eval_B500_on_D.json`, `results/task2/eval_ABC500_on_D.json` | Complete for offline Action L1 |
| Compare metrics | `results/task2/act_eval_summary.csv`, `report/figures/task2_action_l1.pdf` | Present |
| Package weights | `weights/cv_hw3_task2_act_weights.tar.gz` | Present |

## AIGC Environment Verification

- `scripts/setup/clone_external.sh` now downloads local archive copies of tiny-cuda-nn, nvdiffrast, envlight, CLIP, and cubvh dependencies so installation does not depend on fragile nested pip/Git clones.
- `scripts/setup/install_threestudio_env.sh` installs local tiny-cuda-nn/nvdiffrast/envlight/CLIP sources, replaces the nerfacc Git requirement with `nerfacc==0.5.2`, pins `libigl==2.4.1` and `huggingface_hub==0.25.2`, and skips optional Gradio/xformers unless explicitly enabled.
- `scripts/setup/install_magic123_env.sh` builds Magic123 in `.venvs/magic123`, pins the older diffusers/transformers/huggingface-hub stack expected by Magic123, installs local nvdiffrast/CLIP/cubvh, installs `onnxruntime` for the `rembg` fallback, patches Magic123 CUDA extensions to C++17, and skips optional Shape-E unless explicitly enabled.

## Submission Metadata

The notebook report still needs real student names/IDs, contribution split, the public GitHub URL, and a permanent cloud link for `weights/cv_hw3_task2_act_weights.tar.gz`.

## Strict Gate

Run:

```bash
python scripts/report/check_submission_ready.py --json results/submission_readiness.json
```

Current expected failure is only unfilled report metadata. The gate should pass after real student names/IDs, GitHub URL, contribution split, and permanent model-weight link are filled in the first markdown cell of `report/HW3_report.ipynb`.
