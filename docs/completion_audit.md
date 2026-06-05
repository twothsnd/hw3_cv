# Completion Audit

This audit records the current local state against `HW3_计算机视觉.pdf`.

## Task 1

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Object A from phone multi-view/video + COLMAP + 2DGS | scripts exist: `extract_frames.py`, `run_colmap.py`, `train_2dgs.sh`, `render_2dgs.sh`; current demo artifact `results/task1/2dgs/object_A/train/ours_latest/fuse_post.ply` is synthetic provenance; separate 2DGS smoke mesh `results/task1/2dgs/object_A_synthetic_smoke/train/ours_200/fuse_post.ply` was trained/exported from synthetic NeRF views | Not complete for final grading until `data/raw/object_A` contains real capture and the pipeline is rerun |
| Object B text-to-3D with threestudio/SDS | `generate_text3d_threestudio.sh` now auto-selects `.venvs/threestudio`; threestudio imports, `launch.py --help`, and `pip check` pass locally; current artifact `results/task1/aigc/object_B_text3d/export/model.obj` is synthetic demo provenance | Environment and pipeline verified; final grading still needs a real SDS run |
| Object C single image to 3D with Magic123 | `generate_image3d_magic123.sh` now auto-selects `.venvs/magic123`; Magic123 imports, `main.py --help`, `preprocess_image.py --help`, CUDA extension imports, and `pip check` pass locally; current artifact `results/task1/aigc/object_C_image3d/model.obj` is synthetic demo provenance | Environment and pipeline verified; final grading still needs `data/raw/object_C/object_c.png` and a real Magic123 run |
| Mip-NeRF 360 background reconstructed with 2DGS | scripts exist; current `background_garden` mesh is synthetic demo provenance; public NerfBaselines sparse garden-n24 thumbnail run produced `results/task1/2dgs/background_garden_sparse_smoke/train/ours_300/fuse_unbounded_post.ply` with 1,768,391 vertices / 3,563,531 faces | Pipeline prepared and public-data smoke tested; final full-resolution Mip-NeRF 360 run still required |
| Fused scene and walkthrough video | `results/task1/fusion_demo/fusion_preview.png`, `fusion_scene.blend`, `fusion_walkthrough.mp4`; sparse-background smoke video `results/task1/fusion_sparse_bg_smoke/fusion_walkthrough.mp4`; 2DGS smoke stats in `results/task1/task1_2dgs_smoke_stats.json` and `results/task1/background_garden_sparse_smoke_stats.json` | Complete as renderer and 2DGS smoke tests; must rerun with real/generated assets |
| Representation unification explanation | `report/main.tex` explains mesh export and Blender fusion | Present |

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

The report still needs real student names/IDs, the public GitHub URL, and a permanent cloud link for `weights/cv_hw3_task2_act_weights.tar.gz`.

## Strict Gate

Run:

```bash
python scripts/report/check_submission_ready.py --json results/submission_readiness.json
```

Current expected failures are the missing phone-captured Task 1 inputs, missing real Task 1 run manifest/COLMAP/fusion artifacts, unfilled report metadata, and the presence of synthetic demo provenance without a real-run manifest. The gate should pass only after `scripts/task1/run_real_pipeline.sh` is rerun with real object A/C data and report metadata is filled.
