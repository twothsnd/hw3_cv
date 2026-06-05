# CV HW3: 2DGS/AIGC Fusion and LeRobot ACT on CALVIN

This repository is the reproducible project scaffold for both required parts of `HW3_计算机视觉.pdf`.

Deadline: 2026-06-23 23:59 Beijing time.

## What Is Implemented

Task 1:

- real object multi-view/video preprocessing
- COLMAP reconstruction wrapper
- 2D Gaussian Splatting training and mesh/video extraction
- text-to-3D generation through threestudio DreamFusion/SDS
- single-image-to-3D generation through Magic123
- mesh normalization, asset statistics, and Blender-based final scene fusion

Task 2:

- CALVIN raw dataset to LeRobotDataset conversion with environment filtering
- HuggingFace CALVIN LeRobot v2.1 subset download and local v3 conversion
- ACT training for environment B and mixed environments A/B/C
- offline Action L1 validation
- optional CALVIN official simulator evaluation through a LeRobot-to-CALVIN adapter
- metric summarization for report tables

The scripts are designed to be run from the repository root.

## Directory Contract

```text
data/
  raw/
    object_A/                # phone video or multi-view photos for real object A
    object_C/object_c.png    # single foreground object image for Magic123
  task1/
    object_A_colmap/
    mipnerf360/garden/
  task2/
    calvin_raw/task_ABCD_D/
    lerobot/calvin_B/
    lerobot/calvin_ABC/
    lerobot/calvin_D_validation/

external/                    # cloned third-party repositories
results/
  task1/
  task2/
weights/                     # final checkpoint packages for cloud upload
report/                      # LaTeX report
```

Large data, generated results, third-party repos, and weights are intentionally ignored by Git.

## Setup

For lightweight utilities, use the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate cv_hw3_utils
```

Clone pinned third-party code:

```bash
bash scripts/setup/clone_external.sh
```

Install common utility dependencies:

```bash
bash scripts/setup/install_utils_env.sh
source .venvs/hw3-utils/bin/activate
```

Install heavy training environments only when you are ready to run them:

```bash
bash scripts/setup/install_2dgs_env.sh
bash scripts/setup/install_lerobot_env.sh
bash scripts/setup/install_threestudio_env.sh
bash scripts/setup/install_magic123_env.sh
```

COLMAP, Blender, and ffmpeg are system-level tools. If they are not available in `PATH`, the repository can install local Blender/ffmpeg binaries and uses the `pycolmap` backend as a COLMAP fallback:

```bash
bash scripts/setup/install_local_tools.sh
export PATH="$PWD/tools/bin:$PATH"
```

## Task 1 Workflow

The real submission requires phone-captured object A multi-view/video data and one foreground image for object C. If those inputs are absent, this smoke test creates explicitly synthetic demo meshes and verifies the fusion renderer:

```bash
PATH="$PWD/tools/bin:$PATH" blender -b \
  --python scripts/task1/create_synthetic_demo_assets.py -- \
  --config configs/fusion_scene_demo.json

python scripts/task1/collect_asset_stats.py \
  --assets \
    object_A=results/task1/2dgs/object_A/train/ours_latest/fuse_post.ply \
    object_B=results/task1/aigc/object_B_text3d/export/model.obj \
    object_C=results/task1/aigc/object_C_image3d/model.obj \
    background=results/task1/2dgs/background_garden/train/ours_latest/fuse_unbounded_post.ply \
  --output results/task1/asset_stats.json

PATH="$PWD/tools/bin:$PATH" bash scripts/task1/render_fusion_blender.sh configs/fusion_scene_demo.json
```

This writes `results/task1/fusion_demo/fusion_preview.png`, `fusion_scene.blend`, and `fusion_walkthrough.mp4`. Replace these synthetic meshes with the real 2DGS/threestudio/Magic123 outputs for final grading.

To separately verify that the 2DGS training and mesh extraction path is executable, generate a synthetic NeRF-style object-A dataset and run a short 2DGS smoke test:

```bash
PATH="$PWD/tools/bin:$PATH" blender -b \
  --python scripts/task1/create_synthetic_nerf_dataset.py -- \
  --output data/task1/synthetic_object_A_nerf \
  --train-views 24 \
  --test-views 6 \
  --resolution 256

source .venvs/2dgs/bin/activate

bash scripts/task1/train_2dgs.sh \
  --dataset data/task1/synthetic_object_A_nerf \
  --output results/task1/2dgs/object_A_synthetic_smoke \
  --gpu 0 \
  --iterations 200 \
  --resolution 2 \
  --no-eval \
  --extra "--test_iterations 200 --save_iterations 200 --quiet"

bash scripts/task1/render_2dgs.sh \
  --dataset data/task1/synthetic_object_A_nerf \
  --model results/task1/2dgs/object_A_synthetic_smoke \
  --gpu 0 \
  --mesh-mode bounded \
  --mesh-res 64 \
  --resolution 2 \
  --extra "--iteration 200 --num_cluster 5"

python scripts/task1/collect_asset_stats.py \
  --assets object_A_2dgs_synthetic_smoke=results/task1/2dgs/object_A_synthetic_smoke/train/ours_200/fuse_post.ply \
  --output results/task1/task1_2dgs_smoke_stats.json
```

This is a pipeline validation artifact only. It does not satisfy the phone-captured object requirement.

To verify the background 2DGS path without downloading the full 12.5GB Mip-NeRF 360 archive, use the small NerfBaselines Mip-NeRF 360 sparse garden-n24 files:

```bash
bash scripts/task1/download_nerfbaselines_mipnerf360_sparse.sh garden 24

source .venvs/2dgs/bin/activate

python scripts/task1/convert_nerfbaselines_sparse_to_2dgs.py \
  --nbv-json data/task1/mipnerf360_sparse_garden_n24/raw/garden-n24-nbv.json \
  --pointcloud data/task1/mipnerf360_sparse_garden_n24/raw/garden-n24-pointcloud.ply \
  --output data/task1/mipnerf360_sparse_garden_n24/2dgs_dataset \
  --overwrite

bash scripts/task1/train_2dgs.sh \
  --dataset data/task1/mipnerf360_sparse_garden_n24/2dgs_dataset \
  --output results/task1/2dgs/background_garden_sparse_smoke \
  --gpu 0 \
  --iterations 300 \
  --resolution 1 \
  --no-eval \
  --extra "--test_iterations 300 --save_iterations 300 --quiet --depth_ratio 0"

bash scripts/task1/render_2dgs.sh \
  --dataset data/task1/mipnerf360_sparse_garden_n24/2dgs_dataset \
  --model results/task1/2dgs/background_garden_sparse_smoke \
  --gpu 0 \
  --mesh-mode unbounded \
  --mesh-res 512 \
  --resolution 1 \
  --extra "--iteration 300 --num_cluster 20"

python scripts/task1/collect_asset_stats.py \
  --assets background_garden_sparse_2dgs=results/task1/2dgs/background_garden_sparse_smoke/train/ours_300/fuse_unbounded_post.ply \
  --output results/task1/background_garden_sparse_smoke_stats.json

PATH="$PWD/tools/bin:$PATH" bash scripts/task1/render_fusion_blender.sh configs/fusion_scene_sparse_bg_smoke.json
```

This writes `results/task1/fusion_sparse_bg_smoke/fusion_walkthrough.mp4`. It is a public-data smoke test based on 163x105 thumbnails, not a substitute for the final full-resolution Mip-NeRF 360 background run.

Prepare object A frames:

```bash
python scripts/task1/extract_frames.py \
  --input data/raw/object_A \
  --output data/task1/object_A_colmap/images \
  --max-frames 180 \
  --resize-long-edge 1600
```

Run COLMAP:

```bash
python scripts/task1/run_colmap.py \
  --images data/task1/object_A_colmap/images \
  --dataset data/task1/object_A_colmap \
  --matcher exhaustive
```

Train 2DGS for object A and for the background:

```bash
bash scripts/task1/train_2dgs.sh \
  --dataset data/task1/object_A_colmap \
  --output results/task1/2dgs/object_A \
  --gpu 0 \
  --iterations 30000

bash scripts/task1/download_mipnerf360.sh garden data/task1/mipnerf360

bash scripts/task1/train_2dgs.sh \
  --dataset data/task1/mipnerf360/garden \
  --output results/task1/2dgs/background_garden \
  --gpu 1 \
  --iterations 30000 \
  --extra "--depth_ratio 0"
```

Extract meshes and trajectory video:

```bash
bash scripts/task1/render_2dgs.sh \
  --dataset data/task1/object_A_colmap \
  --model results/task1/2dgs/object_A \
  --gpu 0 \
  --mesh-mode bounded

bash scripts/task1/render_2dgs.sh \
  --dataset data/task1/mipnerf360/garden \
  --model results/task1/2dgs/background_garden \
  --gpu 1 \
  --mesh-mode unbounded \
  --render-path 1
```

Generate object B:

```bash
bash scripts/task1/generate_text3d_threestudio.sh \
  --prompt "a small ceramic robot toy, high quality DSLR photo" \
  --name object_B_text3d \
  --gpu 2 \
  --max-steps 10000
```

Generate object C:

```bash
bash scripts/task1/generate_image3d_magic123.sh \
  --image data/raw/object_C/object_c.png \
  --text "a high-resolution DSLR image of the object" \
  --name object_C_image3d \
  --gpu 3
```

Collect mesh statistics:

```bash
python scripts/task1/collect_asset_stats.py \
  --assets \
    object_A=results/task1/2dgs/object_A/train/ours_latest/fuse_post.ply \
    object_B=results/task1/aigc/object_B_text3d/export/model.obj \
    object_C=results/task1/aigc/object_C_image3d/model.obj \
    background=results/task1/2dgs/background_garden/train/ours_latest/fuse_unbounded_post.ply \
  --output results/task1/asset_stats.json
```

Render the fused scene in Blender:

```bash
bash scripts/task1/render_fusion_blender.sh configs/fusion_scene.json
```

After real object A/C inputs are available, the full Task 1 path can be launched with:

```bash
PATH="$PWD/tools/bin:$PATH" bash scripts/task1/run_real_pipeline.sh \
  --object-a-input data/raw/object_A \
  --object-c-image data/raw/object_C/object_c.png \
  --object-b-prompt "a small ceramic robot toy, high quality DSLR photo" \
  --scene garden \
  --object-a-gpu 0 \
  --background-gpu 1 \
  --text3d-gpu 2 \
  --image3d-gpu 3
```

For a quick command audit without running heavy training, use:

```bash
bash scripts/task1/run_real_pipeline.sh --dry-run
```

## Task 2 Workflow

Download CALVIN with the official script from `external/calvin/dataset`. Use `ABCD` if you need B-only and A/B/C filtering from the same source, and use the D validation split for zero-shot testing.

For the checked lightweight run, the CALVIN HuggingFace datasets are downloaded as LeRobot v2.1 parquet subsets and converted locally to LeRobot v3:

```bash
source .venvs/lerobot/bin/activate

python scripts/task2/download_hf_lerobot_subset.py \
  --repo-id fywang/calvin-task-D-D-lerobot \
  --output-root data/task2/lerobot_hf/d_eval_20 \
  --episodes 0:20 --revision main --overwrite

python scripts/task2/download_hf_lerobot_subset.py \
  --repo-id fywang/calvin-task-ABCD-D-lerobot \
  --output-root data/task2/lerobot_hf/b_inferred_40 \
  --episodes 6500:6540 --revision main --overwrite

python scripts/task2/download_hf_lerobot_subset.py \
  --repo-id fywang/calvin-task-ABC-D-lerobot \
  --output-root data/task2/lerobot_hf/abc_train_40 \
  --episodes 0:40 --revision main --overwrite

python scripts/task2/convert_hf_v21_subset_to_v30.py \
  --source-root data/task2/lerobot_hf/d_eval_20 \
  --output-root data/task2/lerobot_v30/calvin_D_eval \
  --repo-id cv_hw3/calvin_D_eval --overwrite

python scripts/task2/convert_hf_v21_subset_to_v30.py \
  --source-root data/task2/lerobot_hf/b_inferred_40 \
  --output-root data/task2/lerobot_v30/calvin_B_inferred \
  --repo-id cv_hw3/calvin_B_inferred \
  --drop-features observation.environment_state --overwrite

python scripts/task2/convert_hf_v21_subset_to_v30.py \
  --source-root data/task2/lerobot_hf/abc_train_40 \
  --output-root data/task2/lerobot_v30/calvin_ABC_small \
  --repo-id cv_hw3/calvin_ABC_small --overwrite
```

The `b_inferred_40` range is inferred from the official CALVIN scene order because the HF parquet metadata does not include per-episode environment labels.

Convert environment B:

```bash
python scripts/task2/prepare_calvin_lerobot.py \
  --calvin-root data/task2/calvin_raw/task_ABCD_D \
  --split training \
  --environments B \
  --repo-id cv_hw3/calvin_B \
  --output-root data/task2/lerobot/calvin_B
```

Convert environments A/B/C:

```bash
python scripts/task2/prepare_calvin_lerobot.py \
  --calvin-root data/task2/calvin_raw/task_ABCD_D \
  --split training \
  --environments A B C \
  --repo-id cv_hw3/calvin_ABC \
  --output-root data/task2/lerobot/calvin_ABC
```

Train ACT:

```bash
bash scripts/task2/train_act_lerobot.sh \
  --dataset-repo-id cv_hw3/calvin_B \
  --dataset-root data/task2/lerobot/calvin_B \
  --output results/task2/act_env_B \
  --job-name act_env_B \
  --gpu 0 \
  --steps 100000 \
  --batch-size 64

bash scripts/task2/train_act_lerobot.sh \
  --dataset-repo-id cv_hw3/calvin_ABC \
  --dataset-root data/task2/lerobot/calvin_ABC \
  --output results/task2/act_env_ABC \
  --job-name act_env_ABC \
  --gpu 1 \
  --steps 100000 \
  --batch-size 64
```

Offline zero-shot action-error evaluation on D:

```bash
python scripts/task2/eval_action_l1.py \
  --policy-path results/task2/act_env_B_inferred/checkpoints/000100/pretrained_model \
  --dataset-repo-id cv_hw3/calvin_D_eval \
  --dataset-root data/task2/lerobot_v30/calvin_D_eval \
  --output results/task2/eval_B_on_D.json

python scripts/task2/eval_action_l1.py \
  --policy-path results/task2/act_env_ABC_small/checkpoints/000100/pretrained_model \
  --dataset-repo-id cv_hw3/calvin_D_eval \
  --dataset-root data/task2/lerobot_v30/calvin_D_eval \
  --output results/task2/eval_ABC_on_D.json

python scripts/task2/eval_action_l1.py \
  --policy-path results/task2/act_env_B_inferred_500/checkpoints/000500/pretrained_model \
  --dataset-repo-id cv_hw3/calvin_D_eval \
  --dataset-root data/task2/lerobot_v30/calvin_D_eval \
  --output results/task2/eval_B500_on_D.json

python scripts/task2/eval_action_l1.py \
  --policy-path results/task2/act_env_ABC_small_500/checkpoints/000500/pretrained_model \
  --dataset-repo-id cv_hw3/calvin_D_eval \
  --dataset-root data/task2/lerobot_v30/calvin_D_eval \
  --output results/task2/eval_ABC500_on_D.json

python scripts/task2/summarize_act_metrics.py \
  --inputs results/task2/eval_B_on_D.json results/task2/eval_ABC_on_D.json results/task2/eval_B500_on_D.json results/task2/eval_ABC500_on_D.json \
  --labels B_100 ABC_100 B_500 ABC_500 \
  --output results/task2/act_eval_summary.csv
```

Optional official CALVIN success-rate evaluation:

```bash
python scripts/task2/eval_calvin_lerobot_act.py \
  --calvin-root external/calvin \
  --dataset-path data/task2/calvin_raw/task_D_D \
  --policy-path results/task2/act_env_ABC/checkpoints/last/pretrained_model \
  --eval-log-dir results/task2/calvin_success_env_ABC_on_D
```

## Report

Build the report after metrics and figures are generated:

```bash
bash scripts/report/build_report.sh
```

Package the trained ACT checkpoints and metrics:

```bash
bash scripts/report/package_weights.sh
```

Run the strict final submission gate:

```bash
python scripts/report/check_submission_ready.py --json results/submission_readiness.json
```

The report must include the public GitHub URL and model weight cloud link before submission.

To fill the report front-page metadata reproducibly, copy the example metadata file, edit it, then apply it:

```bash
cp report/metadata.example.json report/metadata.json
python scripts/report/fill_report_metadata.py --metadata report/metadata.json
bash scripts/report/build_report.sh
```

Create a local submission backup bundle:

```bash
bash scripts/report/package_submission_bundle.sh
```
