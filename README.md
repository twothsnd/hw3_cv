# CV HW3: 2DGS/AIGC 3D Fusion and LeRobot ACT Generalization

This repository contains the code for Computer Vision HW3.

It implements two tasks:

1. **Multi-source 3D asset generation and real-scene fusion** with COLMAP, 2D Gaussian Splatting, threestudio/DreamFusion SDS, Magic123, and Blender mesh rendering.
2. **ACT policy cross-environment generalization** with LeRobot on the CALVIN A/B/C/D environment splits.

Large datasets, generated meshes/videos, trained checkpoints, and the final report are not tracked in Git. They should be downloaded or placed locally according to the instructions below.

## Repository Layout

```text
configs/                 # YAML/JSON configs for Task 1 and fusion rendering
docs/                    # audit notes and handoff docs
scripts/
  setup/                 # environment and third-party dependency setup
  task1/                 # COLMAP, 2DGS, AIGC, mesh cleanup, Blender rendering
  task2/                 # CALVIN conversion, ACT training, offline evaluation
  report/                # figure generation, readiness checks, packaging helpers
src/cv_hw3/              # shared Python utilities
environment.yml          # lightweight utility conda environment
README.md
```

Expected local, untracked working directories:

```text
data/                    # raw inputs and converted datasets
external/                # cloned third-party repositories
results/                 # generated reconstructions, logs, metrics, videos
weights/                 # packaged trained model weights for cloud upload
tools/                   # optional local Blender/ffmpeg binaries
.venvs/                  # Python virtual environments
report/                  # local report notebook/PDF and generated figures
```

## Requirements

Tested environment:

- Linux with CUDA GPU
- Python 3.10 for LeRobot utilities
- CUDA-enabled PyTorch environments for 2DGS, threestudio, Magic123, and LeRobot
- COLMAP or `pycolmap`
- Blender
- ffmpeg

Create the lightweight utility environment:

```bash
conda env create -f environment.yml
conda activate cv_hw3_utils
```

Clone third-party code:

```bash
bash scripts/setup/clone_external.sh
```

Install utility dependencies:

```bash
bash scripts/setup/install_utils_env.sh
source .venvs/hw3-utils/bin/activate
```

Install heavy environments only when needed:

```bash
bash scripts/setup/install_2dgs_env.sh
bash scripts/setup/install_lerobot_env.sh
bash scripts/setup/install_threestudio_env.sh
bash scripts/setup/install_magic123_env.sh
```

Install local Blender/ffmpeg helpers if system binaries are unavailable:

```bash
bash scripts/setup/install_local_tools.sh
export PATH="$PWD/tools/bin:$PATH"
```

## Data Preparation

### Task 1 Inputs

Prepare local private inputs:

```text
data/raw/object_A/                # phone video or multi-view photos of object A
data/raw/object_C/object_c.png    # foreground single image for object C
```

Download the Mip-NeRF 360 `garden` scene:

```bash
mkdir -p data/task1/mipnerf360_full
wget -O data/task1/mipnerf360_full/garden.zip \
  https://storage.googleapis.com/gresearch/refraw360/garden.zip
unzip -q data/task1/mipnerf360_full/garden.zip -d data/task1/mipnerf360_full
```

### Task 2 Inputs

This project uses the official helper split dataset:

```text
xiaoma26/calvin-lerobot
```

Download and convert the required subsets:

```bash
source .venvs/lerobot/bin/activate

python scripts/task2/download_xiaoma_calvin_subset.py \
  --repo-id xiaoma26/calvin-lerobot \
  --output-root data/task2/xiaoma_calvin \
  --splits A B C D

python scripts/task2/convert_xiaoma_calvin_to_v30.py \
  --input-root data/task2/xiaoma_calvin \
  --output-root data/task2/lerobot_v30_official \
  --overwrite
```

The final experiments use:

```text
data/task2/lerobot_v30_official/calvin_B_train80
data/task2/lerobot_v30_official/calvin_B_val20
data/task2/lerobot_v30_official/calvin_ABC_train40each
data/task2/lerobot_v30_official/calvin_ABC_val10each
data/task2/lerobot_v30_official/calvin_D_eval_100
```

## Task 1: Train and Render

Extract object-A frames:

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

Train object-A 2DGS:

```bash
source .venvs/2dgs/bin/activate

bash scripts/task1/train_2dgs.sh \
  --dataset data/task1/object_A_colmap \
  --output results/task1/2dgs/object_A \
  --gpu 0 \
  --iterations 20000
```

Train garden 2DGS:

```bash
source .venvs/2dgs/bin/activate

bash scripts/task1/train_2dgs.sh \
  --dataset data/task1/mipnerf360_full/garden \
  --output results/task1/2dgs/background_garden_full \
  --gpu 0 \
  --iterations 30000 \
  --resolution 1 \
  --no-eval \
  --extra "-i images_2 --test_iterations 30000 --save_iterations 30000 --checkpoint_iterations 30000 --quiet --depth_ratio 0"
```

Extract 2DGS meshes:

```bash
bash scripts/task1/render_2dgs.sh \
  --dataset data/task1/object_A_colmap \
  --model results/task1/2dgs/object_A \
  --gpu 0 \
  --mesh-mode bounded

bash scripts/task1/render_2dgs.sh \
  --dataset data/task1/mipnerf360_full/garden \
  --model results/task1/2dgs/background_garden_full \
  --gpu 0 \
  --mesh-mode unbounded \
  --mesh-res 512 \
  --resolution 1 \
  --extra "-i images_2 --iteration 30000 --num_cluster 20 --quiet"
```

Generate object B with threestudio/DreamFusion SDS:

```bash
bash scripts/task1/generate_text3d_threestudio.sh \
  --prompt "a simple small toy car, four clearly separated black wheels, rounded plastic body, compact low car shape, symmetric side profile, single object, centered" \
  --name object_B_text3d \
  --gpu 0 \
  --max-steps 20000
```

Generate object C with Magic123:

```bash
bash scripts/task1/generate_image3d_magic123.sh \
  --image data/raw/object_C/object_c.png \
  --name object_C_image3d \
  --gpu 0 \
  --no-depth \
  --guidance-mode zero123
```

Collect mesh statistics:

```bash
python scripts/task1/collect_asset_stats.py \
  --assets \
    object_A=results/task1/2dgs/object_A/train/ours_latest/fuse_post.ply \
    object_B=results/task1/aigc/object_B_text3d/export/model.obj \
    object_C=results/task1/aigc/object_C_image3d/model.obj \
    background=results/task1/2dgs/background_garden_full/train/ours_latest/fuse_unbounded_post_crop_q02_98.ply \
  --output results/task1/asset_stats.json
```

Render final mesh fusion in Blender:

```bash
PATH="$PWD/tools/bin:$PATH" blender -b \
  --python scripts/task1/render_mesh_table_fusion.py -- \
  --config configs/fusion_mesh_table.json
```

## Task 2: Train and Test

Train B-only ACT with online validation:

```bash
CUDA_VISIBLE_DEVICES=0 .venvs/lerobot/bin/python scripts/task2/train_act_with_online_val.py \
  --train-repo-id cv_hw3/calvin_official_B_train80 \
  --train-root data/task2/lerobot_v30_official/calvin_B_train80 \
  --val-repo-id cv_hw3/calvin_official_B_val20 \
  --val-root data/task2/lerobot_v30_official/calvin_B_val20 \
  --output results/task2/online_act_B_train80_val20_10k \
  --job-name online_act_B_train80_val20_10k \
  --steps 10000 \
  --batch-size 16 \
  --num-workers 8 \
  --log-freq 100 \
  --val-freq 1000 \
  --save-freq 5000 \
  --val-max-batches 0 \
  --device cuda
```

Train A/B/C mixed ACT with online validation:

```bash
CUDA_VISIBLE_DEVICES=0 .venvs/lerobot/bin/python scripts/task2/train_act_with_online_val.py \
  --train-repo-id cv_hw3/calvin_official_ABC_train40each \
  --train-root data/task2/lerobot_v30_official/calvin_ABC_train40each \
  --val-repo-id cv_hw3/calvin_official_ABC_val10each \
  --val-root data/task2/lerobot_v30_official/calvin_ABC_val10each \
  --output results/task2/online_act_ABC_train40each_val10each_10k \
  --job-name online_act_ABC_train40each_val10each_10k \
  --steps 10000 \
  --batch-size 16 \
  --num-workers 8 \
  --log-freq 100 \
  --val-freq 1000 \
  --save-freq 5000 \
  --val-max-batches 0 \
  --device cuda
```

Evaluate both policies on unseen environment D:

```bash
CUDA_VISIBLE_DEVICES=0 .venvs/lerobot/bin/python scripts/task2/eval_action_l1.py \
  --policy-path results/task2/online_act_B_train80_val20_10k/checkpoints/010000/pretrained_model \
  --dataset-repo-id cv_hw3/calvin_official_D_eval_100 \
  --dataset-root data/task2/lerobot_v30_official/calvin_D_eval_100 \
  --batch-size 16 \
  --num-workers 8 \
  --device cuda \
  --output results/task2/online_eval_B_on_D_100.json

CUDA_VISIBLE_DEVICES=0 .venvs/lerobot/bin/python scripts/task2/eval_action_l1.py \
  --policy-path results/task2/online_act_ABC_train40each_val10each_10k/checkpoints/010000/pretrained_model \
  --dataset-repo-id cv_hw3/calvin_official_D_eval_100 \
  --dataset-root data/task2/lerobot_v30_official/calvin_D_eval_100 \
  --batch-size 16 \
  --num-workers 8 \
  --device cuda \
  --output results/task2/online_eval_ABC_on_D_100.json
```

## Report Figures and Checks

Generate figures and rebuild the local notebook report:

```bash
.venvs/lerobot/bin/python scripts/report/make_task2_official_figures.py
.venvs/lerobot/bin/python scripts/report/make_plots.py \
  --act-evals results/task2/online_eval_B_on_D_100.json results/task2/online_eval_ABC_on_D_100.json \
  --act-labels B-only ABC-mixed \
  --output-dir report/figures
.venvs/lerobot/bin/python scripts/report/build_notebook_report.py
```

Package ACT weights for cloud upload:

```bash
bash scripts/report/package_weights.sh
```

Run the final readiness gate:

```bash
.venvs/lerobot/bin/python scripts/report/check_submission_ready.py \
  --json results/submission_readiness.json
```

## Model Weights

Model weights are not tracked in Git.

After running `scripts/report/package_weights.sh`, upload this file to cloud storage:

```text
weights/cv_hw3_task2_act_weights.tar.gz
```

The archive contains:

- B-only ACT 10000-step checkpoint
- ABC-mixed ACT 10000-step checkpoint
- D-environment evaluation JSON files
- online training/validation metric CSV files

When reproducing from a downloaded archive, extract it at the repository root:

```bash
tar -xzf weights/cv_hw3_task2_act_weights.tar.gz -C .
```

## GitHub Submission

Only source code, configs, documentation, and lightweight metadata should be pushed to GitHub. Do not commit:

- `data/`
- `external/`
- `results/`
- `weights/`
- `.venvs/`
- `tools/`
- final report notebook/PDF

The final report and model weights should be submitted separately according to the course requirements.
