#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/submission"
STAGE="${OUT_DIR}/stage"
BUNDLE="${OUT_DIR}/cv_hw3_submission_bundle.tar.gz"

mkdir -p "${OUT_DIR}"
rm -rf "${STAGE}"
mkdir -p "${STAGE}"

copy_path() {
  local src="$1"
  local dst="${STAGE}/${src}"
  if [[ -e "${ROOT_DIR}/${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    cp -a "${ROOT_DIR}/${src}" "${dst}"
  else
    echo "Warning: missing ${src}" >&2
  fi
}

copy_path ".gitignore"
copy_path "HW3_计算机视觉.pdf"
copy_path "README.md"
copy_path "environment.yml"
copy_path "configs"
copy_path "docs"
copy_path "scripts"
copy_path "src"
copy_path "report/main.tex"
copy_path "report/references.bib"
copy_path "report/metadata.example.json"
copy_path "report/figures"
copy_path "report/build/main.pdf"
copy_path "results/submission_manifest.json"
copy_path "results/submission_readiness.json"
copy_path "results/submission_readiness.log"
copy_path "results/task1/asset_stats.json"
copy_path "results/task1/task1_2dgs_smoke_stats.json"
copy_path "results/task1/2dgs/object_A_synthetic_smoke/train/ours_200/fuse_post.ply"
copy_path "results/task1/background_garden_sparse_smoke_stats.json"
copy_path "results/task1/fusion_sparse_bg_smoke/fusion_preview.png"
copy_path "results/task1/fusion_sparse_bg_smoke/fusion_walkthrough.mp4"
copy_path "results/task1/demo_assets/provenance.json"
copy_path "results/task1/fusion_demo/fusion_preview.png"
copy_path "results/task1/fusion_demo/fusion_walkthrough.mp4"
copy_path "results/task1/real_pipeline_dryrun.log"
copy_path "results/task2/act_eval_summary.csv"
copy_path "results/task2/eval_B_on_D.json"
copy_path "results/task2/eval_ABC_on_D.json"
copy_path "results/task2/eval_B500_on_D.json"
copy_path "results/task2/eval_ABC500_on_D.json"
copy_path "results/task2/download_abc_40_full_attempt.json"
copy_path "results/task2/logs"
copy_path "weights/cv_hw3_task2_act_weights.tar.gz"

find "${STAGE}" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${STAGE}" -type f -name "*.pyc" -delete

tar -czf "${BUNDLE}" -C "${STAGE}" .
echo "Submission bundle: ${BUNDLE}"
