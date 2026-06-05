#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/weights"
mkdir -p "${OUT_DIR}"

tar -czf "${OUT_DIR}/cv_hw3_task2_act_weights.tar.gz" \
  -C "${ROOT_DIR}" \
  results/task2/act_env_B_inferred/checkpoints/000100/pretrained_model \
  results/task2/act_env_ABC_small/checkpoints/000100/pretrained_model \
  results/task2/act_env_B_inferred_500/checkpoints/000500/pretrained_model \
  results/task2/act_env_ABC_small_500/checkpoints/000500/pretrained_model \
  results/task2/eval_B_on_D.json \
  results/task2/eval_ABC_on_D.json \
  results/task2/eval_B500_on_D.json \
  results/task2/eval_ABC500_on_D.json \
  results/task2/act_eval_summary.csv \
  results/task2/logs/train_B_losslog.log \
  results/task2/logs/train_ABC_losslog.log \
  results/task2/logs/train_B_500.log \
  results/task2/logs/train_ABC_500.log \
  results/task2/download_abc_40_full_attempt.json \
  report/figures/task2_training_loss.csv \
  results/submission_readiness.json \
  results/submission_manifest.json

echo "Packaged weights: ${OUT_DIR}/cv_hw3_task2_act_weights.tar.gz"
