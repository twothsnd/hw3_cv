#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/weights"
mkdir -p "${OUT_DIR}"

tar -czf "${OUT_DIR}/cv_hw3_task2_act_weights.tar.gz" \
  -C "${ROOT_DIR}" \
  results/task2/online_act_B_train80_val20_10k/checkpoints/010000/pretrained_model \
  results/task2/online_act_ABC_train40each_val10each_10k/checkpoints/010000/pretrained_model \
  results/task2/online_eval_B_on_D_100.json \
  results/task2/online_eval_ABC_on_D_100.json \
  results/task2/act_eval_summary.csv \
  results/task2/online_act_B_train80_val20_10k/online_metrics.csv \
  results/task2/online_act_ABC_train40each_val10each_10k/online_metrics.csv \
  results/task2/logs/online_train_B_train80_val20_10k.log \
  results/task2/logs/online_train_ABC_train40each_val10each_10k.log \
  report/figures/task2_official_training_loss.csv \
  results/submission_readiness.json \
  results/submission_manifest.json

echo "Packaged weights: ${OUT_DIR}/cv_hw3_task2_act_weights.tar.gz"
