#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_REPO_ID=""
DATASET_ROOT=""
OUTPUT=""
JOB_NAME="act_calvin"
GPU="0"
STEPS="100000"
BATCH_SIZE="64"
NUM_WORKERS="8"
LOG_FREQ="200"
SAVE_FREQ="10000"
EVAL_FREQ="0"
WANDB_ENABLE="true"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-repo-id) DATASET_REPO_ID="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --log-freq) LOG_FREQ="$2"; shift 2 ;;
    --save-freq) SAVE_FREQ="$2"; shift 2 ;;
    --eval-freq) EVAL_FREQ="$2"; shift 2 ;;
    --wandb-enable) WANDB_ENABLE="$2"; shift 2 ;;
    --extra) EXTRA_ARGS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${DATASET_REPO_ID}" || -z "${DATASET_ROOT}" || -z "${OUTPUT}" ]]; then
  echo "Usage: train_act_lerobot.sh --dataset-repo-id ID --dataset-root ROOT --output OUT" >&2
  exit 1
fi

resolve_path() {
  if [[ "$1" = /* ]]; then
    printf '%s' "$1"
  else
    printf '%s/%s' "${ROOT_DIR}" "$1"
  fi
}

DATASET_ROOT_ABS="$(resolve_path "${DATASET_ROOT}")"
OUTPUT_ABS="$(resolve_path "${OUTPUT}")"

CMD=(
  lerobot-train
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT_ABS}"
  --policy.type=act
  --policy.push_to_hub=false
  --output_dir="${OUTPUT_ABS}"
  --job_name="${JOB_NAME}"
  --policy.device=cuda
  --steps="${STEPS}"
  --batch_size="${BATCH_SIZE}"
  --num_workers="${NUM_WORKERS}"
  --log_freq="${LOG_FREQ}"
  --save_freq="${SAVE_FREQ}"
  --eval_freq="${EVAL_FREQ}"
  --wandb.enable="${WANDB_ENABLE}"
)

echo "[run] CUDA_VISIBLE_DEVICES=${GPU} ${CMD[*]} ${EXTRA_ARGS}"
CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" ${EXTRA_ARGS}
