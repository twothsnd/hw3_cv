#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${ROOT_DIR}/external/2d-gaussian-splatting"
DATASET=""
OUTPUT=""
GPU="0"
ITERATIONS="30000"
RESOLUTION=""
EVAL_FLAG="--eval"
LAMBDA_NORMAL="0.05"
LAMBDA_DISTORTION="100"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --resolution) RESOLUTION="$2"; shift 2 ;;
    --no-eval) EVAL_FLAG=""; shift ;;
    --lambda-normal) LAMBDA_NORMAL="$2"; shift 2 ;;
    --lambda-distortion) LAMBDA_DISTORTION="$2"; shift 2 ;;
    --extra) EXTRA_ARGS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${DATASET}" || -z "${OUTPUT}" ]]; then
  echo "Usage: train_2dgs.sh --dataset DATASET --output OUTPUT [--gpu 0]" >&2
  exit 1
fi

if [[ ! -d "${REPO}" ]]; then
  echo "Missing ${REPO}. Run scripts/setup/clone_external.sh first." >&2
  exit 1
fi

mkdir -p "$(dirname "${ROOT_DIR}/${OUTPUT}")"

CMD=(python train.py -s "${ROOT_DIR}/${DATASET}" -m "${ROOT_DIR}/${OUTPUT}" --iterations "${ITERATIONS}" --lambda_normal "${LAMBDA_NORMAL}" --lambda_dist "${LAMBDA_DISTORTION}")
if [[ -n "${EVAL_FLAG}" ]]; then
  CMD+=("${EVAL_FLAG}")
fi
if [[ -n "${RESOLUTION}" ]]; then
  CMD+=(-r "${RESOLUTION}")
fi

echo "[run] CUDA_VISIBLE_DEVICES=${GPU} ${CMD[*]} ${EXTRA_ARGS}"
cd "${REPO}"
CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" ${EXTRA_ARGS}
