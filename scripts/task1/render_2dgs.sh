#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${ROOT_DIR}/external/2d-gaussian-splatting"
DATASET=""
MODEL=""
GPU="0"
MESH_MODE="bounded"
MESH_RES="1024"
RENDER_PATH="0"
RESOLUTION=""
DEPTH_RATIO=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --mesh-mode) MESH_MODE="$2"; shift 2 ;;
    --mesh-res) MESH_RES="$2"; shift 2 ;;
    --render-path) RENDER_PATH="$2"; shift 2 ;;
    --resolution) RESOLUTION="$2"; shift 2 ;;
    --depth-ratio) DEPTH_RATIO="$2"; shift 2 ;;
    --extra) EXTRA_ARGS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${DATASET}" || -z "${MODEL}" ]]; then
  echo "Usage: render_2dgs.sh --dataset DATASET --model MODEL [--mesh-mode bounded|unbounded]" >&2
  exit 1
fi

if [[ ! -d "${REPO}" ]]; then
  echo "Missing ${REPO}. Run scripts/setup/clone_external.sh first." >&2
  exit 1
fi

CMD=(python render.py -s "${ROOT_DIR}/${DATASET}" -m "${ROOT_DIR}/${MODEL}" --skip_train --skip_test --mesh_res "${MESH_RES}")
if [[ "${MESH_MODE}" == "unbounded" ]]; then
  CMD+=(--unbounded)
elif [[ "${MESH_MODE}" != "bounded" ]]; then
  echo "--mesh-mode must be bounded or unbounded." >&2
  exit 1
fi
if [[ "${RENDER_PATH}" == "1" ]]; then
  CMD+=(--render_path)
fi
if [[ -n "${RESOLUTION}" ]]; then
  CMD+=(-r "${RESOLUTION}")
fi
if [[ -n "${DEPTH_RATIO}" ]]; then
  CMD+=(--depth_ratio "${DEPTH_RATIO}")
fi

echo "[run] CUDA_VISIBLE_DEVICES=${GPU} ${CMD[*]} ${EXTRA_ARGS}"
cd "${REPO}"
CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" ${EXTRA_ARGS}
