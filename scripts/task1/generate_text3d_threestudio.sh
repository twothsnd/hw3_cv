#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${ROOT_DIR}/external/threestudio"
PROMPT=""
NAME="object_B_text3d"
TAG="dreamfusion_sd"
GPU="0"
MAX_STEPS="10000"
OUTPUT_ROOT="${ROOT_DIR}/results/task1/aigc"
CONFIG="configs/dreamfusion-sd.yaml"
EXPORT_MESH="1"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt) PROMPT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --no-export) EXPORT_MESH="0"; shift ;;
    --extra) EXTRA_ARGS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${PROMPT}" ]]; then
  echo "Usage: generate_text3d_threestudio.sh --prompt PROMPT [--name object_B_text3d]" >&2
  exit 1
fi
if [[ ! -d "${REPO}" ]]; then
  echo "Missing ${REPO}. Run scripts/setup/clone_external.sh first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO}"

python launch.py \
  --config "${CONFIG}" \
  --train \
  --gpu "${GPU}" \
  system.prompt_processor.prompt="${PROMPT}" \
  exp_root_dir="${OUTPUT_ROOT}" \
  name="${NAME}" \
  tag="${TAG}" \
  trainer.max_steps="${MAX_STEPS}" \
  ${EXTRA_ARGS}

TRIAL_DIR="$(find "${OUTPUT_ROOT}/${NAME}" -maxdepth 1 -type d -name "${TAG}@*" | sort | tail -n 1)"
if [[ -z "${TRIAL_DIR}" ]]; then
  echo "Could not find threestudio trial directory under ${OUTPUT_ROOT}/${NAME}" >&2
  exit 1
fi

if [[ "${EXPORT_MESH}" == "1" ]]; then
  python launch.py \
    --config "${TRIAL_DIR}/configs/parsed.yaml" \
    --export \
    --gpu "${GPU}" \
    resume="${TRIAL_DIR}/ckpts/last.ckpt" \
    system.exporter_type=mesh-exporter \
    system.exporter.fmt=obj

  EXPORT_DIR="${OUTPUT_ROOT}/${NAME}/export"
  mkdir -p "${EXPORT_DIR}"
  FOUND_OBJ="$(find "${TRIAL_DIR}" -type f -name "*.obj" | sort | tail -n 1)"
  if [[ -n "${FOUND_OBJ}" ]]; then
    SRC_DIR="$(dirname "${FOUND_OBJ}")"
    find "${SRC_DIR}" -maxdepth 1 -type f \( -name "*.obj" -o -name "*.mtl" -o -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) -exec cp {} "${EXPORT_DIR}/" \;
    cp "${FOUND_OBJ}" "${EXPORT_DIR}/model.obj"
  fi
fi

echo "threestudio result: ${TRIAL_DIR}"
