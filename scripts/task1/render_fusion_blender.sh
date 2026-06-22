#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:-configs/fusion_scene.json}"
BLENDER="${BLENDER:-blender}"

if [[ "${BLENDER}" == "blender" && -x "${ROOT_DIR}/tools/bin/blender" ]]; then
  BLENDER="${ROOT_DIR}/tools/bin/blender"
fi

if ! command -v "${BLENDER}" >/dev/null 2>&1; then
  echo "Blender executable '${BLENDER}' was not found in PATH." >&2
  exit 1
fi

SCRIPT="${ROOT_DIR}/scripts/task1/compose_blender_scene.py"
if grep -q '"background_image"' "${ROOT_DIR}/${CONFIG}" 2>/dev/null || grep -q '"background_image"' "${CONFIG}" 2>/dev/null; then
  SCRIPT="${ROOT_DIR}/scripts/task1/compose_plate_blender_scene.py"
fi

"${BLENDER}" -b --python "${SCRIPT}" -- --config "${CONFIG}"
