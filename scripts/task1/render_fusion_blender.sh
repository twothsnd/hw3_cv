#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:-configs/fusion_scene.json}"
BLENDER="${BLENDER:-blender}"

if ! command -v "${BLENDER}" >/dev/null 2>&1; then
  echo "Blender executable '${BLENDER}' was not found in PATH." >&2
  exit 1
fi

"${BLENDER}" -b --python "${ROOT_DIR}/scripts/task1/compose_blender_scene.py" -- --config "${CONFIG}"
