#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:-garden}"
NVIEWS="${2:-24}"
OUT_ROOT="${3:-data/task1/mipnerf360_sparse_${SCENE}_n${NVIEWS}/raw}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/${OUT_ROOT}"
BASE_URL="${NERFBASELINES_DATA_URL:-https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data/resolve/main/mipnerf360-sparse}"

mkdir -p "${OUT_DIR}"

download_one() {
  local name="$1"
  local url="${BASE_URL}/${name}"
  local dst="${OUT_DIR}/${name}"
  if [[ -s "${dst}" ]]; then
    echo "Already exists: ${dst}"
    return
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --connect-timeout 30 --max-time 300 "${url}" -o "${dst}"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --timeout=30 -O "${dst}" "${url}"
  else
    echo "Need curl or wget." >&2
    exit 1
  fi
}

download_one "${SCENE}-n${NVIEWS}-nbv.json"
download_one "${SCENE}-n${NVIEWS}-pointcloud.ply"

echo "NerfBaselines Mip-NeRF 360 sparse files ready: ${OUT_DIR}"
