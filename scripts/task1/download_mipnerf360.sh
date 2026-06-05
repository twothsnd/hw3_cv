#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:-garden}"
OUT_ROOT="${2:-data/task1/mipnerf360}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/${OUT_ROOT}"
ZIP_PATH="${OUT_DIR}/360_v2.zip"
URL="${MIPNERF360_URL:-https://storage.googleapis.com/gresearch/refraw360/360_v2.zip}"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${ZIP_PATH}" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --continue-at - \
      --speed-limit "${MIPNERF360_SPEED_LIMIT:-200000}" \
      --speed-time "${MIPNERF360_SPEED_TIME:-120}" \
      "${URL}" -o "${ZIP_PATH}"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --timeout=30 --read-timeout=30 -c -O "${ZIP_PATH}" "${URL}"
  else
    echo "Need wget or curl to download Mip-NeRF 360." >&2
    exit 1
  fi
fi

if [[ ! -d "${OUT_DIR}/${SCENE}" ]]; then
  unzip "${ZIP_PATH}" -d "${OUT_DIR}"
fi

if [[ ! -d "${OUT_DIR}/${SCENE}" ]]; then
  echo "Scene ${SCENE} was not found under ${OUT_DIR}. Check dataset contents." >&2
  exit 1
fi

echo "Mip-NeRF 360 scene ready: ${OUT_DIR}/${SCENE}"
