#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS_DIR="${ROOT_DIR}/tools"
BIN_DIR="${TOOLS_DIR}/bin"
mkdir -p "${BIN_DIR}"

if [[ ! -x "${BIN_DIR}/ffmpeg" ]]; then
  if [[ -x "${ROOT_DIR}/.venvs/2dgs/bin/python" ]]; then
    FFMPEG_PATH="$("${ROOT_DIR}/.venvs/2dgs/bin/python" - <<'PY'
import imageio_ffmpeg
print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)"
  elif [[ -x "${ROOT_DIR}/.venvs/hw3-utils/bin/python" ]]; then
    FFMPEG_PATH="$("${ROOT_DIR}/.venvs/hw3-utils/bin/python" - <<'PY'
import imageio_ffmpeg
print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)"
  else
    FFMPEG_PATH=""
  fi
  if [[ -n "${FFMPEG_PATH}" && -x "${FFMPEG_PATH}" ]]; then
    ln -sf "${FFMPEG_PATH}" "${BIN_DIR}/ffmpeg"
  fi
fi

if [[ ! -x "${BIN_DIR}/blender" ]]; then
  BLENDER_VERSION="${BLENDER_VERSION:-4.3.0}"
  BLENDER_MAJOR_MINOR="${BLENDER_VERSION%.*}"
  ARCHIVE="${TOOLS_DIR}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
  URL="${BLENDER_URL:-https://mirrors.aliyun.com/blender/release/Blender${BLENDER_MAJOR_MINOR}/blender-${BLENDER_VERSION}-linux-x64.tar.xz}"
  mkdir -p "${TOOLS_DIR}"
  if [[ ! -f "${ARCHIVE}" ]]; then
    if command -v curl >/dev/null 2>&1; then
      curl -L "${URL}" -o "${ARCHIVE}"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "${ARCHIVE}" "${URL}"
    else
      echo "Need curl or wget to download Blender." >&2
      exit 1
    fi
  fi
  tar -xf "${ARCHIVE}" -C "${TOOLS_DIR}"
  ln -sf "${TOOLS_DIR}/blender-${BLENDER_VERSION}-linux-x64/blender" "${BIN_DIR}/blender"
fi

echo "Local tools installed under ${TOOLS_DIR}."
echo "Use: export PATH=${BIN_DIR}:\$PATH"
