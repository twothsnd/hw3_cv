#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${ROOT_DIR}/.venvs/hw3-utils"
python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV}/bin/python" -m pip install \
  numpy \
  pyyaml \
  pillow \
  plyfile \
  opencv-python \
  pandas \
  matplotlib \
  seaborn \
  trimesh \
  open3d \
  imageio \
  imageio-ffmpeg \
  tqdm

echo "Activate with: source ${VENV}/bin/activate"
