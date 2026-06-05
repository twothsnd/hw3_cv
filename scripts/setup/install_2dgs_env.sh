#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${ROOT_DIR}/external/2d-gaussian-splatting"
VENV="${ROOT_DIR}/.venvs/2dgs"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

if [[ ! -d "${REPO}" ]]; then
  echo "Missing ${REPO}. Run scripts/setup/clone_external.sh first." >&2
  exit 1
fi

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel ninja
"${VENV}/bin/python" -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
"${VENV}/bin/python" -m pip install \
  numpy \
  plyfile \
  tqdm \
  opencv-python \
  open3d \
  scipy \
  scikit-image \
  lpips \
  mediapy \
  trimesh \
  imageio \
  imageio-ffmpeg

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" FORCE_CUDA=1 "${VENV}/bin/python" -m pip install --no-build-isolation "${REPO}/submodules/diff-surfel-rasterization"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" FORCE_CUDA=1 "${VENV}/bin/python" -m pip install --no-build-isolation "${REPO}/submodules/simple-knn"

echo "Activate with: source ${VENV}/bin/activate"
