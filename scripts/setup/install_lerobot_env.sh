#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${ROOT_DIR}/external/lerobot"
VENV="${ROOT_DIR}/.venvs/lerobot"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

if [[ ! -d "${REPO}" ]]; then
  echo "Missing ${REPO}. Run scripts/setup/clone_external.sh first." >&2
  exit 1
fi

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV}/bin/python" -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
"${VENV}/bin/python" -m pip install -e "${REPO}"
"${VENV}/bin/python" -m pip install \
  numpy \
  pandas \
  pyarrow \
  datasets \
  huggingface_hub \
  wandb \
  tqdm \
  pillow \
  opencv-python

echo "Activate with: source ${VENV}/bin/activate"
