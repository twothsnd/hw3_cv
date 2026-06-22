#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${ROOT_DIR}/external/threestudio"
VENV="${ROOT_DIR}/.venvs/threestudio"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
GIT_HTTP_LOW_SPEED_LIMIT="${GIT_HTTP_LOW_SPEED_LIMIT:-1024}"
GIT_HTTP_LOW_SPEED_TIME="${GIT_HTTP_LOW_SPEED_TIME:-60}"
PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-60}"
INSTALL_THREESTUDIO_UI="${INSTALL_THREESTUDIO_UI:-0}"
INSTALL_THREESTUDIO_XFORMERS="${INSTALL_THREESTUDIO_XFORMERS:-0}"

if [[ ! -d "${REPO}" ]]; then
  echo "Missing ${REPO}. Run scripts/setup/clone_external.sh first." >&2
  exit 1
fi

export GIT_HTTP_LOW_SPEED_LIMIT GIT_HTTP_LOW_SPEED_TIME PIP_DEFAULT_TIMEOUT

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip "setuptools<82" wheel ninja
"${VENV}/bin/python" -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"

TMP_REQUIREMENTS="$(mktemp)"
trap 'rm -f "${TMP_REQUIREMENTS}"' EXIT
sed \
  's#git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2#nerfacc==0.5.2#' \
  "${REPO}/requirements.txt" > "${TMP_REQUIREMENTS}"
sed -i 's/^libigl$/libigl==2.4.1/' "${TMP_REQUIREMENTS}"
sed -i 's/^huggingface_hub$/huggingface_hub==0.25.2/' "${TMP_REQUIREMENTS}"

if [[ -f "${ROOT_DIR}/external/tiny-cuda-nn/dependencies/cutlass/CMakeLists.txt" ]]; then
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
  TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-80}" \
    "${VENV}/bin/python" -m pip install --no-build-isolation \
    "${ROOT_DIR}/external/tiny-cuda-nn/bindings/torch"
  sed -i '/git+https:\/\/github.com\/NVlabs\/tiny-cuda-nn/d' "${TMP_REQUIREMENTS}"
fi

if [[ -f "${ROOT_DIR}/external/nvdiffrast/setup.py" ]]; then
  "${VENV}/bin/python" -m pip install --no-build-isolation "${ROOT_DIR}/external/nvdiffrast"
  sed -i '/git+https:\/\/github.com\/NVlabs\/nvdiffrast/d' "${TMP_REQUIREMENTS}"
fi

if [[ -f "${ROOT_DIR}/external/envlight/setup.py" || -f "${ROOT_DIR}/external/envlight/pyproject.toml" ]]; then
  "${VENV}/bin/python" -m pip install "${ROOT_DIR}/external/envlight"
  sed -i '/git+https:\/\/github.com\/ashawkey\/envlight/d' "${TMP_REQUIREMENTS}"
fi

if [[ -f "${ROOT_DIR}/external/CLIP/setup.py" ]]; then
  "${VENV}/bin/python" -m pip install "${ROOT_DIR}/external/CLIP"
  sed -i '/git+https:\/\/github.com\/openai\/CLIP/d' "${TMP_REQUIREMENTS}"
fi

if [[ "${INSTALL_THREESTUDIO_UI}" != "1" ]]; then
  sed -i '/^gradio==/d' "${TMP_REQUIREMENTS}"
fi

if [[ "${INSTALL_THREESTUDIO_XFORMERS}" != "1" ]]; then
  sed -i '/^xformers$/d' "${TMP_REQUIREMENTS}"
fi

"${VENV}/bin/python" -m pip install -r "${TMP_REQUIREMENTS}"

NERFACC_UTILS="$("${VENV}/bin/python" - <<'PY'
from pathlib import Path
import nerfacc

print(Path(nerfacc.__file__).resolve().parent / "cuda/csrc/include/utils_math.cuh")
PY
)"
if [[ -f "${NERFACC_UTILS}" ]]; then
  sed -i \
    -e 's/float lerp(float a, float b, float t)/float nerfacc_lerp(float a, float b, float t)/' \
    -e 's/float2 lerp(float2 a, float2 b, float t)/float2 nerfacc_lerp(float2 a, float2 b, float t)/' \
    -e 's/float3 lerp(float3 a, float3 b, float t)/float3 nerfacc_lerp(float3 a, float3 b, float t)/' \
    -e 's/float4 lerp(float4 a, float4 b, float t)/float4 nerfacc_lerp(float4 a, float4 b, float t)/' \
    "${NERFACC_UTILS}"
fi
"${VENV}/bin/python" -m pip install -e "${REPO}"

echo "Activate with: source ${VENV}/bin/activate"
