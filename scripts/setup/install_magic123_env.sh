#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${ROOT_DIR}/external/Magic123"
VENV="${ROOT_DIR}/.venvs/magic123"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
INSTALL_MAGIC123_SHAPE="${INSTALL_MAGIC123_SHAPE:-0}"
MAX_JOBS="${MAX_JOBS:-4}"

if [[ ! -d "${REPO}" ]]; then
  echo "Missing ${REPO}. Run scripts/setup/clone_external.sh first." >&2
  exit 1
fi

cd "${REPO}"
export TORCH_CUDA_ARCH_LIST MAX_JOBS

for setup_file in raymarching/setup.py shencoder/setup.py freqencoder/setup.py gridencoder/setup.py; do
  if [[ -f "${setup_file}" ]]; then
    perl -0pi -e 's/-std=c\+\+14/-std=c++17/g' "${setup_file}"
  fi
done

if [[ -f preprocess_image.py ]] && ! grep -q "falling back to rembg" preprocess_image.py; then
  perl -0pi -e 's/        rgba = BackgroundRemoval\(\)\(image\)  # \[H, W, 4\]\n/        try:\n            rgba = BackgroundRemoval()(image)  # [H, W, 4]\n        except Exception as exc:\n            print(f'"'"'[WARN] carvekit background removal failed: {exc}'"'"')\n            print('"'"'[INFO] falling back to rembg...'"'"')\n            rgba = get_rgba(image)\n/s' preprocess_image.py
fi

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip "setuptools<82" wheel ninja
"${VENV}/bin/python" -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"

TMP_REQUIREMENTS="$(mktemp)"
trap 'rm -f "${TMP_REQUIREMENTS}"' EXIT
sed '/^torch$/d; /^torchvision$/d' requirements.txt > "${TMP_REQUIREMENTS}"
sed -i 's/^diffusers.*/diffusers==0.17.1/' "${TMP_REQUIREMENTS}"
sed -i 's/^transformers$/transformers==4.30.2/' "${TMP_REQUIREMENTS}"
sed -i 's/^huggingface_hub$/huggingface_hub==0.15.1/' "${TMP_REQUIREMENTS}"
sed -i 's/^accelerate.*/accelerate==0.21.0/' "${TMP_REQUIREMENTS}"
sed -i '/^dearpygui$/d' "${TMP_REQUIREMENTS}"
sed -i '/^carvekit-colab$/d' "${TMP_REQUIREMENTS}"
sed -i '/^jupyterlab$/d' "${TMP_REQUIREMENTS}"
sed -i '/^debugpy-run$/d' "${TMP_REQUIREMENTS}"

if [[ -f "${ROOT_DIR}/external/nvdiffrast/setup.py" ]]; then
  "${VENV}/bin/python" -m pip install --no-build-isolation "${ROOT_DIR}/external/nvdiffrast"
  sed -i '/git+https:\/\/github.com\/NVlabs\/nvdiffrast/d' "${TMP_REQUIREMENTS}"
fi

if [[ -f "${ROOT_DIR}/external/CLIP/setup.py" ]]; then
  "${VENV}/bin/python" -m pip install "${ROOT_DIR}/external/CLIP"
  sed -i '/git+https:\/\/github.com\/openai\/CLIP/d' "${TMP_REQUIREMENTS}"
fi

if [[ -f "${ROOT_DIR}/external/cubvh/setup.py" ]]; then
  if [[ ! -f "${ROOT_DIR}/external/cubvh/third_party/eigen/Eigen/Dense" ]]; then
    mkdir -p "${ROOT_DIR}/external/_archives" "${ROOT_DIR}/external/cubvh/third_party"
    curl -L --retry 5 --retry-delay 5 --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
      -o "${ROOT_DIR}/external/_archives/eigen-github-master.tar.gz" \
      https://codeload.github.com/eigenteam/eigen-git-mirror/tar.gz/refs/heads/master
    rm -rf "${ROOT_DIR}/external/cubvh/third_party/eigen"
    tar -xzf "${ROOT_DIR}/external/_archives/eigen-github-master.tar.gz" \
      -C "${ROOT_DIR}/external/cubvh/third_party"
    mv "${ROOT_DIR}/external/cubvh/third_party/eigen-git-mirror-master" \
      "${ROOT_DIR}/external/cubvh/third_party/eigen"
  fi
  "${VENV}/bin/python" -m pip install --no-build-isolation "${ROOT_DIR}/external/cubvh"
  sed -i '/git+https:\/\/github.com\/ashawkey\/cubvh/d' "${TMP_REQUIREMENTS}"
fi

if [[ "${INSTALL_MAGIC123_SHAPE}" != "1" ]]; then
  sed -i '/git+https:\/\/github.com\/openai\/shap-e/d' "${TMP_REQUIREMENTS}"
fi

"${VENV}/bin/python" -m pip install -r "${TMP_REQUIREMENTS}"
"${VENV}/bin/python" -m pip install onnxruntime

for ext in raymarching shencoder freqencoder gridencoder; do
  PATH="${VENV}/bin:${PATH}" "${VENV}/bin/python" -m pip install --no-build-isolation "${REPO}/${ext}"
done

echo "Activate with: source ${VENV}/bin/activate"
