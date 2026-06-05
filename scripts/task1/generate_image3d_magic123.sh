#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${ROOT_DIR}/external/Magic123"
IMAGE=""
TEXT="a high-resolution DSLR image of the object"
NAME="object_C_image3d"
GPU="0"
COARSE_ITERS="5000"
FINE_ITERS="5000"
OUTPUT_ROOT="${ROOT_DIR}/results/task1/aigc"
EXTRA_COARSE=""
EXTRA_FINE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --text) TEXT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --coarse-iters) COARSE_ITERS="$2"; shift 2 ;;
    --fine-iters) FINE_ITERS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --extra-coarse) EXTRA_COARSE="$2"; shift 2 ;;
    --extra-fine) EXTRA_FINE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${IMAGE}" ]]; then
  echo "Usage: generate_image3d_magic123.sh --image image.png [--text TEXT]" >&2
  exit 1
fi
if [[ ! -d "${REPO}" ]]; then
  echo "Missing ${REPO}. Run scripts/setup/clone_external.sh first." >&2
  exit 1
fi

WORK_DIR="${OUTPUT_ROOT}/${NAME}/data"
OUT_DIR="${OUTPUT_ROOT}/${NAME}"
mkdir -p "${WORK_DIR}" "${OUT_DIR}"
cp "${ROOT_DIR}/${IMAGE}" "${WORK_DIR}/main.png"

cd "${REPO}"
python preprocess_image.py --path "${WORK_DIR}/main.png"

FILENAME="${NAME}"
COARSE_WS="${OUT_DIR}/magic123_${FILENAME}_coarse"
FINE_WS="${OUT_DIR}/magic123_${FILENAME}_dmtet"

CUDA_VISIBLE_DEVICES="${GPU}" python main.py -O \
  --text "${TEXT}" \
  --sd_version 1.5 \
  --image "${WORK_DIR}/rgba.png" \
  --workspace "${COARSE_WS}" \
  --optim adam \
  --iters "${COARSE_ITERS}" \
  --guidance SD zero123 \
  --lambda_guidance 1.0 40 \
  --guidance_scale 100 5 \
  --latent_iter_ratio 0 \
  --normal_iter_ratio 0.2 \
  --t_range 0.2 0.6 \
  --bg_radius -1 \
  --save_mesh \
  ${EXTRA_COARSE}

CUDA_VISIBLE_DEVICES="${GPU}" python main.py -O \
  --text "${TEXT}" \
  --sd_version 1.5 \
  --image "${WORK_DIR}/rgba.png" \
  --workspace "${FINE_WS}" \
  --dmtet \
  --init_ckpt "${COARSE_WS}/checkpoints/magic123_${FILENAME}_coarse.pth" \
  --iters "${FINE_ITERS}" \
  --optim adam \
  --known_view_interval 4 \
  --latent_iter_ratio 0 \
  --guidance SD zero123 \
  --lambda_guidance 1e-3 0.01 \
  --guidance_scale 100 5 \
  --rm_edge \
  --bg_radius -1 \
  --save_mesh \
  ${EXTRA_FINE}

FOUND_OBJ="$(find "${FINE_WS}" -type f -name "*.obj" | sort | tail -n 1)"
if [[ -n "${FOUND_OBJ}" ]]; then
  SRC_DIR="$(dirname "${FOUND_OBJ}")"
  find "${SRC_DIR}" -maxdepth 1 -type f \( -name "*.obj" -o -name "*.mtl" -o -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) -exec cp {} "${OUT_DIR}/" \;
  cp "${FOUND_OBJ}" "${OUT_DIR}/model.obj"
fi

echo "Magic123 result: ${OUT_DIR}"
