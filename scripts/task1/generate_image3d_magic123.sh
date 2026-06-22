#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${ROOT_DIR}/external/Magic123"
PYTHON="${ROOT_DIR}/.venvs/magic123/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi
IMAGE=""
TEXT="a high-resolution DSLR image of the object"
NAME="object_C_image3d"
GPU="0"
COARSE_ITERS="5000"
FINE_ITERS="5000"
OUTPUT_ROOT="${ROOT_DIR}/results/task1/aigc"
EXTRA_COARSE=""
EXTRA_FINE=""
NO_DEPTH="0"
RGBA_SIZE="512"
HF_KEY=""
GUIDANCE_MODE="both"

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
    --no-depth) NO_DEPTH="1"; shift ;;
    --rgba-size) RGBA_SIZE="$2"; shift 2 ;;
    --hf-key) HF_KEY="$2"; shift 2 ;;
    --guidance-mode) GUIDANCE_MODE="$2"; shift 2 ;;
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
if [[ ! -s "${REPO}/pretrained/zero123/105000.ckpt" ]]; then
  echo "Missing ${REPO}/pretrained/zero123/105000.ckpt. Download the Zero-1-to-3 checkpoint before running Magic123." >&2
  exit 1
fi
if [[ "${NO_DEPTH}" != "1" && ! -s "${REPO}/pretrained/midas/dpt_beit_large_512.pt" ]]; then
  echo "Missing ${REPO}/pretrained/midas/dpt_beit_large_512.pt. Use --no-depth or download the MiDaS checkpoint." >&2
  exit 1
fi
if [[ "${GUIDANCE_MODE}" != "both" && "${GUIDANCE_MODE}" != "zero123" ]]; then
  echo "--guidance-mode must be 'both' or 'zero123'" >&2
  exit 1
fi

WORK_DIR="${OUTPUT_ROOT}/${NAME}/data"
OUT_DIR="${OUTPUT_ROOT}/${NAME}"
mkdir -p "${WORK_DIR}" "${OUT_DIR}"
cp "${ROOT_DIR}/${IMAGE}" "${WORK_DIR}/main.png"

cd "${REPO}"
if [[ "${NO_DEPTH}" == "1" ]]; then
  "${PYTHON}" "${ROOT_DIR}/scripts/task1/prepare_magic123_rgba.py" \
    --input "${WORK_DIR}/main.png" \
    --output "${WORK_DIR}/rgba.png" \
    --size "${RGBA_SIZE}"
else
  "${PYTHON}" preprocess_image.py --path "${WORK_DIR}/main.png"
fi

FILENAME="${NAME}"
COARSE_WS="${OUT_DIR}/magic123_${FILENAME}_coarse"
FINE_WS="${OUT_DIR}/magic123_${FILENAME}_dmtet"
DEPTH_ARGS=()
if [[ "${NO_DEPTH}" == "1" ]]; then
  DEPTH_ARGS=(--lambda_depth 0)
fi
HF_ARGS=()
if [[ -n "${HF_KEY}" ]]; then
  HF_ARGS=(--hf_key "${HF_KEY}")
fi
COARSE_GUIDANCE=(--guidance SD zero123 --lambda_guidance 1.0 40 --guidance_scale 100 5)
FINE_GUIDANCE=(--guidance SD zero123 --lambda_guidance 1e-3 0.01 --guidance_scale 100 5)
if [[ "${GUIDANCE_MODE}" == "zero123" ]]; then
  COARSE_GUIDANCE=(--guidance zero123 --lambda_guidance 40 --guidance_scale 5)
  FINE_GUIDANCE=(--guidance zero123 --lambda_guidance 0.01 --guidance_scale 5)
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" main.py -O \
  --text "${TEXT}" \
  --sd_version 1.5 \
  --image "${WORK_DIR}/rgba.png" \
  --workspace "${COARSE_WS}" \
  "${HF_ARGS[@]}" \
  --optim adam \
  --iters "${COARSE_ITERS}" \
  "${COARSE_GUIDANCE[@]}" \
  --latent_iter_ratio 0 \
  --normal_iter_ratio 0.2 \
  --t_range 0.2 0.6 \
  --bg_radius -1 \
  --save_mesh \
  "${DEPTH_ARGS[@]}" \
  ${EXTRA_COARSE}

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" main.py -O \
  --text "${TEXT}" \
  --sd_version 1.5 \
  --image "${WORK_DIR}/rgba.png" \
  --workspace "${FINE_WS}" \
  "${HF_ARGS[@]}" \
  --dmtet \
  --init_ckpt "${COARSE_WS}/checkpoints/magic123_${FILENAME}_coarse.pth" \
  --iters "${FINE_ITERS}" \
  --optim adam \
  --known_view_interval 4 \
  --latent_iter_ratio 0 \
  "${FINE_GUIDANCE[@]}" \
  --rm_edge \
  --bg_radius -1 \
  --save_mesh \
  "${DEPTH_ARGS[@]}" \
  ${EXTRA_FINE}

FOUND_OBJ="$(find "${FINE_WS}" -type f -name "*.obj" | sort | tail -n 1)"
if [[ -n "${FOUND_OBJ}" ]]; then
  SRC_DIR="$(dirname "${FOUND_OBJ}")"
  find "${SRC_DIR}" -maxdepth 1 -type f \( -name "*.obj" -o -name "*.mtl" -o -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) -exec cp {} "${OUT_DIR}/" \;
  cp "${FOUND_OBJ}" "${OUT_DIR}/model.obj"
fi

echo "Magic123 result: ${OUT_DIR}"
