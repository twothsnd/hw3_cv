#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_2DGS="${ROOT_DIR}/.venvs/2dgs/bin/python"
if [[ ! -x "${PYTHON_2DGS}" ]]; then
  PYTHON_2DGS="python"
fi

OBJECT_A_INPUT="data/raw/object_A"
OBJECT_C_IMAGE="data/raw/object_C/object_c.png"
OBJECT_B_PROMPT="a small ceramic robot toy, high quality DSLR photo"
OBJECT_C_TEXT="a high-resolution DSLR image of the object"
SCENE="garden"

OBJECT_A_GPU="0"
BACKGROUND_GPU="1"
TEXT3D_GPU="2"
IMAGE3D_GPU="3"

ITERATIONS="30000"
BACKGROUND_ITERATIONS="30000"
OBJECT_A_FPS="5"
OBJECT_A_MATCHER="sequential"
OBJECT_A_CAMERA_MODEL="SIMPLE_PINHOLE"
MAX_FRAMES="180"
RESIZE_LONG_EDGE="1600"
OBJECT_A_MESH_RES="256"
TEXT3D_STEPS="10000"
MAGIC123_COARSE_ITERS="5000"
MAGIC123_FINE_ITERS="5000"

RUN_OBJECT_A="1"
RUN_BACKGROUND="1"
RUN_TEXT3D="1"
RUN_IMAGE3D="1"
RUN_FUSION="1"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --object-a-input) OBJECT_A_INPUT="$2"; shift 2 ;;
    --object-c-image) OBJECT_C_IMAGE="$2"; shift 2 ;;
    --object-b-prompt) OBJECT_B_PROMPT="$2"; shift 2 ;;
    --object-c-text) OBJECT_C_TEXT="$2"; shift 2 ;;
    --scene) SCENE="$2"; shift 2 ;;
    --object-a-gpu) OBJECT_A_GPU="$2"; shift 2 ;;
    --background-gpu) BACKGROUND_GPU="$2"; shift 2 ;;
    --text3d-gpu) TEXT3D_GPU="$2"; shift 2 ;;
    --image3d-gpu) IMAGE3D_GPU="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --background-iterations) BACKGROUND_ITERATIONS="$2"; shift 2 ;;
    --object-a-fps) OBJECT_A_FPS="$2"; shift 2 ;;
    --object-a-matcher) OBJECT_A_MATCHER="$2"; shift 2 ;;
    --object-a-camera-model) OBJECT_A_CAMERA_MODEL="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --resize-long-edge) RESIZE_LONG_EDGE="$2"; shift 2 ;;
    --object-a-mesh-res) OBJECT_A_MESH_RES="$2"; shift 2 ;;
    --text3d-steps) TEXT3D_STEPS="$2"; shift 2 ;;
    --magic123-coarse-iters) MAGIC123_COARSE_ITERS="$2"; shift 2 ;;
    --magic123-fine-iters) MAGIC123_FINE_ITERS="$2"; shift 2 ;;
    --skip-object-a) RUN_OBJECT_A="0"; shift ;;
    --skip-background) RUN_BACKGROUND="0"; shift ;;
    --skip-text3d) RUN_TEXT3D="0"; shift ;;
    --skip-image3d) RUN_IMAGE3D="0"; shift ;;
    --skip-fusion) RUN_FUSION="0"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

abs_path() {
  if [[ "$1" = /* ]]; then
    printf '%s' "$1"
  else
    printf '%s/%s' "${ROOT_DIR}" "$1"
  fi
}

run_cmd() {
  echo "[run] $*"
  if [[ "${DRY_RUN}" == "0" ]]; then
    "$@"
  fi
}

has_object_a_media() {
  local input_abs
  input_abs="$(abs_path "${OBJECT_A_INPUT}")"
  if [[ -f "${input_abs}" ]]; then
    case "${input_abs,,}" in
      *.mp4|*.mov|*.m4v|*.avi|*.mkv|*.webm) return 0 ;;
    esac
    return 1
  fi
  if [[ -d "${input_abs}" ]]; then
    find "${input_abs}" -maxdepth 1 -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' -o -iname '*.mp4' -o -iname '*.mov' -o -iname '*.m4v' -o -iname '*.avi' -o -iname '*.mkv' -o -iname '*.webm' \) \
      | head -n 1 | grep -q .
  else
    return 1
  fi
}

OBJECT_C_ABS="$(abs_path "${OBJECT_C_IMAGE}")"
if [[ "${RUN_OBJECT_A}" == "1" ]] && ! has_object_a_media; then
  echo "Missing object A phone multi-view photos or video under ${OBJECT_A_INPUT}" >&2
  exit 1
fi
if [[ "${RUN_IMAGE3D}" == "1" ]] && [[ ! -f "${OBJECT_C_ABS}" ]]; then
  echo "Missing object C single foreground image: ${OBJECT_C_IMAGE}" >&2
  exit 1
fi

OBJECT_A_DATASET="data/task1/object_A_colmap"
OBJECT_A_IMAGES="data/task1/object_A_frames"
OBJECT_A_MODEL="results/task1/2dgs/object_A"
BACKGROUND_DATA_ROOT="data/task1/mipnerf360"
BACKGROUND_DATASET="${BACKGROUND_DATA_ROOT}/${SCENE}"
BACKGROUND_MODEL="results/task1/2dgs/background_${SCENE}"
OBJECT_B_MODEL="results/task1/aigc/object_B_text3d"
OBJECT_C_MODEL="results/task1/aigc/object_C_image3d"
FUSION_CONFIG="configs/fusion_scene.json"

mkdir -p "${ROOT_DIR}/results/task1"

if [[ "${RUN_OBJECT_A}" == "1" ]]; then
  run_cmd "${PYTHON_2DGS}" "${ROOT_DIR}/scripts/task1/extract_frames.py" \
    --input "$(abs_path "${OBJECT_A_INPUT}")" \
    --output "${ROOT_DIR}/${OBJECT_A_IMAGES}" \
    --fps "${OBJECT_A_FPS}" \
    --max-frames "${MAX_FRAMES}" \
    --resize-long-edge "${RESIZE_LONG_EDGE}" \
    --overwrite

  run_cmd "${PYTHON_2DGS}" "${ROOT_DIR}/scripts/task1/run_colmap.py" \
    --images "${ROOT_DIR}/${OBJECT_A_IMAGES}" \
    --dataset "${ROOT_DIR}/${OBJECT_A_DATASET}" \
    --matcher "${OBJECT_A_MATCHER}" \
    --camera-model "${OBJECT_A_CAMERA_MODEL}" \
    --backend auto \
    --overwrite

  run_cmd bash "${ROOT_DIR}/scripts/task1/train_2dgs.sh" \
    --dataset "${OBJECT_A_DATASET}" \
    --output "${OBJECT_A_MODEL}" \
    --gpu "${OBJECT_A_GPU}" \
    --iterations "${ITERATIONS}"

  run_cmd bash "${ROOT_DIR}/scripts/task1/render_2dgs.sh" \
    --dataset "${OBJECT_A_DATASET}" \
    --model "${OBJECT_A_MODEL}" \
    --gpu "${OBJECT_A_GPU}" \
    --mesh-mode bounded \
    --mesh-res "${OBJECT_A_MESH_RES}"
fi

if [[ "${RUN_BACKGROUND}" == "1" ]]; then
  run_cmd bash "${ROOT_DIR}/scripts/task1/download_mipnerf360.sh" "${SCENE}" "${BACKGROUND_DATA_ROOT}"
  run_cmd bash "${ROOT_DIR}/scripts/task1/train_2dgs.sh" \
    --dataset "${BACKGROUND_DATASET}" \
    --output "${BACKGROUND_MODEL}" \
    --gpu "${BACKGROUND_GPU}" \
    --iterations "${BACKGROUND_ITERATIONS}" \
    --extra "--depth_ratio 0"

  run_cmd bash "${ROOT_DIR}/scripts/task1/render_2dgs.sh" \
    --dataset "${BACKGROUND_DATASET}" \
    --model "${BACKGROUND_MODEL}" \
    --gpu "${BACKGROUND_GPU}" \
    --mesh-mode unbounded \
    --render-path 1
fi

if [[ "${RUN_TEXT3D}" == "1" ]]; then
  run_cmd bash "${ROOT_DIR}/scripts/task1/generate_text3d_threestudio.sh" \
    --prompt "${OBJECT_B_PROMPT}" \
    --name object_B_text3d \
    --gpu "${TEXT3D_GPU}" \
    --max-steps "${TEXT3D_STEPS}"
fi

if [[ "${RUN_IMAGE3D}" == "1" ]]; then
  run_cmd bash "${ROOT_DIR}/scripts/task1/generate_image3d_magic123.sh" \
    --image "${OBJECT_C_IMAGE}" \
    --text "${OBJECT_C_TEXT}" \
    --name object_C_image3d \
    --gpu "${IMAGE3D_GPU}" \
    --coarse-iters "${MAGIC123_COARSE_ITERS}" \
    --fine-iters "${MAGIC123_FINE_ITERS}"
fi

run_cmd "${PYTHON_2DGS}" "${ROOT_DIR}/scripts/task1/collect_asset_stats.py" \
  --assets \
    "object_A=${OBJECT_A_MODEL}/train/ours_latest/fuse_post.ply" \
    "object_B=${OBJECT_B_MODEL}/export/model.obj" \
    "object_C=${OBJECT_C_MODEL}/model.obj" \
    "background=${BACKGROUND_MODEL}/train/ours_latest/fuse_unbounded_post.ply" \
  --output "${ROOT_DIR}/results/task1/asset_stats.json"

if [[ "${RUN_FUSION}" == "1" ]]; then
  run_cmd bash "${ROOT_DIR}/scripts/task1/render_fusion_blender.sh" "${FUSION_CONFIG}"
fi

MANIFEST="${ROOT_DIR}/results/task1/real_run_manifest.json"
if [[ "${DRY_RUN}" == "0" ]]; then
  python - <<PY
from pathlib import Path
import json, time
root = Path("${ROOT_DIR}")
manifest = {
    "created_at_unix": time.time(),
    "object_a_input": "${OBJECT_A_INPUT}",
    "object_a_camera_model": "${OBJECT_A_CAMERA_MODEL}",
    "object_a_fps": "${OBJECT_A_FPS}",
    "object_a_matcher": "${OBJECT_A_MATCHER}",
    "object_a_mesh_res": "${OBJECT_A_MESH_RES}",
    "object_c_image": "${OBJECT_C_IMAGE}",
    "object_b_prompt": "${OBJECT_B_PROMPT}",
    "object_c_text": "${OBJECT_C_TEXT}",
    "scene": "${SCENE}",
    "object_a_model": "${OBJECT_A_MODEL}",
    "background_model": "${BACKGROUND_MODEL}",
    "object_b_model": "${OBJECT_B_MODEL}",
    "object_c_model": "${OBJECT_C_MODEL}",
    "fusion_config": "${FUSION_CONFIG}",
}
path = Path("${MANIFEST}")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(f"Wrote {path}")
PY
else
  echo "[dry-run] would write ${MANIFEST}"
fi
