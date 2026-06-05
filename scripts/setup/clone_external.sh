#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"
mkdir -p "${EXTERNAL_DIR}"

clone_or_update() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local recursive="${4:-0}"
  local dest="${EXTERNAL_DIR}/${name}"

  if [[ ! -d "${dest}/.git" ]]; then
    if [[ "${recursive}" == "1" ]]; then
      git clone --recursive "${url}" "${dest}"
    else
      git clone "${url}" "${dest}"
    fi
  fi

  git -C "${dest}" fetch origin "${commit}" --depth 1 || git -C "${dest}" fetch origin
  git -C "${dest}" checkout "${commit}"
  if [[ "${recursive}" == "1" ]]; then
    git -C "${dest}" submodule update --init --recursive
  fi
}

download_archive_if_missing() {
  local name="$1"
  local url="$2"
  local sentinel="${3:-}"
  local dest="${EXTERNAL_DIR}/${name}"
  local safe_name="${name//\//_}"
  local archive="${EXTERNAL_DIR}/_archives/${safe_name}.tar.gz"
  local tmp_dir
  local extracted

  if [[ -d "${dest}" ]] && { [[ -z "${sentinel}" ]] || [[ -e "${dest}/${sentinel}" ]]; }; then
    return
  fi

  mkdir -p "${EXTERNAL_DIR}/_archives" "$(dirname "${dest}")"
  tmp_dir="$(mktemp -d "${EXTERNAL_DIR}/.extract.${safe_name}.XXXXXX")"
  curl -L --retry 5 --retry-delay 5 --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
    -o "${archive}" "${url}"
  tar -xzf "${archive}" -C "${tmp_dir}"
  extracted="$(find "${tmp_dir}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  if [[ -z "${extracted}" ]]; then
    echo "Archive ${archive} did not contain a top-level directory." >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi
  rm -rf "${dest}"
  mv "${extracted}" "${dest}"
  rm -rf "${tmp_dir}"
}

patch_2dgs_compat() {
  local reader="${EXTERNAL_DIR}/2d-gaussian-splatting/scene/dataset_readers.py"
  if [[ -f "${reader}" ]]; then
    perl -0pi -e 's/dtype=np\.byte\), "RGB"/dtype=np.uint8), "RGB"/g' "${reader}"
  fi
  local mesh_utils="${EXTERNAL_DIR}/2d-gaussian-splatting/utils/mesh_utils.py"
  if [[ -f "${mesh_utils}" ]] && ! grep -q "def as_open3d_mesh" "${mesh_utils}"; then
    perl -0pi -e 's/import trimesh\n/import trimesh\n\ndef as_open3d_mesh(mesh):\n    if hasattr(mesh, "as_open3d"):\n        return mesh.as_open3d\n    o3d_mesh = o3d.geometry.TriangleMesh()\n    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices, copy=True))\n    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces, copy=True))\n    if hasattr(mesh, "vertex_normals") and len(mesh.vertex_normals) == len(mesh.vertices):\n        o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(np.array(mesh.vertex_normals, copy=True))\n    if hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) == len(mesh.vertices):\n        colors = np.array(mesh.visual.vertex_colors, copy=True)[:, :3] \\/ 255.0\n        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)\n    return o3d_mesh\n/s' "${mesh_utils}"
    perl -0pi -e 's/mesh = mesh\.as_open3d/mesh = as_open3d_mesh(mesh)/g' "${mesh_utils}"
  fi
}

patch_magic123_compat() {
  local repo="${EXTERNAL_DIR}/Magic123"
  local setup_file
  for setup_file in \
    "${repo}/raymarching/setup.py" \
    "${repo}/shencoder/setup.py" \
    "${repo}/freqencoder/setup.py" \
    "${repo}/gridencoder/setup.py"; do
    if [[ -f "${setup_file}" ]]; then
      perl -0pi -e 's/-std=c\+\+14/-std=c++17/g' "${setup_file}"
    fi
  done

  local preprocess="${repo}/preprocess_image.py"
  if [[ -f "${preprocess}" ]] && ! grep -q "falling back to rembg" "${preprocess}"; then
    perl -0pi -e 's/        rgba = BackgroundRemoval\(\)\(image\)  # \[H, W, 4\]\n/        try:\n            rgba = BackgroundRemoval()(image)  # [H, W, 4]\n        except Exception as exc:\n            print(f'"'"'[WARN] carvekit background removal failed: {exc}'"'"')\n            print('"'"'[INFO] falling back to rembg...'"'"')\n            rgba = get_rgba(image)\n/s' "${preprocess}"
  fi
}

clone_or_update "2d-gaussian-splatting" "https://github.com/hbb1/2d-gaussian-splatting.git" "335ad612f2e783a4e57b9cbc4d1e167bd599fc98" "1"
clone_or_update "threestudio" "https://github.com/threestudio-project/threestudio.git" "28d9d80d9d00f308244adfcf3be8b17ca0cb6465" "0"
clone_or_update "Magic123" "https://github.com/guochengqian/Magic123.git" "c2eb289f0b9e03e5cf39cf1417f05ca33e9eb0a5" "0"
clone_or_update "lerobot" "https://github.com/huggingface/lerobot.git" "8fff0fde7c79f23a93d845d1a50e985de01f8b8a" "0"
clone_or_update "calvin" "https://github.com/mees/calvin.git" "fa03f01f19c65920e18cf37398a9ce859274af76" "1"

download_archive_if_missing "tiny-cuda-nn" "https://codeload.github.com/NVlabs/tiny-cuda-nn/tar.gz/749dd70c5afc5a9dadb85e5652ed65d55e0ba187" "CMakeLists.txt"
download_archive_if_missing "tiny-cuda-nn/dependencies/cutlass" "https://codeload.github.com/NVIDIA/cutlass/tar.gz/82f5075946e2569589439d500733b700a3141374" "CMakeLists.txt"
download_archive_if_missing "tiny-cuda-nn/dependencies/fmt" "https://codeload.github.com/fmtlib/fmt/tar.gz/fa2eb2d2e3ec5c21629f8ccd88ae05ec40b963fa" "include/fmt/format.h"
download_archive_if_missing "tiny-cuda-nn/dependencies/cmrc" "https://codeload.github.com/vector-of-bool/cmrc/tar.gz/952ffddba731fc110bd50409e8d2b8a06abbd237" "CMakeLists.txt"
download_archive_if_missing "nvdiffrast" "https://codeload.github.com/NVlabs/nvdiffrast/tar.gz/refs/heads/main" "setup.py"
download_archive_if_missing "envlight" "https://codeload.github.com/ashawkey/envlight/tar.gz/refs/heads/main" "setup.py"
download_archive_if_missing "CLIP" "https://codeload.github.com/openai/CLIP/tar.gz/refs/heads/main" "setup.py"
download_archive_if_missing "cubvh" "https://codeload.github.com/ashawkey/cubvh/tar.gz/refs/heads/main" "setup.py"

patch_2dgs_compat
patch_magic123_compat

echo "External repositories are pinned under ${EXTERNAL_DIR}."
