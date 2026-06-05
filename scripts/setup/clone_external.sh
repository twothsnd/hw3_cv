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

clone_or_update "2d-gaussian-splatting" "https://github.com/hbb1/2d-gaussian-splatting.git" "335ad612f2e783a4e57b9cbc4d1e167bd599fc98" "1"
clone_or_update "threestudio" "https://github.com/threestudio-project/threestudio.git" "28d9d80d9d00f308244adfcf3be8b17ca0cb6465" "0"
clone_or_update "Magic123" "https://github.com/guochengqian/Magic123.git" "c2eb289f0b9e03e5cf39cf1417f05ca33e9eb0a5" "0"
clone_or_update "lerobot" "https://github.com/huggingface/lerobot.git" "8fff0fde7c79f23a93d845d1a50e985de01f8b8a" "0"
clone_or_update "calvin" "https://github.com/mees/calvin.git" "fa03f01f19c65920e18cf37398a9ce859274af76" "1"
patch_2dgs_compat

echo "External repositories are pinned under ${EXTERNAL_DIR}."
