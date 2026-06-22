#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/submission"
STAGE="${OUT_DIR}/stage"
BUNDLE="${OUT_DIR}/cv_hw3_submission_bundle.tar.gz"

mkdir -p "${OUT_DIR}"
rm -rf "${STAGE}"
mkdir -p "${STAGE}"

copy_path() {
  local src="$1"
  local dst="${STAGE}/${src}"
  if [[ -e "${ROOT_DIR}/${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    cp -a "${ROOT_DIR}/${src}" "${dst}"
  else
    echo "Warning: missing ${src}" >&2
  fi
}

task1_manifest_artifacts() {
  (cd "${ROOT_DIR}" && python - <<'PY'
import json
from pathlib import Path

manifest_path = Path("results/task1/real_run_manifest.json")
if not manifest_path.exists():
    raise SystemExit(0)

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
paths = []

object_b = manifest.get("object_b", {}).get("threestudio", {})
for key in ("command", "config", "metrics", "preview", "test_video", "export"):
    value = object_b.get(key)
    if value:
        paths.append(value)

object_a = manifest.get("object_a", {}).get("2dgs", {})
for key in ("checkpoint", "point_cloud", "official_tsdf_mesh", "final_mesh", "mesh_preview", "canonical_checkpoint", "official_tsdf_source", "official_tsdf_clean_mesh"):
    value = object_a.get(key)
    if value:
        paths.append(value)

background = manifest.get("background", {})
for key in ("dataset_manifest",):
    value = background.get(key)
    if value:
        paths.append(value)
background_2dgs = background.get("2dgs", {})
for key in ("checkpoint", "mesh", "point_cloud", "raw_mesh", "rgb_render_contact_sheet", "final_mesh_fusion_background", "mesh_preview"):
    value = background_2dgs.get(key)
    if value:
        paths.append(value)

object_c = manifest.get("object_c", {})
for key in ("prepared_rgba", "mesh", "material", "mesh_preview"):
    value = object_c.get(key)
    if value:
        paths.append(value)

fusion = manifest.get("fusion", {})
for key in ("config", "preview", "video", "blend"):
    value = fusion.get(key)
    if value:
        paths.append(value)

for path in dict.fromkeys(paths):
    print(path)
PY
  )
}

copy_path ".gitignore"
copy_path "HW3_计算机视觉.pdf"
copy_path "README.md"
copy_path "environment.yml"
copy_path "configs"
copy_path "docs"
copy_path "scripts"
copy_path "src"
copy_path "report/HW3_report.ipynb"
copy_path "report/main.tex"
copy_path "report/references.bib"
copy_path "report/metadata.example.json"
copy_path "report/figures"
copy_path "report/build/main.pdf"
copy_path "results/submission_manifest.json"
copy_path "results/submission_readiness.json"
copy_path "results/submission_readiness.log"
copy_path "data/task1/frames_manifest.json"
copy_path "data/task1/object_A_colmap/colmap_manifest.json"
copy_path "results/task1/asset_stats.json"
copy_path "results/task1/real_run_manifest.json"
copy_path "data/task1/mipnerf360_full/garden/garden_manifest.json"
copy_path "results/task1/2dgs/object_A_rgba_cropped_alpha_clean/train/ours_latest/fuse_post_2dgs_tsdf_main_component.ply"
copy_path "results/task1/2dgs/background_garden_full/train/ours_latest/fuse_unbounded_post_crop_q02_98.ply"
copy_path "results/task1/previews/background_garden_full_2dgs_contact.png"
copy_path "results/task1/fusion_mesh/garden_scene_upright_topopen_q05.ply"
copy_path "results/task1/aigc/object_B_text3d/export/model.obj"
copy_path "results/task1/aigc/object_C_image3d/model.obj"
copy_path "results/task1/aigc/object_C_image3d/model.mtl"
while IFS= read -r manifest_artifact; do
  [[ -n "${manifest_artifact}" ]] && copy_path "${manifest_artifact}"
done < <(task1_manifest_artifacts)
copy_path "results/task1/fusion_mesh/fusion_mesh_preview.png"
copy_path "results/task1/fusion_mesh/fusion_mesh_walkthrough.mp4"
copy_path "results/task1/fusion_mesh/fusion_mesh_scene.blend"
copy_path "results/task2/act_eval_summary.csv"
copy_path "results/task2/online_eval_B_on_D_100.json"
copy_path "results/task2/online_eval_ABC_on_D_100.json"
copy_path "results/task2/online_act_B_train80_val20_10k/online_metrics.csv"
copy_path "results/task2/online_act_ABC_train40each_val10each_10k/online_metrics.csv"
copy_path "results/task2/logs"
copy_path "results/task2/wandb_offline"
copy_path "weights/cv_hw3_task2_act_weights.tar.gz"

find "${STAGE}" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${STAGE}" -type f -name "*.pyc" -delete

tar -czf "${BUNDLE}" -C "${STAGE}" .
echo "Submission bundle: ${BUNDLE}"
