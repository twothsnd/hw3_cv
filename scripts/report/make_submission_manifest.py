#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACTS = [
    "report/HW3_report.ipynb",
    "report/build/main.pdf",
    "report/figures/task1_fusion_preview.png",
    "report/figures/task1_asset_mesh_stats.pdf",
    "report/figures/task1_asset_mesh_stats.png",
    "report/figures/task2_official_training_loss.pdf",
    "report/figures/task2_official_training_loss.png",
    "report/figures/task2_official_training_loss.csv",
    "report/figures/task2_official_validation_metrics.pdf",
    "report/figures/task2_official_validation_metrics.png",
    "report/figures/task2_official_eval_metrics.pdf",
    "report/figures/task2_official_eval_metrics.png",
    "data/task1/frames_manifest.json",
    "data/task1/object_A_colmap/colmap_manifest.json",
    "data/task1/mipnerf360_full/garden/garden_manifest.json",
    "results/task1/asset_stats.json",
    "results/task1/real_run_manifest.json",
    "results/task1/2dgs/object_A_rgba_cropped_alpha_clean/train/ours_latest/fuse_post_2dgs_tsdf_main_component.ply",
    "results/task1/2dgs/background_garden_full/train/ours_latest/fuse_unbounded_post_crop_q02_98.ply",
    "results/task1/previews/background_garden_full_2dgs_contact.png",
    "results/task1/fusion_mesh/garden_scene_upright_topopen_q05.ply",
    "results/task1/aigc/object_B_text3d/export/model.obj",
    "results/task1/aigc/object_C_image3d/model.obj",
    "results/task1/fusion_mesh/fusion_mesh_preview.png",
    "results/task1/fusion_mesh/fusion_mesh_walkthrough.mp4",
    "results/task1/fusion_mesh/fusion_mesh_scene.blend",
    "results/task2/logs/online_train_B_train80_val20_10k.log",
    "results/task2/logs/online_train_ABC_train40each_val10each_10k.log",
    "results/task2/wandb_offline",
    "results/task2/online_eval_B_on_D_100.json",
    "results/task2/online_eval_ABC_on_D_100.json",
    "results/task2/act_eval_summary.csv",
    "results/task2/online_act_B_train80_val20_10k/online_metrics.csv",
    "results/task2/online_act_ABC_train40each_val10each_10k/online_metrics.csv",
    "results/task2/online_act_B_train80_val20_10k/checkpoints/010000/pretrained_model/model.safetensors",
    "results/task2/online_act_ABC_train40each_val10each_10k/checkpoints/010000/pretrained_model/model.safetensors",
    "results/submission_readiness.json",
]

REQUIRED_PRIVATE_INPUTS = [
    ("object_A_phone_capture", "data/raw/object_A"),
    ("object_C_single_foreground_image", "data/raw/object_C/object_c.png"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a manifest for generated HW3 submission artifacts.")
    parser.add_argument("--output", default="results/submission_manifest.json", type=Path)
    parser.add_argument("--artifacts", nargs="*", default=None)
    return parser.parse_args()


def dynamic_task1_artifacts(root: Path) -> list[str]:
    manifest_path = root / "results/task1/real_run_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    artifacts: list[str] = []

    object_a = manifest.get("object_a", {}).get("2dgs", {})
    for key in ("checkpoint", "point_cloud", "official_tsdf_mesh", "final_mesh", "mesh_preview", "canonical_checkpoint", "official_tsdf_source", "official_tsdf_clean_mesh"):
        value = object_a.get(key)
        if isinstance(value, str) and value:
            artifacts.append(value)

    background = manifest.get("background", {})
    for key in ("dataset_manifest",):
        value = background.get(key)
        if isinstance(value, str) and value:
            artifacts.append(value)
    background_2dgs = background.get("2dgs", {})
    for key in ("checkpoint", "mesh", "point_cloud", "raw_mesh", "rgb_render_contact_sheet", "final_mesh_fusion_background", "mesh_preview"):
        value = background_2dgs.get(key)
        if isinstance(value, str) and value:
            artifacts.append(value)

    fusion = manifest.get("fusion", {})
    for key in ("config", "preview", "video", "blend"):
        value = fusion.get(key)
        if isinstance(value, str) and value:
            artifacts.append(value)

    object_b = manifest.get("object_b", {}).get("threestudio", {})
    for key in ("command", "config", "metrics", "preview", "test_video", "export"):
        value = object_b.get(key)
        if isinstance(value, str) and value:
            artifacts.append(value)

    object_c = manifest.get("object_c", {})
    for key in ("prepared_rgba", "mesh", "material", "mesh_preview"):
        value = object_c.get(key)
        if isinstance(value, str) and value:
            artifacts.append(value)

    return artifacts


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def describe_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
        return {
            "path": str(path),
            "exists": True,
            "type": "directory",
            "file_count": len(files),
            "nonempty": bool(files),
        }
    item: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "type": "file",
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix.lower() == ".json":
        try:
            item["json"] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            item["json_error"] = str(exc)
    return item


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    artifact_list = args.artifacts or [*DEFAULT_ARTIFACTS, *dynamic_task1_artifacts(root)]
    artifacts = {name: describe_path(root / name) for name in dict.fromkeys(artifact_list)}
    private_inputs = {name: describe_path(root / path) for name, path in REQUIRED_PRIVATE_INPUTS}
    manifest = {
        "note": (
            "This manifest records current generated artifacts. "
            "The phone-captured Task 1 inputs are private required inputs and are separately audited below."
        ),
        "artifacts": artifacts,
        "required_private_inputs": private_inputs,
        "completion_notes": [
            "Object B is a completed threestudio DreamFusion/SDS run using local Stable Diffusion v1.5 fp16 weights.",
            "The Task 1 background source, dataset, 2DGS model, and exported mesh are recorded in results/task1/real_run_manifest.json.",
            "The main report is report/HW3_report.ipynb; report/build/main.pdf is retained as a PDF backup.",
            "Student names, IDs, GitHub URL, and permanent model-weight URL must be filled by the submitter in the notebook metadata cell.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
