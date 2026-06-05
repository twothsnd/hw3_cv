#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACTS = [
    "report/build/main.pdf",
    "report/figures/task1_fusion_preview.png",
    "report/figures/task1_asset_mesh_stats.pdf",
    "report/figures/task2_training_loss.pdf",
    "report/figures/task2_training_loss.csv",
    "report/figures/task2_action_l1.pdf",
    "results/task1/fusion_demo/fusion_preview.png",
    "results/task1/fusion_demo/fusion_walkthrough.mp4",
    "results/task1/fusion_demo/fusion_scene.blend",
    "results/task1/asset_stats.json",
    "results/task1/task1_2dgs_smoke_stats.json",
    "results/task1/2dgs/object_A_synthetic_smoke/train/ours_200/fuse_post.ply",
    "results/task1/background_garden_sparse_smoke_stats.json",
    "results/task1/2dgs/background_garden_sparse_smoke/train/ours_300/fuse_unbounded_post.ply",
    "results/task1/fusion_sparse_bg_smoke/fusion_preview.png",
    "results/task1/fusion_sparse_bg_smoke/fusion_walkthrough.mp4",
    "results/task2/logs/train_B_losslog.log",
    "results/task2/logs/train_ABC_losslog.log",
    "results/task2/logs/train_B_500.log",
    "results/task2/logs/train_ABC_500.log",
    "results/task2/eval_B_on_D.json",
    "results/task2/eval_ABC_on_D.json",
    "results/task2/eval_B500_on_D.json",
    "results/task2/eval_ABC500_on_D.json",
    "results/task2/act_eval_summary.csv",
    "results/task2/download_abc_40_full_attempt.json",
    "results/task2/act_env_B_inferred/checkpoints/000100/pretrained_model/model.safetensors",
    "results/task2/act_env_ABC_small/checkpoints/000100/pretrained_model/model.safetensors",
    "results/task2/act_env_B_inferred_500/checkpoints/000500/pretrained_model/model.safetensors",
    "results/task2/act_env_ABC_small_500/checkpoints/000500/pretrained_model/model.safetensors",
    "results/submission_readiness.json",
]

REQUIRED_PRIVATE_INPUTS = [
    ("object_A_phone_capture", "data/raw/object_A"),
    ("object_C_single_foreground_image", "data/raw/object_C/object_c.png"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a manifest for generated HW3 submission artifacts.")
    parser.add_argument("--output", default="results/submission_manifest.json", type=Path)
    parser.add_argument("--artifacts", nargs="*", default=DEFAULT_ARTIFACTS)
    return parser.parse_args()


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
    artifacts = {name: describe_path(root / name) for name in args.artifacts}
    private_inputs = {name: describe_path(root / path) for name, path in REQUIRED_PRIVATE_INPUTS}
    manifest = {
        "note": (
            "This manifest records current generated artifacts. "
            "The phone-captured Task 1 inputs are private required inputs and are separately audited below."
        ),
        "artifacts": artifacts,
        "required_private_inputs": private_inputs,
        "known_completion_gap": (
            "Task 1 real phone-captured object A multi-view/video and object C foreground image "
            "are not present when their audited paths are missing or empty."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
