#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path
from typing import Any


MEDIA_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly check whether the HW3 submission is ready.")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--allow-demo-task1", action="store_true", help="Do not fail on synthetic Task 1 demo artifacts.")
    return parser.parse_args()


def exists_file(path: str) -> bool:
    p = Path(path)
    return p.exists() and p.is_file() and p.stat().st_size > 0


def exists_dir_with_files(path: str, exts: set[str] | None = None) -> bool:
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return False
    for child in p.rglob("*"):
        if child.is_file() and (exts is None or child.suffix.lower() in exts):
            return True
    return False


def object_a_input_ready(path: str) -> bool:
    p = Path(path)
    if p.is_file():
        return p.suffix.lower() in MEDIA_EXTS and p.stat().st_size > 0
    return exists_dir_with_files(path, MEDIA_EXTS)


def json_has_number(path: str, key: str) -> bool:
    if not exists_file(path):
        return False
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8")).get(key)
    except Exception:
        return False
    return isinstance(value, (int, float))


def tar_contains(path: str, members: list[str]) -> bool:
    if not exists_file(path):
        return False
    try:
        with tarfile.open(path, "r:gz") as tf:
            names = set(tf.getnames())
    except Exception:
        return False
    return all(any(name == member or name.startswith(member.rstrip("/") + "/") for name in names) for member in members)


def report_metadata_ready(path: str) -> bool:
    if not exists_file(path):
        return False
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    placeholders = [
        "Student Name",
        "Student ID",
        "your-account",
        "your-cloud-link",
        "fill in concrete contributions",
    ]
    return not any(token in text for token in placeholders)


def check(name: str, ok: bool, evidence: str, severity: str = "error") -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "evidence": evidence, "severity": severity}


def main() -> None:
    args = parse_args()
    checks = [
        check("object A phone capture exists", object_a_input_ready("data/raw/object_A"), "data/raw/object_A"),
        check("object C foreground image exists", exists_file("data/raw/object_C/object_c.png"), "data/raw/object_C/object_c.png"),
        check("Task 1 real-run manifest exists", exists_file("results/task1/real_run_manifest.json"), "results/task1/real_run_manifest.json"),
        check("Object A COLMAP manifest exists", exists_file("data/task1/object_A_colmap/colmap_manifest.json"), "data/task1/object_A_colmap/colmap_manifest.json"),
        check("Object A 2DGS mesh exists", exists_file("results/task1/2dgs/object_A/train/ours_latest/fuse_post.ply"), "results/task1/2dgs/object_A/train/ours_latest/fuse_post.ply"),
        check("Background 2DGS mesh exists", exists_file("results/task1/2dgs/background_garden/train/ours_latest/fuse_unbounded_post.ply"), "results/task1/2dgs/background_garden/train/ours_latest/fuse_unbounded_post.ply"),
        check("Object B mesh exists", exists_file("results/task1/aigc/object_B_text3d/export/model.obj"), "results/task1/aigc/object_B_text3d/export/model.obj"),
        check("Object C mesh exists", exists_file("results/task1/aigc/object_C_image3d/model.obj"), "results/task1/aigc/object_C_image3d/model.obj"),
        check("Final real fusion video exists", exists_file("results/task1/fusion/fusion_walkthrough.mp4"), "results/task1/fusion/fusion_walkthrough.mp4"),
        check("Task 2 B checkpoint exists", exists_file("results/task2/act_env_B_inferred/checkpoints/000100/pretrained_model/model.safetensors"), "results/task2/act_env_B_inferred/checkpoints/000100/pretrained_model/model.safetensors"),
        check("Task 2 ABC checkpoint exists", exists_file("results/task2/act_env_ABC_small/checkpoints/000100/pretrained_model/model.safetensors"), "results/task2/act_env_ABC_small/checkpoints/000100/pretrained_model/model.safetensors"),
        check("Task 2 B 500-step checkpoint exists", exists_file("results/task2/act_env_B_inferred_500/checkpoints/000500/pretrained_model/model.safetensors"), "results/task2/act_env_B_inferred_500/checkpoints/000500/pretrained_model/model.safetensors"),
        check("Task 2 ABC 500-step checkpoint exists", exists_file("results/task2/act_env_ABC_small_500/checkpoints/000500/pretrained_model/model.safetensors"), "results/task2/act_env_ABC_small_500/checkpoints/000500/pretrained_model/model.safetensors"),
        check("B-on-D Action L1 JSON exists", json_has_number("results/task2/eval_B_on_D.json", "mean_action_l1"), "results/task2/eval_B_on_D.json"),
        check("ABC-on-D Action L1 JSON exists", json_has_number("results/task2/eval_ABC_on_D.json", "mean_action_l1"), "results/task2/eval_ABC_on_D.json"),
        check("B500-on-D Action L1 JSON exists", json_has_number("results/task2/eval_B500_on_D.json", "mean_action_l1"), "results/task2/eval_B500_on_D.json"),
        check("ABC500-on-D Action L1 JSON exists", json_has_number("results/task2/eval_ABC500_on_D.json", "mean_action_l1"), "results/task2/eval_ABC500_on_D.json"),
        check("Task 2 training loss figure exists", exists_file("report/figures/task2_training_loss.pdf"), "report/figures/task2_training_loss.pdf"),
        check("Task 2 Action L1 figure exists", exists_file("report/figures/task2_action_l1.pdf"), "report/figures/task2_action_l1.pdf"),
        check("Report PDF exists", exists_file("report/build/main.pdf"), "report/build/main.pdf"),
        check("Report metadata is filled", report_metadata_ready("report/main.tex"), "report/main.tex"),
        check(
            "Weight package contains both ACT policies and metrics",
            tar_contains(
                "weights/cv_hw3_task2_act_weights.tar.gz",
                [
                    "results/task2/act_env_B_inferred/checkpoints/000100/pretrained_model",
                    "results/task2/act_env_ABC_small/checkpoints/000100/pretrained_model",
                    "results/task2/act_env_B_inferred_500/checkpoints/000500/pretrained_model",
                    "results/task2/act_env_ABC_small_500/checkpoints/000500/pretrained_model",
                    "results/task2/eval_B_on_D.json",
                    "results/task2/eval_ABC_on_D.json",
                    "results/task2/eval_B500_on_D.json",
                    "results/task2/eval_ABC500_on_D.json",
                ],
            ),
            "weights/cv_hw3_task2_act_weights.tar.gz",
        ),
        check("README exists", exists_file("README.md"), "README.md"),
    ]

    if Path("results/task1/demo_assets/provenance.json").exists() and not Path("results/task1/real_run_manifest.json").exists():
        checks.append(
            check(
                "Task 1 is not synthetic demo only",
                args.allow_demo_task1,
                "results/task1/demo_assets/provenance.json exists without real_run_manifest.json",
            )
        )

    failures = [item for item in checks if not item["ok"] and item["severity"] == "error"]
    result = {"ready": not failures, "num_checks": len(checks), "num_failures": len(failures), "checks": checks}

    for item in checks:
        status = "OK" if item["ok"] else "FAIL"
        print(f"[{status}] {item['name']} :: {item['evidence']}")
    print(f"ready={result['ready']} failures={result['num_failures']}/{result['num_checks']}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {args.json}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
