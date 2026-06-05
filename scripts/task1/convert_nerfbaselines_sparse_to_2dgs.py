#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cv_hw3.common import relative_or_absolute, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NerfBaselines mipnerf360-sparse JSON thumbnails to a 2DGS NeRF-style dataset."
    )
    parser.add_argument("--nbv-json", required=True, type=Path)
    parser.add_argument("--pointcloud", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--pose-convention",
        default="opencv-c2w",
        choices=["opencv-c2w", "opengl-c2w"],
        help="NerfBaselines documents poses as OpenCV camera-to-world by default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def decode_thumbnail(url: str) -> Image.Image:
    prefix = "data:image/jpeg;base64,"
    if not url.startswith(prefix):
        raise ValueError("Expected thumbnail_url to be a base64 JPEG data URL")
    raw = base64.b64decode(url[len(prefix) :])
    return Image.open(BytesIO(raw)).convert("RGB")


def pose_to_transform(pose: list[float], pose_convention: str) -> list[list[float]]:
    if len(pose) != 12:
        raise ValueError(f"Expected 12 pose values, got {len(pose)}")
    mat = [
        [float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3])],
        [float(pose[4]), float(pose[5]), float(pose[6]), float(pose[7])],
        [float(pose[8]), float(pose[9]), float(pose[10]), float(pose[11])],
        [0.0, 0.0, 0.0, 1.0],
    ]
    if pose_convention == "opencv-c2w":
        for row in range(3):
            mat[row][1] *= -1.0
            mat[row][2] *= -1.0
    return mat


def camera_angle_x(camera: dict[str, Any], rendered_width: int) -> float:
    source_width = float(camera["image_size"][0])
    fx = float(camera["intrinsics"][0]) * rendered_width / source_width
    return 2.0 * math.atan(rendered_width / (2.0 * fx))


def write_2dgs_pointcloud(src: Path, dst: Path) -> None:
    ply = PlyData.read(src)
    vertices = ply["vertex"]
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype("f4")
    rgb = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype("u1")
    normals = np.zeros_like(xyz, dtype="f4")
    elements = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    elements[:] = list(map(tuple, np.concatenate((xyz, normals, rgb), axis=1)))
    PlyData([PlyElement.describe(elements, "vertex")]).write(dst)


def convert_split(cameras: list[dict[str, Any]], output: Path, split: str, pose_convention: str) -> dict[str, Any]:
    split_dir = output / split
    split_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    angle_x_values = []
    for idx, camera in enumerate(cameras):
        image = decode_thumbnail(camera["thumbnail_url"])
        name = Path(camera["image_name"]).stem
        rel = f"{split}/{idx:03d}_{name}"
        image.save(output / f"{rel}.png")
        angle_x_values.append(camera_angle_x(camera, image.width))
        frames.append(
            {
                "file_path": rel,
                "transform_matrix": pose_to_transform(camera["pose"], pose_convention),
                "source_image_name": camera["image_name"],
            }
        )
    return {
        "camera_angle_x": float(sum(angle_x_values) / len(angle_x_values)),
        "frames": frames,
    }


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.nbv_json.read_text(encoding="utf-8"))
    train = convert_split(data["train"]["cameras"], output, "train", args.pose_convention)
    test = convert_split(data["test"]["cameras"], output, "test", args.pose_convention)
    write_json(output / "transforms_train.json", train)
    write_json(output / "transforms_test.json", test)
    write_2dgs_pointcloud(args.pointcloud, output / "points3d.ply")

    manifest = {
        "type": "nerfbaselines_mipnerf360_sparse_thumbnail_dataset",
        "source_json": relative_or_absolute(args.nbv_json),
        "source_pointcloud": relative_or_absolute(args.pointcloud),
        "scene": data.get("metadata", {}).get("scene"),
        "split": "n24",
        "pose_convention_input": args.pose_convention,
        "train_views": len(train["frames"]),
        "test_views": len(test["frames"]),
        "thumbnail_resolution": Image.open(output / (train["frames"][0]["file_path"] + ".png")).size,
        "metadata": data.get("metadata", {}),
    }
    write_json(output / "dataset_manifest.json", manifest)
    print(f"Wrote 2DGS NeRF-style dataset to {output}")


if __name__ == "__main__":
    main()
