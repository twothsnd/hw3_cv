#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fallback Task 1 fusion renderer without Blender.")
    parser.add_argument("config", type=Path, nargs="?", default=Path("configs/fusion_scene.json"))
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--object-faces", type=int, default=12000)
    parser.add_argument("--background-points", type=int, default=24000)
    return parser.parse_args()


def rotation_matrix(degrees: list[float]) -> np.ndarray:
    rx, ry, rz = [math.radians(v) for v in degrees]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return mz @ my @ mx


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    vertices: list[list[float]] = []
    colors: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
            elif line.startswith("f "):
                face = [int(token.split("/")[0]) - 1 for token in line.split()[1:4]]
                faces.append(face)
    vertex_colors = np.asarray(colors, dtype=np.float64) if len(colors) == len(vertices) else None
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32), vertex_colors


def read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if path.suffix.lower() == ".obj":
        return read_obj(path)
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64) if mesh.has_vertex_colors() else None
    return vertices, faces, colors


def apply_transform(vertices: np.ndarray, asset: dict) -> np.ndarray:
    scale = np.asarray(asset.get("scale", [1, 1, 1]), dtype=np.float64)
    rot = rotation_matrix(asset.get("rotation_degrees", [0, 0, 0]))
    scaled = vertices * scale[None, :]
    if "center_at" in asset:
        local_center = (scaled.min(axis=0) + scaled.max(axis=0)) * 0.5
        translation = np.asarray(asset["center_at"], dtype=np.float64) - rot @ local_center
    else:
        translation = np.asarray(asset.get("location", [0, 0, 0]), dtype=np.float64)
    return scaled @ rot.T + translation[None, :]


def sample_faces(faces: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(faces) <= limit:
        return faces
    return faces[rng.choice(len(faces), size=limit, replace=False)]


def make_asset(asset: dict, rng: np.random.Generator, object_face_limit: int, background_point_limit: int) -> dict:
    path = Path(asset["path"])
    if not path.is_absolute():
        path = ROOT / path
    vertices, faces, vertex_colors = read_mesh(path)
    vertices = apply_transform(vertices, asset)

    if asset.get("override_material", False) or vertex_colors is None or not asset.get("use_vertex_colors", False):
        color = np.asarray(asset.get("material_color", [0.7, 0.7, 0.7, 1.0])[:3], dtype=np.float64)
        vertex_colors = np.repeat(color[None, :], len(vertices), axis=0)

    if asset["name"] == "background":
        count = min(background_point_limit, len(vertices))
        idx = rng.choice(len(vertices), size=count, replace=False)
        return {"name": asset["name"], "type": "points", "vertices": vertices[idx], "colors": vertex_colors[idx]}

    faces_sample = sample_faces(faces, object_face_limit, rng)
    face_colors = np.clip(vertex_colors[faces_sample].mean(axis=1), 0, 1)
    return {
        "name": asset["name"],
        "type": "mesh",
        "vertices": vertices,
        "faces": faces_sample,
        "face_colors": face_colors,
    }


def render_frame(fig, ax, assets: list[dict], azim: float, elevation: float, limits: tuple[tuple[float, float], ...]) -> np.ndarray:
    ax.clear()
    fig.patch.set_facecolor((0.94, 0.94, 0.91))
    ax.set_facecolor((0.94, 0.94, 0.91))
    for asset in assets:
        if asset["type"] == "points":
            vertices = asset["vertices"]
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=asset["colors"], s=0.25, alpha=0.28, depthshade=False)
        else:
            vertices = asset["vertices"]
            faces = asset["faces"]
            poly = Poly3DCollection(
                vertices[faces],
                facecolors=np.clip(asset["face_colors"] * 0.88, 0, 1),
                edgecolors="none",
                linewidths=0.0,
                alpha=1.0,
            )
            ax.add_collection3d(poly)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.view_init(elev=elevation, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect((limits[0][1] - limits[0][0], limits[1][1] - limits[1][0], limits[2][1] - limits[2][0]))
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba()).reshape(h, w, 4)
    return cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)

    assets = [make_asset(asset, rng, args.object_faces, args.background_points) for asset in config["assets"]]
    output_dir = Path(config.get("output_dir", "results/task1/fusion"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    width, height = config.get("resolution", [1280, 720])
    frames = args.frames or int(config.get("frames", 144))
    fps = int(config.get("fps", 24))
    center = np.asarray(config.get("camera", {}).get("center", [0.0, 0.0, 0.55]), dtype=np.float64)
    limits = ((center[0] - 2.5, center[0] + 2.5), (center[1] - 2.35, center[1] + 2.35), (-1.35, 2.45))

    fig = plt.figure(figsize=(width / 160, height / 160), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    preview = render_frame(fig, ax, assets, azim=-55.0, elevation=20.0, limits=limits)
    cv2.imwrite(str(output_dir / "fusion_preview.png"), preview)

    writer = cv2.VideoWriter(
        str(output_dir / "fusion_walkthrough.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to open cv2 VideoWriter for fusion_walkthrough.mp4")
    for frame in range(frames):
        azim = -55.0 + 360.0 * frame / frames
        image = render_frame(fig, ax, assets, azim=azim, elevation=20.0, limits=limits)
        writer.write(image)
    writer.release()
    plt.close(fig)
    print(f"Wrote {output_dir / 'fusion_preview.png'}")
    print(f"Wrote {output_dir / 'fusion_walkthrough.mp4'} frames={frames} fps={fps}")


if __name__ == "__main__":
    main()
