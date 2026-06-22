#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import open3d as o3d
except Exception:  # pragma: no cover - PLY previews require open3d
    o3d = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a compact multi-view preview for one OBJ/PLY mesh.")
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--faces", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--color", nargs=3, type=float, default=[0.18, 0.48, 0.88])
    parser.add_argument("--use-vertex-colors", action="store_true")
    parser.add_argument("--views", default="-65,18;25,18;110,24", help="Semicolon-separated azimuth,elevation pairs.")
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--cell-width", type=float, default=3.5)
    parser.add_argument("--cell-height", type=float, default=3.2)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def normalized_vertices(vertices: np.ndarray) -> np.ndarray:
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    shifted = vertices - center[None, :]
    scale = np.max(vertices.max(axis=0) - vertices.min(axis=0))
    if scale <= 0:
        return shifted
    return shifted / scale


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
    if o3d is None:
        raise RuntimeError("open3d is required for non-OBJ mesh previews")
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64) if mesh.has_vertex_colors() else None
    return vertices, faces, colors


def sample_faces(faces: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(faces) <= limit:
        return faces
    return faces[rng.choice(len(faces), size=limit, replace=False)]


def parse_views(views: str) -> list[tuple[float, float, float]]:
    parsed = []
    for item in views.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(",")
        if len(parts) == 2:
            azim, elev = parts
            roll = 0.0
        elif len(parts) == 3:
            azim, elev, roll = parts
        else:
            raise ValueError(f"Invalid view spec: {item}")
        parsed.append((float(azim), float(elev), float(roll)))
    return parsed or [(-65, 18, 0), (25, 18, 0), (110, 24, 0)]


def render_preview(
    mesh_path: Path,
    output: Path,
    face_limit: int,
    seed: int,
    color: list[float],
    use_vertex_colors: bool,
    views_arg: str,
    cols: int,
    cell_width: float,
    cell_height: float,
    dpi: int,
) -> None:
    vertices, faces, vertex_colors = read_mesh(mesh_path)
    if len(vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")
    vertices = normalized_vertices(vertices)
    rng = np.random.default_rng(seed)
    faces = sample_faces(faces, face_limit, rng) if len(faces) else faces

    if use_vertex_colors and vertex_colors is not None and len(faces):
        face_colors = np.clip(vertex_colors[faces].mean(axis=1), 0, 1)
    else:
        face_colors = np.repeat(np.asarray(color, dtype=np.float64)[None, :], len(faces), axis=0)

    output.parent.mkdir(parents=True, exist_ok=True)
    views = parse_views(views_arg)
    cols = max(1, min(cols, len(views)))
    rows = int(np.ceil(len(views) / cols))
    fig = plt.figure(figsize=(cell_width * cols, cell_height * rows), dpi=dpi)
    for idx, (azim, elev, roll) in enumerate(views, start=1):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        ax.set_facecolor((0.96, 0.96, 0.94))
        if len(faces):
            poly = Poly3DCollection(
                vertices[faces],
                facecolors=np.clip(face_colors * 0.9, 0, 1),
                edgecolors="none",
                linewidths=0.0,
            )
            ax.add_collection3d(poly)
        else:
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.5, c=[color], depthshade=False)
        ax.view_init(elev=elev, azim=azim, roll=roll)
        ax.set_xlim(-0.58, 0.58)
        ax.set_ylim(-0.58, 0.58)
        ax.set_zlim(-0.58, 0.58)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.0, hspace=0.0)
    fig.savefig(output)
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    args = parse_args()
    render_preview(
        args.mesh,
        args.output,
        args.faces,
        args.seed,
        args.color,
        args.use_vertex_colors,
        args.views,
        args.cols,
        args.cell_width,
        args.cell_height,
        args.dpi,
    )


if __name__ == "__main__":
    main()
