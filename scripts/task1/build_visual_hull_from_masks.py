#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap
from PIL import Image
from skimage import measure


@dataclass(frozen=True)
class View:
    name: str
    matrix: np.ndarray
    center: np.ndarray
    f: float
    cx: float
    cy: float
    width: int
    height: int
    rgb: np.ndarray
    mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a foreground visual hull mesh from COLMAP poses and RGBA/mask images."
    )
    parser.add_argument("--sparse", required=True, type=Path, help="COLMAP sparse model directory.")
    parser.add_argument("--images", required=True, type=Path, help="RGBA images whose alpha channel is the mask.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--resolution", type=int, default=192, help="Longest voxel-grid side.")
    parser.add_argument("--vote-threshold", type=float, default=0.72)
    parser.add_argument("--mask-threshold", type=int, default=16)
    parser.add_argument("--mask-dilate", type=int, default=4)
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    parser.add_argument("--margin", type=float, default=0.28, help="BBox margin as a fraction of longest robust side.")
    parser.add_argument("--chunk-size", type=int, default=262144)
    parser.add_argument("--smooth-iters", type=int, default=4)
    parser.add_argument("--keep-clusters", type=int, default=1)
    return parser.parse_args()


def load_rgba(path: Path, mask_threshold: int, mask_dilate: int) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(path).convert("RGBA")
    arr = np.asarray(image)
    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    mask = arr[:, :, 3] > mask_threshold
    if mask_dilate > 0:
        k = 2 * mask_dilate + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return rgb, mask


def load_views(args: argparse.Namespace) -> tuple[pycolmap.Reconstruction, list[View]]:
    reconstruction = pycolmap.Reconstruction(str(args.sparse))
    views: list[View] = []
    for image in sorted(reconstruction.images.values(), key=lambda item: item.name):
        image_path = args.images / image.name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing RGBA image for COLMAP view {image.name}: {image_path}")
        camera = reconstruction.cameras[image.camera_id]
        if str(camera.model) != "CameraModelId.SIMPLE_PINHOLE":
            raise ValueError(f"Only SIMPLE_PINHOLE is supported, got {camera.model} for {image.name}")
        rgb, mask = load_rgba(image_path, args.mask_threshold, args.mask_dilate)
        if rgb.shape[1] != camera.width or rgb.shape[0] != camera.height:
            raise ValueError(
                f"Image/camera size mismatch for {image.name}: "
                f"image={rgb.shape[1]}x{rgb.shape[0]} camera={camera.width}x{camera.height}"
            )
        f, cx, cy = [float(v) for v in camera.params]
        views.append(
            View(
                name=image.name,
                matrix=np.asarray(image.cam_from_world().matrix(), dtype=np.float64),
                center=np.asarray(image.projection_center(), dtype=np.float64),
                f=f,
                cx=cx,
                cy=cy,
                width=int(camera.width),
                height=int(camera.height),
                rgb=rgb,
                mask=mask,
            )
        )
    if not views:
        raise RuntimeError(f"No registered COLMAP images found in {args.sparse}")
    return reconstruction, views


def robust_bounds(reconstruction: pycolmap.Reconstruction, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray([point.xyz for point in reconstruction.points3D.values()], dtype=np.float64)
    if points.size == 0:
        centers = np.asarray([image.projection_center() for image in reconstruction.images.values()], dtype=np.float64)
        center = centers.mean(axis=0)
        radius = np.linalg.norm(centers - center, axis=1).mean() * 0.2
        return center - radius, center + radius

    lo = np.percentile(points, args.percentile_low, axis=0)
    hi = np.percentile(points, args.percentile_high, axis=0)
    size = hi - lo
    margin = max(float(size.max()) * args.margin, 1e-3)
    return lo - margin, hi + margin


def make_grid(lo: np.ndarray, hi: np.ndarray, longest_resolution: int) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray, np.ndarray]:
    size = hi - lo
    longest = float(size.max())
    dims = np.maximum(32, np.ceil(size / longest * longest_resolution).astype(int))
    axes = [np.linspace(lo[i], hi[i], int(dims[i]), dtype=np.float32) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    spacing = np.asarray(
        [
            (hi[i] - lo[i]) / max(int(dims[i]) - 1, 1)
            for i in range(3)
        ],
        dtype=np.float64,
    )
    return grid, (int(dims[0]), int(dims[1]), int(dims[2])), spacing, lo.astype(np.float64)


def project_points(points: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = view.matrix[:, :3]
    t = view.matrix[:, 3]
    cam = points @ r.T + t
    z = cam[:, 2]
    positive = z > 1e-6
    u = view.f * cam[:, 0] / np.maximum(z, 1e-6) + view.cx
    v = view.f * cam[:, 1] / np.maximum(z, 1e-6) + view.cy
    inside = (
        positive
        & (u >= 0)
        & (u <= view.width - 1)
        & (v >= 0)
        & (v <= view.height - 1)
    )
    return u, v, inside


def carve_volume(grid: np.ndarray, dims: tuple[int, int, int], views: list[View], args: argparse.Namespace) -> np.ndarray:
    votes = np.zeros(len(grid), dtype=np.uint16)
    for view in views:
        for start in range(0, len(grid), args.chunk_size):
            end = min(start + args.chunk_size, len(grid))
            points = grid[start:end]
            u, v, inside = project_points(points, view)
            hit = np.zeros(end - start, dtype=bool)
            if np.any(inside):
                uu = np.rint(u[inside]).astype(np.int32)
                vv = np.rint(v[inside]).astype(np.int32)
                local = view.mask[vv, uu]
                hit_indices = np.flatnonzero(inside)
                hit[hit_indices] = local
            votes[start:end] += hit.astype(np.uint16)
    required = int(math.ceil(len(views) * args.vote_threshold))
    occupied = votes >= required
    volume = occupied.reshape(dims)
    if np.count_nonzero(volume) == 0:
        raise RuntimeError(
            f"Visual hull is empty: views={len(views)} required_votes={required} "
            f"threshold={args.vote_threshold}"
        )
    return volume


def keep_largest_components(mesh: o3d.geometry.TriangleMesh, keep_clusters: int) -> o3d.geometry.TriangleMesh:
    if keep_clusters <= 0 or len(mesh.triangles) == 0:
        return mesh
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels_np = np.asarray(labels)
    counts_np = np.asarray(counts)
    if counts_np.size <= keep_clusters:
        return mesh
    keep = set(np.argsort(counts_np)[-keep_clusters:].tolist())
    remove = np.asarray([label not in keep for label in labels_np], dtype=bool)
    mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    return mesh


def color_mesh_vertices(mesh: o3d.geometry.TriangleMesh, views: list[View], chunk_size: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    colors = np.zeros((len(vertices), 3), dtype=np.float64)
    weights = np.zeros(len(vertices), dtype=np.float64)

    for view in views:
        for start in range(0, len(vertices), chunk_size):
            end = min(start + chunk_size, len(vertices))
            pts = vertices[start:end]
            nrm = normals[start:end]
            u, v, inside = project_points(pts, view)
            if not np.any(inside):
                continue
            to_cam = view.center[None, :] - pts
            to_cam /= np.maximum(np.linalg.norm(to_cam, axis=1, keepdims=True), 1e-9)
            facing = np.sum(nrm * to_cam, axis=1) > 0.15
            ok = inside & facing
            if not np.any(ok):
                continue
            uu = np.rint(u[ok]).astype(np.int32)
            vv = np.rint(v[ok]).astype(np.int32)
            fg = view.mask[vv, uu]
            if not np.any(fg):
                continue
            local_indices = np.flatnonzero(ok)[fg]
            sample = view.rgb[vv[fg], uu[fg]]
            w = np.sum(nrm[local_indices] * to_cam[local_indices], axis=1).clip(0.05, 1.0)
            colors[start + local_indices] += sample * w[:, None]
            weights[start + local_indices] += w

    missing = weights <= 1e-8
    if np.any(~missing):
        colors[~missing] /= weights[~missing, None]
    if np.any(missing):
        fallback = colors[~missing].mean(axis=0) if np.any(~missing) else np.array([0.7, 0.7, 0.7])
        colors[missing] = fallback
    return colors.clip(0, 1)


def volume_to_mesh(volume: np.ndarray, lo: np.ndarray, spacing: np.ndarray, args: argparse.Namespace) -> o3d.geometry.TriangleMesh:
    padded = np.pad(volume.astype(np.float32), 1, mode="constant", constant_values=0)
    padded_lo = lo - spacing
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=tuple(spacing.tolist()))
    verts = verts + padded_lo[None, :]

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh = keep_largest_components(mesh, args.keep_clusters)
    if args.smooth_iters > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smooth_iters)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def mesh_stats(mesh: o3d.geometry.TriangleMesh) -> dict[str, object]:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) == 0:
        bbox_min = bbox_max = bbox_size = [0.0, 0.0, 0.0]
    else:
        bbox_min_np = vertices.min(axis=0)
        bbox_max_np = vertices.max(axis=0)
        bbox_min = bbox_min_np.tolist()
        bbox_max = bbox_max_np.tolist()
        bbox_size = (bbox_max_np - bbox_min_np).tolist()
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(triangles)),
        "bbox": {"min": bbox_min, "max": bbox_max},
        "bbox_size": bbox_size,
    }


def main() -> None:
    args = parse_args()
    reconstruction, views = load_views(args)
    lo, hi = robust_bounds(reconstruction, args)
    grid, dims, spacing, lo = make_grid(lo, hi, args.resolution)
    volume = carve_volume(grid, dims, views, args)
    mesh = volume_to_mesh(volume, lo, spacing, args)
    colors = color_mesh_vertices(mesh, views, args.chunk_size)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.output), mesh, write_ascii=False)

    manifest = {
        "output": str(args.output),
        "sparse": str(args.sparse),
        "images": str(args.images),
        "num_views": len(views),
        "grid_dims": list(dims),
        "occupied_voxels": int(np.count_nonzero(volume)),
        "vote_threshold": args.vote_threshold,
        "mask_threshold": args.mask_threshold,
        "mask_dilate": args.mask_dilate,
        "bounds": {"min": lo.tolist(), "max": hi.tolist()},
        "spacing": spacing.tolist(),
        "mesh": mesh_stats(mesh),
    }
    manifest_path = args.manifest or args.output.with_suffix(".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
