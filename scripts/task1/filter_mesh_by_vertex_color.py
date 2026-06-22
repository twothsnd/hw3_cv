#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove near-white background surfaces from a colored mesh.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--saturation-threshold", type=float, default=0.08)
    parser.add_argument("--brightness-threshold", type=float, default=0.90)
    parser.add_argument("--min-colored-vertices", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--keep-clusters", type=int, default=12)
    parser.add_argument("--min-cluster-triangles", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh = o3d.io.read_triangle_mesh(str(args.input))
    if not mesh.has_triangles() or not mesh.has_vertices():
        raise SystemExit(f"Input mesh is empty: {args.input}")
    if not mesh.has_vertex_colors():
        raise SystemExit(f"Input mesh has no vertex colors: {args.input}")

    colors = np.asarray(mesh.vertex_colors)
    triangles = np.asarray(mesh.triangles)
    saturation = colors.max(axis=1) - colors.min(axis=1)
    brightness = colors.mean(axis=1)
    colored_vertex = (saturation >= args.saturation_threshold) | (brightness <= args.brightness_threshold)
    tri_colored_counts = colored_vertex[triangles].sum(axis=1)
    keep_triangles = tri_colored_counts >= args.min_colored_vertices

    filtered = o3d.geometry.TriangleMesh(mesh)
    filtered.remove_triangles_by_mask((~keep_triangles).tolist())
    filtered.remove_unreferenced_vertices()
    triangle_clusters, cluster_n_triangles, _ = filtered.cluster_connected_triangles()
    cluster_ids = np.asarray(triangle_clusters)
    cluster_sizes = np.asarray(cluster_n_triangles)
    keep_cluster_ids = set(
        int(idx)
        for idx in np.argsort(cluster_sizes)[::-1][: args.keep_clusters]
        if cluster_sizes[idx] >= args.min_cluster_triangles
    )
    if keep_cluster_ids:
        keep_after_cluster = np.array([cluster_id in keep_cluster_ids for cluster_id in cluster_ids])
        filtered.remove_triangles_by_mask((~keep_after_cluster).tolist())

    filtered.remove_unreferenced_vertices()
    filtered.compute_vertex_normals()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(args.output), filtered, write_ascii=False, compressed=False)
    if not ok:
        raise SystemExit(f"Failed to write {args.output}")

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "saturation_threshold": args.saturation_threshold,
        "brightness_threshold": args.brightness_threshold,
        "min_colored_vertices": args.min_colored_vertices,
        "keep_clusters": args.keep_clusters,
        "min_cluster_triangles": args.min_cluster_triangles,
        "input_vertices": len(mesh.vertices),
        "input_triangles": len(mesh.triangles),
        "colored_vertices": int(colored_vertex.sum()),
        "kept_triangles_before_cluster": int(keep_triangles.sum()),
        "output_vertices": len(filtered.vertices),
        "output_triangles": len(filtered.triangles),
        "output_bbox_min": filtered.get_min_bound().tolist() if len(filtered.vertices) else None,
        "output_bbox_max": filtered.get_max_bound().tolist() if len(filtered.vertices) else None,
        "output_extent": filtered.get_axis_aligned_bounding_box().get_extent().tolist()
        if len(filtered.vertices)
        else None,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
