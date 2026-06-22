#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep the largest connected triangle components of a mesh.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--keep-clusters", type=int, default=1)
    parser.add_argument("--min-triangles", type=int, default=0)
    parser.add_argument("--smooth-taubin-iters", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input mesh not found: {args.input}")

    mesh = o3d.io.read_triangle_mesh(str(args.input))
    input_vertices = len(mesh.vertices)
    input_triangles = len(mesh.triangles)
    if input_triangles == 0:
        raise SystemExit(f"Input mesh has no triangles: {args.input}")

    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    cleaned_vertices = len(mesh.vertices)
    cleaned_triangles = len(mesh.triangles)

    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    order = np.argsort(-cluster_n_triangles)
    keep_ids = []
    for cluster_id in order:
        if len(keep_ids) >= args.keep_clusters:
            break
        if int(cluster_n_triangles[cluster_id]) < args.min_triangles:
            continue
        keep_ids.append(int(cluster_id))

    if not keep_ids:
        raise SystemExit("No connected components passed the filtering thresholds.")

    keep_mask = np.isin(triangle_clusters, np.asarray(keep_ids))
    filtered = o3d.geometry.TriangleMesh(mesh)
    filtered.remove_triangles_by_mask((~keep_mask).tolist())
    filtered.remove_unreferenced_vertices()
    filtered.remove_degenerate_triangles()
    filtered.remove_duplicated_triangles()
    filtered.remove_duplicated_vertices()
    if args.smooth_taubin_iters > 0:
        filtered = filtered.filter_smooth_taubin(number_of_iterations=args.smooth_taubin_iters)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.output), filtered, write_ascii=False)

    vertices = np.asarray(filtered.vertices)
    clusters = []
    triangles = np.asarray(mesh.triangles)
    for cluster_id in order.tolist():
        tri_idx = np.where(triangle_clusters == cluster_id)[0]
        vert_idx = np.unique(triangles[tri_idx].reshape(-1))
        cluster_vertices = np.asarray(mesh.vertices)[vert_idx]
        clusters.append(
            {
                "id": int(cluster_id),
                "triangles": int(cluster_n_triangles[cluster_id]),
                "area": float(cluster_area[cluster_id]),
                "bbox_min": cluster_vertices.min(axis=0).tolist(),
                "bbox_max": cluster_vertices.max(axis=0).tolist(),
                "kept": int(cluster_id) in keep_ids,
            }
        )

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "keep_clusters": args.keep_clusters,
        "min_triangles": args.min_triangles,
        "smooth_taubin_iters": args.smooth_taubin_iters,
        "kept_cluster_ids": keep_ids,
        "input_vertices": input_vertices,
        "input_triangles": input_triangles,
        "cleaned_vertices": cleaned_vertices,
        "cleaned_triangles": cleaned_triangles,
        "output_vertices": len(filtered.vertices),
        "output_triangles": len(filtered.triangles),
        "output_has_vertex_colors": len(filtered.vertex_colors) > 0,
        "output_bbox_min": vertices.min(axis=0).tolist() if len(vertices) else None,
        "output_bbox_max": vertices.max(axis=0).tolist() if len(vertices) else None,
        "clusters": clusters,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Kept clusters {keep_ids}: {input_triangles} -> {len(filtered.triangles)} triangles")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
