#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cv_hw3.common import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Center and scale a mesh for Blender scene fusion.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-size", default=1.0, type=float, help="Largest bbox side after normalization.")
    return parser.parse_args()


def normalize_with_trimesh(input_path: Path, output_path: Path, target_size: float) -> None:
    import trimesh

    mesh = trimesh.load(input_path, force="mesh")
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    extent = (bounds[1] - bounds[0]).max()
    if extent <= 0:
        raise ValueError(f"Degenerate mesh extent: {input_path}")
    mesh.apply_translation(-center)
    mesh.apply_scale(target_size / extent)
    ensure_dir(output_path.parent)
    mesh.export(output_path)


def normalize_obj_fallback(input_path: Path, output_path: Path, target_size: float) -> None:
    vertices: list[tuple[float, float, float]] = []
    lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not vertices:
        raise ValueError(f"No OBJ vertices found in {input_path}")
    mins = [min(v[i] for v in vertices) for i in range(3)]
    maxs = [max(v[i] for v in vertices) for i in range(3)]
    center = [(mins[i] + maxs[i]) * 0.5 for i in range(3)]
    extent = max(maxs[i] - mins[i] for i in range(3))
    scale = target_size / extent
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith("v "):
                parts = line.split()
                xyz = [(float(parts[i + 1]) - center[i]) * scale for i in range(3)]
                f.write(f"v {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}\n")
            else:
                f.write(line + "\n")


def main() -> None:
    args = parse_args()
    try:
        normalize_with_trimesh(args.input, args.output, args.target_size)
    except ImportError:
        if args.input.suffix.lower() != ".obj":
            raise SystemExit("trimesh is required to normalize non-OBJ meshes.")
        normalize_obj_fallback(args.input, args.output, args.target_size)
    print(f"Wrote normalized mesh: {args.output}")


if __name__ == "__main__":
    main()
