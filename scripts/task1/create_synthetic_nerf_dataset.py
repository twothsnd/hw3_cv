#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a small Blender/NeRF synthetic dataset for 2DGS smoke tests.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-views", default=24, type=int)
    parser.add_argument("--test-views", default=6, type=int)
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--radius", default=3.0, type=float)
    parser.add_argument("--seed", default=7, type=int)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def material(name: str, color: tuple[float, float, float, float]):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def assign(obj, mat) -> None:
    obj.data.materials.append(mat)


def add_cube(name: str, loc, scale, mat):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.location = loc
    obj.scale = scale
    assign(obj, mat)
    return obj


def add_cylinder(name: str, vertices: int, radius: float, depth: float, loc, mat):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=loc)
    obj = bpy.context.object
    obj.name = name
    assign(obj, mat)
    return obj


def create_scene() -> None:
    clear_scene()
    red = material("mug_red", (0.78, 0.12, 0.08, 1.0))
    cream = material("mug_cream", (0.96, 0.88, 0.70, 1.0))
    dark = material("dark_lines", (0.08, 0.08, 0.08, 1.0))
    blue = material("blue_marker", (0.1, 0.38, 0.85, 1.0))
    white = material("white_tile", (0.88, 0.88, 0.82, 1.0))
    gray = material("gray_tile", (0.36, 0.38, 0.40, 1.0))

    # Checker table gives 2DGS/pose-estimation smoke tests strong visual texture.
    for ix in range(-4, 5):
        for iy in range(-4, 5):
            mat = white if (ix + iy) % 2 == 0 else gray
            add_cube(f"tile_{ix}_{iy}", (ix * 0.28, iy * 0.28, -0.035), (0.14, 0.14, 0.018), mat)

    add_cylinder("mug_body", 96, 0.42, 0.82, (0, 0, 0.41), red)
    add_cylinder("mug_rim", 96, 0.45, 0.06, (0, 0, 0.84), cream)
    add_cylinder("mug_base", 96, 0.36, 0.06, (0, 0, 0.05), cream)

    bpy.ops.mesh.primitive_torus_add(major_radius=0.26, minor_radius=0.045, major_segments=80, minor_segments=16)
    handle = bpy.context.object
    handle.name = "mug_handle"
    handle.location = (0.42, 0, 0.46)
    handle.rotation_euler = (math.radians(90), 0, 0)
    handle.scale = (0.80, 1.0, 1.20)
    assign(handle, red)

    for z in [0.25, 0.48, 0.68]:
        ring = add_cylinder(f"stripe_{z:.2f}", 96, 0.424, 0.028, (0, 0, z), dark)
        ring.scale = (1.01, 1.01, 1.0)
    for angle in range(0, 360, 45):
        x = 0.425 * math.cos(math.radians(angle))
        y = 0.425 * math.sin(math.radians(angle))
        marker = add_cube(f"marker_{angle}", (x, y, 0.50), (0.035, 0.012, 0.12), blue)
        marker.rotation_euler[2] = math.radians(angle)

    bpy.ops.object.light_add(type="AREA", location=(0, -3.5, 4.0))
    light = bpy.context.object
    light.name = "Key_Area"
    light.data.energy = 550
    light.data.size = 4.0
    bpy.ops.object.light_add(type="POINT", location=(-2, 2, 2.2))
    fill = bpy.context.object
    fill.name = "Fill_Point"
    fill.data.energy = 90


def look_at(obj, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera(resolution: int):
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera_data.lens = 55
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.film_transparent = False
    for engine in ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"]:
        try:
            bpy.context.scene.render.engine = engine
            break
        except TypeError:
            continue
    if hasattr(bpy.context.scene, "eevee") and hasattr(bpy.context.scene.eevee, "taa_render_samples"):
        bpy.context.scene.eevee.taa_render_samples = 16
    return camera


def camera_matrix(camera) -> list[list[float]]:
    return [[float(v) for v in row] for row in camera.matrix_world]


def render_split(output: Path, split: str, count: int, radius: float, camera, angle_offset: float) -> dict:
    frames = []
    target = Vector((0.0, 0.0, 0.42))
    split_dir = output / split
    split_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    for i in range(count):
        theta = angle_offset + 2.0 * math.pi * i / count
        height = 0.92 + 0.20 * math.sin(theta * 1.7)
        camera.location = (radius * math.cos(theta), radius * math.sin(theta), height)
        look_at(camera, target)
        rel = f"{split}/r_{i:03d}"
        scene.render.filepath = str(output / f"{rel}.png")
        bpy.ops.render.render(write_still=True)
        frames.append({"file_path": rel, "transform_matrix": camera_matrix(camera)})
    return {"camera_angle_x": float(camera.data.angle_x), "frames": frames}


def write_point_cloud(path: Path, seed: int, count: int = 8000) -> None:
    rng = random.Random(seed)
    points = []
    for _ in range(count):
        if rng.random() < 0.65:
            theta = rng.random() * 2.0 * math.pi
            radius = 0.42 + rng.uniform(-0.035, 0.035)
            z = rng.uniform(0.05, 0.84)
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            r, g, b = (198, 38, 28)
        else:
            x = rng.uniform(-1.15, 1.15)
            y = rng.uniform(-1.15, 1.15)
            z = rng.uniform(-0.04, 0.02)
            tile = int((x + 1.2) / 0.28) + int((y + 1.2) / 0.28)
            r, g, b = (225, 225, 210) if tile % 2 == 0 else (92, 96, 102)
        points.append((x, y, z, 0.0, 0.0, 0.0, r, g, b))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property float nx\nproperty float ny\nproperty float nz\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for row in points:
            handle.write("%f %f %f %f %f %f %d %d %d\n" % row)


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else root_dir() / args.output
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    create_scene()
    camera = setup_camera(args.resolution)
    train = render_split(output, "train", args.train_views, args.radius, camera, angle_offset=0.0)
    test = render_split(output, "test", args.test_views, args.radius, camera, angle_offset=math.pi / args.test_views)
    (output / "transforms_train.json").write_text(json.dumps(train, indent=2) + "\n", encoding="utf-8")
    (output / "transforms_test.json").write_text(json.dumps(test, indent=2) + "\n", encoding="utf-8")
    write_point_cloud(output / "points3d.ply", args.seed)
    manifest = {
        "type": "synthetic_nerf_2dgs_smoke",
        "train_views": args.train_views,
        "test_views": args.test_views,
        "resolution": args.resolution,
        "radius": args.radius,
        "seed": args.seed,
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote synthetic NeRF dataset to {output}")


if __name__ == "__main__":
    main()
