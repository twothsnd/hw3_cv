#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic Task-1 demo meshes when real captures are absent.")
    parser.add_argument("--config", default="configs/fusion_scene_demo.json", type=Path)
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
    obj.scale = scale
    assign(obj, mat)
    return obj


def add_cylinder(name: str, vertices: int, radius: float, depth: float, loc, mat):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=loc)
    obj = bpy.context.object
    obj.name = name
    assign(obj, mat)
    return obj


def add_uv_sphere(name: str, radius: float, loc, scale, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=radius, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    assign(obj, mat)
    return obj


def add_cone(name: str, radius1: float, radius2: float, depth: float, loc, mat):
    bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=radius1, radius2=radius2, depth=depth, location=loc)
    obj = bpy.context.object
    obj.name = name
    assign(obj, mat)
    return obj


def select_all_meshes() -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.select_set(True)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if meshes:
        bpy.context.view_layer.objects.active = meshes[0]


def export_selected(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    select_all_meshes()
    if path.suffix.lower() == ".obj":
        if hasattr(bpy.ops.wm, "obj_export"):
            bpy.ops.wm.obj_export(filepath=str(path), export_selected_objects=True)
        else:
            bpy.ops.export_scene.obj(filepath=str(path), use_selection=True)
    elif path.suffix.lower() == ".ply":
        if hasattr(bpy.ops.wm, "ply_export"):
            bpy.ops.wm.ply_export(filepath=str(path), export_selected_objects=True)
        else:
            bpy.ops.export_mesh.ply(filepath=str(path), use_selection=True)
    else:
        raise ValueError(f"Unsupported export format: {path}")


def create_background(path: Path) -> None:
    clear_scene()
    grass = material("grass", (0.20, 0.45, 0.22, 1.0))
    path_mat = material("stone_path", (0.55, 0.52, 0.47, 1.0))
    wood = material("wood", (0.45, 0.28, 0.14, 1.0))
    leaf = material("leaf", (0.12, 0.38, 0.18, 1.0))
    add_cube("lawn", (0, 0, -0.03), (4.8, 4.8, 0.03), grass)
    for i in range(-5, 6):
        add_cube(f"path_tile_{i}", (0.38 * i, -0.15 * math.sin(i), 0.01), (0.17, 0.42, 0.025), path_mat)
    for i, (x, y) in enumerate([(-1.9, -1.5), (1.8, -1.25), (-2.1, 1.45), (1.55, 1.55)]):
        add_cylinder(f"tree_{i}_trunk", 18, 0.08, 0.65, (x, y, 0.32), wood)
        add_cone(f"tree_{i}_canopy_low", 0.45, 0.12, 0.65, (x, y, 0.85), leaf)
        add_cone(f"tree_{i}_canopy_high", 0.34, 0.05, 0.55, (x, y, 1.18), leaf)
    export_selected(path)


def create_object_a(path: Path) -> None:
    clear_scene()
    ceramic = material("object_A_ceramic_red", (0.78, 0.16, 0.12, 1.0))
    rim = material("object_A_rim", (0.95, 0.92, 0.84, 1.0))
    cup = add_cylinder("reconstructed_mug_body", 80, 0.45, 0.9, (0, 0, 0.45), ceramic)
    cup.scale = (1.0, 1.0, 1.0)
    add_cylinder("reconstructed_mug_rim", 80, 0.48, 0.05, (0, 0, 0.92), rim)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.28, minor_radius=0.055, major_segments=64, minor_segments=12)
    handle = bpy.context.object
    handle.name = "reconstructed_mug_handle"
    handle.location = (0.46, 0, 0.52)
    handle.rotation_euler = (math.radians(90), 0, 0)
    handle.scale = (0.78, 1.0, 1.25)
    assign(handle, ceramic)
    export_selected(path)


def create_object_b(path: Path) -> None:
    clear_scene()
    blue = material("text3d_robot_blue", (0.18, 0.48, 0.88, 1.0))
    dark = material("text3d_robot_dark", (0.08, 0.10, 0.12, 1.0))
    glow = material("text3d_robot_screen", (0.40, 0.95, 0.82, 1.0))
    add_cube("robot_body", (0, 0, 0.42), (0.38, 0.28, 0.42), blue)
    add_cube("robot_head", (0, 0, 0.95), (0.34, 0.30, 0.24), blue)
    add_cube("robot_screen", (0, -0.31, 0.96), (0.22, 0.02, 0.10), glow)
    add_cylinder("robot_left_eye", 24, 0.035, 0.025, (-0.10, -0.335, 1.01), dark).rotation_euler[0] = math.radians(90)
    add_cylinder("robot_right_eye", 24, 0.035, 0.025, (0.10, -0.335, 1.01), dark).rotation_euler[0] = math.radians(90)
    add_cylinder("robot_left_arm", 24, 0.055, 0.42, (-0.48, 0, 0.55), blue).rotation_euler[1] = math.radians(90)
    add_cylinder("robot_right_arm", 24, 0.055, 0.42, (0.48, 0, 0.55), blue).rotation_euler[1] = math.radians(90)
    add_cylinder("robot_left_wheel", 32, 0.13, 0.08, (-0.22, -0.01, 0.07), dark).rotation_euler[1] = math.radians(90)
    add_cylinder("robot_right_wheel", 32, 0.13, 0.08, (0.22, -0.01, 0.07), dark).rotation_euler[1] = math.radians(90)
    export_selected(path)


def create_object_c(path: Path) -> None:
    clear_scene()
    orange = material("image3d_vase_orange", (0.90, 0.42, 0.10, 1.0))
    cream = material("image3d_vase_highlight", (0.96, 0.82, 0.56, 1.0))
    add_uv_sphere("vase_lower_body", 0.5, (0, 0, 0.35), (0.85, 0.85, 0.62), orange)
    add_cylinder("vase_neck", 64, 0.22, 0.62, (0, 0, 0.86), orange)
    add_cylinder("vase_mouth", 64, 0.30, 0.06, (0, 0, 1.18), cream)
    for angle in [0, 120, 240]:
        x = 0.30 * math.cos(math.radians(angle))
        y = 0.30 * math.sin(math.radians(angle))
        add_uv_sphere(f"painted_dot_{angle}", 0.055, (x, y, 0.58), (1.0, 1.0, 0.35), cream)
    export_selected(path)


def write_demo_config(path: Path, root: Path) -> None:
    config = {
        "output_dir": "results/task1/fusion_demo",
        "resolution": [960, 540],
        "fps": 24,
        "frames": 72,
        "camera": {"center": [0.0, 0.0, 0.7], "radius": 5.2, "height": 2.1, "focal_length": 30},
        "assets": [
            {
                "name": "background_demo",
                "path": "results/task1/2dgs/background_garden/train/ours_latest/fuse_unbounded_post.ply",
                "location": [0, 0, 0],
                "rotation_degrees": [0, 0, 0],
                "scale": [1, 1, 1],
                "material_color": [0.25, 0.50, 0.28, 1],
            },
            {
                "name": "object_A_demo_2dgs",
                "path": "results/task1/2dgs/object_A/train/ours_latest/fuse_post.ply",
                "location": [-0.9, 0.2, 0.1],
                "rotation_degrees": [0, 0, -10],
                "scale": [0.6, 0.6, 0.6],
                "material_color": [0.78, 0.16, 0.12, 1],
            },
            {
                "name": "object_B_demo_text3d",
                "path": "results/task1/aigc/object_B_text3d/export/model.obj",
                "location": [0.25, -0.45, 0.08],
                "rotation_degrees": [0, 0, 25],
                "scale": [0.62, 0.62, 0.62],
                "material_color": [0.18, 0.48, 0.88, 1],
            },
            {
                "name": "object_C_demo_image3d",
                "path": "results/task1/aigc/object_C_image3d/model.obj",
                "location": [0.95, 0.25, 0.07],
                "rotation_degrees": [0, 0, -20],
                "scale": [0.65, 0.65, 0.65],
                "material_color": [0.90, 0.42, 0.10, 1],
            },
        ],
    }
    path = path if path.is_absolute() else root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = root_dir()
    paths = {
        "background": root / "results/task1/2dgs/background_garden/train/ours_latest/fuse_unbounded_post.ply",
        "object_A": root / "results/task1/2dgs/object_A/train/ours_latest/fuse_post.ply",
        "object_B": root / "results/task1/aigc/object_B_text3d/export/model.obj",
        "object_C": root / "results/task1/aigc/object_C_image3d/model.obj",
    }
    create_background(paths["background"])
    create_object_a(paths["object_A"])
    create_object_b(paths["object_B"])
    create_object_c(paths["object_C"])
    provenance = {
        "note": "Synthetic/demo assets generated because required phone-captured object_A/object_C inputs were absent.",
        "object_A": str(paths["object_A"].relative_to(root)),
        "object_B_prompt": "a small ceramic robot toy, high quality DSLR photo",
        "object_C_proxy": "single-image-to-3D proxy mesh for an orange vase-like object",
        "background_proxy": "garden-like mesh proxy for fusion/rendering smoke test",
    }
    provenance_path = root / "results/task1/demo_assets/provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    write_demo_config(args.config, root)
    print(f"Wrote synthetic demo assets and {args.config}")


if __name__ == "__main__":
    main()
