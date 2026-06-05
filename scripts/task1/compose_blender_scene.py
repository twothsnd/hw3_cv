#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose task-1 meshes into one Blender scene and render a video.")
    parser.add_argument("--config", required=True, type=Path)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def import_mesh(path: Path):
    ext = path.suffix.lower()
    if ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    elif ext == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.ply(filepath=str(path))
    elif ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise ValueError(f"Unsupported mesh format: {path}")
    return list(bpy.context.selected_objects)


def make_material(name: str, color: list[float]):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def apply_transform(obj, asset: dict) -> None:
    obj.location = asset.get("location", [0, 0, 0])
    obj.scale = asset.get("scale", [1, 1, 1])
    rot = [math.radians(v) for v in asset.get("rotation_degrees", [0, 0, 0])]
    obj.rotation_euler = rot


def setup_camera(config: dict) -> None:
    cam_cfg = config["camera"]
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera_data.lens = cam_cfg.get("focal_length", 28)

    center = Vector(cam_cfg.get("center", [0.0, 0.0, 0.0]))
    radius = float(cam_cfg.get("radius", 6.0))
    height = float(cam_cfg.get("height", 2.0))
    frames = int(config.get("frames", 144))
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frames

    for frame in range(1, frames + 1):
        theta = 2.0 * math.pi * (frame - 1) / frames
        camera.location = (center.x + radius * math.cos(theta), center.y + radius * math.sin(theta), height)
        direction = center - camera.location
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)


def setup_lighting() -> None:
    sun = bpy.data.lights.new("Sun", type="SUN")
    sun.energy = 2.0
    sun_obj = bpy.data.objects.new("Sun", sun)
    bpy.context.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(45), 0, math.radians(35))

    area = bpy.data.lights.new("Key_Area", type="AREA")
    area.energy = 500.0
    area.size = 5.0
    area_obj = bpy.data.objects.new("Key_Area", area)
    bpy.context.collection.objects.link(area_obj)
    area_obj.location = (0, -4, 5)


def setup_render_engine(scene) -> None:
    for engine in ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"]:
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    if hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 16
        if hasattr(scene.eevee, "taa_samples"):
            scene.eevee.taa_samples = 16


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for asset in config["assets"]:
        mesh_path = Path(asset["path"])
        if not mesh_path.is_absolute():
            mesh_path = root / mesh_path
        if not mesh_path.exists():
            raise FileNotFoundError(mesh_path)
        objects = import_mesh(mesh_path)
        material = make_material(asset["name"] + "_mat", asset.get("material_color", [0.8, 0.8, 0.8, 1.0]))
        for obj in objects:
            obj.name = asset["name"] + "_" + obj.name
            apply_transform(obj, asset)
            if not obj.data.materials:
                obj.data.materials.append(material)

    setup_lighting()
    setup_camera(config)

    resolution = config.get("resolution", [1280, 720])
    scene = bpy.context.scene
    setup_render_engine(scene)
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.fps = int(config.get("fps", 24))

    output_dir = Path(config.get("output_dir", "results/task1/fusion"))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scene.frame_set(1)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_dir / "fusion_preview.png")
    bpy.ops.render.render(write_still=True)

    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = str(output_dir / "fusion_walkthrough.mp4")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "fusion_scene.blend"))
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
