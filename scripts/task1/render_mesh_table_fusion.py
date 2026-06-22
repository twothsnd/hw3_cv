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
    parser = argparse.ArgumentParser(description="Render the mesh-only Task 1 garden table fusion preview.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preview-only", action="store_true")
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


def color_attribute_name(obj) -> str | None:
    color_attributes = getattr(obj.data, "color_attributes", None)
    if color_attributes and len(color_attributes) > 0:
        active = getattr(color_attributes, "active_color", None) or color_attributes[0]
        return active.name
    vertex_colors = getattr(obj.data, "vertex_colors", None)
    if vertex_colors and len(vertex_colors) > 0:
        return vertex_colors.active.name if vertex_colors.active else vertex_colors[0].name
    return None


def vertex_color_material(name: str, attribute_name: str, alpha: float = 1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = (1.0, 1.0, 1.0, alpha)
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled is None:
        return mat
    attr = nodes.new(type="ShaderNodeAttribute")
    attr.attribute_name = attribute_name
    mat.node_tree.links.new(attr.outputs["Color"], principled.inputs["Base Color"])
    if "Alpha" in principled.inputs:
        principled.inputs["Alpha"].default_value = alpha
    if "Roughness" in principled.inputs:
        principled.inputs["Roughness"].default_value = 0.82
    if alpha < 0.999:
        mat.blend_method = "BLEND"
        mat.use_screen_refraction = False
        mat.show_transparent_back = True
    return mat


def plain_material(name: str, color: list[float], alpha: float | None = None):
    rgba = list(color)
    if alpha is not None:
        rgba = rgba[:3] + [alpha]
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = rgba
    mat.use_nodes = True
    principled = mat.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = rgba
        if "Alpha" in principled.inputs:
            principled.inputs["Alpha"].default_value = rgba[3]
        if "Roughness" in principled.inputs:
            principled.inputs["Roughness"].default_value = 0.8
    if rgba[3] < 0.999:
        mat.blend_method = "BLEND"
        mat.use_screen_refraction = False
    return mat


def setup_render(scene, config: dict) -> None:
    for engine in ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"]:
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    if hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 48
        if hasattr(scene.eevee, "taa_samples"):
            scene.eevee.taa_samples = 48
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = True
    resolution = config.get("resolution", [1280, 720])
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.fps = int(config.get("fps", 24))
    scene.frame_start = 1
    scene.frame_end = int(config.get("frames", 1))
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = tuple(config.get("world_color", [0.96, 0.96, 0.93]))
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass


def look_at(camera, target: Vector) -> None:
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera(config: dict):
    cam_cfg = config["camera"]
    data = bpy.data.cameras.new("Camera")
    data.type = cam_cfg.get("type", "ORTHO")
    if data.type == "ORTHO":
        data.ortho_scale = float(cam_cfg.get("orthographic_scale", 3.5))
    else:
        data.lens = float(cam_cfg.get("focal_length", 35.0))
    camera = bpy.data.objects.new("Camera", data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    target = Vector(cam_cfg.get("target", [0.0, 0.0, 0.0]))
    frames = int(config.get("frames", 1))
    orbit = float(cam_cfg.get("orbit_degrees", 0.0))
    if frames > 1 and abs(orbit) > 1e-6:
        base = Vector(cam_cfg.get("location", [0.0, -3.5, 2.8]))
        offset = base - target
        radius = math.hypot(offset.x, offset.y)
        start = math.atan2(offset.y, offset.x)
        z = base.z
        for frame in range(1, frames + 1):
            theta = start + math.radians(orbit) * (frame - 1) / max(frames - 1, 1)
            camera.location = (target.x + radius * math.cos(theta), target.y + radius * math.sin(theta), z)
            look_at(camera, target)
            camera.keyframe_insert(data_path="location", frame=frame)
            camera.keyframe_insert(data_path="rotation_euler", frame=frame)
        if camera.animation_data and camera.animation_data.action:
            for fcurve in camera.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = "LINEAR"
    else:
        camera.location = Vector(cam_cfg.get("location", [0.0, -3.5, 2.8]))
        look_at(camera, target)
    return camera


def setup_lighting(config: dict) -> None:
    sun_cfg = config.get("sun", {})
    sun = bpy.data.lights.new("Sun", type="SUN")
    sun.energy = float(sun_cfg.get("energy", 1.2))
    sun_obj = bpy.data.objects.new("Sun", sun)
    bpy.context.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = tuple(math.radians(v) for v in sun_cfg.get("rotation_degrees", [48.0, 0.0, 35.0]))

    area_cfg = config.get("area_light", {})
    area = bpy.data.lights.new("Key_Area", type="AREA")
    area.energy = float(area_cfg.get("energy", 450.0))
    area.size = float(area_cfg.get("size", 5.0))
    area_obj = bpy.data.objects.new("Key_Area", area)
    bpy.context.collection.objects.link(area_obj)
    area_obj.location = tuple(area_cfg.get("location", [0.0, -3.0, 4.2]))


def combined_bbox(objects) -> tuple[Vector, Vector]:
    bpy.context.view_layer.update()
    corners: list[Vector] = []
    for obj in objects:
        if hasattr(obj, "bound_box"):
            corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not corners:
        zero = Vector((0.0, 0.0, 0.0))
        return zero, zero
    return (
        Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners))),
        Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners))),
    )


def parent_to_empty(objects, name: str):
    empty = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(empty)
    for obj in objects:
        obj.parent = empty
        obj.matrix_parent_inverse = empty.matrix_world.inverted()
    return empty


def import_background(root: Path, config: dict) -> None:
    bg = config["background"]
    path = Path(bg["path"])
    if not path.is_absolute():
        path = root / path
    objects = import_mesh(path)
    alpha = float(bg.get("alpha", 1.0))
    fallback = plain_material("garden_mesh_fallback", bg.get("material_color", [0.70, 0.70, 0.64, 1.0]), alpha)
    for obj in objects:
        obj.name = "garden_table_mesh_" + obj.name
        attr = color_attribute_name(obj) if bg.get("use_vertex_colors", True) else None
        obj.data.materials.clear()
        obj.data.materials.append(vertex_color_material(obj.name + "_vertex_colors", attr, alpha) if attr else fallback)
        obj.hide_select = True


def import_asset(root: Path, asset: dict, frames: int) -> None:
    path = Path(asset["path"])
    if not path.is_absolute():
        path = root / path
    objects = import_mesh(path)
    fallback = plain_material(asset["name"] + "_fallback", asset.get("material_color", [0.8, 0.8, 0.8, 1.0]))
    for obj in objects:
        obj.name = asset["name"] + "_" + obj.name
        attr = color_attribute_name(obj) if asset.get("use_vertex_colors", False) else None
        if attr:
            obj.data.materials.clear()
            obj.data.materials.append(vertex_color_material(obj.name + "_vertex_colors", attr, 1.0))
        elif not obj.data.materials:
            obj.data.materials.append(fallback)

    empty = parent_to_empty(objects, asset["name"] + "_placement")
    empty.rotation_euler = tuple(math.radians(v) for v in asset.get("rotation_degrees", [0.0, 0.0, 0.0]))
    empty.location = (0.0, 0.0, 0.0)
    empty.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()

    mn, mx = combined_bbox(objects)
    height = max(mx.z - mn.z, 1e-6)
    target_height = float(asset.get("height", 0.5))
    factor = target_height / height
    empty.scale = (factor, factor, factor)
    bpy.context.view_layer.update()

    mn, mx = combined_bbox(objects)
    center = (mn + mx) * 0.5
    target = Vector(asset.get("location", [0.0, 0.0, 0.04]))
    empty.location += Vector((target.x - center.x, target.y - center.y, target.z - mn.z))
    bpy.context.view_layer.update()

    spin = float(asset.get("spin_degrees", 0.0))
    if frames > 1 and abs(spin) > 1e-6:
        mn, mx = combined_bbox(objects)
        pivot = (mn + mx) * 0.5
        turntable = bpy.data.objects.new(asset["name"] + "_turntable", None)
        bpy.context.collection.objects.link(turntable)
        turntable.location = (pivot.x, pivot.y, target.z)
        matrix = empty.matrix_world.copy()
        empty.parent = turntable
        empty.matrix_world = matrix
        turntable.rotation_euler = (0.0, 0.0, 0.0)
        turntable.keyframe_insert(data_path="rotation_euler", frame=1)
        turntable.rotation_euler = (0.0, 0.0, math.radians(spin))
        turntable.keyframe_insert(data_path="rotation_euler", frame=frames)
        if turntable.animation_data and turntable.animation_data.action:
            for fcurve in turntable.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = "LINEAR"


def add_shadow(shadow: dict) -> None:
    mat = plain_material("soft_table_shadow", [0.0, 0.0, 0.0, float(shadow.get("alpha", 0.22))])
    bpy.ops.mesh.primitive_circle_add(
        vertices=96,
        radius=1.0,
        fill_type="TRIFAN",
        location=tuple(shadow.get("location", [0.0, 0.0, 0.025])),
        rotation=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.object
    obj.name = shadow.get("name", "soft_table_shadow")
    obj.scale = tuple(shadow.get("scale", [0.5, 0.22, 1.0]))
    obj.data.materials.append(mat)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    scene = bpy.context.scene
    setup_render(scene, config)
    setup_camera(config)
    setup_lighting(config)
    import_background(root, config)
    for shadow in config.get("shadows", []):
        add_shadow(shadow)
    frames = int(config.get("frames", 1))
    for asset in config["assets"]:
        import_asset(root, asset, frames)

    output_dir = Path(config.get("output_dir", "results/task1/fusion_mesh"))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scene.frame_set(1)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_dir / "fusion_mesh_preview.png")
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "fusion_mesh_scene.blend"))
    if args.preview_only or not config.get("render_animation", False):
        return

    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = str(output_dir / "fusion_mesh_walkthrough.mp4")
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
