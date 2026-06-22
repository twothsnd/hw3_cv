#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose Task 1 meshes over a 2DGS-rendered garden backplate.")
    parser.add_argument("--config", required=True, type=Path)
    argv = []
    if "--" in __import__("sys").argv:
        raw = __import__("sys").argv
        argv = raw[raw.index("--") + 1 :]
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


def active_color_attribute_name(obj) -> str | None:
    color_attributes = getattr(obj.data, "color_attributes", None)
    if color_attributes and len(color_attributes) > 0:
        active = getattr(color_attributes, "active_color", None) or color_attributes[0]
        return active.name
    vertex_colors = getattr(obj.data, "vertex_colors", None)
    if vertex_colors and len(vertex_colors) > 0:
        return vertex_colors.active.name if vertex_colors.active else vertex_colors[0].name
    return None


def make_vertex_color_material(name: str, attribute_name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled is None:
        return mat
    attr = nodes.new(type="ShaderNodeAttribute")
    attr.attribute_name = attribute_name
    mat.node_tree.links.new(attr.outputs["Color"], principled.inputs["Base Color"])
    if "Roughness" in principled.inputs:
        principled.inputs["Roughness"].default_value = 0.72
    return mat


def make_plain_material(name: str, color: list[float]):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    principled = mat.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = color
        if "Roughness" in principled.inputs:
            principled.inputs["Roughness"].default_value = 0.72
    return mat


def make_image_material(name: str, image_path: Path):
    image = bpy.data.images.load(str(image_path))
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for node in list(nodes):
        nodes.remove(node)
    tex = nodes.new(type="ShaderNodeTexImage")
    tex.image = image
    emission = nodes.new(type="ShaderNodeEmission")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def make_shadow_material(name: str, alpha: float):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (0.0, 0.0, 0.0, alpha)
    mat.use_nodes = True
    principled = mat.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, alpha)
        if "Alpha" in principled.inputs:
            principled.inputs["Alpha"].default_value = alpha
        if "Roughness" in principled.inputs:
            principled.inputs["Roughness"].default_value = 1.0
    mat.blend_method = "BLEND"
    mat.use_screen_refraction = False
    return mat


def setup_render_engine(scene) -> None:
    for engine in ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"]:
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    if hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 32
        if hasattr(scene.eevee, "taa_samples"):
            scene.eevee.taa_samples = 32
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass


def setup_camera(config: dict):
    resolution = config.get("resolution", [1280, 720])
    aspect = float(resolution[0]) / float(resolution[1])
    scale = float(config.get("orthographic_scale", 4.0))
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = scale
    camera_data.lens = 70
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.location = Vector(config.get("camera_location", [0.0, -8.0, 0.0]))
    target = Vector(config.get("camera_target", [0.0, 0.0, 0.0]))
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return camera, scale * aspect, scale


def add_backplate(image_path: Path, plane_width: float, plane_height: float, y: float) -> None:
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, y, 0.0), rotation=(math.radians(90.0), 0.0, 0.0))
    plate = bpy.context.object
    plate.name = "garden_2dgs_render_backplate"
    plate.scale = (plane_width, plane_height, 1.0)
    plate.data.materials.append(make_image_material("garden_2dgs_backplate_mat", image_path))


def setup_lighting() -> None:
    sun = bpy.data.lights.new("Sun", type="SUN")
    sun.energy = 1.1
    sun_obj = bpy.data.objects.new("Sun", sun)
    bpy.context.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(42), 0.0, math.radians(30))

    area = bpy.data.lights.new("Object_Key_Area", type="AREA")
    area.energy = 350.0
    area.size = 4.0
    area_obj = bpy.data.objects.new("Object_Key_Area", area)
    bpy.context.collection.objects.link(area_obj)
    area_obj.location = (0.0, -4.2, 2.5)


def combined_bbox(objects) -> tuple[Vector, Vector]:
    bpy.context.view_layer.update()
    corners = []
    for obj in objects:
        if not hasattr(obj, "bound_box"):
            continue
        corners.extend([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
    if not corners:
        return Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0))
    mn = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    mx = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return mn, mx


def parent_to_empty(objects, name: str):
    empty = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(empty)
    for obj in objects:
        obj.parent = empty
        obj.matrix_parent_inverse = empty.matrix_world.inverted()
    return empty


def fit_and_place(empty, objects, asset: dict) -> None:
    base_rot = [math.radians(v) for v in asset.get("rotation_degrees", [0.0, 0.0, 0.0])]
    empty.rotation_euler = base_rot
    empty.scale = (1.0, 1.0, 1.0)
    empty.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()

    mn, mx = combined_bbox(objects)
    height = max(mx.z - mn.z, 1e-6)
    desired_height = float(asset.get("height", 1.0))
    factor = desired_height / height
    empty.scale = (factor, factor, factor)
    bpy.context.view_layer.update()

    mn, mx = combined_bbox(objects)
    center = (mn + mx) * 0.5
    target_x = float(asset.get("x", 0.0))
    target_y = float(asset.get("y", -0.25))
    baseline_z = float(asset.get("baseline_z", -0.35))
    empty.location += Vector((target_x - center.x, target_y - center.y, baseline_z - mn.z))
    bpy.context.view_layer.update()


def add_fake_shadow(asset: dict, material) -> None:
    shadow = asset.get("shadow", {})
    if not shadow:
        return
    x = float(asset.get("x", 0.0))
    y = float(asset.get("shadow_y", 0.15))
    z = float(asset.get("baseline_z", -0.35)) + float(shadow.get("z_offset", 0.02))
    bpy.ops.mesh.primitive_circle_add(
        vertices=96,
        radius=1.0,
        fill_type="TRIFAN",
        location=(x, y, z),
        rotation=(math.radians(90.0), 0.0, 0.0),
    )
    circle = bpy.context.object
    circle.name = asset["name"] + "_soft_contact_shadow"
    circle.scale = (float(shadow.get("width", 0.58)), float(shadow.get("height", 0.12)), 1.0)
    circle.data.materials.append(material)


def import_asset(root: Path, asset: dict, frames: int) -> None:
    mesh_path = Path(asset["path"])
    if not mesh_path.is_absolute():
        mesh_path = root / mesh_path
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    objects = import_mesh(mesh_path)
    fallback = make_plain_material(asset["name"] + "_mat", asset.get("material_color", [0.8, 0.8, 0.8, 1.0]))
    for obj in objects:
        obj.name = asset["name"] + "_" + obj.name
        color_attribute = active_color_attribute_name(obj) if asset.get("use_vertex_colors", False) else None
        if color_attribute:
            obj.data.materials.clear()
            obj.data.materials.append(make_vertex_color_material(obj.name + "_vertex_color_mat", color_attribute))
        elif not obj.data.materials:
            obj.data.materials.append(fallback)

    empty = parent_to_empty(objects, asset["name"] + "_placement")
    fit_and_place(empty, objects, asset)
    spin = float(asset.get("spin_degrees", 0.0))
    if frames > 1 and abs(spin) > 1e-6:
        mn, mx = combined_bbox(objects)
        pivot = (mn + mx) * 0.5
        turntable = bpy.data.objects.new(asset["name"] + "_turntable", None)
        bpy.context.collection.objects.link(turntable)
        turntable.location = (pivot.x, pivot.y, float(asset.get("baseline_z", pivot.z)))
        empty_world = empty.matrix_world.copy()
        empty.parent = turntable
        empty.matrix_world = empty_world
        turntable.rotation_euler = (0.0, 0.0, 0.0)
        turntable.keyframe_insert(data_path="rotation_euler", frame=1)
        turntable.rotation_euler = (0.0, 0.0, math.radians(spin))
        turntable.keyframe_insert(data_path="rotation_euler", frame=frames)
        if turntable.animation_data and turntable.animation_data.action:
            for fcurve in turntable.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = "LINEAR"


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    scene = bpy.context.scene
    setup_render_engine(scene)
    resolution = config.get("resolution", [1280, 720])
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.fps = int(config.get("fps", 24))
    frames = int(config.get("frames", 72))
    scene.frame_start = 1
    scene.frame_end = frames

    _, plane_width, plane_height = setup_camera(config)
    background = Path(config["background_image"])
    if not background.is_absolute():
        background = root / background
    add_backplate(background, plane_width, plane_height, float(config.get("backplate_y", 1.0)))
    setup_lighting()

    shadow_mat = make_shadow_material("soft_contact_shadow_mat", float(config.get("shadow_alpha", 0.23)))
    for asset in config["assets"]:
        add_fake_shadow(asset, shadow_mat)
        import_asset(root, asset, frames)

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
    if config.get("render_animation", True):
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
