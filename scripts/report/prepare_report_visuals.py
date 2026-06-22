#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "figures"


def load(path: str | Path) -> Image.Image:
    im = Image.open(ROOT / path).convert("RGB")
    return im


def fit(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = im.size
    tw, th = size
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


def crop_left_panel(im: Image.Image) -> Image.Image:
    """threestudio preview files concatenate RGB/normal/mask; keep RGB only."""
    w, h = im.size
    return im.crop((0, 0, w // 3, h))


def trim_white(im: Image.Image, threshold: int = 248, pad: int = 18) -> Image.Image:
    gray = im.convert("L")
    mask = gray.point(lambda px: 255 if px < threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return im
    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(im.width, right + pad)
    bottom = min(im.height, bottom + pad)
    return im.crop((left, top, right, bottom))


def cover(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = im.size
    tw, th = size
    scale = max(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left = max(0, (nw - tw) // 2)
    top = max(0, (nh - th) // 2)
    return resized.crop((left, top, left + tw, top + th))


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.rectangle((xy[0], xy[1], xy[0] + 12 + len(text) * 8, xy[1] + 24), fill=(255, 255, 255))
    draw.text((xy[0] + 6, xy[1] + 5), text, fill=(0, 0, 0))


def grid(items: list[tuple[str, Image.Image]], cols: int, cell: tuple[int, int], title: str, output: str) -> None:
    rows = (len(items) + cols - 1) // cols
    title_h = 46
    gap = 12
    w = cols * cell[0] + (cols + 1) * gap
    h = title_h + rows * cell[1] + (rows + 1) * gap
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 14), title, fill=(0, 0, 0))
    for idx, (name, im) in enumerate(items):
        r, c = divmod(idx, cols)
        x = gap + c * (cell[0] + gap)
        y = title_h + gap + r * (cell[1] + gap)
        tile = fit(im, cell)
        canvas.paste(tile, (x, y))
        label(draw, (x + 8, y + 8), name)
    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT / output)


def compact_four_view(im: Image.Image, output: str, title: str, cell: tuple[int, int] = (520, 360)) -> None:
    w, h = im.size
    labels = ["front-oblique", "back-oblique", "side", "top"]
    boxes = [
        (0, 0, w // 2, h // 2),
        (w // 2, 0, w, h // 2),
        (0, h // 2, w // 2, h),
        (w // 2, h // 2, w, h),
    ]
    items = []
    for name, box in zip(labels, boxes):
        crop = im.crop(box)
        # Source preview already has large per-view text headers. Remove them
        # before content trimming so the object, not the title, drives the crop.
        crop = crop.crop((0, min(90, crop.height // 4), crop.width, crop.height))
        crop = trim_white(crop, threshold=252, pad=24)
        items.append((name, crop))
    grid(items, cols=2, cell=cell, title=title, output=output)


def save_fit(path: str | Path, output: str, size: tuple[int, int], trim: bool = False) -> None:
    im = load(path)
    if trim:
        im = trim_white(im)
    OUT.mkdir(parents=True, exist_ok=True)
    fit(im, size).save(OUT / output)


def refresh_fusion_video_frames() -> None:
    ffmpeg = ROOT / "tools" / "bin" / "ffmpeg"
    video = ROOT / "results/task1/fusion_mesh/fusion_mesh_walkthrough.mp4"
    out_dir = ROOT / "results/task1/previews/fusion_walkthrough_current_frames"
    if not ffmpeg.exists() or not video.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("frame_*.png"):
        path.unlink()
    subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            "select='eq(n,0)+eq(n,23)+eq(n,47)+eq(n,71)+eq(n,95)',setpts=N/FRAME_RATE/TB",
            "-vsync",
            "0",
            str(out_dir / "frame_%03d.png"),
        ],
        check=True,
    )


def make_task2_dataset_samples() -> None:
    python = ROOT / ".venvs" / "lerobot" / "bin" / "python"
    script = ROOT / "scripts" / "report" / "make_task2_official_figures.py"
    if not python.exists() or not script.exists():
        return
    subprocess.run([str(python), str(script)], check=True)


def render_colored_surface_mesh(mesh: str, output_name: str, views: str, cols: int = 2) -> None:
    python = ROOT / ".venvs" / "2dgs" / "bin" / "python"
    script = ROOT / "scripts/task1/render_mesh_preview.py"
    mesh_path = ROOT / mesh
    output = OUT / output_name
    if not python.exists() or not script.exists() or not mesh_path.exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(python),
            str(script),
            "--mesh",
            str(mesh_path),
            "--output",
            str(output),
            "--faces",
            "100000",
            "--use-vertex-colors",
            f"--views={views}",
            "--cols",
            str(cols),
            "--cell-width",
            "2.5",
            "--cell-height",
            "2.0",
            "--dpi",
            "140",
        ],
        check=True,
    )


def main() -> None:
    refresh_fusion_video_frames()
    make_task2_dataset_samples()

    save_fit("results/task1/object_A_frame_contact_sheet.jpg", "task1_object_A_raw_frames.png", (760, 900), trim=True)
    save_fit("results/task1/object_A_masked_cropped_contact_sheet.jpg", "task1_object_A_masked_cropped.png", (760, 900), trim=True)
    save_fit("results/task1/object_A_2dgs_gt_render_contact_sheet.jpg", "task1_object_A_2dgs_render_check.png", (1280, 720), trim=True)
    render_colored_surface_mesh(
        "results/task1/2dgs/object_A_rgba_cropped_alpha_clean/train/ours_latest/fuse_post_2dgs_tsdf_main_component.ply",
        "task1_object_A_final_surface_mesh.png",
        "0,-90,-90;-65,28,0;25,24,0;110,26,0",
        cols=4,
    )

    # Object B: training progression from the actual toy-car four-wheel runs.
    runs = {
        "8k": "toycar_fourwheels_sd_fp16_8000@20260621-121800/save/it8000-test",
        "12k": "toycar_fourwheels_sd_fp16_12000_resume@20260621-124502/save/it12000-test",
        "16k": "toycar_fourwheels_sd_fp16_16000_resume@20260621-130252/save/it16000-test",
        "20k": "toycar_fourwheels_sd_fp16_20000_resume@20260621-131749/save/it20000-test",
    }
    view_ids = [0, 40, 80]
    b_items = []
    for step, rel in runs.items():
        for view_id in view_ids:
            b_items.append(
                (
                    f"{step} view {view_id}",
                    crop_left_panel(load(f"results/task1/aigc/object_B_text3d/{rel}/{view_id}.png")),
                )
            )
    grid(
        b_items,
        cols=4,
        cell=(260, 220),
        title="Object B SDS training progression: 4 checkpoints x 3 views",
        output="task1_object_B_training_progress.png",
    )

    grid(
        [
            ("20k textured mesh", trim_white(load("results/task1/object_B_fourwheels_20000_mesh_preview.png"))),
            ("20k solid geometry", trim_white(load("results/task1/object_B_solid_mesh_preview.png"))),
        ],
        cols=2,
        cell=(620, 420),
        title="Object B final text-to-3D result",
        output="task1_object_B_final.png",
    )

    save_fit("results/task1/aigc/object_C_image3d/data/rgba.png", "task1_object_C_rgba_input.png", (260, 280), trim=True)
    save_fit(
        "results/task1/aigc/object_C_image3d/magic123_object_C_image3d_coarse/results/images/magic123_object_C_image3d_coarse_ep0050_0000_lambertian.jpg",
        "task1_object_C_coarse_render.png",
        (260, 280),
        trim=True,
    )
    save_fit(
        "results/task1/aigc/object_C_image3d/magic123_object_C_image3d_dmtet/results/images/magic123_object_C_image3d_dmtet_ep0050_0000_lambertian.jpg",
        "task1_object_C_dmtet_render.png",
        (260, 280),
        trim=True,
    )
    render_colored_surface_mesh(
        "results/task1/aigc/object_C_image3d/model.obj",
        "task1_object_C_final_mesh.png",
        "0,90,90;-65,18,0;25,18,0;115,20,0",
        cols=4,
    )

    grid(
        [
            ("2DGS RGB renders", load("results/task1/previews/background_garden_full_2dgs_contact.png")),
            ("exported mesh render", load("results/task1/fusion_mesh/fusion_mesh_preview.png")),
        ],
        cols=2,
        cell=(620, 380),
        title="Garden background: 2DGS rendered views and final mesh-fusion scene",
        output="task1_background_pipeline.png",
    )

    frames = [
        ("frame 1", load("results/task1/previews/fusion_walkthrough_current_frames/frame_001.png")),
        ("frame 24", load("results/task1/previews/fusion_walkthrough_current_frames/frame_002.png")),
        ("frame 48", load("results/task1/previews/fusion_walkthrough_current_frames/frame_003.png")),
        ("frame 72", load("results/task1/previews/fusion_walkthrough_current_frames/frame_004.png")),
        ("frame 96", load("results/task1/previews/fusion_walkthrough_current_frames/frame_005.png")),
        ("preview", load("results/task1/fusion_mesh/fusion_mesh_preview.png")),
    ]
    grid(
        frames,
        cols=3,
        cell=(420, 240),
        title="Final fusion walkthrough frames",
        output="task1_fusion_walkthrough_grid.png",
    )


if __name__ == "__main__":
    main()
