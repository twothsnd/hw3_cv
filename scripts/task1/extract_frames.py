#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cv_hw3.common import ensure_dir, write_json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frames for COLMAP/2DGS from photos or a phone video.")
    parser.add_argument("--input", required=True, type=Path, help="Input image directory or video file.")
    parser.add_argument("--output", required=True, type=Path, help="Output image directory.")
    parser.add_argument("--fps", default=2.0, type=float, help="Frame extraction FPS for video input.")
    parser.add_argument("--max-frames", default=0, type=int, help="Maximum frames/images to keep. 0 keeps all.")
    parser.add_argument("--resize-long-edge", default=0, type=int, help="Resize so the long edge is at most this value.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output directory first.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ffmpeg_filter(args: argparse.Namespace) -> str:
    filters = [f"fps={args.fps:g}"]
    if args.resize_long_edge > 0:
        size = args.resize_long_edge
        filters.append(f"scale='if(gt(iw,ih),{size},-2)':'if(gt(ih,iw),{size},-2)'")
    return ",".join(filters)


def extract_video(args: argparse.Namespace) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return extract_video_opencv(args)
    ensure_dir(args.output)
    pattern = str(args.output / "frame_%05d.jpg")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(args.input),
        "-vf",
        ffmpeg_filter(args),
        "-q:v",
        "2",
        pattern,
    ]
    print("[run] " + " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)
    frames = sorted(args.output.glob("frame_*.jpg"))
    if args.max_frames and len(frames) > args.max_frames:
        keep = set(sample_paths(frames, args.max_frames))
        for path in frames:
            if path not in keep:
                path.unlink()
    return len(list(args.output.glob("frame_*.jpg")))


def extract_video_opencv(args: argparse.Namespace) -> int:
    try:
        import cv2
        from PIL import Image
    except Exception as exc:
        raise SystemExit("ffmpeg was not found and OpenCV/Pillow fallback is unavailable.") from exc

    ensure_dir(args.output)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.input}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / max(args.fps, 1e-6))))
    written = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            if args.resize_long_edge > 0:
                long_edge = max(image.size)
                if long_edge > args.resize_long_edge:
                    scale = args.resize_long_edge / long_edge
                    new_size = (round(image.size[0] * scale), round(image.size[1] * scale))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
            if not args.dry_run:
                written += 1
                image.save(args.output / f"frame_{written:05d}.jpg", quality=95)
            if args.max_frames and written >= args.max_frames:
                break
        frame_idx += 1
    cap.release()
    print(f"[opencv] sampled every {step} frames from {total} input frames")
    return written


def sample_paths(paths: list[Path], max_items: int) -> list[Path]:
    if max_items <= 0 or len(paths) <= max_items:
        return paths
    step = len(paths) / max_items
    selected: list[Path] = []
    for i in range(max_items):
        selected.append(paths[min(len(paths) - 1, math.floor(i * step))])
    return selected


def copy_images(args: argparse.Namespace) -> int:
    if not args.input.is_dir():
        raise SystemExit(f"Expected an image directory, got {args.input}")
    ensure_dir(args.output)
    sources = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    sources = sample_paths(sources, args.max_frames)
    if not sources:
        raise SystemExit(f"No images found in {args.input}")

    use_pil = args.resize_long_edge > 0
    image_mod = None
    if use_pil:
        try:
            from PIL import Image

            image_mod = Image
        except Exception:
            print("Pillow is not installed; images will be copied without resizing.")
            use_pil = False

    for index, src in enumerate(sources):
        dst = args.output / f"frame_{index:05d}{src.suffix.lower()}"
        if args.dry_run:
            print(f"[copy] {src} -> {dst}")
            continue
        if use_pil and image_mod is not None:
            image = image_mod.open(src).convert("RGB")
            long_edge = max(image.size)
            if long_edge > args.resize_long_edge:
                scale = args.resize_long_edge / long_edge
                new_size = (round(image.size[0] * scale), round(image.size[1] * scale))
                image = image.resize(new_size)
            image.save(dst, quality=95)
        else:
            shutil.copy2(src, dst)
    return len(sources)


def main() -> None:
    args = parse_args()
    if args.overwrite and args.output.exists() and not args.dry_run:
        shutil.rmtree(args.output)
    input_path = args.input
    suffix = input_path.suffix.lower()
    if input_path.is_dir():
        images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        videos = sorted(p for p in input_path.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if not images and len(videos) == 1:
            args.input = videos[0]
            input_path = args.input
            suffix = input_path.suffix.lower()
    if input_path.is_file() and suffix in VIDEO_EXTS:
        count = extract_video(args)
        mode = "video"
    else:
        count = copy_images(args)
        mode = "images"
    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "mode": mode,
        "num_frames": count,
        "fps": args.fps if mode == "video" else None,
        "resize_long_edge": args.resize_long_edge or None,
    }
    if not args.dry_run:
        write_json(args.output.parent / "frames_manifest.json", manifest)
    print(f"Prepared {count} frames in {args.output}")


if __name__ == "__main__":
    main()
