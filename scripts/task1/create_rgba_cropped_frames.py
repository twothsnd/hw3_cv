#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create globally cropped RGBA frames from RGB frames and foreground masks. "
            "The output filenames keep the input extension so existing COLMAP image names still match."
        )
    )
    parser.add_argument("--images", required=True, type=Path, help="Directory of original RGB frames.")
    parser.add_argument("--masks", required=True, type=Path, help="Directory of foreground masks.")
    parser.add_argument(
        "--crop-manifest",
        required=True,
        type=Path,
        help="Manifest written by crop_masked_object_frames.py.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Directory for cropped RGBA frames.")
    parser.add_argument("--manifest", required=True, type=Path, help="Path to write output manifest.")
    parser.add_argument("--alpha-threshold", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def crop_with_padding(image: Image.Image, crop_xyxy: list[int], fill: int | tuple[int, ...]) -> Image.Image:
    x0, y0, x1, y1 = crop_xyxy
    width, height = image.size
    crop_w = x1 - x0
    crop_h = y1 - y0
    canvas = Image.new(image.mode, (crop_w, crop_h), fill)

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(width, x1)
    src_y1 = min(height, y1)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return canvas

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    canvas.paste(image.crop((src_x0, src_y0, src_x1, src_y1)), (dst_x0, dst_y0))
    return canvas


def main() -> None:
    args = parse_args()
    if not args.images.is_dir():
        raise SystemExit(f"Image directory not found: {args.images}")
    if not args.masks.is_dir():
        raise SystemExit(f"Mask directory not found: {args.masks}")
    if not args.crop_manifest.is_file():
        raise SystemExit(f"Crop manifest not found: {args.crop_manifest}")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is nonempty: {args.output}. Use --overwrite.")

    crop_manifest = json.loads(args.crop_manifest.read_text(encoding="utf-8"))
    crop_xyxy = [int(v) for v in crop_manifest["crop_xyxy_before_resize"]]
    size = int(crop_manifest["size"])

    args.output.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    rows = []
    alpha_pixels = 0
    total_pixels = 0
    for image_path in image_paths:
        mask_path = args.masks / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise SystemExit(f"Missing mask for {image_path.name}: {mask_path}")

        rgb = Image.open(image_path).convert("RGB")
        alpha = Image.open(mask_path).convert("L")
        rgb_crop = crop_with_padding(rgb, crop_xyxy, (255, 255, 255))
        alpha_crop = crop_with_padding(alpha, crop_xyxy, 0)
        if rgb_crop.size != (size, size):
            rgb_crop = rgb_crop.resize((size, size), Image.Resampling.LANCZOS)
            alpha_crop = alpha_crop.resize((size, size), Image.Resampling.LANCZOS)

        rgba = rgb_crop.convert("RGBA")
        rgba.putalpha(alpha_crop)

        out_path = args.output / image_path.name
        rgba.save(out_path, format="PNG")
        foreground = int((np.asarray(alpha_crop) > args.alpha_threshold).sum())
        alpha_pixels += foreground
        total_pixels += alpha_crop.size[0] * alpha_crop.size[1]
        rows.append(
            {
                "source": str(image_path),
                "mask": str(mask_path),
                "rgba_image": str(out_path),
                "saved_format": "PNG",
                "filename_kept_for_colmap": image_path.name,
                "foreground_pixels": foreground,
            }
        )

    manifest = {
        "images": str(args.images),
        "masks": str(args.masks),
        "crop_manifest": str(args.crop_manifest),
        "output": str(args.output),
        "num_frames": len(rows),
        "size": size,
        "crop_xyxy_before_resize": crop_xyxy,
        "alpha_threshold": args.alpha_threshold,
        "foreground_ratio": alpha_pixels / float(total_pixels) if total_pixels else 0.0,
        "frames": rows,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} RGBA cropped frames to {args.output}")
    print(f"Mean foreground ratio: {manifest['foreground_ratio']:.4f}")


if __name__ == "__main__":
    main()
