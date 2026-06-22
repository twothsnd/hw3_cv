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
        description="Crop masked object frames with one global square crop derived from alpha masks."
    )
    parser.add_argument("--images", required=True, type=Path, help="Directory of RGB masked frames.")
    parser.add_argument("--masks", required=True, type=Path, help="Directory of alpha masks with matching stems.")
    parser.add_argument("--output", required=True, type=Path, help="Directory for cropped RGB frames.")
    parser.add_argument("--manifest", required=True, type=Path, help="Path to write crop manifest JSON.")
    parser.add_argument("--threshold", type=int, default=20, help="Alpha threshold for foreground bbox.")
    parser.add_argument("--margin", type=float, default=0.18, help="Fractional margin around the global bbox.")
    parser.add_argument("--size", type=int, default=1024, help="Output square image size.")
    parser.add_argument("--background", default="white", choices=["white", "gray", "black"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def bg_rgb(name: str) -> tuple[int, int, int]:
    if name == "black":
        return (0, 0, 0)
    if name == "gray":
        return (128, 128, 128)
    return (255, 255, 255)


def bbox_from_mask(mask_path: Path, threshold: int) -> list[int] | None:
    alpha = np.asarray(Image.open(mask_path).convert("L"))
    ys, xs = np.where(alpha > threshold)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def crop_with_padding(image: Image.Image, crop_xyxy: list[int], background: tuple[int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = crop_xyxy
    width, height = image.size
    crop_w = x1 - x0
    crop_h = y1 - y0
    canvas = Image.new("RGB", (crop_w, crop_h), background)

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
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is nonempty: {args.output}. Use --overwrite.")
    args.output.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    frame_rows: list[dict[str, object]] = []
    boxes = []
    for image_path in image_paths:
        mask_path = args.masks / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise SystemExit(f"Missing mask for {image_path.name}: {mask_path}")
        bbox = bbox_from_mask(mask_path, args.threshold)
        if bbox is None:
            continue
        boxes.append(bbox)
        frame_rows.append({"source": str(image_path), "mask": str(mask_path), "bbox_xyxy": bbox})

    if not boxes:
        raise SystemExit("No foreground masks found.")

    boxes_np = np.asarray(boxes)
    gx0, gy0 = boxes_np[:, 0].min(), boxes_np[:, 1].min()
    gx1, gy1 = boxes_np[:, 2].max(), boxes_np[:, 3].max()
    union_w = int(gx1 - gx0)
    union_h = int(gy1 - gy0)
    side = int(np.ceil(max(union_w, union_h) * (1.0 + args.margin)))
    if side % 2:
        side += 1

    cx = int(round((gx0 + gx1) / 2.0))
    cy = int(round((gy0 + gy1) / 2.0))
    crop_xyxy = [cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2]
    background = bg_rgb(args.background)

    written_rows = []
    for row in frame_rows:
        image_path = Path(str(row["source"]))
        image = Image.open(image_path).convert("RGB")
        cropped = crop_with_padding(image, crop_xyxy, background)
        if cropped.size != (args.size, args.size):
            cropped = cropped.resize((args.size, args.size), Image.Resampling.LANCZOS)
        out_path = args.output / image_path.name
        cropped.save(out_path, quality=95)
        written_rows.append({**row, "cropped_image": str(out_path)})

    manifest = {
        "images": str(args.images),
        "masks": str(args.masks),
        "output": str(args.output),
        "threshold": args.threshold,
        "margin": args.margin,
        "background": args.background,
        "size": args.size,
        "num_frames": len(written_rows),
        "global_bbox_xyxy": [int(gx0), int(gy0), int(gx1), int(gy1)],
        "crop_xyxy_before_resize": [int(v) for v in crop_xyxy],
        "scale_to_output": args.size / float(side),
        "frames": written_rows,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(written_rows)} cropped frames to {args.output}")
    print(f"Global bbox {manifest['global_bbox_xyxy']} -> crop {manifest['crop_xyxy_before_resize']}")


if __name__ == "__main__":
    main()
