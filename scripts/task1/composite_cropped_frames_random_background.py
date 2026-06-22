#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Composite original object frames over deterministic random flat backgrounds."
    )
    parser.add_argument("--originals", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--crop-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--alpha-threshold", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def crop_with_padding(image: Image.Image, crop_xyxy: list[int], background: tuple[int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = crop_xyxy
    width, height = image.size
    crop_w = x1 - x0
    crop_h = y1 - y0
    canvas = Image.new(image.mode, (crop_w, crop_h), background)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(width, x1)
    src_y1 = min(height, y1)
    if src_x1 > src_x0 and src_y1 > src_y0:
        canvas.paste(image.crop((src_x0, src_y0, src_x1, src_y1)), (src_x0 - x0, src_y0 - y0))
    return canvas


def choose_background(rng: random.Random) -> tuple[int, int, int]:
    # Medium-bright colors avoid being mistaken for object whites while staying featureless for COLMAP.
    base = [rng.randint(70, 210) for _ in range(3)]
    if max(base) - min(base) < 45:
        base[rng.randrange(3)] = rng.choice([60, 220])
    return tuple(base)


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is nonempty: {args.output}. Use --overwrite.")
    crop_manifest = json.loads(args.crop_manifest.read_text(encoding="utf-8"))
    crop_xyxy = [int(v) for v in crop_manifest["crop_xyxy_before_resize"]]
    output_size = int(crop_manifest["size"])

    args.output.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []

    for frame in crop_manifest["frames"]:
        stem = Path(frame["cropped_image"]).stem
        original_path = args.originals / f"{stem}.jpg"
        if not original_path.exists():
            original_path = args.originals / f"{stem}.png"
        mask_path = args.masks / f"{stem}.png"
        if not original_path.exists():
            raise SystemExit(f"Missing original frame for {stem}: {args.originals}")
        if not mask_path.exists():
            raise SystemExit(f"Missing mask for {stem}: {mask_path}")

        original = Image.open(original_path).convert("RGB")
        alpha = Image.open(mask_path).convert("L")
        bg = choose_background(rng)
        rgb_crop = crop_with_padding(original, crop_xyxy, bg)
        alpha_crop = crop_with_padding(alpha, crop_xyxy, 0)
        if rgb_crop.size != (output_size, output_size):
            rgb_crop = rgb_crop.resize((output_size, output_size), Image.Resampling.LANCZOS)
            alpha_crop = alpha_crop.resize((output_size, output_size), Image.Resampling.LANCZOS)

        alpha_np = np.asarray(alpha_crop)
        alpha_crop = Image.fromarray(np.where(alpha_np > args.alpha_threshold, alpha_np, 0).astype(np.uint8), "L")
        bg_img = Image.new("RGB", (output_size, output_size), bg)
        composed = Image.composite(rgb_crop, bg_img, alpha_crop)
        out_path = args.output / f"{stem}.jpg"
        composed.save(out_path, quality=95)
        rows.append(
            {
                "source": str(original_path),
                "mask": str(mask_path),
                "image": str(out_path),
                "background_rgb": list(bg),
            }
        )

    manifest = {
        "originals": str(args.originals),
        "masks": str(args.masks),
        "crop_manifest": str(args.crop_manifest),
        "output": str(args.output),
        "seed": args.seed,
        "alpha_threshold": args.alpha_threshold,
        "num_frames": len(rows),
        "frames": rows,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} random-background frames to {args.output}")


if __name__ == "__main__":
    main()
