#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create object-only frames for Object A using rembg masks.")
    parser.add_argument("--input", required=True, type=Path, help="Directory of extracted Object A frames.")
    parser.add_argument("--output", required=True, type=Path, help="Directory for RGB frames composited on a plain background.")
    parser.add_argument("--mask-output", type=Path, default=None, help="Optional directory for alpha masks.")
    parser.add_argument("--background", default="white", choices=["white", "black", "gray"])
    parser.add_argument("--alpha-threshold", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def background_rgba(name: str) -> tuple[int, int, int, int]:
    if name == "white":
        return (255, 255, 255, 255)
    if name == "black":
        return (0, 0, 0, 255)
    return (128, 128, 128, 255)


def bbox_from_alpha(alpha: np.ndarray, threshold: int) -> list[int] | None:
    ys, xs = np.where(alpha > threshold)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def main() -> None:
    args = parse_args()
    if not args.input.is_dir():
        raise SystemExit(f"Input frame directory not found: {args.input}")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is nonempty: {args.output}. Use --overwrite.")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mask_output is not None:
        args.mask_output.mkdir(parents=True, exist_ok=True)

    from rembg import new_session, remove

    session = new_session("u2net")
    frame_paths = sorted(path for path in args.input.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not frame_paths:
        raise SystemExit(f"No input frames found in {args.input}")

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "mask_output": str(args.mask_output) if args.mask_output else None,
        "background": args.background,
        "num_frames": 0,
        "frames": [],
    }
    bg_color = background_rgba(args.background)

    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        rgba = remove(image, session=session).convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"))
        bbox = bbox_from_alpha(alpha, args.alpha_threshold)
        foreground_pixels = int((alpha > args.alpha_threshold).sum())

        bg = Image.new("RGBA", rgba.size, bg_color)
        composited = Image.alpha_composite(bg, rgba).convert("RGB")
        out_path = args.output / frame_path.name
        composited.save(out_path, quality=95)

        mask_path = None
        if args.mask_output is not None:
            mask_path = args.mask_output / f"{frame_path.stem}.png"
            rgba.getchannel("A").save(mask_path)

        manifest["frames"].append(
            {
                "source": str(frame_path),
                "image": str(out_path),
                "mask": str(mask_path) if mask_path else None,
                "bbox_xyxy": bbox,
                "foreground_pixels": foreground_pixels,
                "foreground_ratio": foreground_pixels / float(alpha.size),
            }
        )

    manifest["num_frames"] = len(manifest["frames"])
    (args.output.parent / "object_A_masked_frames_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} with {manifest['num_frames']} masked frames")


if __name__ == "__main__":
    main()
