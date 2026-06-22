#!/usr/bin/env python3
"""Prepare a square RGBA input for Magic123 without requiring depth weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def remove_background(image: Image.Image) -> Image.Image:
    """Return an RGBA image with foreground alpha."""

    try:
        from rembg import remove

        rgba = remove(image.convert("RGB"))
        if isinstance(rgba, Image.Image):
            return rgba.convert("RGBA")
        return Image.open(rgba).convert("RGBA")
    except Exception as exc:
        print(f"[WARN] rembg failed, using border-color fallback: {exc}")

    rgb = np.asarray(image.convert("RGB")).astype(np.int16)
    h, w = rgb.shape[:2]
    border = np.concatenate(
        [
            rgb[: max(1, h // 20), :, :].reshape(-1, 3),
            rgb[-max(1, h // 20) :, :, :].reshape(-1, 3),
            rgb[:, : max(1, w // 20), :].reshape(-1, 3),
            rgb[:, -max(1, w // 20) :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(rgb - bg, axis=2)
    alpha = (dist > 32).astype(np.uint8) * 255
    alpha_img = Image.fromarray(alpha, mode="L").filter(ImageFilter.MedianFilter(5))
    rgba = image.convert("RGBA")
    rgba.putalpha(alpha_img)
    return rgba


def crop_and_pad(rgba: Image.Image, size: int, padding_ratio: float) -> Image.Image:
    alpha = np.asarray(rgba.getchannel("A"))
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0 or len(ys) == 0:
        cropped = rgba
    else:
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        pad = int(max(x1 - x0, y1 - y0) * padding_ratio)
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(rgba.width, x1 + pad)
        y1 = min(rgba.height, y1 + pad)
        cropped = rgba.crop((x0, y0, x1, y1))

    scale = min(size / cropped.width, size / cropped.height)
    new_size = (max(1, int(round(cropped.width * scale))), max(1, int(round(cropped.height * scale))))
    cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    offset = ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
    canvas.alpha_composite(cropped, offset)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--padding-ratio", type=float, default=0.12)
    args = parser.parse_args()

    image = Image.open(args.input).convert("RGB")
    rgba = remove_background(image)
    rgba = crop_and_pad(rgba, args.size, args.padding_ratio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(args.output)
    mask = np.asarray(rgba.getchannel("A")) > 16
    print(f"Prepared {args.output} size={rgba.size} foreground_pixels={int(mask.sum())}")


if __name__ == "__main__":
    main()
