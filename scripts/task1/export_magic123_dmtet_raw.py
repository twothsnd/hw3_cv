#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
MAGIC123 = ROOT / "external" / "Magic123"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Magic123 DMTet checkpoint as a simple OBJ, bypassing xatlas/mesh cleanup."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", default="model")
    parser.add_argument("--tet-grid-size", default=256, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default=262144, type=int)
    return parser.parse_args()


def make_opt(device: torch.device, tet_grid_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        base_mesh=None,
        bg_radius=-1.0,
        blob_density=10.0,
        blob_radius=0.2,
        bound=1.0,
        cuda_ray=True,
        density_activation="exp",
        density_thresh=10.0,
        device=device,
        dmtet=True,
        grid_type="hashgrid",
        h=1024,
        min_near=0.1,
        taichi_ray=False,
        tet_grid_size=tet_grid_size,
        tet_mlp=False,
        w=1024,
    )


def write_obj(
    path: Path,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    colors: torch.Tensor,
    material_name: str,
) -> None:
    mean_color = colors.mean(dim=0).clamp(0, 1).tolist()
    mtl_path = path.with_suffix(".mtl")
    tex_name = mtl_path.name

    with mtl_path.open("w", encoding="utf-8") as f:
        f.write(f"newmtl {material_name}\n")
        f.write(f"Ka {mean_color[0]:.6f} {mean_color[1]:.6f} {mean_color[2]:.6f}\n")
        f.write(f"Kd {mean_color[0]:.6f} {mean_color[1]:.6f} {mean_color[2]:.6f}\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("illum 1\n")

    vertices_np = vertices.cpu().numpy()
    colors_np = colors.cpu().numpy()
    faces_np = faces.cpu().numpy()
    with path.open("w", encoding="utf-8") as f:
        f.write(f"mtllib {tex_name}\n")
        f.write(f"usemtl {material_name}\n")
        for vertex, color in zip(vertices_np, colors_np):
            f.write(
                "v "
                f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} "
                f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
            )
        for face in faces_np:
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(MAGIC123))
    os.chdir(MAGIC123)

    from nerf.network_grid import NeRFNetwork

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    opt = make_opt(device, args.tet_grid_size)
    model = NeRFNetwork(opt).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if "tet_scale" in ckpt:
        new_scale = torch.as_tensor(ckpt["tet_scale"], device=device)
        model.dmtet.verts *= new_scale / model.dmtet.tet_scale
        model.dmtet.tet_scale = new_scale
    model.eval()

    with torch.no_grad():
        vertices, faces = model.dmtet.get_verts_face()
        if vertices.numel() == 0 or faces.numel() == 0:
            raise RuntimeError("DMTet checkpoint exported an empty mesh.")

        color_batches = []
        for start in range(0, vertices.shape[0], args.batch_size):
            end = min(start + args.batch_size, vertices.shape[0])
            color_batches.append(model.density(vertices[start:end])["albedo"].detach().float().cpu())
        colors = torch.cat(color_batches, dim=0)

    obj_path = output_dir / f"{args.name}.obj"
    write_obj(obj_path, vertices.detach().cpu(), faces.detach().cpu().long(), colors, args.name)
    print(
        f"Wrote {obj_path} "
        f"vertices={vertices.shape[0]} faces={faces.shape[0]} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


if __name__ == "__main__":
    main()
