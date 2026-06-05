#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cv_hw3.common import ensure_dir, run_cmd, which_or_raise, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COLMAP and create a dataset layout accepted by 2DGS.")
    parser.add_argument("--images", required=True, type=Path, help="Directory containing input images.")
    parser.add_argument("--dataset", required=True, type=Path, help="Output COLMAP dataset directory.")
    parser.add_argument("--camera-model", default="OPENCV", help="COLMAP camera model.")
    parser.add_argument("--matcher", default="exhaustive", choices=["exhaustive", "sequential"])
    parser.add_argument("--single-camera", default=1, type=int, choices=[0, 1])
    parser.add_argument("--colmap", default="colmap", help="COLMAP executable.")
    parser.add_argument("--backend", default="auto", choices=["auto", "colmap", "pycolmap"])
    parser.add_argument("--copy-images", action="store_true", help="Copy instead of symlinking images.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def prepare_images(src: Path, dst: Path, copy_images: bool, dry_run: bool) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        return
    if dry_run:
        print(f"[prepare] {src} -> {dst}")
        return
    if copy_images:
        shutil.copytree(src, dst)
    else:
        dst.symlink_to(src.resolve(), target_is_directory=True)


def main() -> None:
    args = parse_args()
    if not args.images.exists():
        raise SystemExit(f"Image directory not found: {args.images}")
    colmap = args.colmap if Path(args.colmap).exists() else shutil.which(args.colmap)
    backend = args.backend
    if backend == "auto":
        backend = "colmap" if colmap else "pycolmap"
    if backend == "colmap" and not colmap:
        colmap = which_or_raise(args.colmap)

    if args.overwrite and args.dataset.exists() and not args.dry_run:
        shutil.rmtree(args.dataset)
    ensure_dir(args.dataset)
    images_dir = args.dataset / "images"
    prepare_images(args.images, images_dir, args.copy_images, args.dry_run)

    database = args.dataset / "database.db"
    sparse_dir = args.dataset / "sparse"
    ensure_dir(sparse_dir)

    timings: dict[str, float] = {}
    if backend == "pycolmap":
        timings.update(run_pycolmap(args, database, images_dir, sparse_dir))
    else:
        timings.update(run_colmap_binary(args, colmap, database, images_dir, sparse_dir))

    manifest = {
        "images": str(images_dir),
        "dataset": str(args.dataset),
        "database": str(database),
        "sparse": str(sparse_dir),
        "camera_model": args.camera_model,
        "matcher": args.matcher,
        "single_camera": bool(args.single_camera),
        "backend": backend,
        "timings": timings,
    }
    if not args.dry_run:
        write_json(args.dataset / "colmap_manifest.json", manifest)
    print(f"COLMAP dataset ready: {args.dataset}")


def run_colmap_binary(
    args: argparse.Namespace,
    colmap: str,
    database: Path,
    images_dir: Path,
    sparse_dir: Path,
) -> dict[str, float]:
    timings: dict[str, float] = {}
    timings["feature_extractor_seconds"] = run_cmd(
        [
            colmap,
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--ImageReader.camera_model",
            args.camera_model,
            "--ImageReader.single_camera",
            str(args.single_camera),
        ],
        dry_run=args.dry_run,
    )

    matcher_cmd = "exhaustive_matcher" if args.matcher == "exhaustive" else "sequential_matcher"
    timings["matcher_seconds"] = run_cmd(
        [colmap, matcher_cmd, "--database_path", str(database)],
        dry_run=args.dry_run,
    )

    timings["mapper_seconds"] = run_cmd(
        [
            colmap,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse_dir),
        ],
        dry_run=args.dry_run,
    )
    return timings


def run_pycolmap(
    args: argparse.Namespace,
    database: Path,
    images_dir: Path,
    sparse_dir: Path,
) -> dict[str, float]:
    import time

    try:
        import pycolmap
    except Exception as exc:
        raise SystemExit("pycolmap is not installed. Install it or provide a COLMAP binary.") from exc

    if args.dry_run:
        print(f"[pycolmap] extract/match/map {images_dir} -> {sparse_dir}")
        return {"feature_extractor_seconds": 0.0, "matcher_seconds": 0.0, "mapper_seconds": 0.0}

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = args.camera_model
    camera_mode = pycolmap.CameraMode.SINGLE if args.single_camera else pycolmap.CameraMode.AUTO

    timings: dict[str, float] = {}
    start = time.perf_counter()
    pycolmap.extract_features(
        database_path=database,
        image_path=images_dir,
        camera_mode=camera_mode,
        reader_options=reader_options,
    )
    timings["feature_extractor_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    if args.matcher == "exhaustive":
        pycolmap.match_exhaustive(database)
    else:
        pycolmap.match_sequential(database)
    timings["matcher_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    reconstructions = pycolmap.incremental_mapping(
        database_path=database,
        image_path=images_dir,
        output_path=sparse_dir,
    )
    timings["mapper_seconds"] = time.perf_counter() - start
    if not reconstructions:
        raise SystemExit("pycolmap mapping did not produce any reconstruction.")
    return timings


if __name__ == "__main__":
    main()
