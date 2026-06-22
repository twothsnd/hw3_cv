#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
    parser.add_argument("--feature-num-threads", type=int, default=4, help="SIFT extraction thread limit.")
    parser.add_argument("--matcher-num-threads", type=int, default=4, help="SIFT matching thread limit.")
    parser.add_argument("--mapper-num-threads", type=int, default=4, help="Incremental mapper thread limit.")
    parser.add_argument("--exhaustive-block-size", type=int, default=25, help="Exhaustive matcher block size.")
    parser.add_argument("--sequential-overlap", type=int, default=10, help="Sequential matcher overlap.")
    parser.add_argument("--sift-max-num-features", type=int, default=8192, help="Maximum SIFT features per image.")
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

    sparse_stats = canonicalize_best_sparse_model(sparse_dir, args.dry_run)
    manifest = {
        "images": str(images_dir),
        "dataset": str(args.dataset),
        "database": str(database),
        "sparse": str(sparse_dir),
        "sparse_model": str(sparse_dir / "0"),
        "sparse_stats": sparse_stats,
        "camera_model": args.camera_model,
        "matcher": args.matcher,
        "single_camera": bool(args.single_camera),
        "backend": backend,
        "options": {
            "feature_num_threads": args.feature_num_threads,
            "matcher_num_threads": args.matcher_num_threads,
            "mapper_num_threads": args.mapper_num_threads,
            "exhaustive_block_size": args.exhaustive_block_size,
            "sequential_overlap": args.sequential_overlap,
            "sift_max_num_features": args.sift_max_num_features,
        },
        "timings": timings,
    }
    if not args.dry_run:
        write_json(args.dataset / "colmap_manifest.json", manifest)
    print(f"COLMAP dataset ready: {args.dataset}")
    if backend == "pycolmap" and not args.dry_run:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


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
            "--SiftExtraction.num_threads",
            str(args.feature_num_threads),
            "--SiftExtraction.max_num_features",
            str(args.sift_max_num_features),
        ],
        dry_run=args.dry_run,
    )

    matcher_cmd = "exhaustive_matcher" if args.matcher == "exhaustive" else "sequential_matcher"
    matcher_args = [
        colmap,
        matcher_cmd,
        "--database_path",
        str(database),
        "--SiftMatching.num_threads",
        str(args.matcher_num_threads),
    ]
    if args.matcher == "exhaustive":
        matcher_args.extend(["--ExhaustiveMatching.block_size", str(args.exhaustive_block_size)])
    else:
        matcher_args.extend(["--SequentialMatching.overlap", str(args.sequential_overlap)])
    timings["matcher_seconds"] = run_cmd(matcher_args, dry_run=args.dry_run)

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
            "--Mapper.num_threads",
            str(args.mapper_num_threads),
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
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.num_threads = args.feature_num_threads
    extraction_options.sift.max_num_features = args.sift_max_num_features
    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.num_threads = args.matcher_num_threads
    sequential_pairing_options = pycolmap.SequentialPairingOptions()
    sequential_pairing_options.overlap = args.sequential_overlap
    sequential_pairing_options.num_threads = args.matcher_num_threads
    exhaustive_pairing_options = pycolmap.ExhaustivePairingOptions()
    exhaustive_pairing_options.block_size = args.exhaustive_block_size
    mapping_options = pycolmap.IncrementalPipelineOptions()
    mapping_options.num_threads = args.mapper_num_threads
    mapping_options.mapper.num_threads = args.mapper_num_threads

    timings: dict[str, float] = {}
    start = time.perf_counter()
    pycolmap.extract_features(
        database_path=database,
        image_path=images_dir,
        camera_mode=camera_mode,
        reader_options=reader_options,
        extraction_options=extraction_options,
    )
    timings["feature_extractor_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    if args.matcher == "exhaustive":
        pycolmap.match_exhaustive(
            database,
            matching_options=matching_options,
            pairing_options=exhaustive_pairing_options,
        )
    else:
        pycolmap.match_sequential(
            database,
            matching_options=matching_options,
            pairing_options=sequential_pairing_options,
        )
    timings["matcher_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    reconstructions = pycolmap.incremental_mapping(
        database_path=database,
        image_path=images_dir,
        output_path=sparse_dir,
        options=mapping_options,
    )
    timings["mapper_seconds"] = time.perf_counter() - start
    if not reconstructions:
        raise SystemExit("pycolmap mapping did not produce any reconstruction.")
    return timings


def canonicalize_best_sparse_model(sparse_dir: Path, dry_run: bool) -> dict[str, int | str]:
    if dry_run:
        return {}
    model_dirs = sorted(path for path in sparse_dir.iterdir() if path.is_dir())
    if not model_dirs:
        raise SystemExit(f"No sparse reconstructions found in {sparse_dir}")
    try:
        import pycolmap
    except Exception:
        best = sparse_dir / "0"
        if not best.exists():
            raise SystemExit(f"pycolmap unavailable and {best} does not exist.")
        return {"selected_model": "0"}

    stats: list[tuple[int, int, Path]] = []
    for model_dir in model_dirs:
        try:
            reconstruction = pycolmap.Reconstruction(model_dir)
        except Exception:
            continue
        stats.append((reconstruction.num_reg_images(), reconstruction.num_points3D(), model_dir))
    if not stats:
        raise SystemExit(f"No readable sparse reconstructions found in {sparse_dir}")

    reg_images, points3d, best_model = max(stats, key=lambda item: (item[0], item[1]))
    canonical = sparse_dir / "0"
    if best_model != canonical:
        if canonical.exists():
            shutil.rmtree(canonical)
        shutil.copytree(best_model, canonical)
    return {
        "selected_model": best_model.name,
        "canonical_model": "0",
        "registered_images": reg_images,
        "points3D": points3d,
    }


if __name__ == "__main__":
    main()
