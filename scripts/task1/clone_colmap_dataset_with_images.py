#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone a COLMAP dataset while replacing the images directory."
    )
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--note", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    if not args.source_dataset.is_dir():
        raise SystemExit(f"Source dataset not found: {args.source_dataset}")
    if not args.images.is_dir():
        raise SystemExit(f"Replacement images directory not found: {args.images}")
    if args.output_dataset.exists() and any(args.output_dataset.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output dataset is nonempty: {args.output_dataset}. Use --overwrite.")

    if args.output_dataset.exists() and args.overwrite:
        shutil.rmtree(args.output_dataset)
    args.output_dataset.mkdir(parents=True, exist_ok=True)

    source_sparse = args.source_dataset / "sparse"
    if not source_sparse.is_dir():
        raise SystemExit(f"Source sparse directory not found: {source_sparse}")
    shutil.copytree(source_sparse, args.output_dataset / "sparse")

    source_database = args.source_dataset / "database.db"
    if source_database.exists():
        shutil.copy2(source_database, args.output_dataset / "database.db")

    images_link = args.output_dataset / "images"
    images_link.symlink_to(args.images.resolve(), target_is_directory=True)

    source_manifest_path = args.source_dataset / "colmap_manifest.json"
    source_manifest = {}
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    manifest = {
        "dataset": rel_or_abs(args.output_dataset),
        "images": rel_or_abs(images_link),
        "replacement_images_source": rel_or_abs(args.images),
        "source_dataset": rel_or_abs(args.source_dataset),
        "source_manifest": rel_or_abs(source_manifest_path) if source_manifest_path.exists() else None,
        "sparse": rel_or_abs(args.output_dataset / "sparse"),
        "sparse_model": rel_or_abs(args.output_dataset / "sparse" / "0"),
        "database": rel_or_abs(args.output_dataset / "database.db") if source_database.exists() else None,
        "note": args.note,
        "source_sparse_stats": source_manifest.get("sparse_stats", {}),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Cloned {args.source_dataset} -> {args.output_dataset}")
    print(f"Images now point to {args.images}")


if __name__ == "__main__":
    main()
