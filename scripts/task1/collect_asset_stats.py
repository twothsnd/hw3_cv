#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cv_hw3.common import parse_key_value_items, write_json
from cv_hw3.mesh_io import mesh_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect mesh/file statistics for task-1 assets.")
    parser.add_argument("--assets", nargs="+", required=True, help="KEY=mesh_path entries.")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets = parse_key_value_items(args.assets)
    stats = {name: mesh_stats(path) for name, path in assets.items()}
    write_json(args.output, stats)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
