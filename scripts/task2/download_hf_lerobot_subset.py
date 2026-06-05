#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cv_hw3.common import ensure_dir, write_json


def parse_episodes(values: list[str]) -> list[int]:
    episodes: set[int] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                start, end = part.split(":", 1)
                episodes.update(range(int(start), int(end)))
            else:
                episodes.add(int(part))
    return sorted(episodes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a local subset of an HF LeRobot dataset.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--episodes", nargs="+", required=True, help="Episode ids or half-open ranges, e.g. 0:100 250.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--include-stats", action="store_true", help="Download full meta/episodes_stats.jsonl if present.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise SystemExit("huggingface_hub is required.") from exc

    episodes = parse_episodes(args.episodes)
    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root)

    downloaded: list[str] = []
    for meta_file in ["meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"]:
        path = hf_hub_download(
            args.repo_id,
            meta_file,
            repo_type="dataset",
            revision=args.revision,
            local_dir=args.output_root,
        )
        downloaded.append(path)
    if args.include_stats:
        try:
            path = hf_hub_download(
                args.repo_id,
                "meta/episodes_stats.jsonl",
                repo_type="dataset",
                revision=args.revision,
                local_dir=args.output_root,
            )
            downloaded.append(path)
        except Exception as exc:
            print(f"Warning: could not download episodes_stats.jsonl: {exc}")

    info = json.loads((args.output_root / "meta/info.json").read_text(encoding="utf-8"))
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    for episode in episodes:
        rel = data_path.format(episode_chunk=episode // chunks_size, episode_index=episode)
        path = hf_hub_download(
            args.repo_id,
            rel,
            repo_type="dataset",
            revision=args.revision,
            local_dir=args.output_root,
        )
        downloaded.append(path)

    write_json(
        args.output_root / "subset_manifest.json",
        {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "episodes": episodes,
            "num_episodes": len(episodes),
            "downloaded": downloaded,
        },
    )
    print(f"Downloaded {len(episodes)} episodes to {args.output_root}")


if __name__ == "__main__":
    main()
