#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT_REPO = "xiaoma26/calvin-lerobot"


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
    parser = argparse.ArgumentParser(description="Download selected episodes from xiaoma26/calvin-lerobot splits.")
    parser.add_argument("--split", required=True, choices=["splitA", "splitB", "splitC", "splitD"])
    parser.add_argument("--output-root", required=True, type=Path, help="Local root containing splitA/splitB/... folders.")
    parser.add_argument("--episodes", nargs="+", required=True, help="Episode ids or half-open ranges, e.g. 0:100 250.")
    parser.add_argument("--repo-id", default=ROOT_REPO)
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise SystemExit("huggingface_hub is required.") from exc

    split_root = args.output_root / args.split
    split_root.mkdir(parents=True, exist_ok=True)
    episodes = parse_episodes(args.episodes)

    downloaded: list[str] = []
    meta_names = ["info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl", "modality.json", "conversion.json"]
    for name in meta_names:
        rel = f"{args.split}/meta/{name}"
        path = hf_hub_download(
            args.repo_id,
            rel,
            repo_type="dataset",
            revision=args.revision,
            local_dir=args.output_root,
        )
        downloaded.append(path)

    info = json.loads((split_root / "meta/info.json").read_text(encoding="utf-8"))
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    for episode in episodes:
        rel_in_split = data_path.format(episode_chunk=episode // chunks_size, episode_index=episode)
        rel = f"{args.split}/{rel_in_split}"
        path = hf_hub_download(
            args.repo_id,
            rel,
            repo_type="dataset",
            revision=args.revision,
            local_dir=args.output_root,
        )
        downloaded.append(path)
        print(f"downloaded {rel}")

    manifest = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "split": args.split,
        "episodes": episodes,
        "num_episodes": len(episodes),
        "downloaded": downloaded,
    }
    (split_root / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloaded {len(episodes)} {args.split} episodes to {split_root}")


if __name__ == "__main__":
    main()
