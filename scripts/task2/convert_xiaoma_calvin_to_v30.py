#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from PIL import Image

from cv_hw3.common import ensure_dir, write_json

FEATURE_MAP = {
    "image": "observation.images.top",
    "wrist_image": "observation.images.wrist",
    "state": "observation.state",
    "actions": "action",
}


def parse_episodes(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
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


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must be SPLIT=PATH, e.g. splitB=data/task2/.../splitB")
    split, path = value.split("=", 1)
    return split, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert xiaoma26/calvin-lerobot official splits to local LeRobot v3.")
    parser.add_argument("--source", action="append", required=True, type=parse_source, help="SPLIT=PATH; repeat for A+B+C.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episodes", nargs="+", default=None, help="Optional episode ids/ranges applied to every source.")
    parser.add_argument("--max-episodes-per-source", default=0, type=int)
    parser.add_argument("--frame-stride", default=1, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def episode_path(source_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    rel = data_path.format(episode_chunk=episode_index // chunks_size, episode_index=episode_index)
    return source_root / rel


def discover_episode_ids(source_root: Path, info: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for path in sorted((source_root / "data").rglob("episode_*.parquet")):
        try:
            episode_index = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        if episode_path(source_root, info, episode_index).exists():
            ids.append(episode_index)
    return ids


def feature_shape_chw(feature: dict[str, Any]) -> tuple[int, int, int]:
    shape = tuple(int(v) for v in feature["shape"])
    if len(shape) != 3:
        raise ValueError(f"Image feature has invalid shape: {shape}")
    names = [str(v).lower() for v in feature.get("names", [])]
    if names == ["height", "width", "channel"] or shape[-1] in {1, 3, 4}:
        h, w, c = shape
        return (3 if c == 4 else c, h, w)
    if names == ["channel", "height", "width"] or shape[0] in {1, 3, 4}:
        c, h, w = shape
        return (3 if c == 4 else c, h, w)
    raise ValueError(f"Could not infer image layout: {feature}")


def build_features(source_features: dict[str, Any], columns: list[str]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for source_key, target_key in FEATURE_MAP.items():
        if source_key not in source_features or source_key not in columns:
            raise ValueError(f"Missing required source feature {source_key}")
        feature = source_features[source_key]
        if feature["dtype"] == "image":
            features[target_key] = {
                "dtype": "image",
                "shape": feature_shape_chw(feature),
                "names": ["channel", "height", "width"],
            }
        else:
            features[target_key] = {
                "dtype": feature["dtype"],
                "shape": tuple(int(v) for v in feature["shape"]),
                "names": feature.get("names"),
            }
    return features


def decode_image(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))
        if value.get("path"):
            return np.asarray(Image.open(value["path"]).convert("RGB"))
    if isinstance(value, (bytes, bytearray)):
        return np.asarray(Image.open(io.BytesIO(value)).convert("RGB"))
    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"))
    arr = np.asarray(value)
    if arr.ndim != 3:
        raise ValueError(f"Expected image value, got shape {arr.shape}")
    return arr


def to_chw_uint8(value: Any, expected_shape: tuple[int, int, int]) -> np.ndarray:
    arr = decode_image(value)
    if arr.shape[0] not in {1, 3, 4} and arr.shape[-1] in {1, 3, 4}:
        arr = np.transpose(arr, (2, 0, 1))
    if arr.shape[0] == 4:
        arr = arr[:3]
    if arr.dtype != np.uint8:
        max_value = float(np.nanmax(arr)) if arr.size else 1.0
        if max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if tuple(arr.shape) != expected_shape:
        raise ValueError(f"Image shape {arr.shape} does not match expected {expected_shape}")
    return arr


def to_array(value: Any, dtype: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.dtype(dtype))
    if arr.shape != expected_shape:
        raise ValueError(f"Array shape {arr.shape} does not match expected {expected_shape}")
    return arr


def task_lookup(source_root: Path) -> tuple[dict[int, str], dict[int, str]]:
    tasks_by_index = {int(row["task_index"]): str(row["task"]) for row in load_jsonl(source_root / "meta/tasks.jsonl")}
    episode_tasks: dict[int, str] = {}
    for row in load_jsonl(source_root / "meta/episodes.jsonl"):
        tasks = row.get("tasks") or []
        if tasks:
            episode_tasks[int(row["episode_index"])] = str(tasks[0])
    return tasks_by_index, episode_tasks


def make_dataset(repo_id: str, root: Path, fps: int, robot_type: str | None, features: dict[str, Any]):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        raise SystemExit("LeRobot is not installed. Activate the LeRobot environment first.") from exc

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        robot_type=robot_type or "franka_emika",
        features=features,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )


def convert_frame(row: pd.Series, features: dict[str, Any], task: str) -> dict[str, Any]:
    frame: dict[str, Any] = {"task": task}
    for source_key, target_key in FEATURE_MAP.items():
        feature = features[target_key]
        if feature["dtype"] == "image":
            frame[target_key] = to_chw_uint8(row[source_key], tuple(feature["shape"]))
        else:
            frame[target_key] = to_array(row[source_key], feature["dtype"], tuple(feature["shape"]))
    return frame


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be >= 1")

    sources = args.source
    first_split, first_root = sources[0]
    first_info = load_json(first_root / "meta/info.json")
    selected = parse_episodes(args.episodes)
    first_ids = selected if selected is not None else discover_episode_ids(first_root, first_info)
    if args.max_episodes_per_source:
        first_ids = first_ids[: args.max_episodes_per_source]
    if not first_ids:
        raise SystemExit("No episodes found in first source")
    first_df = pd.read_parquet(episode_path(first_root, first_info, first_ids[0]))
    features = build_features(first_info["features"], list(first_df.columns))

    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root.parent)
    dataset = make_dataset(args.repo_id, args.output_root, int(first_info.get("fps", 10)), first_info.get("robot_type"), features)

    converted: list[dict[str, Any]] = []
    skipped_missing: list[dict[str, Any]] = []
    total_frames = 0
    for split, source_root in sources:
        info = load_json(source_root / "meta/info.json")
        source_episode_ids = selected if selected is not None else discover_episode_ids(source_root, info)
        if args.max_episodes_per_source:
            source_episode_ids = source_episode_ids[: args.max_episodes_per_source]
        tasks_by_index, episode_tasks = task_lookup(source_root)
        for source_episode_index in source_episode_ids:
            path = episode_path(source_root, info, source_episode_index)
            if not path.exists():
                skipped_missing.append({"split": split, "episode_index": source_episode_index})
                continue
            df = pd.read_parquet(path)
            if not len(df):
                continue
            task_index = int(df.iloc[0]["task_index"]) if "task_index" in df.columns else -1
            task = tasks_by_index.get(task_index) or episode_tasks.get(source_episode_index) or "calvin_task"
            frames_added = 0
            for _, row in df.iloc[:: args.frame_stride].iterrows():
                dataset.add_frame(convert_frame(row, features, task))
                frames_added += 1
            if frames_added == 0:
                continue
            dataset.save_episode()
            converted.append(
                {
                    "split": split,
                    "source_episode_index": source_episode_index,
                    "local_episode_index": len(converted),
                    "task": task,
                    "frames": frames_added,
                }
            )
            total_frames += frames_added
            print(f"converted {split} episode {source_episode_index} -> local {len(converted) - 1} ({frames_added} frames)")

    manifest = {
        "source_dataset": "xiaoma26/calvin-lerobot",
        "sources": [{"split": split, "root": str(root)} for split, root in sources],
        "repo_id": args.repo_id,
        "output_root": str(args.output_root),
        "frame_stride": args.frame_stride,
        "feature_map": FEATURE_MAP,
        "features": features,
        "converted_episodes": converted,
        "num_converted_episodes": len(converted),
        "num_frames": total_frames,
        "skipped_missing_episodes": skipped_missing,
    }
    write_json(args.output_root / "conversion_manifest.json", manifest)
    print(f"Converted {len(converted)} episodes / {total_frames} frames to {args.output_root}")


if __name__ == "__main__":
    main()
