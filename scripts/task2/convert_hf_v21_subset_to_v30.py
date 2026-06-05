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

DEFAULT_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a local LeRobot v2.1 HF parquet subset to v3.0.")
    parser.add_argument("--source-root", required=True, type=Path, help="Downloaded v2.1 subset root.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output v3 local dataset root.")
    parser.add_argument("--repo-id", required=True, help="Local LeRobot repo_id, e.g. cv_hw3/calvin_D_eval.")
    parser.add_argument("--episodes", nargs="+", default=None, help="Optional source episode ids/ranges.")
    parser.add_argument("--drop-features", nargs="*", default=[], help="Source feature keys to exclude.")
    parser.add_argument("--frame-stride", default=1, type=int, help="Use >1 for quick experiments.")
    parser.add_argument("--max-episodes", default=0, type=int, help="0 converts all selected parquet episodes.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def feature_shape_chw(feature: dict[str, Any]) -> tuple[int, int, int]:
    shape = tuple(int(v) for v in feature["shape"])
    names = feature.get("names") or []
    if len(shape) != 3:
        raise ValueError(f"Image feature has invalid shape: {shape}")
    lowered = [str(name).lower() for name in names]
    if lowered == ["height", "width", "channel"] or shape[-1] in {1, 3, 4}:
        h, w, c = shape
        return (3 if c == 4 else c, h, w)
    if lowered == ["channel", "height", "width"] or shape[0] in {1, 3, 4}:
        c, h, w = shape
        return (3 if c == 4 else c, h, w)
    raise ValueError(f"Could not infer channel dimension for image feature: {feature}")


def build_features(
    source_features: dict[str, Any], columns: list[str], drop_features: set[str]
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for key, feature in source_features.items():
        if key in DEFAULT_FEATURES or key in drop_features or key not in columns:
            continue
        dtype = feature["dtype"]
        if dtype == "image":
            features[key] = {
                "dtype": "image",
                "shape": feature_shape_chw(feature),
                "names": ["channel", "height", "width"],
            }
        elif dtype in {"float32", "float64", "int64", "int32", "bool"}:
            features[key] = {
                "dtype": dtype,
                "shape": tuple(int(v) for v in feature["shape"]),
                "names": feature.get("names"),
            }
        else:
            raise ValueError(f"Unsupported feature dtype for {key}: {dtype}")
    required = {"observation.state", "observation.images.top", "observation.images.wrist", "action"}
    missing = sorted(required - set(features))
    if missing:
        raise ValueError(f"Missing required CALVIN features in parquet/source metadata: {missing}")
    return features


def episode_path(source_root: Path, data_path: str, chunks_size: int, episode_index: int) -> Path:
    rel = data_path.format(episode_chunk=episode_index // chunks_size, episode_index=episode_index)
    return source_root / rel


def discover_episode_ids(source_root: Path, info: dict[str, Any]) -> list[int]:
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    ids = []
    for path in sorted((source_root / "data").rglob("episode_*.parquet")):
        stem = path.stem
        try:
            episode_index = int(stem.split("_")[-1])
        except ValueError:
            continue
        if episode_path(source_root, data_path, chunks_size, episode_index).exists():
            ids.append(episode_index)
    return ids


def decode_image(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            return np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))
        if "path" in value and value["path"]:
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
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {arr.shape}")
    if arr.shape[0] not in {1, 3, 4} and arr.shape[-1] in {1, 3, 4}:
        arr = np.transpose(arr, (2, 0, 1))
    if arr.shape[0] == 4:
        arr = arr[:3]
    if arr.dtype != np.uint8:
        max_value = float(np.nanmax(arr)) if arr.size else 1.0
        if max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"Image shape {arr.shape} does not match expected {expected_shape}")
    return arr


def to_array(value: Any, dtype: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.dtype(dtype))
    if arr.shape != expected_shape:
        raise ValueError(f"Array shape {arr.shape} does not match expected {expected_shape}")
    return arr


def make_dataset(repo_id: str, root: Path, fps: int, robot_type: str | None, features: dict[str, Any]):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        raise SystemExit("LeRobot is not installed. Activate the LeRobot environment first.") from exc

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        robot_type=robot_type or "calvin_franka",
        features=features,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )


def task_lookup(source_root: Path) -> tuple[dict[int, str], dict[int, str]]:
    tasks_by_index = {
        int(row["task_index"]): str(row["task"]) for row in load_jsonl(source_root / "meta/tasks.jsonl")
    }
    episode_tasks: dict[int, str] = {}
    for row in load_jsonl(source_root / "meta/episodes.jsonl"):
        tasks = row.get("tasks") or []
        if tasks:
            episode_tasks[int(row["episode_index"])] = str(tasks[0])
    return tasks_by_index, episode_tasks


def convert_frame(row: pd.Series, features: dict[str, Any], task: str) -> dict[str, Any]:
    frame: dict[str, Any] = {"task": task}
    for key, feature in features.items():
        if feature["dtype"] == "image":
            frame[key] = to_chw_uint8(row[key], tuple(feature["shape"]))
        else:
            frame[key] = to_array(row[key], feature["dtype"], tuple(feature["shape"]))
    return frame


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be >= 1")
    source_root = args.source_root
    info = load_json(source_root / "meta/info.json")
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")

    selected = parse_episodes(args.episodes)
    source_episode_ids = selected if selected is not None else discover_episode_ids(source_root, info)
    if args.max_episodes:
        source_episode_ids = source_episode_ids[: args.max_episodes]
    if not source_episode_ids:
        raise SystemExit(f"No parquet episodes found under {source_root}")

    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root.parent)

    first_path = episode_path(source_root, data_path, chunks_size, source_episode_ids[0])
    first_df = pd.read_parquet(first_path)
    features = build_features(info["features"], list(first_df.columns), set(args.drop_features))
    dataset = make_dataset(args.repo_id, args.output_root, int(info.get("fps", 10)), info.get("robot_type"), features)
    tasks_by_index, episode_tasks = task_lookup(source_root)

    converted: list[dict[str, Any]] = []
    skipped_missing = []
    total_frames = 0
    for source_episode_index in source_episode_ids:
        path = episode_path(source_root, data_path, chunks_size, source_episode_index)
        if not path.exists():
            skipped_missing.append(source_episode_index)
            continue
        df = pd.read_parquet(path)
        task_index = int(df.iloc[0]["task_index"]) if "task_index" in df.columns and len(df) else -1
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
                "source_episode_index": source_episode_index,
                "local_episode_index": len(converted),
                "task": task,
                "frames": frames_added,
            }
        )
        total_frames += frames_added
        print(f"converted source episode {source_episode_index} -> local {len(converted) - 1} ({frames_added} frames)")

    manifest = {
        "source_root": str(source_root),
        "source_codebase_version": info.get("codebase_version"),
        "repo_id": args.repo_id,
        "output_root": str(args.output_root),
        "frame_stride": args.frame_stride,
        "drop_features": sorted(args.drop_features),
        "converted_episodes": converted,
        "num_converted_episodes": len(converted),
        "num_frames": total_frames,
        "skipped_missing_episodes": skipped_missing,
        "features": features,
    }
    write_json(args.output_root / "conversion_manifest.json", manifest)
    print(f"Converted {len(converted)} episodes / {total_frames} frames to {args.output_root}")


if __name__ == "__main__":
    main()
