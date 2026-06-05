#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from cv_hw3.common import ensure_dir, write_json


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    task: str
    task_id: str | None = None


@dataclass
class SceneIndex:
    intervals: list[tuple[int, int, str]]
    labels: list[str] | None = None

    def env_for(self, start: int, end: int) -> str | None:
        mid = (start + end) // 2
        if self.labels is not None and 0 <= mid < len(self.labels):
            return extract_env_label(self.labels[mid])
        for left, right, env in self.intervals:
            if left <= mid <= right:
                return env
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CALVIN language segments into a LeRobotDataset.")
    parser.add_argument("--calvin-root", required=True, type=Path, help="CALVIN task root, e.g. task_ABCD_D.")
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--environments", nargs="+", default=["B"], choices=["A", "B", "C", "D"])
    parser.add_argument("--repo-id", required=True, help="LeRobot repo_id metadata, e.g. cv_hw3/calvin_B.")
    parser.add_argument("--output-root", required=True, type=Path, help="Local LeRobot dataset root.")
    parser.add_argument("--fps", default=10, type=int)
    parser.add_argument("--max-episodes", default=0, type=int, help="0 converts all matching segments.")
    parser.add_argument("--frame-stride", default=1, type=int, help="Use >1 for a quick debug subset.")
    parser.add_argument("--allow-missing-scene-info", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_env_label(value: Any) -> str | None:
    text = str(value)
    for env in "ABCD":
        patterns = [
            rf"(?:^|[_\-/\s]){env}(?:$|[_\-/\s])",
            rf"scene[_\-/]?{env}(?:$|[_\-/\s])",
            rf"calvin_scene[_\-/]?{env}(?:$|[_\-/\s])",
        ]
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            return env
    if text.strip().upper() in {"A", "B", "C", "D"}:
        return text.strip().upper()
    return None


def as_interval_list(value: Any) -> list[tuple[int, int]]:
    if isinstance(value, dict):
        intervals: list[tuple[int, int]] = []
        for nested in value.values():
            intervals.extend(as_interval_list(nested))
        return intervals
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 1 and len(arr) >= 2 and all(np.isscalar(x) for x in arr[:2]):
        return [(int(arr[0]), int(arr[1]))]
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return [(int(row[0]), int(row[1])) for row in arr]
    intervals = []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for item in value:
            intervals.extend(as_interval_list(item))
    return intervals


def load_scene_index(split_dir: Path) -> SceneIndex | None:
    path = split_dir / "scene_info.npy"
    if not path.exists():
        return None
    raw = np.load(path, allow_pickle=True)
    obj: Any
    if raw.shape == () and hasattr(raw, "item"):
        obj = raw.item()
    else:
        obj = raw

    intervals: list[tuple[int, int, str]] = []
    labels: list[str] | None = None
    if isinstance(obj, dict):
        for key, value in obj.items():
            env = extract_env_label(key)
            if env is None:
                env = extract_env_label(value)
            if env is None:
                continue
            for start, end in as_interval_list(value):
                intervals.append((start, end, env))
    else:
        flat = np.asarray(obj, dtype=object).reshape(-1)
        env_labels = [extract_env_label(item) for item in flat]
        if any(label is not None for label in env_labels):
            labels = [label or "" for label in env_labels]
    return SceneIndex(intervals=intervals, labels=labels)


def find_split_dir(calvin_root: Path, split: str) -> Path:
    candidates = [
        calvin_root / split,
        calvin_root / f"task_ABCD_D/{split}",
        calvin_root / f"task_ABC_D/{split}",
        calvin_root / f"task_D_D/{split}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"Could not locate split '{split}' under {calvin_root}")


def load_annotations(split_dir: Path) -> list[Segment]:
    candidates = [
        split_dir / "lang_annotations" / "auto_lang_ann.npy",
        split_dir.parent / "lang_annotations" / "auto_lang_ann.npy",
    ]
    ann_path = next((p for p in candidates if p.exists()), None)
    if ann_path is not None:
        raw = np.load(ann_path, allow_pickle=True).item()
        info = raw.get("info", {})
        language = raw.get("language", {})
        indices = first_not_none(info, ["indx", "indices"])
        if indices is None:
            raise SystemExit(f"Language annotation file has no info.indx: {ann_path}")
        texts = first_not_none(language, ["ann", "annotations", "lang_ann"])
        tasks = first_not_none(language, ["task", "tasks"])
        if tasks is None:
            tasks = first_not_none(info, ["task", "tasks"])
        segments: list[Segment] = []
        for i, pair in enumerate(indices):
            task = str(texts[i]) if texts is not None and i < len(texts) else "calvin_task"
            task_id = str(tasks[i]) if tasks is not None and i < len(tasks) else None
            segments.append(Segment(start=int(pair[0]), end=int(pair[1]), task=task, task_id=task_id))
        return segments

    ep_ranges = split_dir / "ep_start_end_ids.npy"
    if ep_ranges.exists():
        ranges = np.load(ep_ranges, allow_pickle=True)
        return [Segment(int(pair[0]), int(pair[1]), "calvin_play") for pair in ranges]
    raise SystemExit(f"No language annotations or ep_start_end_ids.npy found in {split_dir}")


def first_not_none(mapping: dict[str, Any], keys: Iterable[str]) -> Any | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def first_existing(data: Any, keys: Iterable[str]) -> np.ndarray:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of the keys were found: {list(keys)}")


def to_chw_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
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
    return arr


def load_frame(npz_path: Path, task: str) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        state = first_existing(data, ["robot_obs"]).astype(np.float32)
        scene_state = first_existing(data, ["scene_obs"]).astype(np.float32)
        action = first_existing(data, ["rel_actions", "actions"]).astype(np.float32)
        static = to_chw_uint8(first_existing(data, ["rgb_static", "static_rgb"]))
        wrist = to_chw_uint8(first_existing(data, ["rgb_gripper", "rgb_wrist", "wrist_rgb"]))
    return {
        "observation.state": state,
        "observation.environment_state": scene_state,
        "observation.images.top": static,
        "observation.images.wrist": wrist,
        "action": action,
        "task": task,
    }


def make_dataset(repo_id: str, root: Path, fps: int):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        raise SystemExit("LeRobot is not installed. Activate the LeRobot environment first.") from exc

    features = {
        "observation.state": {"dtype": "float32", "shape": (15,), "names": None},
        "observation.environment_state": {"dtype": "float32", "shape": (24,), "names": None},
        "observation.images.top": {"dtype": "image", "shape": (3, 200, 200), "names": ["channel", "height", "width"]},
        "observation.images.wrist": {"dtype": "image", "shape": (3, 84, 84), "names": ["channel", "height", "width"]},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }
    kwargs = {
        "repo_id": repo_id,
        "fps": fps,
        "robot_type": "calvin_franka",
        "features": features,
        "use_videos": False,
    }
    try:
        return LeRobotDataset.create(root=root, **kwargs)
    except TypeError:
        try:
            return LeRobotDataset.create(local_dir=root, **kwargs)
        except TypeError:
            return LeRobotDataset.create(**kwargs)


def save_episode(dataset: Any, episode_index: int, task: str) -> None:
    variants = [
        {"episode_index": episode_index, "task": task},
        {"task": task},
        {},
    ]
    last_error: TypeError | None = None
    for kwargs in variants:
        try:
            dataset.save_episode(**kwargs)
            return
        except TypeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def main() -> None:
    args = parse_args()
    split_dir = find_split_dir(args.calvin_root, args.split)
    allowed_envs = set(args.environments)
    scene_index = load_scene_index(split_dir)
    if scene_index is None and allowed_envs and not args.allow_missing_scene_info:
        raise SystemExit(
            f"{split_dir / 'scene_info.npy'} is missing. Refusing to create an environment-filtered dataset."
        )

    if args.overwrite and args.output_root.exists():
        import shutil

        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root.parent)

    segments = load_annotations(split_dir)
    dataset = make_dataset(args.repo_id, args.output_root, args.fps)

    converted = 0
    skipped_env = 0
    skipped_missing = 0
    for segment in segments:
        env = scene_index.env_for(segment.start, segment.end) if scene_index is not None else None
        if env is not None and env not in allowed_envs:
            skipped_env += 1
            continue
        if env is None and scene_index is not None:
            skipped_env += 1
            continue
        frames_added = 0
        for frame_idx in range(segment.start, segment.end + 1, args.frame_stride):
            npz_path = split_dir / f"episode_{frame_idx:07d}.npz"
            if not npz_path.exists():
                skipped_missing += 1
                continue
            dataset.add_frame(load_frame(npz_path, segment.task))
            frames_added += 1
        if frames_added == 0:
            continue
        save_episode(dataset, converted, segment.task)
        converted += 1
        if args.max_episodes and converted >= args.max_episodes:
            break

    if hasattr(dataset, "finalize"):
        dataset.finalize()

    manifest = {
        "calvin_root": str(args.calvin_root),
        "split_dir": str(split_dir),
        "split": args.split,
        "repo_id": args.repo_id,
        "output_root": str(args.output_root),
        "environments": sorted(allowed_envs),
        "converted_episodes": converted,
        "skipped_environment_segments": skipped_env,
        "skipped_missing_frames": skipped_missing,
        "frame_stride": args.frame_stride,
    }
    write_json(args.output_root / "conversion_manifest.json", manifest)
    print(f"Converted {converted} episodes to {args.output_root}")


if __name__ == "__main__":
    main()
