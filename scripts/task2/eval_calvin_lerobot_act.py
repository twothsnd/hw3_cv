#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CALVIN official success-rate evaluation with a LeRobot ACT policy.")
    parser.add_argument("--calvin-root", required=True, type=Path, help="Local clone of mees/calvin.")
    parser.add_argument("--dataset-path", required=True, type=Path, help="CALVIN dataset path containing validation/.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-repo-id", default=None, help="Optional LeRobot dataset repo_id for stats/processors.")
    parser.add_argument("--dataset-root", default=None, type=Path, help="Optional LeRobot dataset root for stats/processors.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-log-dir", required=True, type=Path)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def move_to_device(batch: Any, device: str) -> Any:
    import torch

    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    return batch


def image_to_chw(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {arr.shape}")
    if arr.shape[0] not in {1, 3, 4} and arr.shape[-1] in {1, 3, 4}:
        arr = np.transpose(arr, (2, 0, 1))
    if arr.shape[0] == 4:
        arr = arr[:3]
    return arr


def get_nested(mapping: dict[str, Any], *keys: str) -> Any | None:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


class LeRobotACTCalvinModel:
    def __init__(
        self,
        policy_path: str,
        device: str,
        dataset_repo_id: str | None = None,
        dataset_root: Path | None = None,
    ) -> None:
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy

        self.torch = torch
        self.device = device
        self.policy = ACTPolicy.from_pretrained(policy_path)
        self.policy.to(device)
        self.policy.eval()
        self.preprocessor = None
        self.postprocessor = None

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            from lerobot.policies.factory import make_pre_post_processors

            metadata_kwargs: dict[str, Any] = {}
            if dataset_root is not None:
                metadata_kwargs["root"] = dataset_root
            if dataset_repo_id is not None:
                metadata = LeRobotDatasetMetadata(dataset_repo_id, **metadata_kwargs)
                self.preprocessor, self.postprocessor = make_pre_post_processors(
                    self.policy.config, dataset_stats=metadata.stats
                )
        except Exception as exc:
            print(f"Warning: using policy without LeRobot pre/post processors: {exc}")

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _build_batch(self, obs: dict[str, Any], goal: str) -> dict[str, Any]:
        torch = self.torch
        rgb_obs = obs.get("rgb_obs", {}) if isinstance(obs.get("rgb_obs", {}), dict) else {}
        static = rgb_obs.get("rgb_static", obs.get("rgb_static"))
        wrist = rgb_obs.get("rgb_gripper", obs.get("rgb_gripper", obs.get("rgb_wrist")))
        robot_obs = obs.get("robot_obs", get_nested(obs, "state_obs", "robot_obs"))
        scene_obs = obs.get("scene_obs", get_nested(obs, "state_obs", "scene_obs"))
        if static is None or wrist is None or robot_obs is None or scene_obs is None:
            raise KeyError(f"Unexpected CALVIN observation keys: {list(obs.keys())}")

        batch = {
            "observation.state": torch.as_tensor(robot_obs, dtype=torch.float32).unsqueeze(0),
            "observation.environment_state": torch.as_tensor(scene_obs, dtype=torch.float32).unsqueeze(0),
            "observation.images.top": torch.as_tensor(image_to_chw(static)).unsqueeze(0),
            "observation.images.wrist": torch.as_tensor(image_to_chw(wrist)).unsqueeze(0),
            "task": [goal],
        }
        return batch

    def step(self, obs: dict[str, Any], goal: str) -> np.ndarray:
        batch = self._build_batch(obs, goal)
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        batch = move_to_device(batch, self.device)
        with self.torch.no_grad():
            action = self.policy.select_action(batch)
        if self.postprocessor is not None:
            action = self.postprocessor(action)
        if isinstance(action, dict):
            action = action.get("action", next(iter(action.values())))
        if self.torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        return np.asarray(action).reshape(-1)[:7]


def main() -> None:
    args = parse_args()
    calvin_root = args.calvin_root.resolve()
    sys.path.insert(0, str(calvin_root / "calvin_models"))
    sys.path.insert(0, str(calvin_root))

    try:
        from calvin_agent.evaluation.evaluate_policy import evaluate_policy, make_env
    except Exception as exc:
        raise SystemExit("Could not import CALVIN evaluation code. Check --calvin-root and CALVIN installation.") from exc

    model = LeRobotACTCalvinModel(
        policy_path=args.policy_path,
        device=args.device,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
    )
    env = make_env(args.dataset_path)
    evaluate_policy(
        model,
        env,
        epoch="lerobot_act",
        eval_log_dir=args.eval_log_dir,
        debug=args.debug,
        create_plan_tsne=False,
    )


if __name__ == "__main__":
    main()
