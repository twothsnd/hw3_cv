#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cv_hw3.common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LeRobot ACT checkpoint with offline Action L1 on a dataset.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--max-batches", default=0, type=int, help="0 evaluates all batches.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def move_to_device(batch: Any, device: str) -> Any:
    import torch

    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    return batch


def metric_value(output: Any, key_candidates: list[str]) -> float | None:
    import torch

    if torch.is_tensor(output):
        return float(output.detach().mean().cpu())
    if isinstance(output, tuple):
        for item in output:
            if isinstance(item, dict):
                value = metric_value(item, key_candidates)
                if value is not None:
                    return value
        for item in output:
            value = metric_value(item, key_candidates)
            if value is not None:
                return value
    if isinstance(output, dict):
        for key in key_candidates:
            if key in output:
                value = output[key]
                if torch.is_tensor(value):
                    return float(value.detach().mean().cpu())
                return float(value)
        if "loss" in output:
            value = output["loss"]
            if torch.is_tensor(value):
                return float(value.detach().mean().cpu())
            return float(value)
    return None


def main() -> None:
    args = parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader
        from lerobot.datasets.factory import resolve_delta_timestamps
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.act.modeling_act import ACTPolicy
    except Exception as exc:
        raise SystemExit("LeRobot/PyTorch is not installed. Activate the LeRobot environment first.") from exc

    policy = ACTPolicy.from_pretrained(args.policy_path)
    policy.to(args.device)
    policy.train()

    dataset_kwargs = {"repo_id": args.dataset_repo_id, "root": args.dataset_root}
    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)
    if delta_timestamps is not None:
        dataset_kwargs["delta_timestamps"] = delta_timestamps
    try:
        dataset = LeRobotDataset(**dataset_kwargs)
    except TypeError:
        dataset_kwargs.pop("delta_timestamps", None)
        dataset = LeRobotDataset(**dataset_kwargs)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    total_l1 = 0.0
    total_loss = 0.0
    l1_batches = 0
    loss_batches = 0
    total_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            batch = move_to_device(batch, args.device)
            output = policy(batch)
            l1 = metric_value(output, ["l1_loss", "loss_l1", "action_l1", "l1"])
            loss = metric_value(output, ["loss"])
            if l1 is not None:
                total_l1 += l1
                l1_batches += 1
            if loss is not None:
                total_loss += loss
                loss_batches += 1
            total_batches += 1

    if total_batches == 0:
        raise SystemExit("No evaluation batches were produced.")
    result = {
        "policy_path": args.policy_path,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "num_batches": total_batches,
        "mean_action_l1": total_l1 / l1_batches if l1_batches else None,
        "mean_loss": total_loss / loss_batches if loss_batches else None,
    }
    write_json(args.output, result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
