#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external/lerobot/src"))

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LeRobot ACT with online offline-dataset validation.")
    parser.add_argument("--train-repo-id", required=True)
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--val-repo-id", required=True)
    parser.add_argument("--val-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--steps", default=10000, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--log-freq", default=100, type=int)
    parser.add_argument("--val-freq", default=1000, type=int)
    parser.add_argument("--save-freq", default=5000, type=int)
    parser.add_argument("--val-max-batches", default=0, type=int, help="0 evaluates the full validation set.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=1000, type=int)
    parser.add_argument("--wandb-enable", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def move_to_device(batch, device: torch.device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    return batch


@torch.no_grad()
def evaluate(policy, preprocessor, loader: DataLoader, device: torch.device, max_batches: int = 0) -> dict[str, float]:
    # ACT's training loss needs the VAE branch outputs, which this LeRobot
    # implementation only returns in train mode. no_grad keeps validation
    # side-effect free while preserving the same loss path as training.
    policy.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_kld = 0.0
    n_loss = 0
    n_l1 = 0
    n_kld = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        batch = preprocessor(batch)
        batch = move_to_device(batch, device)
        loss, output = policy.forward(batch)
        total_loss += float(loss.detach().cpu())
        n_loss += 1
        if "l1_loss" in output:
            total_l1 += float(output["l1_loss"])
            n_l1 += 1
        if "kld_loss" in output:
            total_kld += float(output["kld_loss"])
            n_kld += 1
    policy.train()
    return {
        "val_loss": total_loss / max(1, n_loss),
        "val_l1_loss": total_l1 / max(1, n_l1),
        "val_kld_loss": total_kld / max(1, n_kld),
        "val_batches": n_loss,
    }


def append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = resolve(args.output)
    if output_dir.exists():
        raise SystemExit(f"Output already exists: {output_dir}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=args.train_repo_id, root=str(resolve(args.train_root))),
        policy=ACTConfig(device=str(device), push_to_hub=False),
        output_dir=output_dir,
        job_name=args.job_name,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        eval_freq=0,
        wandb=WandBConfig(enable=args.wandb_enable, project="cv_hw3_task2_act"),
    )
    cfg.validate()
    output_dir.mkdir(parents=True)

    logging.info("Creating train dataset")
    train_dataset = make_dataset(cfg)

    val_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=args.val_repo_id, root=str(resolve(args.val_root))),
        policy=cfg.policy,
        output_dir=output_dir / "_val_cfg_unused",
        job_name=args.job_name + "_val",
        steps=1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_freq=0,
        wandb=WandBConfig(enable=False),
    )
    val_cfg.validate()
    logging.info("Creating validation dataset")
    val_dataset = make_dataset(val_cfg)

    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=train_dataset.meta, rename_map=cfg.rename_map).to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        dataset_stats=train_dataset.meta.stats,
    )
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    train_iter = cycle(train_loader)

    metrics_csv = output_dir / "online_metrics.csv"
    fields = [
        "step",
        "train_loss",
        "train_l1_loss",
        "train_kld_loss",
        "grad_norm",
        "lr",
        "val_loss",
        "val_l1_loss",
        "val_kld_loss",
        "val_batches",
        "update_s",
        "val_s",
    ]
    manifest = {
        "train_repo_id": args.train_repo_id,
        "train_root": str(resolve(args.train_root)),
        "val_repo_id": args.val_repo_id,
        "val_root": str(resolve(args.val_root)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "log_freq": args.log_freq,
        "val_freq": args.val_freq,
        "save_freq": args.save_freq,
        "val_max_batches": args.val_max_batches,
        "device": str(device),
        "seed": args.seed,
        "note": "Validation is online: current in-memory policy is evaluated on the held-out validation dataset during training.",
    }
    (output_dir / "online_val_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    logging.info("Start training with online validation")
    pbar = tqdm(total=args.steps, desc=args.job_name, unit="step")
    last_train: dict[str, float] = {}
    for step in range(1, args.steps + 1):
        start = time.perf_counter()
        policy.train()
        batch = next(train_iter)
        batch = preprocessor(batch)
        batch = move_to_device(batch, device)
        loss, output_dict = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.optimizer.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if lr_scheduler is not None:
            lr_scheduler.step()
        update_s = time.perf_counter() - start
        last_train = {
            "train_loss": float(loss.detach().cpu()),
            "train_l1_loss": float(output_dict.get("l1_loss", 0.0)),
            "train_kld_loss": float(output_dict.get("kld_loss", 0.0)),
            "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "update_s": update_s,
        }

        should_val = args.val_freq > 0 and step % args.val_freq == 0
        should_log = args.log_freq > 0 and step % args.log_freq == 0
        should_save = step % args.save_freq == 0 or step == args.steps
        if should_val or step == 1:
            val_start = time.perf_counter()
            val_metrics = evaluate(policy, preprocessor, val_loader, device, args.val_max_batches)
            row = {
                "step": step,
                **last_train,
                **val_metrics,
                "val_s": time.perf_counter() - val_start,
            }
            append_csv(metrics_csv, fields, row)
            logging.info(
                "step=%s train_loss=%.4f val_l1=%.4f val_loss=%.4f val_batches=%s",
                step,
                row["train_loss"],
                row["val_l1_loss"],
                row["val_loss"],
                row["val_batches"],
            )
        elif should_log:
            row = {"step": step, **last_train, "val_loss": "", "val_l1_loss": "", "val_kld_loss": "", "val_batches": "", "val_s": ""}
            append_csv(metrics_csv, fields, row)

        if should_save:
            checkpoint_dir = get_step_checkpoint_dir(output_dir, args.steps, step)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                cfg=cfg,
                policy=policy,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
            update_last_checkpoint(checkpoint_dir)

        pbar.update(1)
    pbar.close()
    logging.info("Done: %s", output_dir)


if __name__ == "__main__":
    main()
