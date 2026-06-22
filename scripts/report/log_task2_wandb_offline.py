#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import wandb


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", str(ROOT / "results/task2/wandb_offline"))
    os.environ.setdefault("WANDB_SILENT", "true")

    run = wandb.init(
        project="cv_hw3_task2_act",
        name="official_calvin_act_B_vs_ABC",
        dir=str(ROOT / "results/task2/wandb_offline"),
        config={
            "policy": "LeRobot ACT",
            "datasets": {
                "B_only": "xiaoma26/calvin-lerobot splitB first 100 episodes",
                "ABC_mixed": "splitA/splitB/splitC first 50 episodes each",
                "D_eval": "splitD first 100 episodes",
            },
            "batch_size": 16,
            "learning_rate": 1e-5,
            "optimizer": "AdamW",
            "training_steps": 10000,
            "loss": "Action L1 + ACT auxiliary losses",
            "chunk_size": 100,
        },
    )

    train_csv = ROOT / "report/figures/task2_official_training_loss.csv"
    with train_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            step = int(float(row["step"]))
            wandb.log(
                {
                    f"train/{model}/loss": float(row["loss"]),
                    f"train/{model}/grad_norm": float(row["grad_norm"]),
                    f"train/{model}/lr": float(row["lr"]),
                },
                step=step,
            )

    eval_paths = {
        "B-only": ROOT / "results/task2/official_eval_B_on_D_100.json",
        "ABC-mixed": ROOT / "results/task2/official_eval_ABC_on_D_100.json",
    }
    for model, path in eval_paths.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        wandb.log(
            {
                f"eval_D/{model}/Action_L1": float(data["mean_action_l1"]),
                f"eval_D/{model}/mean_loss": float(data["mean_loss"]),
                f"eval_D/{model}/num_batches": int(data["num_batches"]),
            },
            step=10000,
        )

    wandb.save(str(train_csv))
    for path in eval_paths.values():
        wandb.save(str(path))
    run.finish()


if __name__ == "__main__":
    main()
