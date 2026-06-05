#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report figures from task metrics.")
    parser.add_argument("--asset-stats", type=Path, default=Path("results/task1/asset_stats.json"))
    parser.add_argument("--act-evals", nargs="*", type=Path, default=[])
    parser.add_argument("--act-labels", nargs="*", default=[])
    parser.add_argument("--train-logs", nargs="*", type=Path, default=[])
    parser.add_argument("--train-labels", nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("report/figures"))
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_asset_stats(asset_stats: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if not asset_stats.exists():
        return
    data = load_json(asset_stats)
    labels = []
    vertices = []
    faces = []
    for label, stats in data.items():
        if not stats.get("exists"):
            continue
        labels.append(label)
        vertices.append(stats.get("vertices") or 0)
        faces.append(stats.get("faces") or 0)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = range(len(labels))
    ax.bar([i - 0.18 for i in x], vertices, width=0.36, label="vertices")
    ax.bar([i + 0.18 for i in x], faces, width=0.36, label="faces")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "task1_asset_mesh_stats.pdf")
    plt.close(fig)


def plot_act_evals(paths: list[Path], labels: list[str], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if not paths:
        return
    if labels and len(labels) != len(paths):
        raise SystemExit("--act-labels length must match --act-evals.")
    labels = labels or [p.stem for p in paths]
    values = []
    for path in paths:
        data = load_json(path)
        values.append(data.get("mean_action_l1") or data.get("mean_loss") or 0)
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.bar(labels, values, color=["#3b6ea8", "#d78233", "#5c9a61", "#8f5a9f"][: len(values)])
    ax.set_ylabel("Validation metric")
    ax.set_title("Zero-shot evaluation on CALVIN D")
    fig.tight_layout()
    fig.savefig(output_dir / "task2_action_l1.pdf")
    plt.close(fig)


def parse_train_log(path: Path) -> list[tuple[int, float]]:
    pattern = re.compile(r"step:(\d+).*?\bloss:([0-9.]+)")
    points: list[tuple[int, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            points.append((int(match.group(1)), float(match.group(2))))
    return points


def plot_train_losses(paths: list[Path], labels: list[str], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if not paths:
        return
    if labels and len(labels) != len(paths):
        raise SystemExit("--train-labels length must match --train-logs.")
    labels = labels or [p.stem for p in paths]

    all_rows = []
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for label, path in zip(labels, paths):
        points = parse_train_log(path)
        if not points:
            continue
        steps, losses = zip(*points)
        ax.plot(steps, losses, marker="o", linewidth=1.7, label=label)
        all_rows.extend({"label": label, "step": step, "loss": loss} for step, loss in points)
    if not all_rows:
        plt.close(fig)
        return
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.set_title("ACT training loss")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "task2_training_loss.pdf")
    plt.close(fig)

    with (output_dir / "task2_training_loss.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "step", "loss"])
        writer.writeheader()
        writer.writerows(all_rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_asset_stats(args.asset_stats, args.output_dir)
    plot_act_evals(args.act_evals, args.act_labels, args.output_dir)
    plot_train_losses(args.train_logs, args.train_labels, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
