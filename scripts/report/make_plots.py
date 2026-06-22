#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

PASTEL = {
    "teal": "#93D8D0",
    "peach": "#F7C59F",
    "lavender": "#C8B6E2",
    "blue": "#9ED1F0",
    "pink": "#F4A6C1",
}


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
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = range(len(labels))
    width = 0.24
    ax.bar([i - width / 2 for i in x], vertices, width=width, label="vertices", color=PASTEL["teal"], edgecolor="#ffffff", linewidth=0.8)
    ax.bar([i + width / 2 for i in x], faces, width=width, label="faces", color=PASTEL["peach"], edgecolor="#ffffff", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "task1_asset_mesh_stats.pdf")
    fig.savefig(output_dir / "task1_asset_mesh_stats.png", dpi=180)
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
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    bars = ax.bar(labels, values, width=0.34, color=[PASTEL["teal"], PASTEL["pink"], PASTEL["lavender"], PASTEL["blue"]][: len(values)], edgecolor="#ffffff", linewidth=0.8)
    ax.set_ylabel("Validation metric")
    ax.set_title("Zero-shot evaluation on CALVIN D")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "task2_action_l1.pdf")
    fig.savefig(output_dir / "task2_action_l1.png", dpi=180)
    plt.close(fig)


def plot_task2_dataset_sizes(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    datasets = [
        ("B-only train", Path("data/task2/lerobot_v30_official/calvin_B_100/meta/info.json")),
        ("ABC-mixed train", Path("data/task2/lerobot_v30_official/calvin_ABC_50each/meta/info.json")),
        ("D zero-shot eval", Path("data/task2/lerobot_v30_official/calvin_D_eval_100/meta/info.json")),
    ]
    labels, episodes, frames = [], [], []
    for label, path in datasets:
        if not path.exists():
            continue
        data = load_json(path)
        labels.append(label)
        episodes.append(data["total_episodes"])
        frames.append(data["total_frames"])
    if not labels:
        return

    x = range(len(labels))
    fig, ax1 = plt.subplots(figsize=(6.8, 3.6))
    ax2 = ax1.twinx()
    width = 0.24
    ax1.bar([i - width / 2 for i in x], episodes, width=width, color=PASTEL["teal"], label="episodes", edgecolor="#ffffff", linewidth=0.8)
    ax2.bar([i + width / 2 for i in x], frames, width=width, color=PASTEL["peach"], label="frames", edgecolor="#ffffff", linewidth=0.8)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=16, ha="right")
    ax1.set_ylabel("Episodes")
    ax2.set_ylabel("Frames")
    ax1.set_title("CALVIN splits used for Task 2")
    ax1.set_axisbelow(True)
    ax1.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "task2_dataset_sizes.png", dpi=180)
    fig.savefig(output_dir / "task2_dataset_sizes.pdf")
    plt.close(fig)


def plot_task2_action_stats(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    datasets = [
        ("B train", Path("data/task2/lerobot_v30_official/calvin_B_100/meta/stats.json")),
        ("ABC train", Path("data/task2/lerobot_v30_official/calvin_ABC_50each/meta/stats.json")),
        ("D eval", Path("data/task2/lerobot_v30_official/calvin_D_eval_100/meta/stats.json")),
    ]
    labels = [f"a{i}" for i in range(7)]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharex=True)
    x = list(range(7))
    colors = [PASTEL["teal"], PASTEL["peach"], PASTEL["lavender"]]
    width = 0.18
    any_data = False
    for idx, (name, path) in enumerate(datasets):
        if not path.exists():
            continue
        stats = load_json(path).get("action", {})
        mean = stats.get("mean")
        std = stats.get("std")
        if mean is None or std is None:
            continue
        any_data = True
        offset = (idx - 1) * width
        axes[0].bar([v + offset for v in x], mean, width=width * 0.86, color=colors[idx], label=name, edgecolor="#ffffff", linewidth=0.8)
        axes[1].bar([v + offset for v in x], std, width=width * 0.86, color=colors[idx], label=name, edgecolor="#ffffff", linewidth=0.8)
    if not any_data:
        plt.close(fig)
        return
    for ax, title, ylabel in [
        (axes[0], "Action mean", "Mean"),
        (axes[1], "Action std", "Std"),
    ]:
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Action dimension")
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("Action distribution statistics")
    fig.tight_layout()
    fig.savefig(output_dir / "task2_action_stats.png", dpi=180)
    fig.savefig(output_dir / "task2_action_stats.pdf")
    plt.close(fig)


def plot_task2_act_pipeline(output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    boxes = [
        ("Top RGB\n3 x 200 x 200", (0.03, 0.66), 0.14, 0.18),
        ("Wrist RGB\n3 x 84 x 84", (0.03, 0.39), 0.14, 0.18),
        ("Robot state\n15-d", (0.03, 0.12), 0.14, 0.18),
        ("ResNet18\nvisual encoders", (0.25, 0.52), 0.15, 0.20),
        ("Visual tokens\n+ state token", (0.47, 0.52), 0.15, 0.20),
        ("ACT Transformer\n4 enc / 1 dec\nlatent dim 32", (0.68, 0.50), 0.16, 0.24),
        ("100-step\naction chunk\n100 x 7", (0.88, 0.50), 0.10, 0.24),
    ]
    fig, ax = plt.subplots(figsize=(12.0, 3.2))
    ax.set_axis_off()

    def add_box(text: str, xy: tuple[float, float], w: float, h: float) -> None:
        patch = FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.1,
            edgecolor="#333333",
            facecolor="#f7f7f7",
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=10, transform=ax.transAxes)

    for text, xy, w, h in boxes:
        add_box(text, xy, w, h)

    def arrow(src: tuple[float, float], dst: tuple[float, float]) -> None:
        ax.add_patch(
            FancyArrowPatch(
                src,
                dst,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.1,
                color="#555555",
                transform=ax.transAxes,
            )
        )

    arrow((0.17, 0.75), (0.25, 0.64))
    arrow((0.17, 0.48), (0.25, 0.61))
    arrow((0.17, 0.21), (0.47, 0.58))
    arrow((0.40, 0.62), (0.47, 0.62))
    arrow((0.62, 0.62), (0.68, 0.62))
    arrow((0.84, 0.62), (0.88, 0.62))
    ax.text(0.5, 0.93, "ACT policy used in Task 2", ha="center", va="center", fontsize=14, transform=ax.transAxes)
    fig.subplots_adjust(left=0.015, right=0.995, top=0.90, bottom=0.04)
    fig.savefig(output_dir / "task2_act_pipeline.png", dpi=180)
    fig.savefig(output_dir / "task2_act_pipeline.pdf")
    plt.close(fig)


def parse_train_log(path: Path) -> list[dict[str, float]]:
    progress_re = re.compile(r"\|\s*(\d+)/(\d+)\s")
    loss_re = re.compile(r"\bloss:([0-9.]+)")
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        progress = progress_re.search(line)
        loss = loss_re.search(line)
        if not progress or not loss:
            continue
        rows.append(
            {
                "step": float(progress.group(1)),
                "total_steps": float(progress.group(2)),
                "loss": float(loss.group(1)),
            }
        )
    return rows


def plot_train_losses(paths: list[Path], labels: list[str], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if not paths:
        return
    if labels and len(labels) != len(paths):
        raise SystemExit("--train-labels length must match --train-logs.")
    labels = labels or [p.stem for p in paths]

    series: list[tuple[str, list[dict[str, float]]]] = []
    csv_rows: list[dict[str, int | float | str]] = []
    for label, path in zip(labels, paths):
        rows = parse_train_log(path)
        if not rows:
            continue
        series.append((label, rows))
        csv_rows.extend(
            {
                "label": label,
                "step": int(row["step"]),
                "loss": row["loss"],
            }
            for row in rows
        )

    if not series:
        return

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=False)
    for label, rows in series:
        steps = [row["step"] for row in rows]
        losses = [row["loss"] for row in rows]
        ax_full.plot(steps, losses, linewidth=1.8, label=label)

        zoom = [row for row in rows if row["step"] >= 500]
        ax_zoom.plot([row["step"] for row in zoom], [row["loss"] for row in zoom], linewidth=1.8, label=label)

    ax_full.set_title("Full training")
    ax_full.set_xlabel("Step")
    ax_full.set_ylabel("Training loss")
    ax_full.set_xlim(0, 10000)
    ax_full.grid(True, linewidth=0.4, alpha=0.35)

    ax_zoom.set_title("After warm-up")
    ax_zoom.set_xlabel("Step")
    ax_zoom.set_xlim(500, 10000)
    ax_zoom.set_ylim(0, 2.2)
    ax_zoom.grid(True, linewidth=0.4, alpha=0.35)
    ax_zoom.legend(frameon=False)

    fig.suptitle("ACT training loss")
    fig.tight_layout()
    fig.savefig(output_dir / "task2_training_loss.pdf")
    fig.savefig(output_dir / "task2_training_loss.png", dpi=180)
    plt.close(fig)

    with (output_dir / "task2_training_loss.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "step", "loss"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_asset_stats(args.asset_stats, args.output_dir)
    plot_act_evals(args.act_evals, args.act_labels, args.output_dir)
    plot_task2_dataset_sizes(args.output_dir)
    plot_task2_action_stats(args.output_dir)
    plot_task2_act_pipeline(args.output_dir)
    plot_train_losses(args.train_logs, args.train_labels, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
