#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "report/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PASTEL = {
    "teal": "#93D8D0",
    "peach": "#F7C59F",
    "lavender": "#C8B6E2",
    "blue": "#9ED1F0",
    "pink": "#F4A6C1",
}

DATASETS = {
    "B train": ROOT / "data/task2/lerobot_v30_official/calvin_B_train80",
    "B val": ROOT / "data/task2/lerobot_v30_official/calvin_B_val20",
    "ABC train": ROOT / "data/task2/lerobot_v30_official/calvin_ABC_train40each",
    "ABC val": ROOT / "data/task2/lerobot_v30_official/calvin_ABC_val10each",
    "D zero-shot eval": ROOT / "data/task2/lerobot_v30_official/calvin_D_eval_100",
}
SPLITS = {
    "A": ROOT / "data/task2/xiaoma_calvin/splitA",
    "B": ROOT / "data/task2/xiaoma_calvin/splitB",
    "C": ROOT / "data/task2/xiaoma_calvin/splitC",
    "D": ROOT / "data/task2/xiaoma_calvin/splitD",
}
ONLINE_METRICS = {
    "B-only": ROOT / "results/task2/online_act_B_train80_val20_10k/online_metrics.csv",
    "ABC mixed": ROOT / "results/task2/online_act_ABC_train40each_val10each_10k/online_metrics.csv",
}
EVALS = {
    "B-only": ROOT / "results/task2/online_eval_B_on_D_100.json",
    "ABC mixed": ROOT / "results/task2/online_eval_ABC_on_D_100.json",
}


def decode_image(value) -> Image.Image:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4}:
        arr = np.transpose(arr[:3], (1, 2, 0))
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def first_parquet(root: Path) -> Path:
    return sorted((root / "data").rglob("*.parquet"))[0]


def load_info(root: Path) -> dict:
    return json.loads((root / "meta/info.json").read_text(encoding="utf-8"))


def plot_dataset_sizes() -> None:
    rows = []
    for name, root in DATASETS.items():
        info = load_info(root)
        rows.append((name, int(info["total_episodes"]), int(info["total_frames"])))
    labels = [r[0] for r in rows]
    episodes = [r[1] for r in rows]
    frames = [r[2] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(labels))
    width = 0.24
    bars_ep = ax1.bar(x - width / 2, episodes, width=width, label="Episodes", color=PASTEL["teal"], edgecolor="#ffffff", linewidth=0.9)
    ax2 = ax1.twinx()
    bars_fr = ax2.bar(x + width / 2, frames, width=width, label="Frames", color=PASTEL["peach"], edgecolor="#ffffff", linewidth=0.9)
    ax1.set_xticks(x, labels, rotation=10)
    ax1.set_ylabel("Episodes")
    ax2.set_ylabel("Frames")
    ax1.set_title("Official CALVIN Split Subsets")
    ax1.set_axisbelow(True)
    ax1.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", ncol=2)
    ax1.bar_label(bars_ep, padding=2, fontsize=8)
    ax2.bar_label(bars_fr, padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task2_official_dataset_sizes.png", dpi=220)
    fig.savefig(FIG_DIR / "task2_official_dataset_sizes.pdf")
    plt.close(fig)


def plot_split_samples() -> None:
    fig, axes = plt.subplots(len(SPLITS), 2, figsize=(6.8, 8.2))
    for row, (split, root) in enumerate(SPLITS.items()):
        df = pd.read_parquet(first_parquet(root))
        sample = df.iloc[len(df) // 2]
        for col, title, ax in [
            ("image", "static camera", axes[row, 0]),
            ("wrist_image", "wrist camera", axes[row, 1]),
        ]:
            ax.imshow(decode_image(sample[col]))
            ax.set_title(f"Env {split} - {title}", fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task2_official_split_samples.png", dpi=220)
    plt.close(fig)


def read_actions(root: Path, max_files: int = 4) -> np.ndarray:
    arrays = []
    for path in sorted((root / "data").rglob("*.parquet"))[:max_files]:
        df = pd.read_parquet(path, columns=["action"])
        arrays.extend(np.asarray(v, dtype=np.float32) for v in df["action"].tolist())
    return np.vstack(arrays)


def plot_action_stats() -> None:
    names = []
    means = []
    stds = []
    for name, root in DATASETS.items():
        actions = read_actions(root)
        names.append(name)
        means.append(np.mean(np.abs(actions), axis=0))
        stds.append(np.std(actions, axis=0))
    dims = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    x = np.arange(len(dims))
    width = 0.11
    palette = [PASTEL["teal"], PASTEL["peach"], PASTEL["lavender"], PASTEL["blue"], PASTEL["pink"]]
    offsets = (np.arange(len(names)) - (len(names) - 1) / 2) * width
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    for i, name in enumerate(names):
        color = palette[i % len(palette)]
        axes[0].bar(x + offsets[i], means[i], width=width * 0.82, label=name, color=color, edgecolor="#ffffff", linewidth=0.8)
        axes[1].bar(x + offsets[i], stds[i], width=width * 0.82, label=name, color=color, edgecolor="#ffffff", linewidth=0.8)
    axes[0].set_title("Mean absolute action")
    axes[1].set_title("Action standard deviation")
    for ax in axes:
        ax.set_xticks(x, dims, rotation=25)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task2_official_action_stats.png", dpi=220)
    fig.savefig(FIG_DIR / "task2_official_action_stats.pdf")
    plt.close(fig)


def plot_training_loss() -> None:
    frames = []
    for name, path in ONLINE_METRICS.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not df.empty:
            df["model"] = name
            frames.append(df)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(FIG_DIR / "task2_official_training_loss.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    for model, df in data.groupby("model"):
        axes[0].plot(df["step"], df["train_loss"], lw=1.5, label=model)
        axes[1].plot(df["step"], df["train_l1_loss"], lw=1.5, label=model)
    axes[0].set_title("Training total loss")
    axes[1].set_title("Training Action L1")
    for ax in axes:
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task2_official_training_loss.png", dpi=220)
    fig.savefig(FIG_DIR / "task2_official_training_loss.pdf")
    plt.close(fig)

    val = data.dropna(subset=["val_l1_loss"]).copy()
    if not val.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
        for model, df in val.groupby("model"):
            axes[0].plot(df["step"], df["val_l1_loss"], marker="o", lw=1.8, label=model)
            axes[1].plot(df["step"], df["val_loss"], marker="o", lw=1.8, label=model)
        axes[0].set_title("Held-out validation Action L1")
        axes[1].set_title("Held-out validation total loss")
        for ax in axes:
            ax.set_xlabel("Training step")
            ax.set_ylabel("Validation metric")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "task2_official_validation_metrics.png", dpi=220)
        fig.savefig(FIG_DIR / "task2_official_validation_metrics.pdf")
        plt.close(fig)


def plot_eval_metrics() -> None:
    rows = []
    for name, path in EVALS.items():
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append((name, data.get("mean_action_l1"), data.get("mean_loss")))
    if not rows:
        return
    labels = [r[0] for r in rows]
    l1 = [r[1] for r in rows]
    loss = [r[2] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.12
    bars_l1 = ax.bar(x - width / 2, l1, width=width, label="Action L1", color=PASTEL["teal"], edgecolor="#ffffff", linewidth=0.9)
    bars_loss = ax.bar(x + width / 2, loss, width=width, label="Mean loss", color=PASTEL["pink"], edgecolor="#ffffff", linewidth=0.9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("D-set offline metric")
    ax.set_title("Zero-shot Evaluation on Official Env D")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax.bar_label(bars_l1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars_loss, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylim(0, max(l1 + loss) * 1.18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task2_official_eval_metrics.png", dpi=220)
    fig.savefig(FIG_DIR / "task2_official_eval_metrics.pdf")
    plt.close(fig)


def main() -> None:
    plot_dataset_sizes()
    plot_split_samples()
    plot_action_stats()
    plot_training_loss()
    plot_eval_metrics()


if __name__ == "__main__":
    main()
