#!/usr/bin/env python3
"""Plot train and validation loss for one pose-v2 LoRA trial.

Reads metrics.jsonl written by scripts/temp/finetune_pose_v2.py.
The figure is saved next to that file as loss.png.

  python3 scripts/temp/plot_pose_v2_trial.py
  python3 scripts/temp/plot_pose_v2_trial.py runs/ft_pose_v2/trial_20261029-173812-016396
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT / "runs/ft_pose_v2"


def configure_font() -> None:
    for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.family"] = family
        break
    plt.rcParams["axes.unicode_minus"] = False


def latest_trial(run_root: Path) -> Path:
    trials = sorted(p for p in run_root.glob("trial_*") if (p / "metrics.jsonl").is_file())
    if not trials:
        raise SystemExit(f"no trial with metrics.jsonl under {run_root}")
    return trials[-1]


def load_metrics(path: Path) -> tuple[list[dict], list[dict]]:
    train, val = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("phase") == "train":
            train.append(row)
        elif row.get("phase") == "val":
            val.append(row)
    if not train or not val:
        raise SystemExit(f"{path} is missing train or validation rows")
    return train, val


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        count = min(index + 1, window)
        out.append(total / count)
    return out


def best_step(run_dir: Path, val: list[dict]) -> tuple[int, float]:
    best_path = run_dir / "best.json"
    if best_path.is_file():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        return int(best["step"]), float(best["loss"])
    chosen = min(val, key=lambda row: float(row["loss"]))
    return int(chosen["step"]), float(chosen["loss"])


def plot_loss(run_dir: Path, window: int, output: Path) -> None:
    train, val = load_metrics(run_dir / "metrics.jsonl")
    train_steps = [int(row["step"]) for row in train]
    train_loss = [float(row["loss"]) for row in train]
    smooth = rolling_mean(train_loss, window)
    val_steps = [int(row["step"]) for row in val]
    val_loss = [float(row["loss"]) for row in val]
    chosen_step, chosen_loss = best_step(run_dir, val)
    baseline = val_loss[0]

    configure_font()
    figure, axes = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True)
    figure.suptitle(run_dir.name, fontsize=13)

    axes[0].plot(train_steps, train_loss, color="#9bb7d4", linewidth=0.7, label="每步训练 loss")
    axes[0].plot(train_steps, smooth, color="#1f4e79", linewidth=1.6, label=f"{window} 步滑动平均")
    axes[0].set_ylabel("训练 loss")
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].plot(val_steps, val_loss, color="#b85c38", marker="o", linewidth=1.5, label="验证 loss")
    axes[1].axvline(chosen_step, color="#2f6f4e", linestyle="--", linewidth=1.1, label=f"选用第 {chosen_step} 步")
    axes[1].scatter([chosen_step], [chosen_loss], color="#2f6f4e", zorder=3)
    axes[1].annotate(
        f"起点 {baseline:.2f}\n最低 {chosen_loss:.2f}",
        xy=(chosen_step, chosen_loss),
        xytext=(12, 18),
        textcoords="offset points",
        fontsize=9,
        color="#2f6f4e",
    )
    axes[1].set_xlabel("参数更新步数")
    axes[1].set_ylabel("验证 loss")
    axes[1].legend(loc="upper right", frameon=False)
    axes[1].grid(True, axis="y", alpha=0.3)

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, help="trial directory; default is the newest one")
    parser.add_argument("--window", type=int, default=20, help="train-loss moving average, in steps")
    parser.add_argument("--output", type=Path, help="png path; default is <run_dir>/loss.png")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    run_dir = args.run_dir if args.run_dir is not None else latest_trial(DEFAULT_RUN_ROOT)
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
    if not (run_dir / "metrics.jsonl").is_file():
        raise SystemExit(f"missing {run_dir / 'metrics.jsonl'}")
    plot_loss(run_dir, args.window, args.output or (run_dir / "loss.png"))


if __name__ == "__main__":
    main()
