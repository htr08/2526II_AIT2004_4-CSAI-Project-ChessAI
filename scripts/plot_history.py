"""
plot_history.py
---------------
Plot training curves từ checkpoint history hoặc self-play log.

Usage:
    python scripts/plot_history.py --ckpt models/best.pt --output reports/supervised.png
    python scripts/plot_history.py --selfplay-dir models/selfplay --output reports/selfplay.png
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def plot_supervised(ckpt_path: str, output: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    history = ckpt.get("history", [])
    if not history:
        print(f"[plot] no history found in {ckpt_path}")
        return

    epochs = [h["epoch"] for h in history]
    tr_p = [h["train"]["policy_loss"] for h in history]
    tr_v = [h["train"]["value_loss"] for h in history]
    val_p = [h["val"]["policy_loss"] for h in history]
    val_v = [h["val"]["value_loss"] for h in history]
    val_top1 = [h["val"]["top1"] for h in history]
    val_top5 = [h["val"]["top5"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, tr_p, label="train", marker="o")
    axes[0].plot(epochs, val_p, label="val", marker="s")
    axes[0].set_title("Policy loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, tr_v, label="train", marker="o")
    axes[1].plot(epochs, val_v, label="val", marker="s")
    axes[1].set_title("Value loss (MSE)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, val_top1, label="top-1", marker="o")
    axes[2].plot(epochs, val_top5, label="top-5", marker="s")
    axes[2].set_title("Validation accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=120)
    print(f"[plot] saved → {output}")


def plot_selfplay(selfplay_dir: str, output: str) -> None:
    dir = pathlib.Path(selfplay_dir)
    iters = sorted(dir.glob("iter_*.pt"))
    if not iters:
        print(f"[plot] no iter_*.pt files trong {dir}")
        return

    iter_nums = []
    win_rates = []
    accepted = []
    policy_losses = []
    value_losses = []
    elos = []

    for path in iters:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        iter_nums.append(ckpt["iteration"])
        win_rates.append(ckpt["pit"]["decisive_win_rate"])
        accepted.append(ckpt["accepted"])
        policy_losses.append(ckpt["train_metrics"]["policy_loss"])
        value_losses.append(ckpt["train_metrics"]["value_loss"])
        elos.append(ckpt.get("elo"))  # None nếu checkpoint cũ chưa có ELO

    has_elo = all(e is not None for e in elos)
    n_panels = 3 if has_elo else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))

    colors = ["g" if a else "r" for a in accepted]
    axes[0].bar(iter_nums, win_rates, color=colors, alpha=0.7)
    axes[0].axhline(0.55, color="k", linestyle="--", label="accept threshold")
    axes[0].set_title("Pit decisive win rate per iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Win rate (excluding draws)")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iter_nums, policy_losses, label="policy loss", marker="o")
    axes[1].plot(iter_nums, value_losses, label="value loss", marker="s")
    axes[1].set_title("Self-play training losses")
    axes[1].set_xlabel("Iteration")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if has_elo:
        axes[2].plot(iter_nums, elos, label="relative ELO", marker="o", color="purple")
        axes[2].set_title("Relative ELO over iterations")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("ELO (start = 0)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=120)
    print(f"[plot] saved → {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", help="Supervised checkpoint (best.pt or latest.pt)")
    p.add_argument("--selfplay-dir", help="Folder chứa iter_*.pt")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if args.ckpt:
        plot_supervised(args.ckpt, args.output)
    elif args.selfplay_dir:
        plot_selfplay(args.selfplay_dir, args.output)
    else:
        print("Cần --ckpt hoặc --selfplay-dir")
