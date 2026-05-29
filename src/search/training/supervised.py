"""
supervised.py
-------------
Training loop supervised cho PolicyValueNet trên dữ liệu PGN.

Loss = policy_loss (cross-entropy) + value_weight * value_loss (MSE)

Metrics:
- policy top-1 accuracy
- policy top-5 accuracy
- value MAE (mean absolute error)
- legal move rate (% các nước argmax có legal trên position đó)
"""
from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field
from typing import Optional

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...data.action_space import legal_move_mask
from ...data.dataset import build_dataloaders
from ...model.network import PolicyValueNet


@dataclass
class TrainConfig:
    data_path: str = "data/processed/train.pt"
    output_dir: str = "models"
    epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    channels: int = 128
    n_res_blocks: int = 3
    num_workers: int = 2
    val_split: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 50
    save_best: bool = True
    history: list = field(default_factory=list)


def _topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Top-k accuracy (theo policy logits)."""
    _, pred_topk = logits.topk(k, dim=-1)
    correct = pred_topk.eq(targets.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


def evaluate(
    model: PolicyValueNet,
    loader: DataLoader,
    device: str,
    value_weight: float = 1.0,
) -> dict:
    model.eval()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_value_mae = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, policy, value in loader:
            x = x.to(device, non_blocking=True)
            policy = policy.to(device, non_blocking=True)
            value = value.to(device, non_blocking=True)

            policy_logits, value_pred = model(x)
            policy_loss = F.cross_entropy(policy_logits, policy)
            value_loss = F.mse_loss(value_pred, value)

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_top1 += _topk_accuracy(policy_logits, policy, k=1)
            total_top5 += _topk_accuracy(policy_logits, policy, k=5)
            total_value_mae += (value_pred - value).abs().mean().item()
            n_batches += 1

    return {
        "policy_loss": total_policy_loss / max(n_batches, 1),
        "value_loss": total_value_loss / max(n_batches, 1),
        "top1": total_top1 / max(n_batches, 1),
        "top5": total_top5 / max(n_batches, 1),
        "value_mae": total_value_mae / max(n_batches, 1),
    }


def train(cfg: TrainConfig) -> dict:
    output_dir = pathlib.Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] device={cfg.device}")
    print(f"[train] loading data from {cfg.data_path}")
    train_loader, val_loader = build_dataloaders(
        cfg.data_path,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
        num_workers=cfg.num_workers,
    )

    model = PolicyValueNet(channels=cfg.channels, n_res_blocks=cfg.n_res_blocks)
    model.to(cfg.device)
    print(f"[train] model params: {sum(p.numel() for p in model.parameters()):,}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cfg.epochs * len(train_loader)
    )

    best_top1 = 0.0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = {"policy_loss": 0.0, "value_loss": 0.0, "top1": 0.0, "top5": 0.0}
        n = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}")
        for batch_idx, (x, policy, value) in enumerate(pbar):
            x = x.to(cfg.device, non_blocking=True)
            policy = policy.to(cfg.device, non_blocking=True)
            value = value.to(cfg.device, non_blocking=True)

            policy_logits, value_pred = model(x)
            policy_loss = F.cross_entropy(policy_logits, policy)
            value_loss = F.mse_loss(value_pred, value)
            loss = policy_loss + cfg.value_weight * value_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()

            running["policy_loss"] += policy_loss.item()
            running["value_loss"] += value_loss.item()
            running["top1"] += _topk_accuracy(policy_logits, policy, 1)
            running["top5"] += _topk_accuracy(policy_logits, policy, 5)
            n += 1

            if batch_idx % cfg.log_interval == 0:
                pbar.set_postfix(
                    p_loss=f"{running['policy_loss']/n:.3f}",
                    v_loss=f"{running['value_loss']/n:.3f}",
                    top1=f"{running['top1']/n:.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.1e}",
                )

        train_metrics = {k: v / max(n, 1) for k, v in running.items()}
        val_metrics = evaluate(model, val_loader, cfg.device, cfg.value_weight)

        dt = time.time() - t0
        print(
            f"[epoch {epoch}] {dt:.0f}s  "
            f"train: p_loss={train_metrics['policy_loss']:.3f} "
            f"v_loss={train_metrics['value_loss']:.3f} "
            f"top1={train_metrics['top1']:.3f} top5={train_metrics['top5']:.3f}"
        )
        print(
            f"           "
            f"val: p_loss={val_metrics['policy_loss']:.3f} "
            f"v_loss={val_metrics['value_loss']:.3f} "
            f"top1={val_metrics['top1']:.3f} top5={val_metrics['top5']:.3f} "
            f"v_mae={val_metrics['value_mae']:.3f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": scheduler.get_last_lr()[0],
                "time_s": dt,
            }
        )

        # Save latest
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": cfg.__dict__,
                "epoch": epoch,
                "history": history,
            },
            output_dir / "latest.pt",
        )

        if cfg.save_best and val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.__dict__,
                    "epoch": epoch,
                    "val_top1": best_top1,
                    "history": history,
                },
                output_dir / "best.pt",
            )
            print(f"[train] saved best.pt (val top1={best_top1:.3f})")

    return {"best_top1": best_top1, "history": history}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/train.pt")
    p.add_argument("--output-dir", default="models")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--n-res", type=int, default=3)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = TrainConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        channels=args.channels,
        n_res_blocks=args.n_res,
        value_weight=args.value_weight,
        num_workers=args.workers,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    train(cfg)
