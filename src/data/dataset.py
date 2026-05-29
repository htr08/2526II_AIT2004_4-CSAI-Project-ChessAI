"""
dataset.py
----------
PyTorch Dataset + DataLoader cho training data.

Đọc .pt file (sản phẩm của pgn_parser.py), trả về (board_tensor, move_index, result).
"""
from __future__ import annotations

import pathlib
from typing import Optional

import chess
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .encode_board import board_to_tensor_perspective
from .action_space import uci_to_index


class ChessDataset(Dataset):
    """
    Mỗi sample = (board_tensor 12×8×8, policy_target int, value_target float).

    policy_target: index trong [0, 4095] của nước đi expert
    value_target: 1.0 nếu side-to-move THẮNG, -1.0 nếu THUA, 0 nếu hòa
                  (đổi dấu dựa trên side-to-move để value head luôn dự đoán
                   từ góc nhìn của bên đang đi)
    """

    def __init__(
        self,
        pt_path: str | pathlib.Path,
        perspective: bool = True,
    ):
        data = torch.load(pt_path, weights_only=False)
        self.fens: list[str] = data["fens"]
        self.moves: list[str] = data["moves"]
        self.results: list[int] = data["results"]
        self.perspective = perspective

        assert (
            len(self.fens) == len(self.moves) == len(self.results)
        ), "Mismatched dataset lengths"

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, idx: int):
        board = chess.Board(self.fens[idx])
        if self.perspective:
            x = board_to_tensor_perspective(board)
        else:
            from .encode_board import board_to_tensor
            x = board_to_tensor(board)

        move = chess.Move.from_uci(self.moves[idx])
        policy = uci_to_index(self.moves[idx])

        # Value target: đổi dấu theo side-to-move
        # result: 1 = white wins, -1 = black wins, 0 = draw
        # value (from side-to-move POV) = result nếu trắng đi, -result nếu đen đi
        result = self.results[idx]
        if board.turn == chess.WHITE:
            value = float(result)
        else:
            value = float(-result)

        return x, policy, torch.tensor(value, dtype=torch.float32)


def build_dataloaders(
    pt_path: str | pathlib.Path,
    batch_size: int = 256,
    val_split: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Split 90/10 train/val, return DataLoader pair."""
    full = ChessDataset(pt_path)
    n_val = int(len(full) * val_split)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(
        f"[dataset] train={n_train:,}  val={n_val:,}  "
        f"batch={batch_size}  workers={num_workers}"
    )
    return train_loader, val_loader


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.data.dataset <path_to.pt>")
        sys.exit(0)
    ds = ChessDataset(sys.argv[1])
    print(f"Dataset size: {len(ds):,}")
    x, policy, value = ds[0]
    print(f"Sample 0: x.shape={tuple(x.shape)}, policy={policy}, value={value.item()}")
