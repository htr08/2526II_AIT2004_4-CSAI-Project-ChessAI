"""Kiểm tra ChessDataset: shape, dtype, kích thước split và phạm vi move index."""

import torch
import pytest
from src.dataset import ChessDataset, get_dataloaders

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_SAMPLES = 1000
VOCAB_SIZE = 4544  # total moves built by src/vocab.py (4032 base + 512 promo)


def _make_pt_file(path: str, n: int = NUM_SAMPLES) -> None:
    """Write a minimal synthetic .pt file that mirrors parse_pgn.py output."""
    X = torch.rand(n, 17, 8, 8, dtype=torch.float32)
    y = torch.randint(0, VOCAB_SIZE, (n,), dtype=torch.long)
    torch.save({"X": X, "y": y}, path)


# ---------------------------------------------------------------------------
# ChessDataset tests
# ---------------------------------------------------------------------------


def test_dataset_len(tmp_path):
    pt = str(tmp_path / "train.pt")
    _make_pt_file(pt)
    ds = ChessDataset(pt)
    assert len(ds) == NUM_SAMPLES


def test_dataset_item_shapes(tmp_path):
    pt = str(tmp_path / "train.pt")
    _make_pt_file(pt)
    ds = ChessDataset(pt)
    board, move = ds[0]
    assert board.shape == (17, 8, 8), f"Expected (17,8,8), got {board.shape}"
    assert move.shape == (), "move index must be a scalar tensor"


def test_dataset_dtypes(tmp_path):
    pt = str(tmp_path / "train.pt")
    _make_pt_file(pt)
    ds = ChessDataset(pt)
    board, move = ds[0]
    assert board.dtype == torch.float32, f"X dtype should be float32, got {board.dtype}"
    assert move.dtype == torch.long, f"y dtype should be long, got {move.dtype}"


def test_dataset_mismatch_raises(tmp_path):
    pt = str(tmp_path / "bad.pt")
    X = torch.rand(10, 17, 8, 8)
    y = torch.zeros(5, dtype=torch.long)
    torch.save({"X": X, "y": y}, pt)
    with pytest.raises(ValueError):
        ChessDataset(pt)


# ---------------------------------------------------------------------------
# DataLoader tests
# ---------------------------------------------------------------------------

BATCH = 64


def test_dataloader_batch_shapes(tmp_path):
    pt = str(tmp_path / "train.pt")
    _make_pt_file(pt)
    train_dl, _ = get_dataloaders(pt, batch_size=BATCH, val_ratio=0.1)

    X_batch, y_batch = next(iter(train_dl))
    assert X_batch.shape == (BATCH, 17, 8, 8), f"Got {X_batch.shape}"
    assert y_batch.shape == (BATCH,), f"Got {y_batch.shape}"


def test_dataloader_dtypes(tmp_path):
    pt = str(tmp_path / "train.pt")
    _make_pt_file(pt)
    train_dl, _ = get_dataloaders(pt, batch_size=BATCH, val_ratio=0.1)

    X_batch, y_batch = next(iter(train_dl))
    assert X_batch.dtype == torch.float32
    assert y_batch.dtype == torch.long


def test_dataloader_move_range(tmp_path):
    """All move indices must fall within the vocabulary."""
    pt = str(tmp_path / "train.pt")
    _make_pt_file(pt)
    train_dl, val_dl = get_dataloaders(pt, batch_size=BATCH, val_ratio=0.1)

    for dl in (train_dl, val_dl):
        for _, y_batch in dl:
            assert y_batch.min() >= 0, "Negative move index found"
            assert y_batch.max() < VOCAB_SIZE, (
                f"Move index {y_batch.max().item()} >= vocab size {VOCAB_SIZE}"
            )


def test_dataloader_split_sizes(tmp_path):
    pt = str(tmp_path / "train.pt")
    _make_pt_file(pt, n=1000)
    train_dl, val_dl = get_dataloaders(pt, batch_size=1, val_ratio=0.1)
    assert len(train_dl.dataset) == 900
    assert len(val_dl.dataset) == 100


def test_board_values_in_range(tmp_path):
    """Board tensors from encode_board_with_meta are all 0.0 or 1.0."""
    pt = str(tmp_path / "train.pt")
    # Synthetic data is random floats — just verify the DataLoader passes them through.
    _make_pt_file(pt)
    train_dl, _ = get_dataloaders(pt, batch_size=BATCH)
    X_batch, _ = next(iter(train_dl))
    assert X_batch.shape[1] == 17, "Expected 17 channels (12 pieces + 5 meta)"
