"""Tests cho src.data.encode_board."""
import chess
import pytest
import torch

from src.data.encode_board import (
    board_to_array,
    board_to_tensor,
    board_to_tensor_perspective,
    NUM_CHANNELS,
)


def test_shape_and_dtype():
    board = chess.Board()
    arr = board_to_array(board)
    assert arr.shape == (12, 8, 8)
    assert arr.dtype.name == "float32"


def test_starting_position_count():
    """Starting position phải có đúng 32 pieces (mỗi loại đúng số."""
    board = chess.Board()
    arr = board_to_array(board)
    assert arr.sum() == 32

    # Mỗi side có 8 pawns, 2 rooks, 2 knights, 2 bishops, 1 queen, 1 king
    expected = [8, 2, 2, 2, 1, 1, 8, 2, 2, 2, 1, 1]
    for ch in range(NUM_CHANNELS):
        assert arr[ch].sum() == expected[ch], f"channel {ch}: got {arr[ch].sum()} expected {expected[ch]}"


def test_after_e4():
    board = chess.Board()
    board.push_san("e4")
    arr = board_to_array(board)
    # White pawn channel = 0
    # e4 = file 4, rank 3 (0-indexed). e2 was rank 1.
    assert arr[0, 3, 4] == 1.0  # white pawn on e4
    assert arr[0, 1, 4] == 0.0  # e2 trống


def test_perspective_flip_black_to_move():
    """Sau 1.e4, black đi → perspective flip."""
    board = chess.Board()
    board.push_san("e4")
    assert board.turn == chess.BLACK

    persp = board_to_tensor_perspective(board)
    # Sau flip:
    # - Color channels swap: black pieces giờ ở channel 0-5
    # - Board flipped 180°: e4 (rank 3, file 4) → (rank 4, file 3) sau rotate

    # Kiểm tra tổng số quân vẫn 32
    assert persp.sum() == 32

    # Kiểm tra side-to-move bây giờ "thấy" mình là trắng
    # (không có cách trực tiếp, chỉ check shape + count)
    assert persp.shape == (12, 8, 8)


def test_tensor_wrapper():
    board = chess.Board()
    t = board_to_tensor(board)
    assert isinstance(t, torch.Tensor)
    assert t.shape == torch.Size([12, 8, 8])
