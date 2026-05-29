"""Tests cho src.data.action_space."""
import chess
import pytest

from src.data.action_space import (
    NUM_ACTIONS,
    move_to_index,
    index_to_move,
    uci_to_index,
    legal_move_mask,
)


def test_size():
    assert NUM_ACTIONS == 4096


def test_simple_roundtrip():
    board = chess.Board()
    moves = ["e2e4", "g1f3", "d2d4", "b1c3"]
    for uci in moves:
        idx = uci_to_index(uci)
        assert 0 <= idx < NUM_ACTIONS
        # Trên board mới (chưa play move), decode lại đúng UCI
        b = chess.Board()
        back = index_to_move(idx, b).uci()
        assert back == uci, f"roundtrip failed: {uci} → {idx} → {back}"


def test_promotion():
    """Pawn vào rank cuối → index_to_move tự promote queen."""
    # Board: white pawn e7, ready to promote
    board = chess.Board("8/4P3/8/8/8/8/8/k6K w - - 0 1")
    idx = uci_to_index("e7e8q")
    back = index_to_move(idx, board)
    assert back.uci() == "e7e8q"
    assert back.promotion == chess.QUEEN


def test_legal_mask_starting():
    board = chess.Board()
    mask = legal_move_mask(board)
    assert len(mask) == 20  # 20 legal moves trong starting position
    # Mọi index trong range hợp lệ
    for idx in mask:
        assert 0 <= idx < NUM_ACTIONS


def test_castling_index():
    """Castling (O-O) là king e1 → g1."""
    board = chess.Board("r3k2r/pppqppbp/2np1np1/8/8/2NP1NP1/PPPQPPBP/R3K2R w KQkq - 0 1")
    idx = uci_to_index("e1g1")
    assert idx == 4 * 64 + 6  # e1 = square 4, g1 = square 6
