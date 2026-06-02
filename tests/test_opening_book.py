"""Tests cho opening book (không cần torch)."""
import random

import chess

from src.search.opening_book import book_move, in_book, OPENING_LINES


def test_book_first_move_legal_and_known():
    """Nước đầu từ sách phải hợp lệ và nằm trong các nước mở đầu đã định."""
    board = chess.Board()
    mv = book_move(board, rng=random.Random(0))
    assert mv is not None
    assert mv in board.legal_moves
    first_moves = {line[0] for line in OPENING_LINES}
    assert mv.uci() in first_moves


def test_book_follows_a_known_line():
    """Đi theo đúng một line → mỗi bước sách vẫn có nước hợp lệ."""
    line = OPENING_LINES[0]  # Ruy Lopez
    board = chess.Board()
    for uci in line[:-1]:
        board.push_uci(uci)
        mv = book_move(board)
        assert mv is not None
        assert mv in board.legal_moves


def test_book_returns_none_off_book():
    """Thế cờ lạ (không khớp line nào) → None."""
    board = chess.Board()
    board.push_uci("a2a3")  # nước hiếm, không có trong sách
    board.push_uci("a7a6")
    assert book_move(board) is None
    assert not in_book(board)


def test_book_move_matches_played_prefix():
    """Sau 1.e4 c5, sách phải gợi ý nước hợp lệ của một line Sicilian."""
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("c7c5")
    mv = book_move(board, rng=random.Random(1))
    assert mv is not None and mv in board.legal_moves
