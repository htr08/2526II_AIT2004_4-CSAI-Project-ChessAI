"""Kiểm tra encode_board và encode_board_with_meta: shape, dtype, vị trí quân cờ."""

import chess, torch
from src.board import encode_board, encode_board_with_meta

def test_shape():
    board = chess.Board()
    t = encode_board(board)
    assert t.shape == (12, 8, 8)
    assert t.dtype == torch.float32

def test_white_pawns_start():
    board = chess.Board()
    t = encode_board(board)
    # Channel 0 = White Pawns, row index 1 = rank 2 (a2-h2)
    assert t[0, 1, :].sum() == 8

def test_black_pawns_start():
    board = chess.Board()
    t = encode_board(board)
    # Channel 6 = Black Pawns, row index 6 = rank 7 (a7-h7)
    assert t[6, 6, :].sum() == 8

def test_empty_board():
    board = chess.Board(fen="8/8/8/8/8/8/8/8 w - - 0 1")
    t = encode_board(board)
    assert t.sum() == 0

def test_piece_moves_correctly():
    board = chess.Board()
    board.push_uci("e2e4")
    t = encode_board(board)
    # Tốt trắng phải rời e2 (row=1, col=4) và đến e4 (row=3, col=4)
    assert t[0, 1, 4] == 0.0
    assert t[0, 3, 4] == 1.0

def test_meta_shape():
    board = chess.Board()
    t = encode_board_with_meta(board)
    assert t.shape == (17, 8, 8)
    assert t.dtype == torch.float32

def test_meta_base_channels_match():
    board = chess.Board()
    base = encode_board(board)
    meta = encode_board_with_meta(board)
    assert torch.equal(meta[:12], base)

def test_meta_turn_white():
    board = chess.Board()
    t = encode_board_with_meta(board)
    assert t[12].unique().item() == 1.0

def test_meta_turn_black():
    board = chess.Board()
    board.push_uci("e2e4")
    t = encode_board_with_meta(board)
    assert t[12].unique().item() == 0.0

def test_meta_castling_rights_start():
    board = chess.Board()
    t = encode_board_with_meta(board)
    # Vị trí xuất phát: cả 4 quyền nhập thành đều có
    assert t[13].unique().item() == 1.0  # White Kingside
    assert t[14].unique().item() == 1.0  # White Queenside
    assert t[15].unique().item() == 1.0  # Black Kingside
    assert t[16].unique().item() == 1.0  # Black Queenside

def test_meta_castling_rights_lost():
    # FEN không có quyền nhập thành (ký tự '-')
    board = chess.Board(fen="8/8/8/8/8/8/8/4K3 w - - 0 1")
    t = encode_board_with_meta(board)
    assert t[13].unique().item() == 0.0  # White Kingside mất
    assert t[14].unique().item() == 0.0  # White Queenside mất
