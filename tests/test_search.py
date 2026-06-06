"""Kiểm tra minimax: tìm chiếu hết bắt buộc và không trả về nước đi bất hợp lệ."""

from src.search import get_best_move_minimax
import chess

def test_mate_in_one():
    # Scholar's mate setup: White có Qh5, Black phải chặn hoặc thua
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    move = get_best_move_minimax(board, depth=2)
    assert move.uci() == "h5f7"  # Qxf7# checkmate

def test_no_illegal_moves():
    for _ in range(20):
        board = chess.Board()
        for _ in range(10):
            if board.is_game_over(): break
            move = get_best_move_minimax(board, depth=2)
            assert move in board.legal_moves
            board.push(move)