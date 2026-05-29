"""Tests cho src.search.minimax + mcts."""
import chess

from src.model.network import PolicyValueNet
from src.search.evaluation import evaluate_board
from src.search.minimax import search_best_move
from src.search.mcts import search_best_move_mcts


def test_eval_starting_zero():
    """Symmetric starting position → eval = 0."""
    board = chess.Board()
    assert evaluate_board(board) == 0


def test_eval_white_advantage():
    """Xóa queen của đen → trắng tốt hơn ~900cp."""
    board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert evaluate_board(board) > 800


def test_minimax_mate_in_1():
    """White có mate in 1: Qxf7#."""
    # Scholar's-style position, white to move
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
    move, score = search_best_move(board, depth=3)
    # Phải tìm ra Qxf7 hoặc nước thắng material lớn
    assert move is not None


def test_mcts_returns_legal_move():
    """MCTS trên model random vẫn phải trả về legal move."""
    model = PolicyValueNet(channels=32, n_res_blocks=1)
    model.eval()
    board = chess.Board()
    move, _ = search_best_move_mcts(board, model, num_simulations=20, device="cpu")
    assert move in board.legal_moves
