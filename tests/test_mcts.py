"""Kiểm tra MCTS UCB1 và PUCT: nước đi hợp lệ, prior policy, Dirichlet noise."""

from src.mcts import (mcts_search, mcts_search_puct, run_puct_search,
                      add_dirichlet_noise, select_move_with_temperature)
from src.model import DualNet
from src.vocab import load_or_build_move2idx
import chess, time

def test_mcts_legal_move():
    board = chess.Board()
    t0 = time.time()
    move = mcts_search(board, n_simulations=200)
    elapsed = time.time() - t0
    assert move in board.legal_moves
    assert elapsed < 10.0   # <10s cho 200 sims trên CPU
    print(f"200 sims in {elapsed:.2f}s → {move}")

def test_mcts_near_endgame():
    # Position gần checkmate: MCTS nên tìm được trong ít sims
    board = chess.Board("k7/8/1K6/8/8/8/8/7R w - - 0 1")
    move = mcts_search(board, n_simulations=500)
    assert move in board.legal_moves


# ── PUCT (AlphaZero) ──────────────────────────────────────────────────────────

def test_puct_uses_policy_prior():
    """expand_with_policy phải gán prior từ policy net (không đồng đều)."""
    model = DualNet(in_ch=17, num_moves=4544)
    move2idx = load_or_build_move2idx()
    root = run_puct_search(chess.Board(), model, move2idx, n_sims=30)
    priors = [c.prior for c in root.children]
    assert len(root.children) == 20            # 20 nước mở đầu hợp lệ
    assert len(set(priors)) > 1                # prior khác nhau -> policy net được dùng
    assert all(c.visits >= 0 for c in root.children)


def test_puct_returns_legal_move():
    model = DualNet(in_ch=17, num_moves=4544)
    move2idx = load_or_build_move2idx()
    board = chess.Board()
    move = mcts_search_puct(board, model, move2idx, n_sims=30)
    assert move in board.legal_moves


def test_dirichlet_noise_changes_priors():
    model = DualNet(in_ch=17, num_moves=4544)
    move2idx = load_or_build_move2idx()
    root = run_puct_search(chess.Board(), model, move2idx, n_sims=1)
    before = [c.prior for c in root.children]
    add_dirichlet_noise(root)
    after = [c.prior for c in root.children]
    assert before != after
    assert abs(sum(after) - sum(before)) < 1e-3   # vẫn xấp xỉ phân phối xác suất


def test_temperature_sampling_legal():
    model = DualNet(in_ch=17, num_moves=4544)
    move2idx = load_or_build_move2idx()
    board = chess.Board()
    root = run_puct_search(board, model, move2idx, n_sims=30, add_noise=True)
    early = select_move_with_temperature(root, move_count=0)    # sample
    late  = select_move_with_temperature(root, move_count=99)   # argmax
    assert early in board.legal_moves
    assert late in board.legal_moves