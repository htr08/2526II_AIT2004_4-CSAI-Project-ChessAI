"""
minimax.py
----------
Negamax + Alpha-Beta pruning với move ordering từ policy net (optional).

API chính: search_best_move(board, depth, model=None) → chess.Move
"""
from __future__ import annotations

import math
from typing import Optional

import chess

from .evaluation import evaluate_board
from ..data.action_space import move_to_index

# torch + encode_board được lazy-import vào _policy_priors để minimax có thể
# chạy được khi không có torch (chỉ cần khi truyền model vào).


INF = 10**9


def _policy_priors(
    board: chess.Board,
    model,
    device: str = "cpu",
) -> dict[chess.Move, float]:
    """
    Lấy policy logits từ model cho board hiện tại, trả về dict move → prior prob.
    Chỉ lấy các move legal. Dùng cho move ordering trong alpha-beta.
    """
    if model is None:
        return {}
    # Lazy import — chỉ cần khi thực sự dùng model
    import torch
    import torch.nn.functional as F
    from ..data.encode_board import board_to_tensor_perspective

    model.eval()
    with torch.no_grad():
        x = board_to_tensor_perspective(board).unsqueeze(0).to(device)
        out = model(x)
        # Hỗ trợ cả PolicyNet (single output) và PolicyValueNet (tuple)
        logits = out[0] if isinstance(out, tuple) else out
        probs = F.softmax(logits.squeeze(0), dim=-1).cpu().numpy()

    priors = {}
    for move in board.legal_moves:
        idx = move_to_index(move)
        priors[move] = float(probs[idx])
    return priors


def _order_moves(
    board: chess.Board,
    priors: Optional[dict[chess.Move, float]] = None,
) -> list[chess.Move]:
    """
    Sắp xếp moves để alpha-beta cắt nhánh tốt hơn.
    Ưu tiên:
    1. Policy prior (nếu có model)
    2. Capture moves (MVV-LVA — kẹp giá trị bị bắt cao trước)
    3. Còn lại
    """
    moves = list(board.legal_moves)

    if priors:
        # Sort theo prior descending
        moves.sort(key=lambda m: priors.get(m, 0.0), reverse=True)
        return moves

    # Không có model → MVV-LVA + checks first
    def move_score(m: chess.Move) -> int:
        s = 0
        if board.is_capture(m):
            captured = board.piece_at(m.to_square)
            if captured is not None:
                s += 10 * captured.piece_type
            attacker = board.piece_at(m.from_square)
            if attacker is not None:
                s -= attacker.piece_type
            s += 1000  # captures first
        if m.promotion:
            s += 900
        if board.gives_check(m):
            s += 50
        return s

    moves.sort(key=move_score, reverse=True)
    return moves


def negamax(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    color: int,
    model=None,
    device: str = "cpu",
) -> int:
    """
    Negamax với alpha-beta pruning.
    color = +1 nếu node hiện tại là side-to-move của TRẮNG (max),
            -1 nếu là ĐEN (sẽ negate score).
    Trả về evaluation score từ góc nhìn của side-to-move.
    """
    if board.is_game_over(claim_draw=True):
        # Terminal evaluation
        if board.is_checkmate():
            # Side-to-move thua → score rất âm từ góc nhìn của họ
            return -100000 + (10 - depth)  # prefer mate trong ít nước hơn
        return 0  # draw

    if depth == 0:
        return color * evaluate_board(board)

    priors = _policy_priors(board, model, device) if model is not None else None
    moves = _order_moves(board, priors)

    best = -INF
    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, -color, model, device)
        board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break  # beta cutoff
    return best


def search_best_move(
    board: chess.Board,
    depth: int = 3,
    model=None,
    device: str = "cpu",
) -> tuple[Optional[chess.Move], int]:
    """
    Root call: trả về (best_move, score).
    Score là centipawns từ góc nhìn của side-to-move.
    """
    if board.is_game_over(claim_draw=True):
        return None, 0

    color = 1 if board.turn == chess.WHITE else -1
    priors = _policy_priors(board, model, device) if model is not None else None
    moves = _order_moves(board, priors)

    best_move = None
    best_score = -INF
    alpha = -INF
    beta = INF

    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, -color, model, device)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score

    return best_move, best_score


if __name__ == "__main__":
    board = chess.Board()
    move, score = search_best_move(board, depth=3)
    print(f"Best from start: {move.uci()}  score={score}")

    # Mate in 1: scholar's mate position
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3")
    move, score = search_best_move(board, depth=2)
    print(f"Position 2 best: {move.uci()}  score={score}")
