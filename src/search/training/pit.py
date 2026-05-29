"""
pit.py
------
Cho hai phiên bản model đánh nhau N ván, trả về win rate cho model mới.
Dùng để quyết định trong self-play loop có giữ model mới không.

Tiêu chuẩn AlphaZero: nếu new wins ≥55% (không tính draws) thì giữ model mới.
"""
from __future__ import annotations

from typing import Optional

import chess
import torch

from ...model.network import PolicyValueNet
from ..mcts import MCTS


def play_match(
    new_model: PolicyValueNet,
    old_model: PolicyValueNet,
    num_games: int = 10,
    num_simulations: int = 100,
    device: str = "cpu",
    max_moves: int = 200,
    verbose: bool = False,
) -> dict:
    """
    Đánh num_games ván, mỗi side đi white phân nửa số ván.
    Trả về dict với wins/losses/draws cho new_model.
    """
    new_mcts = MCTS(
        new_model, device=device, num_simulations=num_simulations, add_noise=False
    )
    old_mcts = MCTS(
        old_model, device=device, num_simulations=num_simulations, add_noise=False
    )

    new_wins = 0
    old_wins = 0
    draws = 0

    for g in range(num_games):
        new_plays_white = g % 2 == 0
        board = chess.Board()
        move_num = 0
        while not board.is_game_over(claim_draw=True) and move_num < max_moves:
            if (board.turn == chess.WHITE) == new_plays_white:
                mcts = new_mcts
            else:
                mcts = old_mcts
            move, _, _ = mcts.get_action_probs(board, temperature=0.0)
            if move is None:
                break
            board.push(move)
            move_num += 1

        # Result từ góc nhìn của new_model
        if board.is_checkmate():
            white_lost = board.turn == chess.WHITE
            if white_lost and new_plays_white:
                old_wins += 1
            elif (not white_lost) and (not new_plays_white):
                old_wins += 1
            else:
                new_wins += 1
        else:
            draws += 1

        if verbose:
            print(
                f"  game {g+1}: new={'W' if new_plays_white else 'B'}  "
                f"score → new {new_wins}-{old_wins}, draws {draws}"
            )

    total = num_games
    win_rate = new_wins / total if total > 0 else 0.0
    # AlphaZero formula: chỉ tính decisive games
    decisive = new_wins + old_wins
    decisive_win_rate = new_wins / decisive if decisive > 0 else 0.5

    return {
        "new_wins": new_wins,
        "old_wins": old_wins,
        "draws": draws,
        "win_rate": win_rate,
        "decisive_win_rate": decisive_win_rate,
        "accept_threshold": 0.55,
        "accept": decisive_win_rate >= 0.55,
    }
