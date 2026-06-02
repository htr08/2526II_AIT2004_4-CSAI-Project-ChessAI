"""
self_play.py
------------
Sinh self-play games dùng MCTS với current model.
Mỗi game tạo ra list samples (state, policy_target, value_target):
- state: board tensor 12×8×8 (perspective)
- policy_target: probability distribution trên 4096 actions theo MCTS visit counts
- value_target: kết quả game cuối, đổi dấu theo side-to-move

Đây là data cho retrain trong self-play loop.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Optional

import chess
import numpy as np
import torch
from tqdm import tqdm

from ...data.action_space import move_to_index
from ...data.encode_board import board_to_tensor_perspective
from ...model.network import PolicyValueNet, NUM_ACTIONS
from ..mcts import MCTS
from ..evaluation import adjudicate_result


@dataclass
class SelfPlayConfig:
    num_games: int = 20
    num_simulations: int = 100
    max_moves: int = 200            # giới hạn để tránh game vô hạn
    temperature_threshold: int = 15  # 15 nước đầu temp=1 để explore, sau đó temp=0
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    adjudication_margin: int = 100   # material edge (cp) để chấm thắng khi cắt ở max_moves
    num_parallel: int = 1            # số leaf gom batch trong MCTS (1 = như cũ)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def play_one_game(
    model: PolicyValueNet, cfg: SelfPlayConfig, verbose: bool = False
) -> list[tuple[torch.Tensor, np.ndarray, float]]:
    """
    Chơi một self-play game, trả về list (state, policy_target_4096, value_target).
    """
    mcts = MCTS(
        model,
        device=cfg.device,
        c_puct=cfg.c_puct,
        num_simulations=cfg.num_simulations,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_epsilon=cfg.dirichlet_epsilon,
        add_noise=True,
        num_parallel=cfg.num_parallel,
    )

    board = chess.Board()
    history: list[tuple[torch.Tensor, np.ndarray, bool]] = []  # state, policy, turn

    move_num = 0
    while not board.is_game_over(claim_draw=True) and move_num < cfg.max_moves:
        temp = 1.0 if move_num < cfg.temperature_threshold else 0.0
        best_move, probs, moves = mcts.get_action_probs(board, temperature=temp)
        if best_move is None:
            break

        # Build policy target full 4096
        policy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for m, p in zip(moves, probs):
            policy_target[move_to_index(m)] = p

        state = board_to_tensor_perspective(board)
        history.append((state, policy_target, board.turn == chess.WHITE))

        board.push(best_move)
        move_num += 1
        if verbose:
            print(f"  move {move_num}: {best_move.uci()}")

    # Game ended — compute final result từ góc nhìn TRẮNG
    if board.is_game_over(claim_draw=True):
        if board.is_checkmate():
            # Side-to-move thua → bên kia thắng
            white_result = -1.0 if board.turn == chess.WHITE else 1.0
        else:
            white_result = 0.0  # hòa tự nhiên (stalemate, lặp, thiếu quân...)
    else:
        # Game bị cắt ở max_moves → chấm theo material thay vì mặc định hòa
        white_result = adjudicate_result(board, cfg.adjudication_margin)

    # Backfill value_target cho mỗi sample (theo side-to-move tại thời điểm đó)
    samples = []
    for state, policy, white_to_move in history:
        v = white_result if white_to_move else -white_result
        samples.append((state, policy, float(v)))

    if verbose:
        outcome = "1-0" if white_result > 0 else ("0-1" if white_result < 0 else "1/2-1/2")
        print(f"  game ended: {outcome} ({move_num} moves)")

    return samples


def generate_self_play_data(
    model: PolicyValueNet,
    cfg: SelfPlayConfig,
    output_path: Optional[str | pathlib.Path] = None,
    verbose: bool = False,
) -> dict:
    """
    Chạy num_games self-play, ghép tất cả samples lại.
    Lưu thành .pt với states (tensor), policies (tensor), values (tensor).
    """
    model.eval()
    all_states = []
    all_policies = []
    all_values = []

    pbar = tqdm(range(cfg.num_games), desc="self-play games")
    for g in pbar:
        samples = play_one_game(model, cfg, verbose=verbose)
        for state, policy, value in samples:
            all_states.append(state)
            all_policies.append(torch.from_numpy(policy))
            all_values.append(value)
        pbar.set_postfix(samples=len(all_states))

    data = {
        "states": torch.stack(all_states),                   # (N, 12, 8, 8)
        "policies": torch.stack(all_policies),               # (N, 4096)
        "values": torch.tensor(all_values, dtype=torch.float32),  # (N,)
        "meta": {
            "num_games": cfg.num_games,
            "num_simulations": cfg.num_simulations,
        },
    }

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, output_path)
        print(f"[self-play] saved {len(all_states)} samples → {output_path}")

    return data
