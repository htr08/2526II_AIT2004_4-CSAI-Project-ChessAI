"""
mcts.py
-------
Monte Carlo Tree Search dựa trên PolicyValueNet (AlphaZero-style).

Cách hoạt động:
1. Selection: từ root, đi xuống cây theo PUCT (UCB với prior policy).
2. Expansion: gặp leaf chưa expand → expand node bằng cách evaluate bằng net.
3. Backup: lan value lên các node trên đường đi (đổi dấu mỗi level vì side đổi).

Khác minimax: KHÔNG search tới depth cố định, mà phân bổ "simulations" cho
các nhánh hứa hẹn. Càng nhiều simulations → càng mạnh.

Reference: AlphaZero (Silver et al., 2017).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import chess
import numpy as np
import torch
import torch.nn.functional as F

from ..data.action_space import move_to_index
from ..data.encode_board import board_to_tensor_perspective


class MCTSNode:
    """Một node trong cây MCTS."""

    __slots__ = (
        "parent",
        "move",          # move từ parent → node này
        "prior",         # P(s, a) từ policy net
        "children",      # dict[chess.Move, MCTSNode]
        "visit_count",   # N
        "value_sum",     # tổng W (sum of values từ góc nhìn side-to-move tại node này)
        "is_expanded",
    )

    def __init__(self, parent=None, move=None, prior: float = 0.0):
        self.parent: Optional[MCTSNode] = parent
        self.move: Optional[chess.Move] = move
        self.prior: float = prior
        self.children: dict[chess.Move, MCTSNode] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float = 1.5) -> float:
        """
        PUCT formula: Q(s,a) + c_puct · P(s,a) · sqrt(N_parent) / (1 + N(s,a))
        """
        if self.parent is None:
            return 0.0
        u = (
            c_puct
            * self.prior
            * math.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )
        # Q từ góc nhìn của parent — vì side-to-move khác nhau giữa parent/child
        return -self.q_value + u

    def select_child(self, c_puct: float = 1.5):
        """Chọn child có UCB cao nhất."""
        return max(self.children.values(), key=lambda c: c.ucb_score(c_puct))

    def expand(self, board: chess.Board, priors: np.ndarray):
        """Tạo children cho mọi legal move, gán prior từ policy net."""
        legal = list(board.legal_moves)
        if not legal:
            self.is_expanded = True
            return

        # Mask + renormalize priors trên legal moves
        legal_priors = np.array([priors[move_to_index(m)] for m in legal])
        s = legal_priors.sum()
        if s > 1e-8:
            legal_priors /= s
        else:
            legal_priors = np.ones(len(legal)) / len(legal)

        for move, p in zip(legal, legal_priors):
            self.children[move] = MCTSNode(parent=self, move=move, prior=float(p))
        self.is_expanded = True

    def backup(self, value: float):
        """Lan value lên root, đổi dấu mỗi level vì side đổi."""
        node = self
        v = value
        while node is not None:
            node.visit_count += 1
            node.value_sum += v
            v = -v  # đổi dấu cho parent (side khác)
            node = node.parent


class MCTS:
    def __init__(
        self,
        model,
        device: str = "cpu",
        c_puct: float = 1.5,
        num_simulations: int = 200,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        add_noise: bool = False,
    ):
        """
        Args:
            model: PolicyValueNet (eval mode)
            c_puct: hệ số exploration (1.0-2.5 thường dùng)
            num_simulations: số lần simulate per move
            dirichlet_*: noise tại root cho self-play, không dùng khi play
            add_noise: True khi dùng cho self-play (tăng exploration)
        """
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.add_noise = add_noise

    @torch.no_grad()
    def _evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        """Run model → return (policy_probs[4096], value scalar [-1,1])."""
        x = board_to_tensor_perspective(board).unsqueeze(0).to(self.device)
        logits, value = self.model(x)
        probs = F.softmax(logits.squeeze(0), dim=-1).cpu().numpy()
        v = float(value.squeeze(0).cpu().item())
        return probs, v

    def search(self, board: chess.Board) -> MCTSNode:
        """Chạy num_simulations MCTS simulations, trả về root."""
        self.model.eval()
        root = MCTSNode()

        # Expand root
        priors, _ = self._evaluate(board)
        root.expand(board, priors)

        # Add Dirichlet noise tại root cho self-play
        if self.add_noise and root.children:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(root.children)
            )
            for (move, child), n in zip(root.children.items(), noise):
                child.prior = (
                    (1 - self.dirichlet_epsilon) * child.prior
                    + self.dirichlet_epsilon * float(n)
                )

        for _ in range(self.num_simulations):
            node = root
            sim_board = board.copy()

            # Selection: đi xuống cho tới khi gặp leaf
            while node.is_expanded and node.children:
                node = node.select_child(self.c_puct)
                sim_board.push(node.move)

            # Check terminal
            if sim_board.is_game_over(claim_draw=True):
                if sim_board.is_checkmate():
                    # side-to-move bị checkmate → giá trị -1 cho họ
                    value = -1.0
                else:
                    value = 0.0
            else:
                # Expansion + evaluation
                priors, value = self._evaluate(sim_board)
                node.expand(sim_board, priors)

            # Backup
            node.backup(value)

        return root

    def get_action_probs(
        self, board: chess.Board, temperature: float = 1.0
    ) -> tuple[chess.Move, np.ndarray, list[chess.Move]]:
        """
        Chạy MCTS, trả về:
        - best move (theo visit count nếu temperature=0, hoặc sample nếu >0)
        - probability distribution theo visit counts (dùng làm policy target cho training)
        - list các moves tương ứng với probs
        """
        root = self.search(board)
        moves = list(root.children.keys())
        if not moves:
            return None, np.array([]), []

        visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)

        if temperature <= 1e-6:
            # Greedy
            best_idx = int(np.argmax(visits))
            probs = np.zeros_like(visits)
            probs[best_idx] = 1.0
            best_move = moves[best_idx]
        else:
            visits_t = visits ** (1.0 / temperature)
            probs = visits_t / visits_t.sum()
            best_idx = int(np.random.choice(len(moves), p=probs))
            best_move = moves[best_idx]

        return best_move, probs, moves


def search_best_move_mcts(
    board: chess.Board,
    model,
    num_simulations: int = 200,
    device: str = "cpu",
    c_puct: float = 1.5,
) -> tuple[chess.Move, float]:
    """Convenience wrapper: chạy MCTS một lần, trả về best move + root q_value."""
    mcts = MCTS(
        model,
        device=device,
        c_puct=c_puct,
        num_simulations=num_simulations,
        add_noise=False,
    )
    move, probs, moves = mcts.get_action_probs(board, temperature=0.0)
    return move, 0.0  # placeholder score
