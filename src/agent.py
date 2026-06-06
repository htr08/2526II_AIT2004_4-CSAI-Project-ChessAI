"""Agent cờ vua: MinimaxAgent (alpha-beta) và DualNetMinimaxAgent (alpha-beta + DualNet)."""

import chess
import torch
import functools
from src.search import minimax, order_moves_with_model
from src.model import PolicyNet, DualNet
from src.vocab import load_or_build_move2idx
from src.opening_book import OpeningBook


class MinimaxAgent:
    """Agent alpha-beta minimax, dùng PolicyNet sắp xếp nước (nếu có) và opening book."""

    def __init__(self, depth: int = 4, model_path: str | None = None,
                 book_path: str | None = None):
        self.depth = depth
        self.model = None
        self.move2idx = load_or_build_move2idx()
        self.book = OpeningBook(book_path) if book_path else None

        if model_path:
            self.model = PolicyNet()
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
            self.model.eval()
            print(f"Loaded policy model from {model_path}")

    def _make_orderer(self):
        if self.model is None:
            return None  # minimax defaults to material-score ordering
        return functools.partial(
            order_moves_with_model,
            model=self.model,
            move2idx=self.move2idx,
        )

    def select_move(self, board: chess.Board) -> chess.Move:
        if self.book:
            book_move = self.book.get_move(board)
            if book_move:
                return book_move
        _, move = minimax(
            board, self.depth, -999999, 999999,
            move_orderer=self._make_orderer(),
        )
        return move or next(iter(board.legal_moves))


class DualNetMinimaxAgent:
    """Agent alpha-beta dùng DualNet: policy head sắp xếp nước, value head đánh giá lá."""

    def __init__(self, depth: int = 4, model_path: str | None = None,
                 device: str = "cpu", book_path: str | None = None):
        self.depth = depth
        self.device = device
        self.move2idx = load_or_build_move2idx()
        self.book = OpeningBook(book_path) if book_path else None
        self.model = DualNet()
        if model_path:
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
            print(f"Loaded DualNet model from {model_path}")
        self.model.eval().to(device)

    def _move_orderer(self):
        # order_moves_with_model expects model(x) to return a plain tensor.
        # DualNet returns (policy, value), so wrap it.
        class _PolicyOnly:
            def __init__(self, m): self.m = m
            def __call__(self, x): p, _ = self.m(x); return p

        return functools.partial(
            order_moves_with_model,
            model=_PolicyOnly(self.model),
            move2idx=self.move2idx,
        )

    def select_move(self, board: chess.Board) -> chess.Move:
        if self.book:
            book_move = self.book.get_move(board)
            if book_move:
                return book_move
        _, move = minimax(
            board, self.depth, -999999, 999999,
            move_orderer=self._move_orderer(),
            # evaluator=None → uses classical material+PST (fast and reliable)
        )
        return move or next(iter(board.legal_moves))
