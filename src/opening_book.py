"""Đọc opening book Polyglot (.bin) và chọn nước theo xác suất cân bằng."""

import chess
import chess.polyglot
import random


class OpeningBook:
    """Đọc file Polyglot .bin và chọn nước theo xác suất (không luôn chọn nước mạnh nhất)."""

    def __init__(self, path: str):
        self._path = path
        self._reader = chess.polyglot.open_reader(path)

    def get_move(self, board: chess.Board) -> chess.Move | None:
        """Trả về nước đi book theo xác suất tỉ lệ weight, hoặc None nếu không có trong sách."""
        try:
            entries = list(self._reader.find_all(board))
            if not entries:
                return None
            total = sum(e.weight for e in entries)
            r = random.uniform(0, total)
            cumulative = 0
            for e in entries:
                cumulative += e.weight
                if r <= cumulative:
                    move = e.move
                    return move if move in board.legal_moves else None
            return entries[-1].move
        except Exception:
            return None

    def close(self) -> None:
        self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()