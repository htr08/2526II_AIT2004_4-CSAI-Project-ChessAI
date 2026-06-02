"""
opening_book.py
---------------
Opening book đơn giản: hardcode ~15 khai cuộc phổ biến để bot chơi đầu ván
chắc tay hơn (thay vì để mạng/đi search từ nước 1).

Cách dùng: khi tới lượt bot và vẫn còn trong sách, trả về nước đi từ sách;
hết sách thì trả None để caller chuyển sang Minimax/MCTS như thường.

Mỗi "line" là danh sách nước đi UCI tính từ thế cờ đầu. Khớp theo tiền tố:
nếu các nước đã đi trùng đầu một line thì nước kế tiếp của line là gợi ý.
"""
from __future__ import annotations

import random
from typing import Optional

import chess

# ~15 khai cuộc phổ biến (UCI), đủ vài nước đầu mỗi bên.
OPENING_LINES: list[list[str]] = [
    # Ruy Lopez
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"],
    # Italian Game
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6"],
    # Sicilian Najdorf
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"],
    # Sicilian (open)
    ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6"],
    # French Defense
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"],
    # Caro-Kann
    ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4"],
    # Queen's Gambit Declined
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"],
    # Slav Defense
    ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6"],
    # King's Indian Defense
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"],
    # Nimzo-Indian
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"],
    # English Opening
    ["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6"],
    # Reti Opening
    ["g1f3", "d7d5", "c2c4", "e7e6", "g2g3", "g8f6"],
    # London System
    ["d2d4", "d7d5", "g1f3", "g8f6", "c1f4", "e7e6"],
    # Scandinavian
    ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"],
    # Pirc Defense
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6"],
]


def book_move(board: chess.Board, rng: Optional[random.Random] = None) -> Optional[chess.Move]:
    """
    Trả về nước đi từ sách cho thế cờ hiện tại, hoặc None nếu đã ra khỏi sách.

    Khớp theo lịch sử nước đi (board.move_stack). Nếu nhiều line cùng khớp,
    chọn ngẫu nhiên một nước kế tiếp (đa dạng khai cuộc). Chỉ trả nước HỢP LỆ.
    """
    played = [m.uci() for m in board.move_stack]
    n = len(played)

    nexts: list[str] = []
    for line in OPENING_LINES:
        if len(line) > n and line[:n] == played:
            nexts.append(line[n])

    if not nexts:
        return None

    picker = rng or random
    # Lọc trùng nhưng giữ thứ tự, rồi chọn ngẫu nhiên
    unique = list(dict.fromkeys(nexts))
    for uci in picker.sample(unique, len(unique)):
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if mv in board.legal_moves:
            return mv
    return None


def in_book(board: chess.Board) -> bool:
    """True nếu thế cờ hiện tại còn nằm trong sách."""
    return book_move(board) is not None


if __name__ == "__main__":
    b = chess.Board()
    print("Demo: đi theo sách tới khi hết...")
    for _ in range(12):
        mv = book_move(b)
        if mv is None:
            print("  (hết sách)")
            break
        print(f"  {b.fullmove_number}. {mv.uci()}")
        b.push(mv)
