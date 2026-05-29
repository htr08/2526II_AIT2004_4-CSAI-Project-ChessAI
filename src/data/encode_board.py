"""
encode_board.py
---------------
Chuyển python-chess Board → tensor 12×8×8 one-hot.

Layout chuẩn AlphaZero:
- 12 channels = 6 loại quân (P, N, B, R, Q, K) × 2 màu (white, black)
- 8×8 = 64 ô bàn cờ
- Channel index:
    0=WP, 1=WN, 2=WB, 3=WR, 4=WQ, 5=WK,
    6=BP, 7=BN, 8=BB, 9=BR, 10=BQ, 11=BK

Optional extra channels (mở rộng sau nếu cần):
- 12: side to move (1 nếu trắng đi, 0 nếu đen)
- 13-16: castling rights (KQkq)
- 17: en passant square
- 18: 50-move counter
- 19: repetition counter

Phiên bản này chỉ dùng 12 channels cơ bản để đơn giản — đủ cho project.
"""
from __future__ import annotations

import chess
import numpy as np
import torch

# Mapping: (piece_type, color) → channel index
# piece_type: 1=PAWN, 2=KNIGHT, 3=BISHOP, 4=ROOK, 5=QUEEN, 6=KING
# color: True=WHITE, False=BLACK
_PIECE_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

NUM_CHANNELS = 12
BOARD_SIZE = 8


def board_to_array(board: chess.Board) -> np.ndarray:
    """
    Chuyển board → np.ndarray shape (12, 8, 8), dtype float32.

    Row 0 của tensor = rank 1 (hàng dưới cùng từ phía trắng).
    Col 0 = file a.

    Lưu ý quan trọng: nếu side-to-move là đen, ta KHÔNG flip board ở đây.
    Flip board sẽ làm trong dataset/training (xem `board_to_tensor_perspective`).
    """
    arr = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for square, piece in board.piece_map().items():
        channel = _PIECE_CHANNEL[(piece.piece_type, piece.color)]
        rank = chess.square_rank(square)  # 0..7 (rank 1..8)
        file = chess.square_file(square)  # 0..7 (file a..h)
        arr[channel, rank, file] = 1.0
    return arr


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Wrapper trả về torch.Tensor (12, 8, 8)."""
    return torch.from_numpy(board_to_array(board))


def board_to_tensor_perspective(board: chess.Board) -> torch.Tensor:
    """
    Encode board từ góc nhìn của side-to-move.
    Nếu đen đi, FLIP board (rotate 180°) và đảo channel màu để mô hình
    luôn "thấy" mình là trắng. Giúp model học được symmetry, training ổn định hơn.

    Cách flip:
    - Đảo channels: white pieces ↔ black pieces (swap channel [0:6] với [6:12])
    - Flip board theo cả rank và file (rotate 180°)
    """
    arr = board_to_array(board)
    if board.turn == chess.BLACK:
        # Swap color channels
        white = arr[:6].copy()
        black = arr[6:].copy()
        arr[:6] = black
        arr[6:] = white
        # Rotate 180° (flip cả rank và file)
        arr = arr[:, ::-1, ::-1].copy()
    return torch.from_numpy(arr)


def array_to_board_debug(arr: np.ndarray) -> str:
    """
    Debug: in mảng 12×8×8 ra dạng ASCII board (chữ in hoa = trắng).
    Chỉ dùng cho test/debug, không dùng trong production.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    symbols = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
    board_chars = [["." for _ in range(8)] for _ in range(8)]
    for ch in range(NUM_CHANNELS):
        for r in range(8):
            for f in range(8):
                if arr[ch, r, f] > 0.5:
                    board_chars[r][f] = symbols[ch]
    # In từ rank 8 xuống rank 1 (hiển thị quen mắt)
    lines = []
    for r in range(7, -1, -1):
        lines.append(f"{r+1} " + " ".join(board_chars[r]))
    lines.append("  a b c d e f g h")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick smoke test
    board = chess.Board()
    t = board_to_tensor(board)
    print(f"Starting position tensor shape: {t.shape}")
    print(f"dtype: {t.dtype}, sum: {t.sum().item()} (expect 32 = số quân)")
    print(array_to_board_debug(t))
    print()

    # Sau e4
    board.push_san("e4")
    t = board_to_tensor(board)
    print(f"After 1.e4 — sum: {t.sum().item()} (expect 32)")
    print(array_to_board_debug(t))
    print()

    # Test perspective: black đi
    t_persp = board_to_tensor_perspective(board)
    print("Same position, từ góc nhìn của đen (flipped):")
    print(array_to_board_debug(t_persp))
