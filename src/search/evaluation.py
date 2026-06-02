"""
evaluation.py
-------------
Static evaluation function cho Minimax/Alpha-Beta — không dùng neural net.

Eval = material + piece-square tables (PST).
Score được tính từ góc nhìn TRẮNG (positive = trắng tốt, negative = đen tốt).
"""
from __future__ import annotations

import chess


# Material values (centipawns)
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Piece-square tables (góc nhìn TRẮNG, rank 1..8 từ dưới lên)
# Mỗi bảng index theo square (a1=0, b1=1, ..., h8=63)
# Đen sẽ dùng giá trị mirror (lật ngược theo rank).

# Pawn: thưởng tiến lên trung lộ, phạt ô a/h ban đầu
PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_PST = [
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
      0,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

KING_PST_MIDGAME = [
     20, 30, 10,  0,  0, 10, 30, 20,
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
]

PST = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_PST_MIDGAME,
}


def evaluate_board(board: chess.Board) -> int:
    """
    Trả về evaluation score (centipawns), từ góc nhìn TRẮNG.
    +100 = trắng hơn 1 tốt; -100 = đen hơn 1 tốt.

    Xử lý:
    - Material + PST
    - Mate: ±100000
    - Stalemate / draw: 0
    """
    if board.is_checkmate():
        # Side-to-move is checkmated — they lose
        return -100000 if board.turn == chess.WHITE else 100000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if board.can_claim_fifty_moves() or board.can_claim_threefold_repetition():
        return 0

    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUE[piece.piece_type]
        pst = PST[piece.piece_type]
        if piece.color == chess.WHITE:
            score += value + pst[square]
        else:
            # Mirror square cho đen (lật rank)
            mirror = chess.square_mirror(square)
            score -= value + pst[mirror]
    return score


# Ngưỡng material (centipawns) để xử "thắng" khi game bị cắt ở move cap.
# 100 = hơn ~1 tốt. Dưới ngưỡng này coi như hòa.
ADJUDICATION_MARGIN = 100


def adjudicate_result(board: chess.Board, margin: int = ADJUDICATION_MARGIN) -> float:
    """
    Chấm kết quả một ván từ góc nhìn TRẮNG, trả về 1.0 / -1.0 / 0.0.

    Mục đích: phá thế "toàn hòa" trong self-play/pit khi game chạm giới hạn
    số nước (max_moves) mà chưa chiếu hết. Thay vì mặc định hòa, ta chấm theo
    material + PST: bên nào hơn rõ (|score| > margin) thì coi như thắng.

    - Checkmate: ±1.0 (đã đúng dấu sẵn trong evaluate_board → ±100000).
    - Hòa tự nhiên (stalemate, thiếu quân, 50-move, lặp 3 lần): 0.0.
    - Còn lại (game bị cắt): chấm theo material, |score| ≤ margin → hòa.
    """
    score = evaluate_board(board)
    if score > margin:
        return 1.0
    if score < -margin:
        return -1.0
    return 0.0
