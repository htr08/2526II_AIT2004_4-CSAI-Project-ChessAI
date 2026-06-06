"""Tìm kiếm cổ điển: đánh giá material+PST, heuristic sắp xếp nước và alpha-beta minimax."""

import chess
import torch
from src.board import encode_board_with_meta

# Material value (centipawns)
MATERIAL = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
}

# PST format: rank 8 at index 0, rank 1 at index 63 (standard top-down notation).
# For White pieces, apply with sq^56 (flip vertically).
# For Black pieces, apply with sq directly.
# Source: chessprogramming.org/Piece-Square_Tables

PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_PST = [
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

# King middlegame: prefers castled position, avoids center
KING_PST = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

PST = {
    chess.PAWN:   PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK:   ROOK_PST,
    chess.QUEEN:  QUEEN_PST,
    chess.KING:   KING_PST,
}


def evaluate(board: chess.Board) -> int:
    """Trả về điểm theo góc nhìn của bên đang đi (centipawns)."""
    if board.is_checkmate():
        return -100000
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0

    score = 0
    for sq, piece in board.piece_map().items():
        val = MATERIAL[piece.piece_type]
        # White flips vertically (sq^56) so rank 1 maps to PST row for rank 8 slot
        pst_sq = sq ^ 56 if piece.color == chess.WHITE else sq
        pst_bonus = PST[piece.piece_type][pst_sq]
        if piece.color == board.turn:
            score += val + pst_bonus
        else:
            score -= val + pst_bonus
    return score


def order_moves(board):
    """Ưu tiên bắt quân và phong cấp trước — cải thiện alpha-beta pruning."""
    def score_move(move):
        s = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                s += MATERIAL[victim.piece_type] - MATERIAL[attacker.piece_type] // 10
        if move.promotion:
            s += 800
        return -s
    return sorted(board.legal_moves, key=score_move)


def minimax(board, depth, alpha, beta, nodes_searched=None, move_orderer=None, evaluator=None):
    """Negamax alpha-beta. Trả về (score, best_move). move_orderer và evaluator có thể thay thế."""
    if nodes_searched is not None:
        nodes_searched[0] += 1

    if depth == 0 or board.is_game_over(claim_draw=True):
        eval_fn = evaluator if evaluator is not None else evaluate
        return eval_fn(board), None

    orderer = move_orderer if move_orderer is not None else order_moves
    best_move = None
    best_score = -999999

    for move in orderer(board):
        board.push(move)
        score, _ = minimax(board, depth - 1, -beta, -alpha, nodes_searched, move_orderer, evaluator)
        score = -score  # negamax: flip perspective after each ply
        board.pop()

        if score > best_score:
            best_score, best_move = score, move
        alpha = max(alpha, score)
        if alpha >= beta:
            break  # beta cutoff

    return best_score, best_move


def get_best_move_minimax(board, depth=4):
    nodes = [0]
    score, move = minimax(board, depth, -999999, 999999, nodes)
    print(f"depth={depth} nodes={nodes[0]:,} score={score}")
    return move

def order_moves_with_model(board, model, move2idx, top_k=20):
    """Dùng Policy Net để rank moves. Fallback về material ordering nếu model=None."""
    if model is None:
        return order_moves(board)

    moves = list(board.legal_moves)
    if not moves: return moves

    t = encode_board_with_meta(board).unsqueeze(0)
    with torch.no_grad():
        logits = model(t)[0]    # (4352,)
        probs  = torch.softmax(logits, dim=0)

    def move_prob(move):
        idx = move2idx.get(move.uci(), -1)
        return probs[idx].item() if idx >= 0 else 0.0

    # Sắp xếp giảm dần theo prob, chỉ giữ top_k + captures
    scored = sorted(moves, key=move_prob, reverse=True)
    captures = [m for m in moves if board.is_capture(m)]
    top = scored[:top_k]
    return list(dict.fromkeys(captures + top))  # deduplicate
