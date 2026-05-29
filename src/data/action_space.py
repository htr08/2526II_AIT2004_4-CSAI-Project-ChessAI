"""
action_space.py
---------------
Định nghĩa action space cho policy head.

Chiến lược: dùng 4096 = 64 × 64 cho (from_square, to_square).
- Đơn giản, đủ phủ mọi nước đi thường (bao gồm cả en passant, castling).
- Promotion: mặc định queen promotion sẽ map vào index thường (from, to).
  Underpromotion (knight/bishop/rook) sẽ được xử lý riêng nếu cần — phiên bản
  này CHẤP NHẬN mất thông tin underpromotion (extremely rare trong cờ vua thực tế).

Nếu cần đầy đủ 4672 outputs (AlphaZero gốc), xem `policy_index_az` mở rộng.

Cách encode:
    index = from_square * 64 + to_square    (range 0..4095)

Cách decode:
    from_square = index // 64
    to_square   = index % 64

Khi training: nước đi từ PGN → uci → from_sq, to_sq → index.
Khi inference: model output logits[4096] → mask theo legal moves → argmax → uci.
"""
from __future__ import annotations

import chess

NUM_ACTIONS = 64 * 64  # 4096


def move_to_index(move: chess.Move) -> int:
    """
    Chuyển chess.Move → int index trong [0, 4095].
    Promotion bị "lossy" — chỉ giữ from/to. Khi decode lại, mặc định promote queen.
    """
    return move.from_square * 64 + move.to_square


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Chuyển index → chess.Move trên board hiện tại.
    Cần board để xử lý promotion: nếu pawn vào rank cuối → tự động promote queen.

    Trả về None nếu move không hợp lệ (caller phải kiểm tra).
    """
    from_sq = index // 64
    to_sq = index % 64

    # Kiểm tra có phải pawn promotion không
    piece = board.piece_at(from_sq)
    promotion = None
    if piece is not None and piece.piece_type == chess.PAWN:
        to_rank = chess.square_rank(to_sq)
        if (piece.color == chess.WHITE and to_rank == 7) or (
            piece.color == chess.BLACK and to_rank == 0
        ):
            promotion = chess.QUEEN  # mặc định promote queen

    move = chess.Move(from_sq, to_sq, promotion=promotion)
    return move


def legal_move_mask(board: chess.Board) -> list[int]:
    """
    Trả về list các action index hợp lệ tại position hiện tại.
    Dùng để mask policy logits trước khi softmax/argmax.
    """
    return [move_to_index(m) for m in board.legal_moves]


def uci_to_index(uci: str) -> int:
    """Helper: convert UCI string (vd 'e2e4', 'e7e8q') → index."""
    move = chess.Move.from_uci(uci)
    return move_to_index(move)


def index_to_uci(index: int, board: chess.Board) -> str:
    """Helper: convert index → UCI string."""
    return index_to_move(index, board).uci()


if __name__ == "__main__":
    # Smoke test
    board = chess.Board()
    print(f"NUM_ACTIONS = {NUM_ACTIONS}")
    print()

    move = chess.Move.from_uci("e2e4")
    idx = move_to_index(move)
    print(f"e2e4 → index {idx}")
    print(f"  from_sq={move.from_square} ({chess.square_name(move.from_square)})")
    print(f"  to_sq={move.to_square} ({chess.square_name(move.to_square)})")

    # Decode lại
    move_back = index_to_move(idx, board)
    print(f"  decode lại: {move_back.uci()}")
    assert move_back.uci() == "e2e4"

    # Test legal mask
    mask = legal_move_mask(board)
    print(f"\nStarting position có {len(mask)} legal moves (expect 20)")
    assert len(mask) == 20

    # Test promotion
    board2 = chess.Board("8/4P3/8/8/8/8/8/k6K w - - 0 1")  # pawn e7
    promo_idx = uci_to_index("e7e8q")
    decoded = index_to_move(promo_idx, board2)
    print(f"\nPromotion: e7e8q → index {promo_idx} → decode: {decoded.uci()}")
    assert decoded.uci() == "e7e8q", f"got {decoded.uci()}"
    print("OK — promotion handled correctly")
