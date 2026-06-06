"""Mã hóa bàn cờ thành tensor 12 channels (quân cờ) hoặc 17 channels (+ meta: lượt đi, nhập thành)."""

import chess
import numpy as np
import torch

PIECE_ORDER = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN, chess.KING
]

def encode_board(board: chess.Board) -> torch.Tensor:
    """
    Mã hóa 64 ô cờ thành tensor (12, 8, 8) float32.
    - Channels 0-5: Quân Trắng (P, N, B, R, Q, K)
    - Channels 6-11: Quân Đen (P, N, B, R, Q, K)
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        row = sq // 8
        col = sq % 8
        color_offset = 0 if piece.color == chess.WHITE else 6
        piece_idx = PIECE_ORDER.index(piece.piece_type)
        channel = color_offset + piece_idx
        tensor[channel, row, col] = 1.0

    return torch.from_numpy(tensor)

def encode_board_with_meta(board: chess.Board) -> torch.Tensor:
    """
    Mã hóa bàn cờ nâng cao thành tensor (17, 8, 8) float32 bao gồm thông tin trận đấu (Meta):
    - 12 channels đầu: Vị trí các quân cờ (giống hàm encode_board)
    - Channel 12: Lượt đi (To move) -> Toàn bộ là 1.0 nếu là Trắng, 0.0 nếu là Đen
    - Channel 13: Quyền nhập thành cánh Vua của Trắng (White Kingside Castling) -> Toàn 1.0 hoặc 0.0
    - Channel 14: Quyền nhập thành cánh Hậu của Trắng (White Queenside Castling)
    - Channel 15: Quyền nhập thành cánh Vua của Đen (Black Kingside Castling)
    - Channel 16: Quyền nhập thành cánh Hậu của Đen (Black Queenside Castling)
    """
    # 1. Tái sử dụng encode_board để lấy 12 channels cơ bản
    base_np = encode_board(board).numpy()

    # 2. Tạo các kênh Meta thông tin (mỗi kênh kích thước 8x8)
    turn_val = 1.0 if board.turn == chess.WHITE else 0.0
    turn_channel = np.full((1, 8, 8), turn_val, dtype=np.float32)

    # Kiểm tra chi tiết quyền nhập thành của từng bên ở từng cánh
    w_kingside = np.full((1, 8, 8), float(board.has_kingside_castling_rights(chess.WHITE)), dtype=np.float32)
    w_queenside = np.full((1, 8, 8), float(board.has_queenside_castling_rights(chess.WHITE)), dtype=np.float32)
    b_kingside = np.full((1, 8, 8), float(board.has_kingside_castling_rights(chess.BLACK)), dtype=np.float32)
    b_queenside = np.full((1, 8, 8), float(board.has_queenside_castling_rights(chess.BLACK)), dtype=np.float32)

    # 3. Nối tất cả lại thành một khối 17 tầng độc lập
    meta_tensor = np.concatenate([
        base_np, 
        turn_channel, 
        w_kingside, 
        w_queenside, 
        b_kingside, 
        b_queenside
    ], axis=0)

    return torch.from_numpy(meta_tensor)

if __name__ == "__main__":
    # Test nhanh xem hàm chạy ra đúng Shape mong muốn không
    board = chess.Board()
    t1 = encode_board(board)
    t2 = encode_board_with_meta(board)
    print("Shape cơ bản:", t1.shape)
    print("Shape có Meta (Chuẩn mã hóa):", t2.shape)