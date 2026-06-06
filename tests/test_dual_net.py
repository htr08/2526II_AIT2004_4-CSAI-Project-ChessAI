"""Kiểm tra DualNet: shape đầu ra policy/value và tích hợp với Hybrid MCTS."""

import torch
import chess
from src.model import DualNet
from src.board import encode_board
from src.mcts import mcts_search_hybrid

def test_dual_net_shapes():
    """Kiểm tra kích thước đầu ra của hai nhánh Policy và Value phải chuẩn xác."""
    batch_size = 4
    in_channels = 17
    num_moves = 4544
    
    model = DualNet(in_ch=in_channels, num_moves=num_moves)
    dummy_input = torch.zeros(batch_size, in_channels, 8, 8)
    
    policy_out, value_out = model(dummy_input)
    
    assert policy_out.shape == (batch_size, num_moves)
    assert value_out.shape == (batch_size,) # Nhánh Value phải ra vector phẳng kích thước bằng Batch Size
    assert torch.all(value_out >= -1.0) and torch.all(value_out <= 1.0) # Ép trong khoảng Tanh [-1,1]

def test_hybrid_mcts_run():
    """Kiểm tra sự kết hợp giữa Hybrid MCTS và DualNet chạy không bị crash kích thước."""
    board = chess.Board()
    model = DualNet(in_ch=17, num_moves=4544)
    
    # Chỉ cần chạy thử 10 sims để test luồng dữ liệu, không cần chạy nhiều
    move = mcts_search_hybrid(board, model, n_simulations=10)
    assert isinstance(move, chess.Move)
    assert move in board.legal_moves