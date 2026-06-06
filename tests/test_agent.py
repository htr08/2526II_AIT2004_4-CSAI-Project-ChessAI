# tests/test_agent.py
import chess
from src.agent import MinimaxAgent

def test_agent_initialization_without_model():
    """Kiểm tra Agent khởi tạo mượt mà khi không có model (Pure Minimax)."""
    agent = MinimaxAgent(depth=1, model_path=None)
    assert agent.model is None
    assert isinstance(agent.move2idx, dict)

def test_agent_select_move_legal():
    """Kiểm tra nước đi do Agent chọn bắt buộc phải là nước đi hợp lệ (Legal)."""
    board = chess.Board()
    agent = MinimaxAgent(depth=1, model_path=None)
    
    move = agent.select_move(board)
    
    assert isinstance(move, chess.Move)
    assert move in board.legal_moves

def test_agent_mate_in_one():
    """Thử thách Agent giải thế cờ chiếu hết trong 1 nước đi."""
    # Thế cờ Scholar's mate, Trắng đi Qh5xf7# là thắng luôn
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    agent = MinimaxAgent(depth=2, model_path=None) # Depth 2 là đủ để nhìn thấy viễn cảnh chiếu hết
    
    move = agent.select_move(board)
    assert move.uci() == "h5f7"