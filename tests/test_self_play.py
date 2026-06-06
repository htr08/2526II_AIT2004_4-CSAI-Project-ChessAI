"""Kiểm tra generate_games: sinh đúng shape tensor và lưu file .pt."""
import torch
from pathlib import Path
from src.model import DualNet
from scripts.self_play import generate_games

def test_self_play_pipeline_shapes():
    """Kiểm tra luồng tự chơi sinh dữ liệu với mạng DualNet có ra đúng kích thước ma trận."""
    # Khởi tạo mô hình mạng nhỏ (1 block) để chạy thử nghiệm nhanh cho pass test
    model = DualNet(in_ch=17, num_moves=4352, n_res=1)
    device = "cpu"
    
    out_dir = "data/test_selfplay"
    # Chạy sinh dữ liệu cho 1 game duy nhất để test pipeline thông suốt
    generate_games(model, n_games=1, out_dir=out_dir, device=device)
    
    target_file = Path(out_dir) / "selfplay_1games_50sims.pt"
    assert target_file.exists(), "Pipeline lỗi không tạo được file .pt lưu trữ!"
    
    # Đọc lại file dữ liệu vừa sinh ra để nghiệm thu cấu trúc Tensor
    data = torch.load(target_file, map_location="cpu")
    
    # Kiểm tra các chiều tensor (Shape Check) bảo toàn kiến trúc AlphaZero
    assert "X" in data and "policy" in data and "value" in data
    assert data["X"].ndim == 4              # Định dạng: (Số thế cờ, Kênh 17, 8, 8)
    assert data["X"].shape[1] == 17        # Đủ 17 kênh biểu diễn quân cờ
    assert data["policy"].shape[1] == 4544  # Khớp kích thước NUM_MOVES
    assert data["value"].ndim == 1          # Mảng phẳng chứa kết quả từng nước đi
    
    print("\n[PASS] Mọi Tensor dữ liệu sinh ra từ Self-Play đều đạt chuẩn thiết kế!")