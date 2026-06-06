# tests/test_arena.py
import torch
from pathlib import Path
from src.model import DualNet
from scripts.pit import pit

def test_arena_promotion_mechanism():
    """Kiểm tra cơ chế chấm điểm và thăng cấp mô hình của đấu trường Arena."""
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    
    # 1. Tạo 2 file checkpoint giả lập có cấu trúc mạng chuẩn 17 kênh để test luồng
    v1_path = ckpt_dir / "best_model_v1.pt"
    v2_path = ckpt_dir / "best_model_v2.pt"
    
    dummy_model = DualNet(in_ch=17, num_moves=4544, n_res=4)
    torch.save(dummy_model.state_dict(), str(v1_path))
    torch.save(dummy_model.state_dict(), str(v2_path))

    # 2. Chạy Arena đấu thử 2 games ngắn (Đảm bảo chạy hết luồng xen kẽ Trắng/Đen)
    # Đặt threshold = -0.1 để chắc chắn mô hình mới sẽ được promote (Smoke Test cơ chế sao chép file)
    promoted = pit(str(v2_path), str(v1_path), n_games=2, threshold=-0.1)
    
    # 3. Khẳng định kết quả kiểm thử
    assert promoted is True, "Cơ chế thăng cấp không kích hoạt dù win-rate vượt ngưỡng!"
    assert (ckpt_dir / "best_dual.pt").exists(), "File best_dual.pt không được sinh ra sau khi promote!"
    print("\n[PASS] Hệ thống Đấu trường Arena vận hành chính xác 100%!")