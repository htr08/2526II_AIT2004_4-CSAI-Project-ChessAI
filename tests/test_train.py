"""Smoke test train_selfplay: pipeline chạy 1 epoch không crash."""
import torch
import pytest
from pathlib import Path
from src.train import train_selfplay

def test_mixed_loss_training_pipeline():
    """Kiểm tra pipeline huấn luyện hỗn hợp chạy thông suốt mà không crash kích thước ma trận."""
    # 1. Tạo dữ liệu giả lập (Mock data) có cấu trúc chuẩn 17 kênh, 4544 nước đi để test nhanh
    mock_dir = Path("data/test_train")
    mock_dir.mkdir(parents=True, exist_ok=True)
    mock_file = mock_dir / "mock_selfplay.pt"

    num_samples = 16  # Kích thước tập test nhỏ cho chạy nhanh
    mock_data = {
        "X": torch.randn(num_samples, 17, 8, 8),
        "policy": torch.softmax(torch.randn(num_samples, 4544), dim=-1), # Tạo phân phối xác suất mềm chuẩn
        "value": torch.rand(num_samples) * 2 - 1.0 # Giá trị nằm trong khoảng [-1, 1]
    }
    torch.save(mock_data, mock_file)

    # 2. Tạo file checkpoint mồi giả định để hàm load_pretrained_backbone không bị crash
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    # Tạo file rỗng hoặc chỉ chứa dictionary tượng trưng
    torch.save({}, checkpoint_dir / "best_policy.pt")

    # 3. Tiến hành chạy huấn luyện thử nghiệm với 1 epoch, batch_size nhỏ
    try:
        train_selfplay(
            sp_path=str(mock_file),
            epochs=1,
            value_weight=1.0,
            batch_size=4
        )
        pipeline_passed = True
    except Exception as e:
        print(f"Pipeline crashed due to error: {e}")
        pipeline_passed = False

    # 4. Khẳng định kết quả nghiệm thu thành công
    assert pipeline_passed is True, "Hàm mất mát hỗn hợp hoặc luồng huấn luyện bị lỗi chiều ma trận!"
    assert Path("checkpoints/best_model_v2.pt").exists(), "Mô hình v2 đã không được lưu xuống ổ cứng!"