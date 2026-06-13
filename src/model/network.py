"""
network.py
----------
CNN cho chess AI. Hai biến thể:
- PolicyNet: chỉ có policy head (dùng cho supervised pretrain)
- PolicyValueNet: policy + value head (dùng cho MCTS + self-play)

Kiến trúc gọn, đủ chạy nhanh trên CPU và GPU:
    Input: (B, 12, 8, 8)
    Conv block × 4 (12 → 64 → 128 → 128 → 128) + BatchNorm + ReLU
    Residual blocks × 3 (optional, giúp model sâu hơn nhưng vẫn ổn định)

    Policy head: Conv 2×1×1 → Flatten → Linear → 4096
    Value head:  Conv 1×1×1 → Flatten → Linear 64 → Linear 1 → tanh
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_ACTIONS = 4096
INPUT_CHANNELS = 12


class ConvBlock(nn.Module):
    """Conv 3×3 + BN + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block kiểu ResNet."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class PolicyValueNet(nn.Module):
    """
    Two-headed model:
    - policy_logits: (B, 4096), không qua softmax — dùng CrossEntropyLoss hoặc
      log_softmax tùy phía caller.
    - value: (B,) trong khoảng [-1, 1] (tanh output)
    """

    def __init__(
        self,
        channels: int = 128,
        n_res_blocks: int = 3,
        num_actions: int = NUM_ACTIONS,
    ):
        super().__init__()
        self.num_actions = num_actions

        # Trunk
        self.stem = ConvBlock(INPUT_CHANNELS, channels)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(n_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, num_actions)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Trunk
        h = self.stem(x)
        for block in self.res_blocks:
            h = block(h)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return policy_logits, value

    @torch.no_grad()
    def evaluate_position(self, board, device: str = "cpu") -> float:
        """
        Đánh giá một thế cờ bằng value head.

        Trả về scalar trong [-1, 1] theo góc nhìn của BÊN ĐANG ĐI (side-to-move):
        +1 = bên sắp đi đang thắng rõ, -1 = đang thua, 0 = cân bằng.

        Lưu ý: KHÔNG đặt tên `eval` vì nn.Module đã có sẵn `.eval()` (đổi sang
        chế độ inference). Hàm này tự bật eval-mode trước khi chạy.
        """
        # Lazy import để network.py vẫn load được khi chưa cần chess/encode.
        from ..data.encode_board import board_to_tensor_perspective

        self.eval()
        x = board_to_tensor_perspective(board).unsqueeze(0).to(device)
        _, value = self.forward(x)
        return float(value.squeeze(0).cpu().item())

    @torch.no_grad()
    def rate_last_move(self, board, device: str = "cpu") -> float:
        """
        Chấm điểm "nước đi vừa rồi tốt đến đâu" cho NGƯỜI VỪA ĐI.

        Input: `board` là thế cờ SAU khi đã đẩy nước cần chấm (board.turn giờ là
        đối thủ của người vừa đi).

        Trả về scalar trong [-1, 1] theo góc nhìn của NGƯỜI VỪA ĐI:
            +1 = nước vừa đi rất tốt (đang thắng / chiếu hết),
             0 = không đổi cục diện,
            -1 = nước vừa đi rất tệ (đang thua).

        Cách tính: value head cho giá trị theo góc nhìn bên-sắp-đi (= đối thủ),
        nên điểm cho người vừa đi là dấu ngược lại. Thế cờ kết thúc xử riêng:
        chiếu hết -> +1 (người vừa đi vừa chiếu hết); hòa -> 0.
        """
        import chess

        if board.is_checkmate():
            # Bên sắp đi bị chiếu hết -> người vừa đi đã thắng.
            return 1.0
        if board.is_stalemate() or board.is_insufficient_material() \
                or board.can_claim_fifty_moves() or board.can_claim_threefold_repetition():
            return 0.0

        # value theo góc nhìn bên-sắp-đi (đối thủ) -> đảo dấu cho người vừa đi.
        return -self.evaluate_position(board, device=device)


class PolicyNet(nn.Module):
    """
    Phiên bản chỉ có policy head — dùng cho supervised pretrain (Tuần 1).
    Tuần 2 sẽ migrate sang PolicyValueNet.

    Wrapper quanh PolicyValueNet, trả về chỉ policy_logits cho gọn.
    """

    def __init__(self, channels: int = 128, n_res_blocks: int = 3):
        super().__init__()
        self.net = PolicyValueNet(channels=channels, n_res_blocks=n_res_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.net(x)
        return logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = PolicyValueNet()
    print(model)
    print(f"\nTotal params: {count_params(model):,}")

    x = torch.randn(4, 12, 8, 8)
    p, v = model(x)
    print(f"\nInput: {x.shape}")
    print(f"Policy logits: {p.shape}  (expect [4, 4096])")
    print(f"Value: {v.shape}  values: {v.tolist()}")
