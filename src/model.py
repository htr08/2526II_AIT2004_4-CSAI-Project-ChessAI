"""Kiến trúc mạng: PolicyNet (CNN đơn giản) và DualNet (ResNet backbone + policy head + value head)."""

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """CNN backbone + policy head. Input: (B, 17, 8, 8) → Output: (B, num_moves) logits."""

    def __init__(self, in_channels: int = 17, num_moves: int = 4544):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64,         128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,        128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,        64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
        )  # output: (B, 64, 8, 8)

        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_moves),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.backbone(x))

    def predict_move(
        self, board_tensor: torch.Tensor, legal_moves_idx: list[int]
    ) -> int:
        """Trả về index nước đi tốt nhất trong danh sách legal_moves_idx."""
        self.eval()
        with torch.no_grad():
            logits = self(board_tensor.unsqueeze(0))[0]
            mask = torch.full_like(logits, float("-inf"))
            mask[legal_moves_idx] = logits[legal_moves_idx]
            return mask.argmax().item()

class ResBlock(nn.Module):
    """Residual block nhỏ — giúp train ổn hơn PolicyNet phẳng."""
    def __init__(self, channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(x + self.net(x))

class DualNet(nn.Module):
    def __init__(self, in_ch: int = 17, num_moves: int = 4544, n_res: int = 4):
        super().__init__()
        # Shared backbone
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(n_res)])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32*8*8, num_moves)
        )
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, 1), nn.BatchNorm2d(1), nn.ReLU(),
            nn.Flatten(), nn.Linear(8*8, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh()   # output ∈ [-1, 1]
        )

    def forward(self, x):
        z = self.res_blocks(self.stem(x))
        return self.policy_head(z), self.value_head(z).squeeze(-1)

# Load weights từ PolicyNet (tuần 1) vào DualNet — chỉ copy những layer có tên + shape khớp.
# DualNet dùng stem/res_blocks thay vì backbone nên rất ít weight được transfer;
# hàm này chủ yếu hữu ích nếu DualNet sau được refactor lại cùng key names.
def load_pretrained_backbone(model: nn.Module, policy_path: str) -> None:
    old = torch.load(policy_path, map_location="cpu")
    new_state = model.state_dict()
    transferred = {k: v for k, v in old.items()
                   if k in new_state and new_state[k].shape == v.shape}
    new_state.update(transferred)
    model.load_state_dict(new_state, strict=False)
    print(f"Transferred {len(transferred)}/{len(old)} layers from pretrained PolicyNet")

if __name__ == "__main__":
    net = DualNet()
    params = sum(p.numel() for p in net.parameters())
    print(f"DualNet params: {params:,}")

    dummy = torch.zeros(1, 17, 8, 8)
    policy, value = net(dummy)
    print(f"Policy head : {policy.shape}")   # (1, 4544)
    print(f"Value  head : {value.shape}")    # (1,)
