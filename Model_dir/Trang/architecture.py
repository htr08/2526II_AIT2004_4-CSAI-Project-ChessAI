import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """CNN backbone + policy head. Input: (B, 17, 8, 8) → Output: (B, num_moves) logits."""

    def __init__(self, in_channels: int = 17, num_moves: int = 4544):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),
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

    def __init__(self, channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class DualNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 17,
        num_moves: int = 4544,
        num_res_blocks: int = 4,
        base_channels: int = 128,
        board_h: int = 8,
        board_w: int = 8,
    ):
        super().__init__()
        spatial = board_h * board_w

        # Shared backbone
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.res_chain = nn.Sequential(
            *[ResBlock(base_channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(base_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * spatial, num_moves),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * spatial, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),  # output ∈ [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.res_chain(self.stem(x))
        return self.policy_head(z), self.value_head(z).squeeze(-1)


def load_pretrained_backbone(model: nn.Module, policy_path: str) -> None:
    """Load weights từ PolicyNet vào DualNet — chỉ copy những layer có tên + shape khớp."""
    old = torch.load(policy_path, map_location="cpu")
    new_state = model.state_dict()
    transferred = {
        k: v
        for k, v in old.items()
        if k in new_state and new_state[k].shape == v.shape
    }
    new_state.update(transferred)
    model.load_state_dict(new_state, strict=False)
    print(
        f"Transferred {len(transferred)}/{len(old)} layers from pretrained PolicyNet"
    )


if __name__ == "__main__":
    net = DualNet()
    params = sum(p.numel() for p in net.parameters())
    print(f"DualNet params: {params:,}")

    dummy = torch.zeros(1, 17, 8, 8)
    policy, value = net(dummy)
    print(f"Policy head : {policy.shape}")  # (1, 4544)
    print(f"Value  head : {value.shape}")  # (1,)