import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.30),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x) + x


class SpatialMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 16):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).permute(0, 2, 1)
        out, _ = self.mha(seq, seq, seq)
        return out.permute(0, 2, 1).view(B, C, H, W)


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 256,
        num_res_blocks: int = 6,
        num_attn_heads: int = 16,
        policy_channels: int = 73,
        value_hidden: int = 256,
        board_h: int = 8,
        board_w: int = 8,
    ):
        super().__init__()
        spatial = board_h * board_w
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.res_chain = nn.Sequential(
            *[ResBlock(base_channels) for _ in range(num_res_blocks)]
        )
        self.attention = SpatialMultiheadAttention(base_channels, num_attn_heads)
        self.policy_conv = nn.Sequential(
            nn.Conv2d(base_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, policy_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(inplace=True),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * spatial, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 3),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_chain(x)
        x = self.attention(x)
        policy_logits = self.policy_conv(x)
        value = self.value_head(x)
        return policy_logits, value