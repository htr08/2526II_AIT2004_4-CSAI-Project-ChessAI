import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ACTIONS = 4096
INPUT_CHANNELS = 12

class ConvBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):

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


class PolicyNet(nn.Module):

    def __init__(self, channels: int = 128, n_res_blocks: int = 3):
        super().__init__()
        self.net = PolicyValueNet(channels=channels, n_res_blocks=n_res_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.net(x)
        return logits