import torch
import torch.nn as nn
import chess
import numpy as np
from pathlib import Path

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

class Encoding:
    _KNIGHT_DELTAS = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]

    _DIRECTIONS = [
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, 1), (1, -1),
        (-1, 1), (-1, -1)
    ]

    _PROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

    _PIECE_MAP = {
        chess.PAWN:   0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK:   3,
        chess.QUEEN:  4,
        chess.KING:   5,
    }

    def __init__(self):
        pass

    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """Public — dùng bởi Predictor. Trả về (12, 8, 8) float32."""
        return self.__board_to_tensor(board)

    def move_to_plane(self, move: chess.Move) -> tuple[int, int, int]:
        """Public — dùng bởi Predictor. Trả về (plane, fr, fc)."""
        return self.__move_to_plane(move)

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def __board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        state = torch.zeros((12, 8, 8), dtype=torch.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            channel = self._PIECE_MAP[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            row = 7 - (square // 8)
            col = square % 8
            state[channel, row, col] = 1.0
        return state

    def __move_to_plane(self, move: chess.Move) -> tuple[int, int, int]:
        from_sq = move.from_square
        to_sq   = move.to_square

        fr = 7 - chess.square_rank(from_sq)
        fc = chess.square_file(from_sq)
        tr = 7 - chess.square_rank(to_sq)
        tc = chess.square_file(to_sq)
        dr, dc = tr - fr, tc - fc

        for idx, (kdr, kdc) in enumerate(self._KNIGHT_DELTAS):
            if (dr, dc) == (kdr, kdc):
                return 56 + idx, fr, fc

        if move.promotion in self._PROMOTION_PIECES:
            move_dir  = dc + 1
            piece_idx = self._PROMOTION_PIECES.index(move.promotion)
            return 64 + piece_idx * 3 + move_dir, fr, fc

        for dir_idx, (rdir, cdir) in enumerate(self._DIRECTIONS):
            for dist in range(1, 8):
                if (dr, dc) == (rdir * dist, cdir * dist):
                    return dir_idx * 7 + (dist - 1), fr, fc

        raise ValueError(f"Unsupported move: {move}")

class Predictor:
    CHECKPOINT_NAME = "checkpoint_clean.pt"

    def __init__(self, device: torch.device | None = None):
        self.device    = device or torch.device("cpu")
        self.model     = None   # sẽ được build trong __load_checkpoint
        self._encoding = Encoding()
        self.__load_checkpoint()
        self.model.eval()
        print("[Predictor] Ready — model is in eval() mode.")
    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def __load_checkpoint(self):
        checkpoint_path = Path(__file__).parent / self.CHECKPOINT_NAME

        print(f"[Predictor] Looking for checkpoint at: {checkpoint_path}")

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"[Predictor] Checkpoint not found: {checkpoint_path}"
            )

        print(f"[Predictor] Found checkpoint ({checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB) — loading weights...")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # --- extract state_dict ---
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            print(f"[Predictor] Detected wrapped checkpoint, keys: {list(checkpoint.keys())}")
            print(f"[Predictor] Extracting: 'model_state'")
            state_dict = checkpoint["model_state"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            print(f"[Predictor] Detected wrapped checkpoint, keys: {list(checkpoint.keys())}")
            print(f"[Predictor] Extracting: 'model_state_dict'")
            state_dict = checkpoint["model_state_dict"]
        else:
            print(f"[Predictor] Treating checkpoint as bare state_dict.")
            state_dict = checkpoint

        # --- auto-detect architecture from state_dict ---
        num_res_blocks  = len({k.split(".")[1] for k in state_dict if k.startswith("res_chain.")})
        base_channels   = state_dict["stem.0.weight"].shape[0]
        in_channels     = state_dict["stem.0.weight"].shape[1]
        policy_channels = state_dict["policy_conv.3.weight"].shape[0]
        value_hidden    = state_dict["value_head.1.weight"].shape[0]
        num_attn_heads  = 16  # không lưu trong state_dict, giữ default

        print(f"[Predictor] Auto-detected architecture:")
        print(f"            in_channels     = {in_channels}")
        print(f"            base_channels   = {base_channels}")
        print(f"            num_res_blocks  = {num_res_blocks}")
        print(f"            policy_channels = {policy_channels}")
        print(f"            value_hidden    = {value_hidden}")

        # --- rebuild model với đúng config ---
        self.model = PolicyValueNet(
            in_channels     = in_channels,
            base_channels   = base_channels,
            num_res_blocks  = num_res_blocks,
            num_attn_heads  = num_attn_heads,
            policy_channels = policy_channels,
            value_hidden    = value_hidden,
        ).to(self.device)

        num_tensors  = len(state_dict)
        total_params = sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))
        print(f"[Predictor] Tensors : {num_tensors}  |  Params : {total_params:,}")

        self.model.load_state_dict(state_dict)
        print(f"[Predictor] Weights loaded successfully — all keys matched.")
    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def predict(self, board: chess.Board) -> tuple[dict[chess.Move, float], torch.Tensor]:
        x = self._encoding.board_to_tensor(board).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value_logits = self.model(x)

        value_probs   = torch.softmax(value_logits[0], dim=0)
        policy_logits = policy_logits[0]

        legal_moves = list(board.legal_moves)
        move_logits = {
            move: policy_logits[self._encoding.move_to_plane(move)].item()
            for move in legal_moves
        }
        probs  = torch.softmax(torch.tensor(list(move_logits.values())), dim=0).tolist()
        policy = {m: p for m, p in zip(move_logits.keys(), probs)}

        return policy, value_probs

    def get_value_prob(self, board: chess.Board) -> dict[str, float]:
        _, value_probs = self.predict(board)
        return {
            "loss": value_probs[0].item(),
            "draw": value_probs[1].item(),
            "win":  value_probs[2].item(),
        }

    def get_top_moves(self, board: chess.Board, n: int) -> list[tuple[chess.Move, float]]:
        policy, _ = self.predict(board)
        ranked     = sorted(policy.items(), key=lambda x: x[1], reverse=True)
        return ranked[:n]