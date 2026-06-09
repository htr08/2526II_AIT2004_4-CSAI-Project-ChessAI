# !pip install chess -q

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import chess.engine

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

MODEL_PATH = "/kaggle/input/datasets/alinhtrng/model-rl/best_model.pt"
CHANNELS   = 64
RES_BLOCKS = 6
ATTN_HEADS = 4
ATTN_EVERY = 3
C_PUCT     = 5.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")
print(f"✓ Device: {DEVICE}")


# ══════════════════════════════════════════════════════════
# BOARD ENCODING
# ══════════════════════════════════════════════════════════

PIECE_MAP = {
    (chess.PAWN,   chess.WHITE): 0,  (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,  (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,  (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,  (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,  (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10, (chess.KING,   chess.BLACK): 11,
}

def board_to_tensor(board: chess.Board, last_move=None) -> np.ndarray:
    tensor = np.zeros((15, 8, 8), dtype=np.float32)
    flip   = (board.turn == chess.BLACK)
    for sq, piece in board.piece_map().items():
        eff_sq = chess.square_mirror(sq) if flip else sq
        row, col = eff_sq // 8, eff_sq % 8
        color = (not piece.color) if flip else piece.color
        tensor[PIECE_MAP[(piece.piece_type, color)]][row][col] = 1.0
    tensor[12] = 1.0
    if last_move is not None:
        fr = chess.square_mirror(last_move.from_square) if flip else last_move.from_square
        to = chess.square_mirror(last_move.to_square)   if flip else last_move.to_square
        tensor[13][fr // 8][fr % 8] = 1.0
        tensor[14][to // 8][to % 8] = 1.0
    return tensor

def encode_move_canonical(move: chess.Move, flip: bool) -> int:
    if not flip:
        return move.from_square * 64 + move.to_square
    return chess.square_mirror(move.from_square) * 64 + chess.square_mirror(move.to_square)

def decode_move_canonical(move_idx: int, board: chess.Board, flip: bool):
    from_sq = move_idx // 64
    to_sq   = move_idx % 64
    if flip:
        from_sq = chess.square_mirror(from_sq)
        to_sq   = chess.square_mirror(to_sq)
    move = chess.Move(from_sq, to_sq)
    if move in board.legal_moves:
        return move
    for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        pm = chess.Move(from_sq, to_sq, promotion=promo)
        if pm in board.legal_moves:
            return pm
    return None


# ══════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.se    = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // 4), nn.ReLU(),
            nn.Linear(channels // 4, channels), nn.Sigmoid(),
        )
    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x * self.se(x).view(x.size(0), -1, 1, 1)
        return F.relu(x + r)

class BoardAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.attn  = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.ff    = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.ReLU(),
            nn.Linear(channels * 2, channels),
        )
        self.norm2 = nn.LayerNorm(channels)
    def forward(self, x):
        B, C, H, W = x.shape
        s = x.flatten(2).permute(0, 2, 1)
        a, _ = self.attn(s, s, s)
        s = self.norm1(s + a)
        s = self.norm2(s + self.ff(s))
        return s.permute(0, 2, 1).view(B, C, H, W)

class ChessNet(nn.Module):
    def __init__(self, in_channels=15, channels=64,
                 res_blocks=6, attn_heads=4, attn_every=3):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
        )
        layers = []
        for i in range(res_blocks):
            layers.append(ResBlock(channels))
            if (i + 1) % attn_every == 0:
                layers.append(BoardAttention(channels, attn_heads))
        self.backbone    = nn.Sequential(*layers)
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(32)
        self.policy_fc   = nn.Linear(32 * 8 * 8, 4096)
        self.value_conv  = nn.Conv2d(channels, 8, 1, bias=False)
        self.value_bn    = nn.BatchNorm2d(8)
        self.value_fc1   = nn.Linear(8 * 8 * 8, 64)
        self.value_fc2   = nn.Linear(64, 3)   # WDL

    def forward(self, x):
        x = self.input_conv(x)
        x = self.backbone(x)
        p = F.relu(self.policy_bn(self.policy_conv(x))).flatten(1)
        p = self.policy_fc(p)
        v = F.relu(self.value_bn(self.value_conv(x))).flatten(1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        return p, v


# ══════════════════════════════════════════════════════════
# MCTS NODE
# ══════════════════════════════════════════════════════════

class MCTSNode:
    __slots__ = ["parent", "move", "prior_p",
                 "children", "visit_count", "total_value", "avg_value"]
    def __init__(self, parent=None, move=None, prior_p=0.0):
        self.parent      = parent
        self.move        = move
        self.prior_p     = prior_p
        self.children    = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.avg_value   = 0.0
    def is_leaf(self):
        return len(self.children) == 0
    def get_puct(self, c_puct):
        u = (c_puct * self.prior_p
             * math.sqrt(self.parent.visit_count)
             / (1 + self.visit_count))
        return self.avg_value + u


# ══════════════════════════════════════════════════════════
# MCTS ENGINE
# ══════════════════════════════════════════════════════════

class MCTSEngine:
    def __init__(self, model, device, c_puct=C_PUCT,
                 dirichlet_alpha=0.3, dirichlet_eps=0.25):
        self.model           = model
        self.device          = device
        self.c_puct          = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps   = dirichlet_eps

    @torch.no_grad()
    def _evaluate_leaf(self, board, last_move):
        flip   = (board.turn == chess.BLACK)
        tensor = torch.from_numpy(
            board_to_tensor(board, last_move)
        ).unsqueeze(0).to(self.device)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            policy_logits, value_logits = self.model(tensor)
            wdl_probs = torch.softmax(value_logits, dim=-1)
            value     = float((wdl_probs[0, 0] - wdl_probs[0, 2]).item())
        policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        legal_moves  = list(board.legal_moves)
        priors, p_sum = {}, 0.0
        for m in legal_moves:
            p         = max(float(policy_probs[encode_move_canonical(m, flip)]), 0.0)
            priors[m] = p
            p_sum    += p
        if p_sum > 1e-8:
            for m in priors: priors[m] /= p_sum
        else:
            uniform = 1.0 / max(len(legal_moves), 1)
            for m in legal_moves: priors[m] = uniform
        return priors, value

    def _do_expand(self, node, priors):
        for move, prior in priors.items():
            node.children[move] = MCTSNode(parent=node, move=move, prior_p=prior)

    def _backprop(self, node, value):
        while node.parent is not None:
            node.visit_count  += 1
            node.total_value  -= value
            node.avg_value     = node.total_value / node.visit_count
            value              = -value
            node               = node.parent
        node.visit_count += 1

    def search(self, board, last_move, num_simulations=400, add_noise=False):
        root = MCTSNode()
        if board.is_game_over():
            return root
        priors, _ = self._evaluate_leaf(board.copy(), last_move)
        self._do_expand(root, priors)
        root.visit_count = 1
        if add_noise:
            moves = list(root.children.keys())
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
            for m, n in zip(moves, noise):
                c = root.children[m]
                c.prior_p = (1 - self.dirichlet_eps) * c.prior_p + self.dirichlet_eps * n
        for _ in range(num_simulations):
            node      = root
            board_sim = board.copy()
            last_sim  = last_move
            while not node.is_leaf():
                best_move = max(node.children,
                                key=lambda m: node.children[m].get_puct(self.c_puct))
                node = node.children[best_move]
                board_sim.push(best_move)
                last_sim = best_move
            if board_sim.is_game_over():
                outcome = board_sim.outcome()
                value   = 0.0 if (outcome is None or outcome.winner is None) else -1.0
            else:
                priors, value = self._evaluate_leaf(board_sim, last_sim)
                self._do_expand(node, priors)
            self._backprop(node, value)
        return root

    def best_move(self, board, last_move, num_simulations=400):
        root = self.search(board, last_move, num_simulations, add_noise=False)
        if not root.children:
            legal = list(board.legal_moves)
            return legal[0] if legal else None
        return max(root.children,
                   key=lambda m: root.children[m].visit_count)


# ══════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════

def load_model(path=MODEL_PATH):
    model = ChessNet(channels=CHANNELS, res_blocks=RES_BLOCKS,
                     attn_heads=ATTN_HEADS, attn_every=ATTN_EVERY).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"✓ Model loaded: {path}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ══════════════════════════════════════════════════════════
# VS STOCKFISH
# ══════════════════════════════════════════════════════════

def play_vs_stockfish(model, stockfish_elo=1500,
                      n_games=10, num_simulations=400):
    engine_sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    engine_sf.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
    mcts    = MCTSEngine(model, DEVICE, c_puct=C_PUCT)
    results = {"win": 0, "draw": 0, "loss": 0}

    for game_idx in range(n_games):
        board        = chess.Board()
        last_move    = None
        bot_is_white = (game_idx % 2 == 0)

        while not board.is_game_over():
            if (board.turn == chess.WHITE) == bot_is_white:
                move = mcts.best_move(board, last_move, num_simulations)
                if move is None or move not in board.legal_moves:
                    move = list(board.legal_moves)[0]
            else:
                result = engine_sf.play(board, chess.engine.Limit(time=0.1))
                move   = result.move
            board.push(move)
            last_move = move

        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            results["draw"] += 1
        elif (outcome.winner == chess.WHITE) == bot_is_white:
            results["win"] += 1
        else:
            results["loss"] += 1
        print(f"  Game {game_idx+1}: {'W' if bot_is_white else 'B'} → {board.result()}")

    engine_sf.quit()
    total    = sum(results.values())
    win_rate = (results["win"] + 0.5 * results["draw"]) / max(total, 1)
    print(f"\nVs Stockfish {stockfish_elo}: {results} | win_rate={win_rate:.3f}")
    return results


# ══════════════════════════════════════════════════════════
# CHẠY
# ══════════════════════════════════════════════════════════

model = load_model(MODEL_PATH)
play_vs_stockfish(model, stockfish_elo=1500, n_games=10, num_simulations=400)