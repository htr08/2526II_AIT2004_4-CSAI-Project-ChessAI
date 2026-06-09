import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

MODEL_PATH = os.environ.get("MODEL_PATH", "best_model.pt")
CHANNELS   = int(os.environ.get("CHANNELS", 64))
RES_BLOCKS = int(os.environ.get("RES_BLOCKS", 6))
ATTN_HEADS = int(os.environ.get("ATTN_HEADS", 4))
ATTN_EVERY = int(os.environ.get("ATTN_EVERY", 3))
C_PUCT     = float(os.environ.get("C_PUCT", 5.0))
MCTS_SIMS  = int(os.environ.get("MCTS_SIMS", 200))

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")
print(f"✓ Device: {DEVICE} | MCTS sims: {MCTS_SIMS}")


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

def board_to_tensor(board, last_move=None):
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

def encode_move_canonical(move, flip):
    if not flip:
        return move.from_square * 64 + move.to_square
    return chess.square_mirror(move.from_square) * 64 + chess.square_mirror(move.to_square)


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
        return F.relu(x * self.se(x).view(x.size(0), -1, 1, 1) + r)

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
        s = self.norm2(self.norm1(s + a) + self.ff(self.norm1(s + a)))
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
        self.value_fc2   = nn.Linear(64, 3)
    def forward(self, x):
        x = self.input_conv(x)
        x = self.backbone(x)
        p = F.relu(self.policy_bn(self.policy_conv(x))).flatten(1)
        p = self.policy_fc(p)
        v = F.relu(self.value_bn(self.value_conv(x))).flatten(1)
        v = self.value_fc2(F.relu(self.value_fc1(v)))
        return p, v


# ══════════════════════════════════════════════════════════
# MCTS
# ══════════════════════════════════════════════════════════

class MCTSNode:
    __slots__ = ["parent", "move", "prior_p",
                 "children", "visit_count", "total_value", "avg_value"]
    def __init__(self, parent=None, move=None, prior_p=0.0):
        self.parent = parent; self.move = move; self.prior_p = prior_p
        self.children = {}; self.visit_count = 0
        self.total_value = 0.0; self.avg_value = 0.0
    def is_leaf(self): return len(self.children) == 0
    def get_puct(self, c_puct):
        return self.avg_value + c_puct * self.prior_p * math.sqrt(
            self.parent.visit_count) / (1 + self.visit_count)

class MCTSEngine:
    def __init__(self, model, device, c_puct=C_PUCT):
        self.model  = model
        self.device = device
        self.c_puct = c_puct

    @torch.no_grad()
    def _evaluate_leaf(self, board, last_move):
        flip   = (board.turn == chess.BLACK)
        tensor = torch.from_numpy(
            board_to_tensor(board, last_move)
        ).unsqueeze(0).to(self.device)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            policy_logits, value_logits = self.model(tensor)
            wdl   = torch.softmax(value_logits, dim=-1)
            value = float((wdl[0, 0] - wdl[0, 2]).item())
        probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        legal = list(board.legal_moves)
        pri   = {m: max(float(probs[encode_move_canonical(m, flip)]), 0.0) for m in legal}
        s     = sum(pri.values())
        if s > 1e-8: pri = {m: p/s for m, p in pri.items()}
        else:        pri = {m: 1/len(legal) for m in legal}
        return pri, value

    def _expand(self, node, priors):
        for m, p in priors.items():
            node.children[m] = MCTSNode(parent=node, move=m, prior_p=p)

    def _backprop(self, node, value):
        while node.parent is not None:
            node.visit_count += 1
            node.total_value -= value
            node.avg_value    = node.total_value / node.visit_count
            value = -value; node = node.parent
        node.visit_count += 1

    def best_move(self, board, last_move, num_simulations=MCTS_SIMS):
        root = MCTSNode()
        if board.is_game_over():
            return None
        pri, _ = self._evaluate_leaf(board.copy(), last_move)
        self._expand(root, pri)
        root.visit_count = 1
        for _ in range(num_simulations):
            node = root; bsim = board.copy(); lsim = last_move
            while not node.is_leaf():
                m    = max(node.children, key=lambda x: node.children[x].get_puct(self.c_puct))
                node = node.children[m]; bsim.push(m); lsim = m
            if bsim.is_game_over():
                out = bsim.outcome()
                val = 0.0 if (out is None or out.winner is None) else -1.0
            else:
                pri, val = self._evaluate_leaf(bsim, lsim)
                self._expand(node, pri)
            self._backprop(node, val)
        if not root.children:
            legal = list(board.legal_moves)
            return legal[0] if legal else None
        return max(root.children, key=lambda m: root.children[m].visit_count)


# ══════════════════════════════════════════════════════════
# LOAD MODEL (1 lần khi khởi động)
# ══════════════════════════════════════════════════════════

model = ChessNet(channels=CHANNELS, res_blocks=RES_BLOCKS,
                 attn_heads=ATTN_HEADS, attn_every=ATTN_EVERY).to(DEVICE)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✓ Model loaded: {MODEL_PATH}")
else:
    print(f"⚠ Model not found at {MODEL_PATH}, using random weights")

model.eval()
engine = MCTSEngine(model, DEVICE, c_puct=C_PUCT)


# ══════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/move", methods=["POST"])
def get_move():
    """
    Nhận FEN + last_move_uci, trả về nước đi tốt nhất của bot.
    Body: { "fen": "...", "last_move": "e2e4" | null }
    Response: { "move": "e7e5", "eval": 0.12 }
    """
    data      = request.get_json()
    fen       = data.get("fen", chess.STARTING_FEN)
    last_uci  = data.get("last_move", None)
    sims      = int(data.get("sims", MCTS_SIMS))

    try:
        board     = chess.Board(fen)
        last_move = chess.Move.from_uci(last_uci) if last_uci else None
    except Exception as e:
        return jsonify({"error": f"Invalid FEN or move: {e}"}), 400

    if board.is_game_over():
        return jsonify({"move": None, "result": board.result()})

    move = engine.best_move(board, last_move, num_simulations=sims)
    if move is None:
        return jsonify({"error": "No legal moves"}), 400

    # Tính eval sau nước đi
    with torch.no_grad():
        tensor = torch.from_numpy(
            board_to_tensor(board, last_move)
        ).unsqueeze(0).to(DEVICE)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            _, v_logits = model(tensor)
            wdl  = torch.softmax(v_logits, dim=-1)
            eval_score = float((wdl[0, 0] - wdl[0, 2]).item())

    return jsonify({
        "move"  : move.uci(),
        "eval"  : round(eval_score, 3),
    })


@app.route("/api/legal_moves", methods=["POST"])
def legal_moves():
    """Trả về list nước đi hợp lệ từ FEN."""
    data = request.get_json()
    fen  = data.get("fen", chess.STARTING_FEN)
    try:
        board = chess.Board(fen)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    moves = [m.uci() for m in board.legal_moves]
    return jsonify({"moves": moves, "is_game_over": board.is_game_over(),
                    "result": board.result() if board.is_game_over() else None})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "device": DEVICE, "sims": MCTS_SIMS})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)