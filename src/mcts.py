"""MCTS hai chế độ: UCB1 rollout (mcts_search) và PUCT AlphaZero (run_puct_search)."""

import math, chess, random, torch
import numpy as np
from typing import Optional

from src.board import encode_board, encode_board_with_meta


class MCTSNode:
    def __init__(self, board: chess.Board, parent: Optional["MCTSNode"] = None,
                 move: Optional[chess.Move] = None, prior: float = 0.0):
        self.board    = board.copy()
        self.parent   = parent
        self.move     = move
        self.children: list["MCTSNode"] = []
        self.visits   = 0
        self.value    = 0.0        # Tổng value tích lũy (theo góc nhìn của chính node này)
        self.prior    = prior      # P(s,a) — policy prob từ DualNet (dùng cho PUCT)
        self.expanded = False      # đã expand-with-policy chưa (AlphaZero style)
        self.untried  = list(board.legal_moves)
        random.shuffle(self.untried)

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def ucb1(self, c: float = 1.41) -> float:
        """UCB1 phiên bản Negamax đồng bộ với vòng lặp Backpropagate.

        node.value lưu từ góc nhìn của chính node này (current player).
        Khi cha chọn con, cha muốn con có điểm THẤP (xấu cho đối thủ = tốt cho ta),
        nên exploit = -(Q/N)."""
        if self.visits == 0:
            return float('inf')
        exploit = -(self.value / self.visits)
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def puct(self, c_puct: float = 1.5) -> float:
        """Công thức selection của AlphaZero: Q(s,a) + U(s,a).

        Q  = -(value/visits)  → flip perspective (giống ucb1, negamax).
        U  = c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child).
        Node chưa thăm: Q=0, vẫn có U > 0 nhờ prior → policy net hướng dẫn search."""
        q = 0.0 if self.visits == 0 else -(self.value / self.visits)
        u = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q + u

    def best_child(self, c: float = 1.41) -> "MCTSNode":
        return max(self.children, key=lambda n: n.ucb1(c))


# ── Phase 1: SELECT ──────────────────────────────────────────────────────────

def select(node: MCTSNode) -> MCTSNode:
    """Đi xuống cây theo UCB1 đến node chưa fully expanded hoặc terminal."""
    while not node.is_terminal():
        if not node.is_fully_expanded():
            return node
        node = node.best_child()
    return node


# ── Phase 2: EXPAND ──────────────────────────────────────────────────────────

def expand(node: MCTSNode) -> MCTSNode:
    """Thêm 1 child chưa được thử, trả về child đó."""
    move = node.untried.pop()
    new_board = node.board.copy()
    new_board.push(move)
    child = MCTSNode(new_board, parent=node, move=move)
    node.children.append(child)
    return child


# ── Phase 3: SIMULATE ────────────────────────────────────────────────────────

def _random_rollout(node: MCTSNode, max_moves: int) -> float:
    """Random rollout. Trả về value từ góc nhìn của current player tại node này.
    +1.0 = current player thắng, -1.0 = thua, 0.0 = hòa / vượt giới hạn."""
    board = node.board.copy()
    current_player = board.turn      # người đang đến lượt tại node này
    depth = 0
    while not board.is_game_over() and depth < max_moves:
        board.push(random.choice(list(board.legal_moves)))
        depth += 1
    result = board.result()          # "1-0" | "0-1" | "1/2-1/2" | "*"
    if result == "1-0":
        return  1.0 if current_player == chess.WHITE else -1.0
    if result == "0-1":
        return -1.0 if current_player == chess.WHITE else  1.0
    return 0.0


def _eval_with_value_head(node: MCTSNode, model: torch.nn.Module, device: str) -> float:
    """Dùng value head của model thay random rollout — O(1), auto-detect 12/17 channels."""
    in_ch = getattr(model.stem[0], 'in_channels', 17)
    encoder = encode_board_with_meta if in_ch == 17 else encode_board
    x = encoder(node.board).unsqueeze(0).to(device)
    with torch.no_grad():
        _, value = model(x)          # value: (1,) scalar, current-player view
    return float(value.item())       # NO negate: UCB1 dùng -(Q/N) để flip perspective


def simulate(node: MCTSNode,
             model: Optional[torch.nn.Module] = None,
             device: str = 'cpu',
             max_moves: int = 50) -> float:
    """Đánh giá node lá: dùng value head nếu có model, ngược lại random rollout."""
    if model is not None:
        return _eval_with_value_head(node, model, device)
    return _random_rollout(node, max_moves)


# ── Phase 4: BACKPROPAGATE ───────────────────────────────────────────────────

def backpropagate(node: MCTSNode, value: float) -> None:
    """Cập nhật ngược lên root. Flip sign ở mỗi tầng vì perspective đổi."""
    while node is not None:
        node.visits += 1
        node.value  += value
        value = -value      # flip: thắng của con = thua của cha
        node = node.parent


# ── Main search ──────────────────────────────────────────────────────────────

def mcts_search(board: chess.Board,
                n_simulations: int = 100,
                model: Optional[torch.nn.Module] = None,
                device: str = 'cpu') -> chess.Move:
    """Chạy MCTS, trả về move có nhiều visits nhất.

    Chọn theo visit count — KHÔNG phải UCB1 cao nhất.
    UCB1 chỉ dùng trong Select để cân bằng exploit vs explore.

    model=None  → random rollout per simulation.
    model=DualNet → value head replaces rollout (fast + accurate)."""
    root = MCTSNode(board)
    if model is not None:
        model.eval()

    for _ in range(n_simulations):
        leaf = select(root)
        if not leaf.is_terminal():
            leaf = expand(leaf)
        value = simulate(leaf, model=model, device=device)
        backpropagate(leaf, value)

    best = max(root.children, key=lambda n: n.visits)
    mode = "value-head" if model is not None else "rollout"
    print(f"MCTS [{mode}]: {n_simulations} sims | best={best.move} visits={best.visits}")
    return best.move


# ── AlphaZero PUCT search ─────────────────────────────────────────────────────

def _encoder_for(model: torch.nn.Module):
    in_ch = getattr(model.stem[0], 'in_channels', 17)
    return encode_board_with_meta if in_ch == 17 else encode_board


def expand_with_policy(node: MCTSNode, model: torch.nn.Module,
                       move2idx: dict, device: str = 'cpu') -> float:
    """Expand TẤT CẢ legal moves cùng lúc, gán prior = policy prob của model.

    Trả về value (value head) từ góc nhìn của current player — KHÔNG random rollout.
    Đây là điểm cốt lõi của AlphaZero: policy net hướng dẫn search qua prior."""
    encoder = _encoder_for(model)
    x = encoder(node.board).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value = model(x)
    probs = torch.softmax(logits[0], dim=0)

    # Lấy prior cho từng nước hợp lệ rồi CHUẨN HÓA trên tập nước hợp lệ
    # (softmax gốc trải trên cả 4544 vocab → tổng prior nước hợp lệ rất nhỏ).
    legal = list(node.board.legal_moves)
    n_out = probs.shape[0]
    def _prior(m):
        idx = move2idx.get(m.uci(), -1)
        return probs[idx].item() if 0 <= idx < n_out else 1e-8
    raw = [_prior(m) for m in legal]
    total = sum(raw) or 1.0
    for move, p in zip(legal, raw):
        child_board = node.board.copy()
        child_board.push(move)
        node.children.append(
            MCTSNode(child_board, parent=node, move=move, prior=p / total))

    node.expanded = True
    return float(value.item())


def add_dirichlet_noise(root: MCTSNode, alpha: float = 0.3, eps: float = 0.25) -> None:
    """Trộn Dirichlet noise vào prior của root children (AlphaZero standard).

    Bắt buộc cho self-play: ép explore, tránh mọi game giống hệt nhau → collapse."""
    if not root.children:
        return
    noise = np.random.dirichlet([alpha] * len(root.children))
    for child, n in zip(root.children, noise):
        child.prior = (1 - eps) * child.prior + eps * n


def select_puct(node: MCTSNode, c_puct: float = 1.5) -> MCTSNode:
    """Đi xuống cây theo PUCT đến node chưa expand hoặc terminal."""
    while node.expanded and not node.is_terminal():
        node = max(node.children, key=lambda n: n.puct(c_puct))
    return node


def run_puct_search(board: chess.Board, model: torch.nn.Module, move2idx: dict,
                    n_sims: int = 100, c_puct: float = 1.5,
                    add_noise: bool = False, device: str = 'cpu') -> MCTSNode:
    """Chạy PUCT-MCTS, trả về root (đã có visit counts trên children).

    add_noise=True → thêm Dirichlet noise ở root (dùng trong self-play)."""
    model.eval()
    root = MCTSNode(board)
    root_value = expand_with_policy(root, model, move2idx, device)
    backpropagate(root, root_value)
    if add_noise:
        add_dirichlet_noise(root)

    for _ in range(n_sims):
        leaf = select_puct(root, c_puct)
        if leaf.is_terminal():
            value = _eval_terminal(leaf)
        else:
            value = expand_with_policy(leaf, model, move2idx, device)
        backpropagate(leaf, value)
    return root


def _eval_terminal(node: MCTSNode) -> float:
    """Value chính xác cho terminal node, góc nhìn current player."""
    result = node.board.result()
    if result == "1-0":
        return 1.0 if node.board.turn == chess.WHITE else -1.0
    if result == "0-1":
        return -1.0 if node.board.turn == chess.WHITE else 1.0
    return 0.0


def select_move_with_temperature(root: MCTSNode, temperature: float = 1.0,
                                  move_count: int = 0, temp_moves: int = 30):
    """Chọn nước từ visit counts. <temp_moves nước đầu: sample (explore);
    sau đó: argmax (chơi tốt nhất). Trả về (move, policy_target_vector_visits)."""
    visits = np.array([c.visits for c in root.children], dtype=np.float64)
    if move_count < temp_moves and temperature > 0:
        probs = visits ** (1.0 / temperature)
        probs /= probs.sum()
        idx = np.random.choice(len(root.children), p=probs)
    else:
        idx = int(visits.argmax())
    return root.children[idx].move


def mcts_search_puct(board: chess.Board, model: torch.nn.Module, move2idx: dict,
                     n_sims: int = 100, c_puct: float = 1.5,
                     device: str = 'cpu') -> chess.Move:
    """AlphaZero-style search cho lúc chơi/đánh giá (deterministic, không noise)."""
    root = run_puct_search(board, model, move2idx, n_sims=n_sims,
                           c_puct=c_puct, add_noise=False, device=device)
    best = max(root.children, key=lambda n: n.visits)
    print(f"MCTS [PUCT]: {n_sims} sims | best={best.move} visits={best.visits}")
    return best.move


def mcts_search_hybrid(board: chess.Board,
                       model: torch.nn.Module,
                       n_simulations: int = 100,
                       device: str = 'cpu') -> chess.Move:
    """MCTS với DualNet value head thay random rollout (AlphaZero style).

    Wrapper của mcts_search — model là bắt buộc (không optional).
    Auto-detect channel encoding (12 hay 17) từ model.stem[0].in_channels."""
    return mcts_search(board, n_simulations=n_simulations, model=model, device=device)

def mcts_batched(board: chess.Board,
                 model: torch.nn.Module,
                 n_sims: int = 100,
                 batch_size: int = 16,
                 device: str = 'cpu') -> chess.Move:
    """Batch MCTS: gom batch_size leaf nodes → 1 lần forward pass → 5–20x nhanh hơn.

    Tự động chọn encoder 12/17 channels dựa vào model.stem[0].in_channels."""
    in_ch = getattr(model.stem[0], 'in_channels', 17)
    encoder = encode_board_with_meta if in_ch == 17 else encode_board

    root = MCTSNode(board)
    model.eval()

    n_rounds = max(1, n_sims // batch_size)
    for _ in range(n_rounds):
        leaves: list[MCTSNode] = []
        for _ in range(batch_size):
            leaf = select(root)
            if not leaf.is_terminal():
                leaf = expand(leaf)
            leaves.append(leaf)

        # Tách terminal nodes (dùng rollout) khỏi non-terminal (dùng model)
        non_terminal = [n for n in leaves if not n.is_terminal()]
        terminal     = [n for n in leaves if n.is_terminal()]

        if non_terminal:
            tensors = torch.stack([encoder(n.board) for n in non_terminal]).to(device)
            with torch.no_grad():
                _, value_batch = model(tensors)
            for node, val in zip(non_terminal, value_batch):
                backpropagate(node, float(val.item()))

        for node in terminal:
            backpropagate(node, _random_rollout(node, max_moves=0))

    best = max(root.children, key=lambda n: n.visits)
    print(f"MCTS batched: {n_rounds * batch_size} sims | best={best.move} visits={best.visits}")
    return best.move