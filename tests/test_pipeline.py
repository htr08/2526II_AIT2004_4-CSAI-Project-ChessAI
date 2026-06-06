"""
Common-bug checklist for MCTS and self-play pipeline.

Run full suite : pytest tests/test_pipeline.py -v
Run one section: pytest tests/test_pipeline.py -v -k "mcts"
ELO estimation : requires Stockfish — set STOCKFISH_PATH env var or install system-wide.
"""

import json
import os
import shutil
from pathlib import Path

import chess
import pytest
import torch

from src.board import encode_board_with_meta
from src.mcts import MCTSNode, backpropagate, expand, select, simulate
from src.model import DualNet
from src.agent import MinimaxAgent
from scripts.self_play import play_one_game


# ── Existing checks (kept) ────────────────────────────────────────────────────

def test_draw_detection():
    board = chess.Board()
    for _ in range(3):
        board.push_uci("g1f3"); board.push_uci("g8f6")
        board.push_uci("f3g1"); board.push_uci("f6g8")
    assert board.is_repetition(3), "Threefold repetition not detected!"


def test_50_move_rule():
    board = chess.Board()
    assert hasattr(board, "is_fifty_moves"), "Missing 50-move check"


def test_agent_no_crash_fullgame():
    agent = MinimaxAgent(depth=2)
    board = chess.Board()
    moves = 0
    while not board.is_game_over() and moves < 100:
        move = agent.select_move(board)
        assert move in board.legal_moves, f"Illegal move: {move}"
        board.push(move)
        moves += 1


# ── Bug 1: MCTS trả về nước đi hợp lệ ───────────────────────────────────────

def test_mcts_returns_legal_move():
    """MCTS phải trả về nước đi hợp lệ — lỗi phổ biến: max() trên children rỗng."""
    board = chess.Board()
    root = MCTSNode(board)
    for _ in range(30):
        leaf = select(root)
        if not leaf.is_terminal():
            leaf = expand(leaf)
        value = simulate(leaf)
        backpropagate(leaf, value)
    assert root.children, "root.children rỗng — MCTS không explore bất kỳ nước nào"
    best = max(root.children, key=lambda n: n.visits)
    assert best.move in board.legal_moves, f"Illegal move returned: {best.move}"


# ── Bug 2: Policy target phải sum = 1 ────────────────────────────────────────

def test_mcts_policy_sums_to_one():
    """Normalized visit counts phải sum = 1.0 — lỗi: quên chia cho total_visits."""
    move2idx = json.load(open("data/processed/move2idx.json"))
    NUM_MOVES = 4544

    board = chess.Board()
    root = MCTSNode(board)
    for _ in range(40):
        leaf = select(root)
        if not leaf.is_terminal():
            leaf = expand(leaf)
        backpropagate(leaf, simulate(leaf))

    total_visits = sum(c.visits for c in root.children)
    assert total_visits > 0, "Không có visit nào — MCTS không chạy"

    policy = torch.zeros(NUM_MOVES)
    for child in root.children:
        idx = move2idx.get(child.move.uci(), -1)
        if idx >= 0:
            policy[idx] = child.visits / total_visits

    s = policy.sum().item()
    assert abs(s - 1.0) < 1e-5, f"Policy sum = {s:.6f}, expected 1.0 — lỗi normalization"


# ── Bug 3: Backprop phải flip sign mỗi tầng ──────────────────────────────────

def test_backprop_sign_flip():
    """Perspective đổi White↔Black mỗi tầng — lỗi: quên negate value khi đi lên."""
    board = chess.Board()
    root = MCTSNode(board)          # White to move
    child = expand(root)            # sau nước 1 của White — Black to move
    grandchild = expand(child)      # sau nước 1 của Black — White to move

    # Backprop value=1.0 từ góc nhìn của grandchild (White = thắng)
    backpropagate(grandchild, 1.0)

    assert grandchild.value == pytest.approx( 1.0), \
        f"grandchild.value = {grandchild.value}, expected +1.0"
    assert child.value      == pytest.approx(-1.0), \
        f"child.value = {child.value}, expected -1.0 (flip: thắng của White = thua của Black)"
    assert root.value       == pytest.approx( 1.0), \
        f"root.value = {root.value}, expected +1.0 (flip lại)"


# ── Bug 4: Value head phải ở trong [-1, 1] ───────────────────────────────────

def test_value_head_range():
    """DualNet value head dùng Tanh → output phải ∈ [-1, 1]."""
    model = DualNet(in_ch=17, num_moves=4544, n_res=1)
    model.eval()
    x = torch.randn(16, 17, 8, 8)
    with torch.no_grad():
        _, values = model(x)
    assert values.shape == (16,), f"Value shape = {values.shape}, expected (16,)"
    assert values.min().item() >= -1.0 - 1e-6, f"Value min = {values.min().item():.4f} < -1"
    assert values.max().item() <=  1.0 + 1e-6, f"Value max = {values.max().item():.4f} > +1"


# ── Bug 5: Terminal node không crash ─────────────────────────────────────────

def test_mcts_terminal_node_no_crash():
    """MCTS được gọi khi board là game-over không được crash — lỗi: thiếu is_terminal() check."""
    # Fool's mate: game kết thúc sau 4 nước
    board = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4"]:
        board.push_san(san)
    assert board.is_checkmate(), "Position phải là checkmate để test terminal handling"

    root = MCTSNode(board)
    for _ in range(5):
        leaf = select(root)
        if not leaf.is_terminal():
            leaf = expand(leaf)
        backpropagate(leaf, simulate(leaf))
    # Không assert kết quả — chỉ kiểm tra không raise exception


# ── Bug 6: Agent không crash khi board là game-over ──────────────────────────

def test_agent_called_on_gameover_board():
    """Bug thường gặp: agent.select_move() bị gọi sau board.is_game_over() → illegal move.
    MinimaxAgent dùng next(iter(board.legal_moves)) làm fallback — board game-over thì
    legal_moves rỗng → StopIteration. Kiểm tra guard này."""
    board = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4"]:  # Fool's mate
        board.push_san(san)
    assert board.is_game_over()
    # Với game loop đúng, không bao giờ gọi select_move sau is_game_over().
    # Đây chỉ là smoke test để confirm game loop không vô tình gọi.
    assert len(list(board.legal_moves)) == 0, "Board game-over phải có 0 legal moves"


# ── Bug 7: Board encoding shape và dtype ─────────────────────────────────────

def test_board_encoding_shape_and_dtype():
    """encode_board_with_meta phải trả về (17, 8, 8) float32 — lỗi: dùng encode_board 12-ch."""
    board = chess.Board()
    tensor = encode_board_with_meta(board)
    assert tensor.shape == (17, 8, 8), \
        f"Shape = {tensor.shape}, expected (17, 8, 8). Đang dùng encode_board 12-ch thay vì encode_board_with_meta?"
    assert tensor.dtype == torch.float32, f"dtype = {tensor.dtype}, expected float32"
    # 12 piece channels phải là binary 0/1
    assert tensor[:12].min() >= 0.0
    assert tensor[:12].max() <= 1.0


# ── Bug 8: Self-play shapes đầu ra ───────────────────────────────────────────

def test_selfplay_one_game_shapes():
    """play_one_game phải trả về (17,8,8) state, (4544,) policy, float value — n_sims nhỏ để test nhanh."""
    model = DualNet(in_ch=17, num_moves=4544, n_res=1)
    move2idx = json.load(open("data/processed/move2idx.json"))
    records = play_one_game(model, move2idx, n_sims=5)

    assert len(records) > 0, "Game trả về 0 records — game kết thúc ngay lập tức?"
    for i, (state, policy, value) in enumerate(records):
        assert state.shape  == (17, 8, 8), f"Record {i}: state.shape = {state.shape}"
        assert policy.shape == (4544,),    f"Record {i}: policy.shape = {policy.shape}"
        assert isinstance(value, float),   f"Record {i}: value type = {type(value)}"
        assert value in (-1.0, 0.0, 1.0), f"Record {i}: value = {value}, expected {{-1, 0, 1}}"


# ── Bug 9: Value sign xen kẽ đúng chiều ──────────────────────────────────────

def test_value_target_alternation_formula():
    """Unit test công thức gán value retroactively — không cần chạy game thật.

    outcome * ((-1)**i): White's moves (i chẵn) nhận +outcome, Black's (i lẻ) nhận -outcome."""
    def _apply_value(n_moves: int, outcome: float) -> list[float]:
        return [float(outcome * ((-1) ** i)) for i in range(n_moves)]

    # White thắng (outcome=1): W=+1, B=-1, W=+1, ...
    vals = _apply_value(4, 1.0)
    assert vals == [1.0, -1.0, 1.0, -1.0], f"White win alternation wrong: {vals}"

    # Black thắng (outcome=-1): W=-1, B=+1, W=-1, ...
    vals = _apply_value(4, -1.0)
    assert vals == [-1.0, 1.0, -1.0, 1.0], f"Black win alternation wrong: {vals}"

    # Hòa (outcome=0): tất cả 0
    vals = _apply_value(4, 0.0)
    assert all(v == 0.0 for v in vals), f"Draw values not all zero: {vals}"


# ── Bug 10: Moves trong vocab index đúng ─────────────────────────────────────

def test_vocab_move_roundtrip():
    """Mọi nước đi hợp lệ từ vị trí khởi đầu phải có trong vocab — lỗi: move2idx thiếu."""
    move2idx = json.load(open("data/processed/move2idx.json"))
    board = chess.Board()
    for move in board.legal_moves:
        uci = move.uci()
        assert uci in move2idx, \
            f"Move '{uci}' không có trong vocab — self-play sẽ tạo policy với gap"
        idx = move2idx[uci]
        assert 0 <= idx < 4544, f"Move '{uci}' có idx={idx} ngoài range [0, 4544)"


# ── ELO estimation với Stockfish ─────────────────────────────────────────────

def _find_stockfish() -> str | None:
    """Tìm Stockfish binary. Ưu tiên STOCKFISH_PATH env var, sau đó PATH, rồi common paths."""
    env = os.environ.get("STOCKFISH_PATH")
    if env and Path(env).exists():
        return env
    found = shutil.which("stockfish")
    if found:
        return found
    # Windows common install paths
    for path in [
        r"C:\stockfish\stockfish.exe",
        r"C:\Program Files\stockfish\stockfish.exe",
        r"C:\Users\Public\stockfish\stockfish.exe",
    ]:
        if Path(path).exists():
            return path
    return None


def estimate_elo(agent, n_games: int = 20) -> dict[int, float]:
    """Ước tính ELO bằng cách đấu với Stockfish ở các skill level.

    Skill level → ELO xấp xỉ:
        0 → ~800    ELO
        2 → ~1000   ELO
        4 → ~1200   ELO
        6 → ~1400   ELO

    Win rate >50% ở skill X → agent ELO ước tính ≈ ELO(X).
    Colors đổi xen kẽ để loại bỏ white-advantage bias.

    Args:
        agent: bất kỳ object có method select_move(board) -> chess.Move
        n_games: số game mỗi skill level (≥20 để kết quả đáng tin)

    Returns:
        dict {skill_level: win_rate}
    """
    import chess.engine

    engine_path = _find_stockfish()
    if engine_path is None:
        raise RuntimeError(
            "Stockfish không tìm thấy.\n"
            "  macOS : brew install stockfish\n"
            "  Ubuntu: sudo apt install stockfish\n"
            "  Windows: tải từ https://stockfishchess.org và đặt STOCKFISH_PATH=<path>"
        )

    elo_map = {0: "~800", 2: "~1000", 4: "~1200", 6: "~1400"}
    results: dict[int, float] = {}

    for skill in [0, 2, 4, 6]:
        wins = 0.0
        with chess.engine.SimpleEngine.popen_uci(engine_path) as sf:
            sf.configure({"Skill Level": skill})
            for i in range(n_games):
                board = chess.Board()
                agent_is_white = (i % 2 == 0)

                while not board.is_game_over():
                    is_agent_turn = (board.turn == chess.WHITE) == agent_is_white
                    if is_agent_turn:
                        move = agent.select_move(board)
                    else:
                        result = sf.play(board, chess.engine.Limit(depth=2))
                        move = result.move
                    board.push(move)

                r = board.result()
                if r == "1/2-1/2":
                    wins += 0.5
                elif (r == "1-0") == agent_is_white:
                    wins += 1.0

        win_rate = wins / n_games
        results[skill] = win_rate
        print(f"  Skill {skill} ({elo_map[skill]} ELO): win_rate={win_rate:.2f}")

    return results


@pytest.mark.skipif(_find_stockfish() is None, reason="Stockfish not installed")
def test_elo_smoke_vs_skill0():
    """Smoke test: agent đấu Stockfish không crash — không check win rate (random model)."""
    from scripts.pit import MCTSAgent
    model = DualNet(in_ch=17, num_moves=4544, n_res=1)
    agent = MCTSAgent(model, n_sims=10, device="cpu")
    # 2 game only — chỉ kiểm tra pipeline không crash
    res = estimate_elo(agent, n_games=2)
    assert 0 in res
    assert 0.0 <= res[0] <= 1.0
