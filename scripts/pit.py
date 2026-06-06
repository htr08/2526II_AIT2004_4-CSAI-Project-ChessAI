"""Arena: so sánh hai checkpoint DualNet, thăng cấp best_dual.pt nếu win_rate vượt ngưỡng."""

import argparse
import shutil
from pathlib import Path

import chess
import torch
from tqdm import tqdm

from src.mcts import mcts_search_puct
from src.model import DualNet
from src.vocab import load_or_build_move2idx

NUM_MOVES  = 4544
BEST_CKPT  = "checkpoints/best_dual.pt"


# ── Agent ────────────────────────────────────────────────────────────────────

class MCTSAgent:
    """Agent MCTS dùng DualNet value head để đánh giá (không random rollout)."""

    def __init__(self, model: DualNet, n_sims: int = 50, device: str = "cpu"):
        self.model    = model.eval().to(device)
        self.n_sims   = n_sims
        self.device   = device
        self.move2idx = load_or_build_move2idx()

    def select_move(self, board: chess.Board) -> chess.Move:
        return mcts_search_puct(board, self.model, self.move2idx,
                                n_sims=self.n_sims, device=self.device)


def _load_agent(ckpt_path: str | None, n_sims: int, device: str) -> MCTSAgent:
    model = DualNet(in_ch=17, num_moves=NUM_MOVES)
    if ckpt_path and Path(ckpt_path).exists():
        model.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True)
        )
    else:
        print(f"  [warn] {ckpt_path} not found — using random weights")
    return MCTSAgent(model, n_sims=n_sims, device=device)


# ── Game logic ────────────────────────────────────────────────────────────────

def play_game(white: MCTSAgent, black: MCTSAgent, max_moves: int = 200) -> str:
    """Chơi 1 ván đến kết thúc, trả về '1-0', '0-1' hoặc '1/2-1/2'."""
    board  = chess.Board()
    agents = {chess.WHITE: white, chess.BLACK: black}
    for _ in range(max_moves):
        if board.is_game_over():
            break
        board.push(agents[board.turn].select_move(board))
    return board.result() if board.is_game_over() else "1/2-1/2"


# ── Arena ─────────────────────────────────────────────────────────────────────

def pit(
    new_ckpt:  str,
    old_ckpt:  str,
    n_games:   int   = 10,
    n_sims:    int   = 50,
    threshold: float = 0.55,
    device:    str   = "cpu",
    best_ckpt: str   = BEST_CKPT,
) -> bool:
    """So sánh new vs old qua n_games ván, xen kẽ màu. Thăng cấp nếu win_rate > threshold."""
    print(f"\nArena  n_games={n_games}  n_sims={n_sims}  threshold={threshold:.0%}")
    print(f"  NEW: {new_ckpt}")
    print(f"  OLD: {old_ckpt}")

    new_agent = _load_agent(new_ckpt, n_sims, device)
    old_agent = _load_agent(old_ckpt, n_sims, device)

    wins = draws = losses = 0

    for i in tqdm(range(n_games), desc="Arena", unit="game"):
        new_is_white = (i % 2 == 0)
        white, black = (new_agent, old_agent) if new_is_white else (old_agent, new_agent)
        result = play_game(white, black)

        if result == "1/2-1/2":
            draws += 1
        elif (result == "1-0") == new_is_white:
            # new won as White  OR  old won as White but new was Black (→ old="0-1")
            wins += 1
        else:
            losses += 1

        tqdm.write(f"  game {i+1:>3d}: {result}  (new={'W' if new_is_white else 'B'})  "
                   f"running: W={wins} D={draws} L={losses}")

    win_rate = (wins + 0.5 * draws) / n_games
    print(f"\nFinal  W={wins}  D={draws}  L={losses}  win_rate={win_rate:.1%}  "
          f"(threshold={threshold:.0%})")

    if win_rate > threshold:
        Path(best_ckpt).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(new_ckpt, best_ckpt)
        print(f"PROMOTED  →  {best_ckpt}")
        return True

    print("Old model retained.")
    return False


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arena: compare two DualNet checkpoints")
    parser.add_argument("--new",       default="checkpoints/candidate_dual.pt",
                        help="Newly trained checkpoint to challenge")
    parser.add_argument("--old",       default=BEST_CKPT,
                        help="Current best checkpoint (default: best_dual.pt)")
    parser.add_argument("--n_games",   type=int,   default=10,
                        help="Number of games (use ≥20 for reliable win-rate estimates)")
    parser.add_argument("--n_sims",    type=int,   default=50,
                        help="MCTS simulations per move")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Win-rate threshold for promotion (default: 0.55)")
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    pit(
        new_ckpt=args.new,
        old_ckpt=args.old,
        n_games=args.n_games,
        n_sims=args.n_sims,
        threshold=args.threshold,
        device=args.device,
    )
