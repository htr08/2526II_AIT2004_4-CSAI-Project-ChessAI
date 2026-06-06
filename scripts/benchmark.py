"""Benchmark model với Stockfish theo skill level, ước tính ELO và ghi log CSV.

Plays n_games alternating colors. Uses Stockfish at the given skill level if
available, otherwise falls back to random-move play.

ELO estimate: opponent_elo + 400 * log10(score / (1 - score))
  where score = (wins + 0.5 * draws) / n_games

Supported architectures:
  --arch dual    DualNet + PUCT-MCTS  (default, self-play model: best_dual.pt)
  --arch policy  PolicyNet + Minimax  (supervised model: best_policy.pt)

Usage:
    # Self-play DualNet (original)
    python scripts/benchmark.py --model checkpoints/best_dual.pt --arch dual \\
           --skill 2 --n_games 10 --log logs/elo_history.csv --iter 1

    # Supervised PolicyNet
    python scripts/benchmark.py --model checkpoints/best_policy.pt --arch policy \\
           --skill 2 --n_games 10 --log logs/elo_supervised.csv --iter 1 --depth 4
"""

import argparse
import math
import os
import random
from pathlib import Path

import chess
import torch
from tqdm import tqdm

from src.mcts import mcts_search_puct
from src.model import DualNet, PolicyNet
from src.agent import MinimaxAgent, DualNetMinimaxAgent
from src.dataset import log_elo
from src.vocab import load_or_build_move2idx

_MOVE2IDX = load_or_build_move2idx()

NUM_MOVES = 4544
# Approximate ELO for Stockfish skill levels and random play
_SKILL_ELO = {0: 800, 1: 900, 2: 1000, 3: 1100, 4: 1200, 5: 1350,
              6: 1500, 7: 1650, 8: 1800, 9: 1900, 10: 2000}
_RANDOM_ELO = 500


def _resolve_stockfish() -> str:
    """Locate a runnable Stockfish binary across platforms.

    Order: PATH (Linux/Kaggle typically `apt-get install stockfish`), then the
    bundled per-OS binary under bin/. The committed bin/stockfish.exe is a Windows
    build, so on Linux we must NOT use it — prefer PATH or bin/stockfish instead.
    """
    import shutil

    on_path = shutil.which("stockfish")
    if on_path:
        return on_path

    base_dir = Path(__file__).resolve().parents[1]
    name = "stockfish.exe" if os.name == "nt" else "stockfish"
    bundled = base_dir / "bin" / name
    if bundled.exists():
        return str(bundled)

    # Last resort: bare name, let popen_uci raise if truly absent.
    return "stockfish"


def _random_move(board: chess.Board) -> chess.Move:
    return random.choice(list(board.legal_moves))


def _mcts_move(model: DualNet, board: chess.Board, n_sims: int, device: str) -> chess.Move:
    return mcts_search_puct(board, model, _MOVE2IDX, n_sims=n_sims, device=device)


def _play_game(
    move_fn,
    opponent_fn,
    model_is_white: bool,
    max_moves: int = 200,
) -> str:
    board = chess.Board()
    for _ in range(max_moves):
        if board.is_game_over():
            break
        if (board.turn == chess.WHITE) == model_is_white:
            move = move_fn(board)
        else:
            move = opponent_fn(board)
        board.push(move)
    return board.result() if board.is_game_over() else "1/2-1/2"


def _load_model(model_path: str, arch: str, n_sims: int, device: str, depth: int):
    """Return a move function (board -> move) for the chosen architecture."""
    if arch == "policy":
        agent = MinimaxAgent(depth=depth, model_path=model_path if Path(model_path).exists() else None)
        if not Path(model_path).exists():
            print(f"[warn] {model_path} not found — using pure minimax (no policy ordering)")
        return agent.select_move

    if arch == "dual_minimax":
        agent = DualNetMinimaxAgent(depth=depth, model_path=model_path, device=device)
        print(f"Arch: dual_minimax  depth={depth}  (DualNet value head + policy ordering)")
        return agent.select_move

    # arch == "dual"
    model = DualNet(in_ch=17, num_moves=NUM_MOVES)
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        print(f"Loaded {model_path}")
    else:
        print(f"[warn] {model_path} not found — benchmarking random DualNet weights")
    model.eval().to(device)
    return lambda board: _mcts_move(model, board, n_sims, device)


def benchmark(
    model_path: str,
    arch: str = "dual",
    skill: int = 2,
    n_games: int = 10,
    n_sims: int = 50,
    depth: int = 4,
    device: str = "cpu",
):
    move_fn = _load_model(model_path, arch, n_sims, device, depth)
    if arch == "dual":
        print(f"Arch: dual  MCTS sims={n_sims}")
    elif arch != "dual_minimax":  # dual_minimax prints inside _load_model
        print(f"Arch: {arch}  minimax depth={depth}")

    engine = None
    try:
        import chess.engine
        sf_cmd = _resolve_stockfish()
        engine = chess.engine.SimpleEngine.popen_uci(sf_cmd)
        engine.configure({"Skill Level": skill})
        limit = chess.engine.Limit(depth=5)
        opponent_fn = lambda b: engine.play(b, limit).move
        opponent_elo = _SKILL_ELO.get(skill, 1000)
        print(f"Opponent: Stockfish skill={skill} (~ELO {opponent_elo})")
    except Exception as e:
        print(f"Stockfish unavailable ({e}) — using random opponent (~ELO {_RANDOM_ELO})")
        opponent_fn = _random_move
        opponent_elo = _RANDOM_ELO

    wins = draws = losses = 0
    try:
        for i in tqdm(range(n_games), desc="Benchmark", unit="game"):
            model_white = (i % 2 == 0)
            result = _play_game(move_fn, opponent_fn, model_white)
            if result == "1/2-1/2":
                draws += 1
            elif (result == "1-0") == model_white:
                wins += 1
            else:
                losses += 1
            tqdm.write(
                f"  game {i+1:>3d}: {result}  model={'W' if model_white else 'B'}"
                f"  running W={wins} D={draws} L={losses}"
            )
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass

    score = (wins + 0.5 * draws) / max(n_games, 1)
    clamped = max(0.01, min(0.99, score))
    est_elo = round(opponent_elo + 400 * math.log10(clamped / (1.0 - clamped)))
    print(f"\nResult  W={wins} D={draws} L={losses}  score={score:.1%}  est_ELO={est_elo}")
    return score, est_elo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark chess model and log ELO")
    parser.add_argument("--model",   default="checkpoints/best_dual.pt")
    parser.add_argument("--arch",    default="dual", choices=["dual", "policy", "dual_minimax"],
                        help="dual=DualNet+MCTS (self-play), policy=PolicyNet+Minimax (supervised)")
    parser.add_argument("--skill",   type=int,   default=2,   help="Stockfish skill level (0-20)")
    parser.add_argument("--n_games", type=int,   default=10)
    parser.add_argument("--n_sims",  type=int,   default=50,  help="MCTS simulations per move (dual only)")
    parser.add_argument("--depth",   type=int,   default=4,   help="Minimax depth (policy only)")
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--log",     default="logs/elo_history.csv", help="CSV for ELO history")
    parser.add_argument("--iter",    type=int,   default=0,   help="Iteration number for logging")
    args = parser.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    score, est_elo = benchmark(
        model_path=args.model,
        arch=args.arch,
        skill=args.skill,
        n_games=args.n_games,
        n_sims=args.n_sims,
        depth=args.depth,
        device=args.device,
    )
    log_elo(args.log, args.iter, round(score, 4), est_elo)
    print(f"Logged to {args.log}")
