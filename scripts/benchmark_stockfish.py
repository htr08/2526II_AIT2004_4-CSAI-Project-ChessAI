"""
benchmark_stockfish.py
----------------------
Đấu bot mình với Stockfish ở depth thấp để ước lượng độ mạnh.

Cần Stockfish binary. Download: https://stockfishchess.org/download/
Windows: --stockfish stockfish.exe
Linux/Mac: --stockfish /usr/local/bin/stockfish (hoặc path tới binary)

Usage:
    python scripts/benchmark_stockfish.py \
        --model models/best.pt --stockfish stockfish.exe \
        --games 10 --stockfish-depth 2 --search mcts --simulations 200
"""
from __future__ import annotations

import argparse
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import chess
import chess.engine
import torch

from src.model.network import PolicyValueNet
from src.search.minimax import search_best_move
from src.search.mcts import search_best_move_mcts


def load_model(path: str, device: str) -> PolicyValueNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = PolicyValueNet(
        channels=cfg.get("channels", 128),
        n_res_blocks=cfg.get("n_res_blocks", 3),
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def bot_move(board, model, search_type, depth, simulations, device):
    if search_type == "minimax":
        move, _ = search_best_move(board, depth=depth, model=model, device=device)
    elif search_type == "mcts":
        move, _ = search_best_move_mcts(
            board, model, num_simulations=simulations, device=device
        )
    else:
        raise ValueError(search_type)
    return move


def play_one_game(
    engine: chess.engine.SimpleEngine,
    model,
    bot_plays_white: bool,
    stockfish_depth: int,
    search_type: str,
    depth: int,
    simulations: int,
    device: str,
    max_moves: int = 200,
) -> str:
    board = chess.Board()
    moves = 0
    while not board.is_game_over(claim_draw=True) and moves < max_moves:
        if (board.turn == chess.WHITE) == bot_plays_white:
            mv = bot_move(board, model, search_type, depth, simulations, device)
        else:
            result = engine.play(board, chess.engine.Limit(depth=stockfish_depth))
            mv = result.move
        if mv is None:
            break
        board.push(mv)
        moves += 1
    return board.result(claim_draw=True), moves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--stockfish", required=True, help="Path tới Stockfish binary")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--stockfish-depth", type=int, default=2)
    p.add_argument("--search", choices=["minimax", "mcts"], default="minimax")
    p.add_argument("--depth", type=int, default=3, help="Minimax depth của bot")
    p.add_argument("--simulations", type=int, default=200, help="MCTS sims của bot")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bench] device={device}")

    model = load_model(args.model, device)
    print(f"[bench] model loaded từ {args.model}")

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    print(f"[bench] stockfish launched: depth={args.stockfish_depth}")

    bot_wins = 0
    sf_wins = 0
    draws = 0
    total_moves = 0

    t0 = time.time()
    for g in range(args.games):
        bot_white = g % 2 == 0
        result, moves = play_one_game(
            engine,
            model,
            bot_plays_white=bot_white,
            stockfish_depth=args.stockfish_depth,
            search_type=args.search,
            depth=args.depth,
            simulations=args.simulations,
            device=device,
        )
        total_moves += moves
        if result == "1-0":
            if bot_white:
                bot_wins += 1
            else:
                sf_wins += 1
        elif result == "0-1":
            if bot_white:
                sf_wins += 1
            else:
                bot_wins += 1
        else:
            draws += 1
        print(
            f"  game {g+1}/{args.games}: bot={'W' if bot_white else 'B'}  "
            f"{result} ({moves} moves)  "
            f"running score → bot {bot_wins} - {sf_wins} sf, draws {draws}"
        )

    engine.quit()
    dt = time.time() - t0

    print("\n=== Benchmark Results ===")
    print(f"Total games: {args.games}  ({dt:.0f}s, ~{total_moves/args.games:.0f} moves/game)")
    print(f"Bot wins:      {bot_wins}")
    print(f"Stockfish:     {sf_wins}")
    print(f"Draws:         {draws}")
    score = bot_wins + 0.5 * draws
    print(f"Bot score:     {score}/{args.games} ({100*score/args.games:.1f}%)")
    # ELO ước lượng đơn giản (so với Stockfish depth N có ELO ~1500-2000 tùy depth)
    sf_elo_estimate = {1: 1320, 2: 1500, 3: 1700, 4: 1900, 5: 2100}.get(
        args.stockfish_depth, 1500 + 200 * args.stockfish_depth
    )
    if args.games > 0 and score > 0 and score < args.games:
        import math
        win_rate = score / args.games
        elo_diff = -400 * math.log10(1 / win_rate - 1)
        print(f"Stockfish depth-{args.stockfish_depth} ≈ {sf_elo_estimate} ELO")
        print(f"Bot ≈ {sf_elo_estimate + elo_diff:.0f} ELO (ước lượng thô)")


if __name__ == "__main__":
    main()
