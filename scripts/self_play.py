"""Sinh dữ liệu self-play: chạy n game PUCT-MCTS, lưu (state, policy, value) vào .pt."""

import argparse
import json
from pathlib import Path

import chess
import torch
from tqdm import tqdm

from src.board import encode_board_with_meta
from src.mcts import run_puct_search, select_move_with_temperature
from src.model import DualNet
from src.opening_book import OpeningBook

NUM_MOVES = 4544
VOCAB_PATH = "data/processed/move2idx.json"


def play_one_game(model, move2idx, n_sims=50, device="cpu", book=None):
    """Chơi 1 game self-play, trả về list (board_tensor, policy_vec, value_target)."""
    board = chess.Board()
    game_data = []  # list of (tensor, policy)
    move_count = 0

    while not board.is_game_over():
        # Opening book: skip MCTS, use uniform policy for book moves
        if book is not None:
            book_move = book.get_move(board)
            if book_move:
                policy = torch.zeros(NUM_MOVES)
                idx = move2idx.get(book_move.uci(), -1)
                if idx >= 0:
                    policy[idx] = 1.0
                game_data.append((encode_board_with_meta(board), policy))
                board.push(book_move)
                continue

        # PUCT search with policy prior + Dirichlet noise at root (AlphaZero)
        root = run_puct_search(board, model, move2idx, n_sims=n_sims,
                               add_noise=True, device=device)

        # Build policy target from normalized visit counts
        policy = torch.zeros(NUM_MOVES)
        total_visits = sum(c.visits for c in root.children)
        if total_visits > 0:
            for child in root.children:
                idx = move2idx.get(child.move.uci(), -1)
                if idx >= 0:
                    policy[idx] = child.visits / total_visits

        game_data.append((encode_board_with_meta(board), policy))

        # Temperature sampling: explore early moves, exploit later → game diversity
        move = select_move_with_temperature(root, temperature=1.0, move_count=move_count)
        board.push(move)
        move_count += 1

    # Retroactively assign value targets from each player's perspective.
    # outcome is +1 if White wins, -1 if Black wins, 0 for draw.
    # Move i is played by White (i even) or Black (i odd).
    # From the current player's view: White's moves get +outcome, Black's get -outcome.
    result = board.result()
    outcome = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)
    records = [
        (state, policy, float(outcome * ((-1) ** i)))
        for i, (state, policy) in enumerate(game_data)
    ]
    return records


def generate_games(model, n_games=100, n_sims=50, out_dir="data/selfplay", out_path=None, device="cpu", book=None):
    """Chạy n_games ván self-play, lưu toàn bộ (state, policy, value) vào file .pt."""
    move2idx = json.load(open(VOCAB_PATH))

    model.eval()
    model.to(device)

    all_states, all_policies, all_values = [], [], []

    for _ in tqdm(range(n_games), desc="Self-play games"):
        records = play_one_game(model, move2idx, n_sims=n_sims, device=device, book=book)
        for state, policy, value in records:
            all_states.append(state)
            all_policies.append(policy)
            all_values.append(value)

    if out_path is None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(out_dir) / f"selfplay_{n_games}games_{n_sims}sims.pt"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "X": torch.stack(all_states),          # (N, 17, 8, 8)
            "policy": torch.stack(all_policies),   # (N, 4544)
            "value": torch.tensor(all_values, dtype=torch.float32),  # (N,)
        },
        out_path,
    )
    n_pos = len(all_states)
    print(f"Saved {n_pos:,} positions from {n_games} games -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-play data generation")
    parser.add_argument("--n_games",    type=int,   default=100,           help="Number of games to play")
    parser.add_argument("--n_sims",     type=int,   default=50,            help="MCTS simulations per move")
    parser.add_argument("--out_dir",    type=str,   default="data/selfplay")
    parser.add_argument("--out",        type=str,   default=None,          help="Exact output path; overrides auto-generated name in --out_dir")
    parser.add_argument("--checkpoint", type=str,   default=None,          help="Path to DualNet checkpoint (.pt)")
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--book",       type=str,   default=None, help="Path to Polyglot .bin opening book")
    args = parser.parse_args()

    model = DualNet(in_ch=17, num_moves=NUM_MOVES)
    if args.checkpoint:
        if Path(args.checkpoint).exists():
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
            print(f"Loaded checkpoint: {args.checkpoint}")
        else:
            print(f"Checkpoint '{args.checkpoint}' not found — using random weights (bootstrap run)")
    else:
        print("No checkpoint supplied — using randomly initialized DualNet")

    book = OpeningBook(args.book) if args.book else None
    generate_games(
        model,
        n_games=args.n_games,
        n_sims=args.n_sims,
        out_dir=args.out_dir,
        out_path=args.out,
        device=args.device,
        book=book,
    )
    if book:
        book.close()
