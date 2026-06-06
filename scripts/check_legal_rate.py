"""Kiểm tra tỉ lệ nước đi hợp lệ của model — mục tiêu >80% sau 5-10 epoch.

If <50% after 5 epochs: the bug is in encode_board_with_meta() or move2idx,
not the model architecture.

Usage:
    python scripts/check_legal_rate.py
    python scripts/check_legal_rate.py --checkpoint checkpoints/best_policy.pt --n 2000
"""

import argparse
import random
import chess
import torch
from src.model import PolicyNet
from src.board import encode_board_with_meta
from src.vocab import load_or_build_move2idx

CHECKPOINT = "checkpoints/best_policy.pt"


def random_board(max_moves: int = 20) -> chess.Board:
    """Play up to max_moves random plies from the start to get varied positions."""
    board = chess.Board()
    for _ in range(random.randint(0, max_moves)):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(random.choice(moves))
    return board


def check_legal_rate(
    model: PolicyNet,
    idx2move: dict[int, str],
    n: int = 1000,
    device: str = "cpu",
) -> float:
    model.eval()
    legal_count = 0
    for _ in range(n):
        board = random_board()
        tensor = encode_board_with_meta(board).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)[0]
        best_idx = logits.argmax().item()
        uci = idx2move.get(best_idx, "")
        try:
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                legal_count += 1
        except ValueError:
            pass
    rate = legal_count / n * 100
    print(f"Legal rate: {legal_count}/{n} = {rate:.1f}%")
    if rate >= 80:
        print("PASS (>= 80%)")
    elif rate >= 50:
        print("MARGINAL (50-80%) — keep training")
    else:
        print("FAIL (<50%) — check encode_board_with_meta() and move2idx")
    return rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    move2idx = load_or_build_move2idx()
    idx2move = {v: k for k, v in move2idx.items()}

    model = PolicyNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    print(f"Loaded checkpoint: {args.checkpoint}")

    check_legal_rate(model, idx2move, n=args.n, device=device)
