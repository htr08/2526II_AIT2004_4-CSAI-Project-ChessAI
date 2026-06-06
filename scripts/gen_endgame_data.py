"""Sinh vị trí tàn cuộc tổng hợp có kết quả rõ ràng để calibrate value head.

These positions are clearly winning (+1) or losing (-1) for the side to move,
helping the value head learn to reach the extremes of [-1, +1].

Usage:
    python scripts/gen_endgame_data.py --out data/processed/endgame.pt
    python scripts/gen_endgame_data.py --out data/processed/endgame.pt --augment 50

The output .pt file has the same format as train_with_value.pt:
    {"X": FloatTensor(N,17,8,8), "y": LongTensor(N,), "v": FloatTensor(N,)}

Mix into training:
    python src/train.py --mode dual_supervised --data data/processed/endgame.pt --value_weight 2.0
"""

import argparse
import sys
from pathlib import Path

import chess
import torch
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.board import encode_board_with_meta
from src.vocab import load_or_build_move2idx

# ---------------------------------------------------------------------------
# Hand-crafted endgame positions: (fen, value_for_side_to_move)
#   +1.0 = side to move is winning decisively
#   -1.0 = side to move is losing decisively
#    0.0 = drawn
# ---------------------------------------------------------------------------
ENDGAME_POSITIONS = [
    # ── White to move, winning ──────────────────────────────────────────
    ("7k/8/8/8/8/8/6QR/7K w - - 0 1", +1.0),   # Q+R vs lone King
    ("6k1/8/8/8/8/8/5RR1/6K1 w - - 0 1", +1.0), # R+R vs lone King
    ("8/8/8/8/8/3k4/3Q4/3K4 w - - 0 1", +1.0),  # Q vs lone King
    ("8/8/8/8/8/8/PPP5/K1k5 w - - 0 1", +1.0),  # passed pawns about to queen
    ("1k6/8/1K6/8/8/8/8/1R6 w - - 0 1", +1.0),  # R+K vs K (Lucena-like)
    ("8/8/8/8/8/2k5/2P5/2K5 w - - 0 1", +1.0),  # K+P vs K, White to push
    ("8/8/8/8/8/8/6PP/5KRk w - - 0 1", +1.0),   # R+pawns vs K
    ("r7/8/8/8/8/8/8/R6K w - - 0 1", +1.0),     # same Rook power but King safety
    ("8/8/8/8/7P/7K/8/7k w - - 0 1", +1.0),     # K+P vs K (opposition won)

    # ── Black to move, losing (= side-to-move value is -1) ──────────────
    ("7K/8/8/8/8/8/6qr/7k b - - 0 1", +1.0),   # mirror: Black Q+R vs White K, Black wins
    ("8/8/8/8/8/3K4/3q4/3k4 b - - 0 1", +1.0),  # Black Q vs White lone K, Black wins
    ("1K6/8/1k6/8/8/8/8/1r6 b - - 0 1", +1.0),  # mirror R+K vs K
    ("8/8/8/8/8/2K5/2p5/2k5 b - - 0 1", +1.0),  # K+p vs K, Black to push

    # ── Drawn positions (value = 0) ──────────────────────────────────────
    ("8/8/8/8/8/k7/p7/K7 b - - 0 1",  0.0),    # K+p vs K, stalemate trap
    ("8/8/8/8/8/8/8/k1K5 w - - 0 1",  0.0),    # lone kings
    ("8/8/4k3/8/8/4K3/8/8 w - - 0 1",  0.0),    # lone kings (center)
    ("8/8/8/3k4/8/3K4/8/8 b - - 0 1",  0.0),    # lone kings

    # ── Losing positions (side to move is losing, value = -1) ───────────
    ("7k/8/8/8/8/8/6QR/6K1 b - - 0 1", -1.0),  # Black King vs White Q+R
    ("6k1/8/8/8/8/8/5RR1/5K2 b - - 0 1", -1.0), # Black King vs White R+R
    ("8/8/8/8/8/3K4/3Q4/3k4 b - - 0 1", -1.0),  # Black King vs White Q
    ("8/8/8/8/8/3k4/3Q4/3K4 b - - 0 1", -1.0),  # same, explicit
    ("1k6/8/1K6/8/8/8/8/1R6 b - - 0 1", -1.0),  # Black King vs White R+K

    # ── Middlegame imbalances ────────────────────────────────────────────
    # White up a Queen
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2NP4/PPP2PPP/R1BQKBNR w KQkq - 0 4", +1.0),
    # Black up material — just a clearly lost White position
    ("8/8/4k3/8/3q4/8/8/4K3 w - - 0 1", -1.0),  # White King vs Black Q
]


def positions_to_tensors(positions, move2idx):
    """Convert list of (fen, value) to (X, y, v) tensors."""
    Xs, ys, vs = [], [], []
    skipped = 0

    for fen, value in positions:
        board = chess.Board(fen)
        if board.is_game_over():
            skipped += 1
            continue

        legal = list(board.legal_moves)
        if not legal:
            skipped += 1
            continue

        X = encode_board_with_meta(board)

        # Pick the best-looking legal move (or just first if no match in vocab)
        y = None
        for move in legal:
            idx = move2idx.get(move.uci())
            if idx is not None:
                y = idx
                break
        if y is None:
            skipped += 1
            continue

        Xs.append(X)
        ys.append(y)
        vs.append(value)

    if skipped:
        print(f"  Skipped {skipped} positions (game over or no vocab match)")

    return (
        torch.stack(Xs),
        torch.tensor(ys, dtype=torch.long),
        torch.tensor(vs, dtype=torch.float32),
    )


def augment_by_random_moves(positions, n_extra: int, move2idx: dict) -> list:
    """For each position, play n random legal moves and add child positions with same value."""
    extra = []
    for fen, value in positions:
        board = chess.Board(fen)
        for _ in range(n_extra):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break
            move = random.choice(legal)
            board.push(move)
            # Value flips when turn changes (relative to side to move)
            extra.append((board.fen(), -value))
            board.pop()
    return extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/processed/endgame.pt")
    parser.add_argument("--augment", type=int, default=20,
                        help="Random child positions per base position (0 to disable)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    move2idx = load_or_build_move2idx()
    print(f"Vocab size: {max(move2idx.values()) + 1}")

    positions = list(ENDGAME_POSITIONS)
    print(f"Base positions: {len(positions)}")

    if args.augment > 0:
        extra = augment_by_random_moves(positions, args.augment, move2idx)
        positions.extend(extra)
        print(f"After augmentation: {len(positions)}")

    X, y, v = positions_to_tensors(positions, move2idx)
    print(f"Final dataset: {len(X)} positions")
    print(f"  Value distribution: "
          f"+1={( v > 0.5).sum().item()}  "
          f"0={(v.abs() < 0.1).sum().item()}  "
          f"-1={(v < -0.5).sum().item()}")

    torch.save({"X": X, "y": y, "v": v}, args.out)
    print(f"Saved → {args.out}")
    print()
    print("Next step — mix into training:")
    print(f"  python src/train.py --mode dual_supervised --data {args.out} --value_weight 2.0 --epochs 5")


if __name__ == "__main__":
    main()
