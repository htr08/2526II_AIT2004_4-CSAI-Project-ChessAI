"""Đánh giá DualNet/PolicyNet trên các vị trí FEN: value score và top-5 nước đi.

Usage:
    python scripts/fen_eval.py                          # uses checkpoints/best_dual.pt
    python scripts/fen_eval.py --checkpoint checkpoints/best_policy.pt --model policy
    python scripts/fen_eval.py --checkpoint checkpoints/best_dual.pt   --model dual

Output per position:
    - Value score  (DualNet only): how the model evaluates the position [-1=Black wins, +1=White wins]
    - Top-5 moves : highest-probability legal moves ranked by policy head
"""

import argparse
import sys
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.board import encode_board_with_meta
from src.model import DualNet, PolicyNet
from src.vocab import load_or_build_move2idx

# ---------------------------------------------------------------------------
# Classic positions to evaluate
# ---------------------------------------------------------------------------
POSITIONS = [
    {
        "name": "Starting position",
        "fen":  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "note": "Expected: near 0.0 (balanced)",
    },
    {
        "name": "Sicilian Defence (after 1.e4 c5)",
        "fen":  "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "note": "Expected: slight White advantage (~+0.1 to +0.2)",
    },
    {
        "name": "Ruy Lopez (after 1.e4 e5 2.Nf3 Nc6 3.Bb5)",
        "fen":  "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "note": "Expected: slight White advantage",
    },
    {
        "name": "Scandinavian Defence (after 1.e4 d5)",
        "fen":  "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        "note": "Expected: slight White advantage",
    },
    {
        "name": "Queen's Gambit (after 1.d4 d5 2.c4)",
        "fen":  "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
        "note": "Expected: roughly balanced",
    },
    {
        "name": "White winning — Queen + Rook vs King",
        "fen":  "7k/8/8/8/8/8/6QR/7K w - - 0 1",
        "note": "Expected: strongly positive (White winning ~+1.0)",
    },
    {
        "name": "Black winning — Queen + Rook vs King",
        "fen":  "7K/8/8/8/8/8/6qr/7k b - - 0 1",
        "note": "Expected: strongly negative (Black winning ~-1.0)",
    },
    {
        "name": "Stalemate-like — King and pawn endgame",
        "fen":  "8/8/8/8/8/k7/p7/K7 b - - 0 1",
        "note": "Expected: negative (Black has advantage)",
    },
]


def load_model(checkpoint: str, model_type: str, num_moves: int, device: str):
    if model_type == "dual":
        model = DualNet(in_ch=17, num_moves=num_moves).to(device)
    else:
        model = PolicyNet(in_channels=17, num_moves=num_moves).to(device)

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def idx_to_uci(idx2move: dict, idx: int) -> str:
    return idx2move.get(idx, f"<unk:{idx}>")


def evaluate_position(model, board: chess.Board, idx2move: dict, model_type: str, device: str):
    tensor = encode_board_with_meta(board).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_type == "dual":
            policy_logits, value = model(tensor)
            value_score = value.item()
        else:
            policy_logits = model(tensor)
            value_score = None

    legal_indices = [
        idx for move in board.legal_moves
        if (uci := move.uci()) in {v: k for k, v in idx2move.items()}
        for idx in [next((k for k, v in idx2move.items() if v == uci), None)]
        if idx is not None
    ]

    probs = F.softmax(policy_logits[0], dim=0)

    if legal_indices:
        legal_probs = [(idx, probs[idx].item()) for idx in legal_indices]
        legal_probs.sort(key=lambda x: x[1], reverse=True)
        top5 = legal_probs[:5]
    else:
        top5 = []

    return value_score, top5


def main():
    parser = argparse.ArgumentParser(description="FEN position evaluator")
    parser.add_argument("--checkpoint", default="checkpoints/best_dual.pt")
    parser.add_argument("--model", choices=["dual", "policy"], default="dual")
    parser.add_argument("--device", default=None, help="cpu or cuda (auto-detect if omitted)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not Path(args.checkpoint).exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    move2idx = load_or_build_move2idx()
    idx2move = {v: k for k, v in move2idx.items()}
    num_moves = max(move2idx.values()) + 1

    model = load_model(args.checkpoint, args.model, num_moves, device)
    print(f"Loaded {args.model} model from '{args.checkpoint}'  (vocab={num_moves})\n")
    print("=" * 70)

    for pos in POSITIONS:
        board = chess.Board(pos["fen"])
        value_score, top5 = evaluate_position(model, board, idx2move, args.model, device)

        print(f"\n{pos['name']}")
        print(f"  FEN  : {pos['fen']}")
        print(f"  Note : {pos['note']}")
        if value_score is not None:
            bar = "#" * int(abs(value_score) * 20)
            side = "White" if value_score > 0 else "Black" if value_score < 0 else "="
            print(f"  Score: {value_score:+.4f}  [{side}]  |{bar:<20}|")
        else:
            print(f"  Score: N/A (PolicyNet has no value head)")

        if top5:
            moves_str = "  ".join(
                f"{idx2move.get(idx, '?')}({p:.1%})" for idx, p in top5
            )
            print(f"  Top-5: {moves_str}")
        else:
            print(f"  Top-5: (no legal moves mapped in vocab)")

    print("\n" + "=" * 70)
    print("Interpretation:")
    print("  Score +1.0 = model thinks White wins decisively")
    print("  Score  0.0 = model thinks position is balanced")
    print("  Score -1.0 = model thinks Black wins decisively")


if __name__ == "__main__":
    main()
