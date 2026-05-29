"""
play_cli.py
-----------
CLI demo: chơi cờ với bot.

Usage:
    python scripts/play_cli.py --model models/best.pt --color white --search mcts --simulations 200
    python scripts/play_cli.py --model models/best.pt --color black --search minimax --depth 3

Input nước đi dạng UCI: e2e4, g1f3, e7e8q (promotion), ...
Hoặc gõ 'quit' để thoát, 'undo' để lùi nước, 'fen' để in FEN hiện tại.
"""
from __future__ import annotations

import argparse
import sys
import pathlib

# Cho phép chạy script trực tiếp mà không cần install package
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import chess
import torch

from src.model.network import PolicyValueNet
from src.search.minimax import search_best_move
from src.search.mcts import search_best_move_mcts


def print_board(board: chess.Board, perspective_white: bool = True) -> None:
    """In bàn cờ ra terminal."""
    print()
    if perspective_white:
        print(board)
    else:
        # Flip cho người chơi đen
        flipped = "\n".join(reversed(str(board).split("\n")))
        print(flipped)
    print()


def load_model(path: str, device: str) -> PolicyValueNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Hỗ trợ cả checkpoint dict {"model_state": ...} và state_dict thuần
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


def bot_move(
    board: chess.Board,
    model: PolicyValueNet | None,
    search_type: str,
    depth: int,
    simulations: int,
    device: str,
) -> chess.Move:
    if search_type == "minimax":
        move, score = search_best_move(board, depth=depth, model=model, device=device)
        print(f"[bot] minimax depth={depth} → {move.uci()}  (eval={score})")
        return move
    elif search_type == "mcts":
        if model is None:
            raise ValueError("MCTS yêu cầu phải có model")
        move, _ = search_best_move_mcts(
            board, model, num_simulations=simulations, device=device
        )
        print(f"[bot] mcts sims={simulations} → {move.uci()}")
        return move
    else:
        raise ValueError(f"Unknown search type: {search_type}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="Path tới .pt checkpoint (optional với minimax)")
    p.add_argument("--color", choices=["white", "black"], default="white",
                   help="Màu của NGƯỜI chơi")
    p.add_argument("--search", choices=["minimax", "mcts"], default="minimax")
    p.add_argument("--depth", type=int, default=3, help="Minimax depth")
    p.add_argument("--simulations", type=int, default=200, help="MCTS simulations")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[play_cli] device={device}")

    model = None
    if args.model:
        model = load_model(args.model, device)
        print(f"[play_cli] loaded model from {args.model}")
    elif args.search == "mcts":
        print("ERROR: --model required for MCTS search")
        sys.exit(1)

    board = chess.Board()
    human_color = chess.WHITE if args.color == "white" else chess.BLACK
    print(f"Bạn chơi {'WHITE' if human_color else 'BLACK'}, bot chơi {'BLACK' if human_color else 'WHITE'}")
    print("Gõ nước đi UCI (vd 'e2e4'), 'undo', 'fen', hoặc 'quit'.")
    print_board(board, perspective_white=(human_color == chess.WHITE))

    while not board.is_game_over(claim_draw=True):
        if board.turn == human_color:
            cmd = input("> Move: ").strip().lower()
            if cmd in ("quit", "q", "exit"):
                print("Bye!")
                return
            if cmd == "undo":
                if len(board.move_stack) >= 2:
                    board.pop()
                    board.pop()
                    print("[undo 2 plies]")
                    print_board(board, perspective_white=(human_color == chess.WHITE))
                continue
            if cmd == "fen":
                print(board.fen())
                continue
            try:
                move = chess.Move.from_uci(cmd)
            except ValueError:
                print("UCI không hợp lệ, vd: 'e2e4' hoặc 'e7e8q'")
                continue
            if move not in board.legal_moves:
                print("Nước đi không hợp lệ trên position hiện tại")
                continue
            board.push(move)
        else:
            move = bot_move(
                board, model, args.search, args.depth, args.simulations, device
            )
            board.push(move)

        print_board(board, perspective_white=(human_color == chess.WHITE))

    # Game over
    result = board.result(claim_draw=True)
    if board.is_checkmate():
        # board.turn là side-to-move (kẻ thua) → bên thắng là phía còn lại
        winner = "Đen" if board.turn == chess.WHITE else "Trắng"
        print(f"Game over — {winner} thắng (checkmate). Kết quả: {result}")
    elif board.is_stalemate():
        print(f"Game over — Pat. Kết quả: {result}")
    else:
        print(f"Game over. Kết quả: {result}")


if __name__ == "__main__":
    main()
