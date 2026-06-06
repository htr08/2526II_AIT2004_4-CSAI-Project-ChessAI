"""
play.py — chơi cờ với AI trên terminal.

Dùng:
  python play.py                          # Human (Trắng) vs AI (Đen), depth=4
  python play.py --side black             # Human (Đen) vs AI (Trắng)
  python play.py --model checkpoints/best_policy.pt --depth 3
  python play.py --ai-vs-ai              # xem AI tự đánh nhau
"""

import argparse
import chess
from src.agent import MinimaxAgent


def print_board(board: chess.Board) -> None:
    print()
    print(board)
    print()


def human_move(board: chess.Board) -> chess.Move:
    while True:
        try:
            uci = input("Nước đi của bạn (UCI, vd e2e4): ").strip()
            if uci.lower() in ("quit", "exit", "q"):
                raise SystemExit(0)
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                return move
            print("  Nước không hợp lệ, thử lại.")
        except ValueError:
            print("  Định dạng sai, nhập kiểu e2e4.")


def play(human_color: chess.Color, agent: MinimaxAgent) -> None:
    board = chess.Board()

    while not board.is_game_over():
        print_board(board)
        side = "Trắng" if board.turn == chess.WHITE else "Đen"

        if board.turn == human_color:
            print(f"Lượt của bạn ({side})")
            move = human_move(board)
        else:
            print(f"AI đang suy nghĩ ({side})...")
            move = agent.select_move(board)
            print(f"  AI đi: {move.uci()}")

        board.push(move)

    print_board(board)
    result = board.result()
    if result == "1-0":
        winner = "Trắng thắng"
    elif result == "0-1":
        winner = "Đen thắng"
    else:
        winner = "Hòa"
    print(f"Kết quả: {result} — {winner}")


def ai_vs_ai(agent_w: MinimaxAgent, agent_b: MinimaxAgent) -> None:
    board = chess.Board()
    move_num = 0

    while not board.is_game_over():
        print_board(board)
        agent = agent_w if board.turn == chess.WHITE else agent_b
        side = "Trắng" if board.turn == chess.WHITE else "Đen"
        move = agent.select_move(board)
        move_num += 1
        print(f"  [{move_num}] {side}: {move.uci()}")
        board.push(move)

    print_board(board)
    print(f"Kết quả: {board.result()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="checkpoints/best_policy.pt",
                    help="Đường dẫn checkpoint (bỏ qua nếu chưa có)")
    ap.add_argument("--depth",   type=int, default=4, help="Độ sâu minimax")
    ap.add_argument("--side",    choices=["white", "black"], default="white",
                    help="Bạn chơi màu nào (mặc định: trắng)")
    ap.add_argument("--book",    default=None, help="File opening book .bin (tuỳ chọn)")
    ap.add_argument("--ai-vs-ai", action="store_true", help="AI tự đánh nhau")
    args = ap.parse_args()

    import os
    model_path = args.model if os.path.exists(args.model) else None
    if not model_path:
        print(f"[warn] Không tìm thấy {args.model} — AI chạy không có model (minimax thuần).")

    agent = MinimaxAgent(depth=args.depth, model_path=model_path, book_path=args.book)

    if args.ai_vs_ai:
        ai_vs_ai(agent, agent)
    else:
        human_color = chess.WHITE if args.side == "white" else chess.BLACK
        play(human_color, agent)


if __name__ == "__main__":
    main()
