"""Parse file PGN Lichess thành tensor (X, y) hoặc (X, y, v) lưu dưới dạng .pt."""

import chess.pgn
import torch
import os
from pathlib import Path
from tqdm import tqdm
from src.board import encode_board_with_meta
from src.vocab import load_or_build_move2idx

MIN_ELO = 1800
MAX_POSITIONS = 200_000

def _result_to_value(result: str, turn: bool) -> float:
    """Convert PGN result string to value from current player's perspective."""
    if result == "1-0":
        return 1.0 if turn == chess.WHITE else -1.0
    if result == "0-1":
        return -1.0 if turn == chess.WHITE else 1.0
    return 0.0  # draw or unknown


def parse_pgn(pgn_path, out_path, with_value: bool = False):
    if not os.path.exists(pgn_path):
        print(f"❌ Không tìm thấy file PGN tại: {pgn_path}")
        return

    positions, labels, values = [], [], []
    move2idx = load_or_build_move2idx()

    print(f"🔄 Đang mở file PGN và trích xuất dữ liệu...")
    with open(pgn_path, encoding="utf-8") as f, tqdm(total=MAX_POSITIONS, desc="Parsing positions") as pbar:
        while len(positions) < MAX_POSITIONS:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            headers = game.headers
            w_elo = int(headers.get("WhiteElo", "0") or "0")
            b_elo = int(headers.get("BlackElo", "0") or "0")

            # Lọc elo cao thủ
            if min(w_elo, b_elo) < MIN_ELO:
                continue

            result = headers.get("Result", "*")
            board = game.board()
            before = len(positions)

            for move in game.mainline_moves():
                if len(positions) >= MAX_POSITIONS:
                    break

                # Mã hóa bàn cờ sang 17 channels xịn
                t = encode_board_with_meta(board)
                idx = move2idx.get(move.uci(), -1)

                if idx >= 0:
                    positions.append(t)
                    labels.append(idx)
                    if with_value:
                        values.append(_result_to_value(result, board.turn))
                board.push(move)

            # Cập nhật thanh tiến trình chuẩn theo số thế cờ tăng lên
            pbar.update(len(positions) - before)

    if not positions:
        print("❌ Không thu hoạch được dữ liệu nào — Hãy kiểm tra lại bộ lọc ELO hoặc đường dẫn file.")
        return

    print("🔄 Đang đóng gói dữ liệu và lưu vào ổ cứng...")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    data = {"X": torch.stack(positions), "y": torch.tensor(labels, dtype=torch.long)}
    if with_value:
        data["v"] = torch.tensor(values, dtype=torch.float32)
    torch.save(data, out_path)
    print(f"🎉 XONG! Đã lưu {len(positions)} thế cờ sạch vào -> {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", default="data/raw/lichess_elite_2022-03.pgn")
    ap.add_argument("--out", default="data/processed/train.pt")
    ap.add_argument("--min_elo", type=int, default=MIN_ELO)
    ap.add_argument("--max_pos", type=int, default=MAX_POSITIONS)
    ap.add_argument("--with_value", action="store_true",
                    help="Also save game outcome per position (needed for DualNet supervised training)")
    args = ap.parse_args()
    MIN_ELO = args.min_elo
    MAX_POSITIONS = args.max_pos
    parse_pgn(args.pgn, args.out, with_value=args.with_value)