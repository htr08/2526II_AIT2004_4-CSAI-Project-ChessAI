"""Xây dựng và load từ điển UCI 4544 nước đi — dùng chung bởi DualNet, MCTS và parse_pgn."""

import chess
import json
from pathlib import Path

VOCAB_PATH = "data/processed/move2idx.json"
IDX2MOVE_PATH = "data/processed/idx2move.json"


def build_move2idx() -> dict:
    """
    Sinh toàn bộ action space cố định theo UCI.
    4032 nước thông thường + promotion variants cho rank 6->7 (trắng) và rank 1->0 (đen).
    """
    move2idx = {}
    idx = 0
    for from_sq in range(64):
        from_rank = from_sq // 8
        for to_sq in range(64):
            if from_sq == to_sq:
                continue
            to_rank = to_sq // 8
            uci = chess.square_name(from_sq) + chess.square_name(to_sq)
            move2idx[uci] = idx
            idx += 1
            is_white_promo = (from_rank == 6 and to_rank == 7)
            is_black_promo = (from_rank == 1 and to_rank == 0)
            if is_white_promo or is_black_promo:
                for promo in ["q", "r", "b", "n"]:
                    move2idx[uci + promo] = idx
                    idx += 1
    return move2idx


def load_or_build_move2idx(vocab_path: str = VOCAB_PATH) -> dict:
    v_path = Path(vocab_path)
    if v_path.exists():
        with open(v_path, encoding="utf-8") as f:
            return json.load(f)

    print("Building move vocabulary...")
    move2idx = build_move2idx()
    v_path.parent.mkdir(parents=True, exist_ok=True)
    with open(v_path, "w", encoding="utf-8") as f:
        json.dump(move2idx, f)
    print(f"Vocabulary: {len(move2idx)} moves -> {vocab_path}")
    return move2idx


def load_idx2move(idx2move_path: str = IDX2MOVE_PATH) -> dict[int, str]:
    """Load mapping index → chuỗi UCI dùng lúc inference."""
    path = Path(idx2move_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{idx2move_path} not found. Run scripts/build_vocab.py first."
        )
    with open(path, encoding="utf-8") as f:
        # JSON keys are always strings; convert back to int for direct indexing.
        return {int(k): v for k, v in json.load(f).items()}
