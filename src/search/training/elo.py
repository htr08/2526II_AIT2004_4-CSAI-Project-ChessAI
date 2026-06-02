"""
elo.py
------
Ước lượng ELO tương đối cho self-play loop.

Ý tưởng: mỗi vòng pit (candidate vs current) cho ta một tỉ số (score) của
candidate. Từ score suy ra chênh lệch ELO giữa candidate và current bằng
công thức ELO chuẩn. Cộng dồn qua các vòng (chỉ khi accept) → đường ELO
"tăng dần" của model giữ lại.

Đây là ELO TƯƠNG ĐỐI (so với chính mình lúc đầu = 0), không phải ELO tuyệt đối
so với người/Stockfish — muốn ELO tuyệt đối thì benchmark với Stockfish.
"""
from __future__ import annotations

import math


def expected_score(rating_a: float, rating_b: float) -> float:
    """Xác suất A thắng B theo ELO (kỳ vọng điểm của A, 0..1)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def score_from_pit(new_wins: int, old_wins: int, draws: int) -> float:
    """
    Tỉ số của candidate trong pit: thắng=1, hòa=0.5, thua=0.
    Trả về trong [0, 1]. Nếu không có ván nào → 0.5 (không có thông tin).
    """
    games = new_wins + old_wins + draws
    if games <= 0:
        return 0.5
    return (new_wins + 0.5 * draws) / games


def elo_diff_from_score(score: float, clamp: float = 1e-4) -> float:
    """
    Chênh lệch ELO (candidate - current) suy từ score.
    score=0.5 → 0; score=0.75 → ~+191; score=1.0 → +inf (đã clamp).

    Clamp tránh chia 0 / log(0) khi score = 0 hoặc 1.
    """
    s = min(max(score, clamp), 1.0 - clamp)
    return -400.0 * math.log10(1.0 / s - 1.0)


def update_elo(
    current_elo: float,
    new_wins: int,
    old_wins: int,
    draws: int,
    accepted: bool,
) -> dict:
    """
    Tính ELO sau một vòng pit.

    Trả về dict:
        score:          tỉ số candidate (0..1)
        elo_diff:       chênh lệch ELO candidate so với current
        candidate_elo:  ELO ước lượng của candidate
        current_elo:    ELO model GIỮ LẠI sau vòng này
                        (= candidate_elo nếu accept, ngược lại giữ nguyên)
    """
    score = score_from_pit(new_wins, old_wins, draws)
    diff = elo_diff_from_score(score)
    candidate_elo = current_elo + diff
    kept = candidate_elo if accepted else current_elo
    return {
        "score": score,
        "elo_diff": diff,
        "candidate_elo": candidate_elo,
        "current_elo": kept,
    }


if __name__ == "__main__":
    # Sanity demo
    for s in (0.0, 0.25, 0.5, 0.55, 0.75, 1.0):
        print(f"score {s:.2f} → elo_diff {elo_diff_from_score(s):+.1f}")
    print()
    elo = 0.0
    for it, (nw, ow, dr, acc) in enumerate(
        [(6, 2, 2, True), (5, 5, 0, False), (7, 1, 2, True)], 1
    ):
        r = update_elo(elo, nw, ow, dr, acc)
        elo = r["current_elo"]
        print(f"iter {it}: score={r['score']:.2f} diff={r['elo_diff']:+.0f} "
              f"accepted={acc} → kept_elo={elo:+.0f}")
