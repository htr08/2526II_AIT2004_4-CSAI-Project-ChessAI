"""Tests cho adjudication (chống self-play toàn hòa) và ELO tracking.

Không cần torch — chỉ dùng python-chess + math thuần.
"""
import chess
import pytest

from src.search.evaluation import adjudicate_result, evaluate_board
from src.search.training.elo import (
    expected_score,
    score_from_pit,
    elo_diff_from_score,
    update_elo,
)


# ---------- adjudication ----------

def test_adjudicate_start_is_draw():
    """Vị trí đầu cân bằng → trong margin → hòa."""
    assert adjudicate_result(chess.Board()) == 0.0


def test_adjudicate_white_up_queen():
    """Trắng hơn hẳn 1 hậu → chấm trắng thắng (+1)."""
    # Đen mất hậu (không có hậu đen trên bàn)
    board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert adjudicate_result(board) == 1.0


def test_adjudicate_black_up_queen():
    """Đen hơn hẳn 1 hậu → chấm đen thắng (-1)."""
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    assert adjudicate_result(board) == -1.0


def test_adjudicate_checkmate():
    """Checkmate vẫn ra đúng dấu (fool's mate: trắng bị chiếu hết)."""
    board = chess.Board()
    for mv in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        board.push_uci(mv)
    assert board.is_checkmate()
    assert adjudicate_result(board) == -1.0  # trắng thua


def test_adjudicate_margin_respected():
    """Hơn đúng 1 tốt (100cp) không vượt margin mặc định 100 → hòa.
    Nhưng với margin nhỏ hơn thì thành thắng."""
    # Trắng dư 1 tốt: thêm tốt trắng ở d4 so với đối xứng. Dùng vị trí đơn giản.
    board = chess.Board("4k3/8/8/8/3P4/8/8/4K3 w - - 0 1")  # chỉ vua + 1 tốt trắng
    s = evaluate_board(board)
    assert s > 0  # trắng hơn
    # với margin lớn (vượt score) → hòa; margin nhỏ → thắng
    assert adjudicate_result(board, margin=10_000) == 0.0
    assert adjudicate_result(board, margin=10) == 1.0


# ---------- ELO ----------

def test_expected_score_symmetry():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1900, 1500) > 0.9
    assert expected_score(1500, 1900) < 0.1


def test_score_from_pit():
    assert score_from_pit(6, 2, 2) == pytest.approx(0.7)   # (6 + 1) / 10
    assert score_from_pit(0, 0, 10) == pytest.approx(0.5)  # toàn hòa = 0.5
    assert score_from_pit(0, 0, 0) == pytest.approx(0.5)   # không ván nào


def test_elo_diff_basic():
    assert elo_diff_from_score(0.5) == pytest.approx(0.0, abs=1e-6)
    assert elo_diff_from_score(0.75) == pytest.approx(190.8, abs=1.0)
    # đối xứng quanh 0.5
    assert elo_diff_from_score(0.6) == pytest.approx(-elo_diff_from_score(0.4), abs=1e-6)


def test_elo_diff_clamped():
    """score=1.0 / 0.0 không được trả về inf."""
    import math
    d_hi = elo_diff_from_score(1.0)
    d_lo = elo_diff_from_score(0.0)
    assert math.isfinite(d_hi) and d_hi > 0
    assert math.isfinite(d_lo) and d_lo < 0


def test_elo_monotonic():
    diffs = [elo_diff_from_score(s) for s in (0.3, 0.4, 0.5, 0.6, 0.7)]
    assert diffs == sorted(diffs)  # tăng dần theo score


def test_update_elo_accept_and_reject():
    # Accept: ELO của model giữ lại = candidate_elo
    r1 = update_elo(0.0, new_wins=7, old_wins=1, draws=2, accepted=True)
    assert r1["elo_diff"] > 0
    assert r1["current_elo"] == pytest.approx(r1["candidate_elo"])

    # Reject: giữ nguyên ELO cũ dù candidate có diff
    r2 = update_elo(r1["current_elo"], new_wins=4, old_wins=6, draws=0, accepted=False)
    assert r2["current_elo"] == pytest.approx(r1["current_elo"])


def test_all_draws_gives_zero_progress():
    """Toàn hòa → diff = 0 → ELO không đổi (đúng: pit không có thông tin)."""
    r = update_elo(50.0, new_wins=0, old_wins=0, draws=10, accepted=False)
    assert r["elo_diff"] == pytest.approx(0.0, abs=1e-6)
    assert r["current_elo"] == pytest.approx(50.0)
