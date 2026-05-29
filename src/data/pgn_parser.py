"""
pgn_parser.py
-------------
Parse PGN games -> list of (board_fen, move_uci, result) samples.

Hỗ trợ cả .pgn (text) và .pgn.zst (zstandard compressed).
Lưu output thành .pt file để load nhanh trong Dataset.
"""
from __future__ import annotations

import io
import pathlib
from typing import Iterable, Optional

import chess
import chess.pgn
from tqdm import tqdm

# torch chỉ cần khi gọi parse_to_file (torch.save) - lazy import để
# generator parse_games dùng được khi chưa cài torch.


def open_pgn(path):
    """Mở file PGN. Tự detect .zst (compressed) hay text thường."""
    path = pathlib.Path(path)
    if path.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as e:
            raise ImportError(
                "Để đọc file .zst cần cài zstandard: pip install zstandard"
            ) from e
        dctx = zstd.ZstdDecompressor()
        return io.TextIOWrapper(
            dctx.stream_reader(open(path, "rb")), encoding="utf-8"
        )
    return open(path, "r", encoding="utf-8")


def parse_games(
    pgn_path,
    max_games=None,
    min_rating=0,
    min_moves=10,
    skip_draws=False,
):
    """
    Generator yield (board_fen, move_uci, result_from_white_pov).

    Args:
        pgn_path: đường dẫn file PGN
        max_games: chỉ parse tối đa max_games (None = tất cả)
        min_rating: chỉ lấy game có cả WhiteElo & BlackElo >= min_rating
        min_moves: bỏ game ngắn hơn min_moves nửa-nước
        skip_draws: nếu True thì bỏ qua game hòa

    Yields (fen, uci_move, result):
        - fen: FEN string của position TRƯỚC khi đi nước này
        - uci_move: nước đi tiếp theo
        - result: 1 nếu trắng thắng, -1 nếu đen thắng, 0 nếu hòa
    """
    handle = open_pgn(pgn_path)
    games_parsed = 0
    games_kept = 0

    pbar = tqdm(desc="Parsing games", unit="game")
    while True:
        try:
            game = chess.pgn.read_game(handle)
        except Exception:
            continue
        if game is None:
            break

        games_parsed += 1
        pbar.update(1)

        if max_games is not None and games_kept >= max_games:
            break

        if min_rating > 0:
            try:
                w_elo = int(game.headers.get("WhiteElo", "0"))
                b_elo = int(game.headers.get("BlackElo", "0"))
                if w_elo < min_rating or b_elo < min_rating:
                    continue
            except ValueError:
                continue

        result_str = game.headers.get("Result", "*")
        if result_str == "1-0":
            result = 1
        elif result_str == "0-1":
            result = -1
        elif result_str == "1/2-1/2":
            if skip_draws:
                continue
            result = 0
        else:
            continue

        board = game.board()
        move_count = 0
        samples_this_game = []
        for move in game.mainline_moves():
            fen = board.fen()
            uci = move.uci()
            samples_this_game.append((fen, uci, result))
            board.push(move)
            move_count += 1

        if move_count < min_moves:
            continue

        games_kept += 1
        for sample in samples_this_game:
            yield sample

    pbar.close()
    handle.close()
    print(f"[pgn_parser] parsed {games_parsed} games, kept {games_kept}")


def parse_to_file(
    pgn_path,
    output_path,
    max_games=None,
    min_rating=0,
    min_moves=10,
    skip_draws=False,
):
    """Parse PGN -> lưu .pt file dạng dict."""
    fens = []
    moves = []
    results = []

    for fen, uci, result in parse_games(
        pgn_path,
        max_games=max_games,
        min_rating=min_rating,
        min_moves=min_moves,
        skip_draws=skip_draws,
    ):
        fens.append(fen)
        moves.append(uci)
        results.append(result)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import torch  # lazy import
    torch.save(
        {
            "fens": fens,
            "moves": moves,
            "results": results,
            "meta": {
                "n_samples": len(fens),
                "max_games": max_games,
                "min_rating": min_rating,
                "min_moves": min_moves,
                "skip_draws": skip_draws,
                "source": str(pgn_path),
            },
        },
        output_path,
    )
    print(
        f"[pgn_parser] saved {len(fens):,} (fen, move) samples -> {output_path}"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Parse PGN file -> .pt dataset cho training."
    )
    p.add_argument("--input", required=True, help="Đường dẫn .pgn hoặc .pgn.zst")
    p.add_argument("--output", required=True, help="Output .pt file")
    p.add_argument("--max-games", type=int, default=None)
    p.add_argument("--min-rating", type=int, default=0)
    p.add_argument("--min-moves", type=int, default=10)
    p.add_argument("--skip-draws", action="store_true")
    args = p.parse_args()

    parse_to_file(
        args.input,
        args.output,
        max_games=args.max_games,
        min_rating=args.min_rating,
        min_moves=args.min_moves,
        skip_draws=args.skip_draws,
    )
