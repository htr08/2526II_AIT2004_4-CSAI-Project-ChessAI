"""Wrapper script cho src.data.pgn_parser."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    from src.data.pgn_parser import parse_to_file
    import argparse

    p = argparse.ArgumentParser(description="Parse PGN file → .pt dataset")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
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
