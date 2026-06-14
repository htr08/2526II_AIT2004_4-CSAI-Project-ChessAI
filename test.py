# test.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "Model_dir" / "UIA_model(Khanh)"))

from Model_dir.khanh.architecture import Predictor
import chess

if __name__ == "__main__":
    print("[test] Initializing model...")
    predictor = Predictor()   # không cần truyền model nữa

    board = chess.Board()

    print("\n[test] --- get_value_prob() ---")
    vp = predictor.get_value_prob(board)
    print(f"  {vp}")

    print("\n[test] --- get_top_moves(5) ---")
    for move, prob in predictor.get_top_moves(board, n=5):
        print(f"  {move.uci()}  {prob:.3%}")